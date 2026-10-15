"""Generate joint audio-video samples with the AVC-Reward LTX-2 LoRA."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def add_ltx_source() -> Path:
    script_dir = Path(__file__).resolve().parent
    root = Path(os.environ.get("LTX_V2_ROOT", script_dir / "ltx_v2")).expanduser().resolve()
    package_root = root / "packages" if (root / "packages").is_dir() else root
    for package in ("ltx-core", "ltx-pipelines", "ltx-trainer"):
        source = package_root / package / "src"
        if not source.is_dir():
            raise FileNotFoundError(
                f"LTX runtime package is missing: {source}. Run generation_model/code/install_ltx_runtime.sh "
                "or set LTX_V2_ROOT to the pinned LTX-2 checkout."
            )
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Merged LTX-2 checkpoint (.safetensors)")
    parser.add_argument("--gemma_path", default=None, help="Gemma 3 text encoder directory; defaults to GEMMA_MODEL_PATH")
    parser.add_argument("--prompt", required=True, help="Text prompt for joint audio-video generation")
    parser.add_argument("--negative_prompt", default=None, help="Negative prompt (defaults to the pinned LTX-2 value)")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of generated video frames")
    parser.add_argument("--height", type=int, default=None, help="Video height; defaults to checkpoint parameters")
    parser.add_argument("--width", type=int, default=None, help="Video width; defaults to checkpoint parameters")
    parser.add_argument("--frame_rate", type=float, default=None, help="Frame rate; defaults to checkpoint parameters")
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Denoising steps; defaults to checkpoint parameters")
    parser.add_argument("--video_guidance_scale", type=float, default=None, help="Video CFG scale; defaults to checkpoint parameters")
    parser.add_argument("--audio_guidance_scale", type=float, default=None, help="Audio CFG scale; defaults to checkpoint parameters")
    parser.add_argument("--stg_scale", type=float, default=None, help="STG scale; defaults to checkpoint parameters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", default="./results", help="Output directory for MP4 and WAV")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Inference precision")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_ltx_source()

    import torch
    import torchaudio
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.types import Audio
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_params
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_trainer.model_loader import load_embeddings_processor, load_model
    from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler

    gemma_path = args.gemma_path or os.environ.get("GEMMA_MODEL_PATH", "")
    if not gemma_path:
        raise ValueError("Gemma path required: pass --gemma_path or set GEMMA_MODEL_PATH.")
    if not torch.cuda.is_available():
        raise RuntimeError("LTX-2 generation requires a CUDA GPU; CPU generation is not supported by this entry point.")

    device = torch.device("cuda")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    negative_prompt = args.negative_prompt if args.negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT

    print(f"Loading merged model from {args.model_path} ...")
    components = load_model(
        checkpoint_path=args.model_path,
        text_encoder_path=gemma_path,
        device="cpu",
        dtype=dtype,
        with_video_vae_encoder=False,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=True,
        with_vocoder=True,
        with_text_encoder=True,
    )
    embeddings_processor = load_embeddings_processor(checkpoint_path=args.model_path, device="cpu", dtype=dtype)
    model = components.transformer.to(device, dtype=dtype).eval()
    sampler = ValidationSampler(
        transformer=model,
        vae_decoder=components.video_vae_decoder,
        vae_encoder=None,
        text_encoder=components.text_encoder,
        embeddings_processor=embeddings_processor,
        audio_decoder=components.audio_vae_decoder,
        vocoder=components.vocoder,
    )

    params = detect_params(args.model_path)
    height = args.height if args.height is not None else params.stage_1_height
    width = args.width if args.width is not None else params.stage_1_width
    frame_rate = args.frame_rate if args.frame_rate is not None else params.frame_rate
    inference_steps = args.num_inference_steps if args.num_inference_steps is not None else params.num_inference_steps
    video_cfg = args.video_guidance_scale if args.video_guidance_scale is not None else params.video_guider_params.cfg_scale
    audio_cfg = args.audio_guidance_scale if args.audio_guidance_scale is not None else params.audio_guider_params.cfg_scale
    stg_scale = args.stg_scale if args.stg_scale is not None else params.video_guider_params.stg_scale
    stg_blocks = list(params.video_guider_params.stg_blocks)

    config = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        height=int(height),
        width=int(width),
        num_frames=args.num_frames,
        frame_rate=float(frame_rate),
        num_inference_steps=int(inference_steps),
        guidance_scale=float(video_cfg),
        video_guidance_scale=float(video_cfg),
        audio_guidance_scale=float(audio_cfg),
        seed=args.seed,
        condition_image=None,
        generate_audio=True,
        stg_scale=float(stg_scale),
        stg_blocks=stg_blocks if stg_blocks else None,
        stg_mode="stg_av",
    )

    print(f"Generating: {args.prompt}")
    print(f"  Resolution: {int(height)}x{int(width)}, frames: {args.num_frames}, steps: {int(inference_steps)}")
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(args.dtype != "fp32"), dtype=dtype):
        sampler._validate_config(config)
        v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg = sampler._get_prompt_embeddings(config, device)
        video_tools = sampler._create_video_latent_tools(config)
        audio_tools = sampler._create_audio_latent_tools(config)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        video_clean_state = video_tools.create_initial_state(device=device, dtype=dtype)
        audio_clean_state = audio_tools.create_initial_state(device=device, dtype=dtype)
        noiser = GaussianNoiser(generator=generator)
        video_state = noiser(latent_state=video_clean_state, noise_scale=1.0)
        audio_state = noiser(latent_state=audio_clean_state, noise_scale=1.0)

        video_state, audio_state = sampler._run_denoising(
            config=config,
            video_state=video_state,
            audio_state=audio_state,
            video_clean_state=video_clean_state,
            audio_clean_state=audio_clean_state,
            v_ctx_pos=v_ctx_pos,
            a_ctx_pos=a_ctx_pos,
            v_ctx_neg=v_ctx_neg,
            a_ctx_neg=a_ctx_neg,
            device=device,
        )
        video_state = video_tools.unpatchify(video_tools.clear_conditioning(video_state))
        audio_state = audio_tools.unpatchify(audio_tools.clear_conditioning(audio_state))
        video = sampler._decode_video(video_state, device, None)
        video = (video.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(1, 2, 3, 0)
        audio_waveform = sampler._decode_audio(audio_state, device)
        audio_rate = sampler._vocoder.output_sampling_rate if sampler._vocoder is not None else 16000
        audio = Audio(waveform=audio_waveform, sampling_rate=audio_rate) if audio_waveform is not None else None

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_prompt = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.prompt[:60]).strip("._-") or "sample"
    video_path = output_dir / f"{safe_prompt}_seed{args.seed}.mp4"
    encode_video(video, int(round(float(frame_rate))), audio, str(video_path), 1)
    print(f"Video saved: {video_path}")

    if audio is not None and audio.waveform is not None:
        wav = audio.waveform
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 2 and wav.shape[0] > wav.shape[1]:
            wav = wav.transpose(0, 1)
        audio_path = video_path.with_suffix(".wav")
        torchaudio.save(str(audio_path), wav.detach().cpu(), sample_rate=int(audio.sampling_rate))
        print(f"Audio saved: {audio_path}")


if __name__ == "__main__":
    main()
