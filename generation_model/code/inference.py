"""Generate joint audio-video samples with benchmark-aligned LTX-2 guidance."""

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
    parser.add_argument("--num_frames", type=int, default=None, help="Frame count; defaults to checkpoint parameters (121 for LTX-2)")
    parser.add_argument("--height", type=int, default=None, help="Video height; defaults to checkpoint parameters")
    parser.add_argument("--width", type=int, default=None, help="Video width; defaults to checkpoint parameters")
    parser.add_argument("--frame_rate", type=float, default=None, help="Frame rate; defaults to checkpoint parameters")
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Denoising steps; defaults to checkpoint parameters")
    parser.add_argument("--video_guidance_scale", type=float, default=None, help="Video CFG scale; defaults to checkpoint parameters")
    parser.add_argument("--audio_guidance_scale", type=float, default=None, help="Audio CFG scale; defaults to checkpoint parameters")
    parser.add_argument("--stg_scale", type=float, default=None, help="Set both modality STG scales; defaults to checkpoint parameters")
    parser.add_argument("--video_stg_scale", type=float, default=None, help="Video STG scale; defaults to checkpoint parameters")
    parser.add_argument("--audio_stg_scale", type=float, default=None, help="Audio STG scale; defaults to checkpoint parameters")
    parser.add_argument("--video_rescale_scale", type=float, default=None, help="Video CFG rescale; defaults to checkpoint parameters")
    parser.add_argument("--audio_rescale_scale", type=float, default=None, help="Audio CFG rescale; defaults to checkpoint parameters")
    parser.add_argument("--a2v_guidance_scale", type=float, default=None, help="Audio-to-video modality scale; defaults to checkpoint parameters")
    parser.add_argument("--v2a_guidance_scale", type=float, default=None, help="Video-to-audio modality scale; defaults to checkpoint parameters")
    parser.add_argument("--video_skip_step", type=int, default=None, help="Video guider skip step; defaults to checkpoint parameters")
    parser.add_argument("--audio_skip_step", type=int, default=None, help="Audio guider skip step; defaults to checkpoint parameters")
    parser.add_argument("--video_stg_blocks", type=int, nargs="+", default=None, help="Video STG blocks; defaults to checkpoint parameters")
    parser.add_argument("--audio_stg_blocks", type=int, nargs="+", default=None, help="Audio STG blocks; defaults to checkpoint parameters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", default="./results", help="Output directory for MP4 and WAV")
    parser.add_argument("--dtype", choices=["bf16"], default="bf16", help="Current LTX-2 Pipeline supports BF16 only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_ltx_source()

    import torch
    import torchaudio
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_params
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

    gemma_path = args.gemma_path or os.environ.get("GEMMA_MODEL_PATH", "")
    if not gemma_path:
        raise ValueError("Gemma path required: pass --gemma_path or set GEMMA_MODEL_PATH.")
    if not torch.cuda.is_available():
        raise RuntimeError("LTX-2 generation requires a CUDA GPU; CPU generation is not supported by this entry point.")

    if args.dtype != "bf16":
        raise ValueError("The benchmark-aligned LTX-2 pipeline uses bfloat16, matching 21_grpo.")

    negative_prompt = args.negative_prompt if args.negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT

    params = detect_params(args.model_path)
    height = args.height if args.height is not None else params.stage_1_height
    width = args.width if args.width is not None else params.stage_1_width
    frame_rate = args.frame_rate if args.frame_rate is not None else params.frame_rate
    inference_steps = args.num_inference_steps if args.num_inference_steps is not None else params.num_inference_steps
    num_frames = args.num_frames if args.num_frames is not None else params.num_frames

    def build_guider_params(base, modality: str) -> MultiModalGuiderParams:
        is_video = modality == "video"
        cfg = getattr(args, f"{modality}_guidance_scale")
        stg = getattr(args, f"{modality}_stg_scale")
        if stg is None:
            stg = args.stg_scale if args.stg_scale is not None else base.stg_scale
        rescale = getattr(args, f"{modality}_rescale_scale")
        modality_scale = args.a2v_guidance_scale if is_video else args.v2a_guidance_scale
        skip_step = getattr(args, f"{modality}_skip_step")
        stg_blocks = getattr(args, f"{modality}_stg_blocks")
        return MultiModalGuiderParams(
            cfg_scale=base.cfg_scale if cfg is None else cfg,
            stg_scale=stg,
            rescale_scale=base.rescale_scale if rescale is None else rescale,
            modality_scale=base.modality_scale if modality_scale is None else modality_scale,
            skip_step=base.skip_step if skip_step is None else skip_step,
            stg_blocks=base.stg_blocks if stg_blocks is None else stg_blocks,
        )

    video_guider_params = build_guider_params(params.video_guider_params, "video")
    audio_guider_params = build_guider_params(params.audio_guider_params, "audio")

    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=args.model_path,
        gemma_root=gemma_path,
        loras=(),
        device=torch.device("cuda"),
        quantization=None,
    )

    print(f"Generating: {args.prompt}")
    print(f"  Resolution: {int(height)}x{int(width)}, frames: {int(num_frames)}, fps: {float(frame_rate)}, steps: {int(inference_steps)}")
    print(f"  Video guider: {video_guider_params}")
    print(f"  Audio guider: {audio_guider_params}")
    # The pinned VAE decoder uses operations that may need ordinary tensors
    # even for inference; no_grad avoids inference-tensor/autograd conflicts.
    with torch.no_grad():
        video, audio = pipeline(
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            seed=args.seed,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            frame_rate=float(frame_rate),
            num_inference_steps=int(inference_steps),
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            images=[],
            enhance_prompt=False,
        )

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
