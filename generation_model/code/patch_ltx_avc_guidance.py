"""Apply AVC-Reward's generation-guidance change to upstream LTX-2.

The modified upstream file is marked at its top. The fetched upstream checkout
contains the LTX-2 license.
"""

from __future__ import annotations

import argparse
from pathlib import Path


NOTICE = (
    "# Modified by AVC-Reward: adds independent video/audio CFG scales. "
    "See the upstream LTX-2 LICENSE.\n"
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one upstream {label} block, found {count}.")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltx-root", type=Path, required=True)
    args = parser.parse_args()

    sampler_path = args.ltx_root / "packages/ltx-trainer/src/ltx_trainer/validation_sampler.py"
    text = sampler_path.read_text(encoding="utf-8")
    if "video_guidance_scale: float | None" in text and "audio_guidance_scale: float | None" in text:
        if not text.startswith(NOTICE):
            raise RuntimeError("LTX-2 already has the guidance fields but lacks the AVC-Reward change notice.")
        print("AVC-Reward guidance patch is already applied.")
        return

    text = replace_once(
        text,
        "    guidance_scale: float = 4.0  # CFG guidance scale\n",
        "    guidance_scale: float = 4.0  # CFG guidance scale (unified fallback)\n"
        "    video_guidance_scale: float | None = None  # Per-modality video CFG\n"
        "    audio_guidance_scale: float | None = None  # Per-modality audio CFG\n",
        "guidance settings",
    )
    text = replace_once(
        text,
        "        cfg_guider = CFGGuider(config.guidance_scale)\n",
        "        video_cfg = config.video_guidance_scale if config.video_guidance_scale is not None else config.guidance_scale\n"
        "        audio_cfg = config.audio_guidance_scale if config.audio_guidance_scale is not None else config.guidance_scale\n"
        "        video_cfg_guider = CFGGuider(video_cfg)\n"
        "        audio_cfg_guider = CFGGuider(audio_cfg)\n",
        "CFG guider setup",
    )
    text = replace_once(
        text,
        "                if cfg_guider.enabled() and v_ctx_neg is not None:\n"
        "                    video_neg = replace(video, context=v_ctx_neg)\n"
        "                    audio_neg = replace(audio, context=a_ctx_neg) if audio is not None else None\n"
        "                    neg_video, neg_audio = x0_model(video=video_neg, audio=audio_neg, perturbations=None)\n\n"
        "                    denoised_video = denoised_video + cfg_guider.delta(pos_video, neg_video)\n"
        "                    if audio is not None and denoised_audio is not None:\n"
        "                        denoised_audio = denoised_audio + cfg_guider.delta(pos_audio, neg_audio)\n",
        "                if (video_cfg_guider.enabled() or audio_cfg_guider.enabled()) and v_ctx_neg is not None:\n"
        "                    video_neg = replace(video, context=v_ctx_neg)\n"
        "                    audio_neg = replace(audio, context=a_ctx_neg) if audio is not None else None\n"
        "                    neg_video, neg_audio = x0_model(video=video_neg, audio=audio_neg, perturbations=None)\n\n"
        "                    if video_cfg_guider.enabled():\n"
        "                        denoised_video = denoised_video + video_cfg_guider.delta(pos_video, neg_video)\n"
        "                    if audio is not None and denoised_audio is not None and audio_cfg_guider.enabled():\n"
        "                        denoised_audio = denoised_audio + audio_cfg_guider.delta(pos_audio, neg_audio)\n",
        "per-modality CFG application",
    )
    text = replace_once(
        text,
        "        if config.guidance_scale != 1.0:\n",
        "        video_cfg = config.video_guidance_scale if config.video_guidance_scale is not None else config.guidance_scale\n"
        "        audio_cfg = config.audio_guidance_scale if config.audio_guidance_scale is not None else config.guidance_scale\n"
        "        if video_cfg != 1.0 or audio_cfg != 1.0:\n",
        "negative-prompt encoding condition",
    )
    sampler_path.write_text(NOTICE + text, encoding="utf-8", newline="\n")
    print(f"Applied AVC-Reward guidance patch to {sampler_path}")


if __name__ == "__main__":
    main()
