from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import requests


SYSTEM_PROMPT = """You are a professional Audio-Video Consistency Evaluation Expert. Your core evaluation principle is based on 'World Construction': Consider audio as the 3D ground truth of the world, and video as a limited 2D observation window. Your task is to assess the quality of audio-video generation by strictly following this 5-layer reasoning pipeline:
1. Blind Audio Analysis: Establish the auditory world cognition independently without visual cues.
2. Deaf Video Analysis: Analyze visual events independently without auditory cues.
3. Auditory-Visual Verification (A2V): Using audio as ground truth, verify if visual events align physically and temporally.
4. Visual-Auditory Verification (V2A): Using video as an anchor, check for missing or physically incorrect auditory feedback.
5. World Construction Verdict: Synthesize findings. Crucially, distinguish between:
   - **In-Window (Diegetic)**: Sounds from visible sources (must sync perfectly).
   - **Off-Screen (Diegetic)**: Sounds from the implied 3D world outside the frame (must match spatial logic).
   - **Narrative (Non-Diegetic)**: BGM or Voiceover (must match mood/semantic context).
   Finally, output scores for Macro-Semantic Alignment, Temporal Synchronization, and Fine-Grained Event Correspondence."""

USER_PROMPT = "<video><audio>Please analyze the audio-video consistency of this material and provide a rating."

SCORE_KEYS = (
    "macro_semantic_alignment",
    "temporal_synchronicity",
    "fine_grained_event_correspondence",
)
SCORE_KEY_ALIASES = {
    "macro_semantic_alignment": (
        "macro_semantic_alignment",
        "macro_semantic",
        "semantic_alignment",
        "macro-semantic alignment",
        "macro semantic alignment",
    ),
    "temporal_synchronicity": (
        "temporal_synchronicity",
        "temporal_synchronization",
        "temporal_sync",
        "temporal",
        "temporal synchronicity",
        "temporal synchronization",
    ),
    "fine_grained_event_correspondence": (
        "fine_grained_event_correspondence",
        "fine_grained_correspondence",
        "fine_grained",
        "fine-grained event correspondence",
        "fine grained event correspondence",
    ),
}

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg", ".mpg", ".m4v")
AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma")


def _file_url(path: str | Path) -> str:
    return Path(path).expanduser().resolve().as_uri()


def find_or_extract_audio(video_path: str | Path, preferred_audio_path: str | Path | None = None) -> tuple[str, bool]:
    """Return an audio path and whether it is a temporary file owned by the caller."""
    if preferred_audio_path:
        audio = Path(preferred_audio_path)
        if audio.exists():
            return str(audio), False

    video = Path(video_path)
    base = video.with_suffix("")
    for ext in AUDIO_EXT:
        sidecar = Path(str(base) + ext)
        if sidecar.exists():
            return str(sidecar), False

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    cmd = [
        os.environ.get("FFMPEG_BIN", "ffmpeg"),
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        tmp_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_path, True


def build_messages(video_path: str | Path, audio_path: str | Path) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": _file_url(video_path)}},
                {"type": "audio_url", "audio_url": {"url": _file_url(audio_path)}},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]


def build_swift_record(video_path: str | Path, audio_path: str | Path, sample_id: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": ""},
        ],
        "videos": [str(Path(video_path).expanduser().resolve())],
        "audios": [str(Path(audio_path).expanduser().resolve())],
    }
    if sample_id is not None:
        record["id"] = sample_id
    return record


def parse_rm_scores(text: str) -> dict[str, float]:
    tail = text.split("</think>", 1)[1] if "</think>" in text else text
    payload = None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", tail):
        try:
            candidate, _ = decoder.raw_decode(tail[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and all(key in candidate for key in SCORE_KEYS):
            payload = candidate
            break
    if payload is None:
        match = re.search(r"\{.*\}", tail, flags=re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None
    if payload is not None and not all(key in payload for key in SCORE_KEYS):
        normalized = dict(payload)
        normalized_payload = {str(key).lower().replace("-", "_").replace(" ", "_"): value for key, value in payload.items()}
        for key in SCORE_KEYS:
            if key in normalized:
                continue
            for alias in SCORE_KEY_ALIASES[key]:
                alias_key = alias.lower().replace("-", "_").replace(" ", "_")
                if alias_key in normalized_payload:
                    normalized[key] = normalized_payload[alias_key]
                    break
        payload = normalized if all(key in normalized for key in SCORE_KEYS) else None
    if payload is None:
        payload = {}
        for key in SCORE_KEYS:
            for alias in SCORE_KEY_ALIASES[key]:
                quoted = re.findall(
                    rf'["\']{re.escape(alias)}["\']\s*:\s*["\']?(-?(?:\d+(?:\.\d*)?|\.\d+))',
                    tail,
                    flags=re.I,
                )
                label = re.findall(
                    rf'(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])\s*[:=]\s*["\']?(-?(?:\d+(?:\.\d*)?|\.\d+))',
                    tail,
                    flags=re.I,
                )
                matches = quoted + label
                if matches:
                    payload[key] = matches[-1]
                    break
    if payload is None:
        raise ValueError(f"reward model output does not contain score fields: {text[:500]}")
    scores = {}
    for key in SCORE_KEYS:
        if key not in payload:
            raise KeyError(f"reward model output missing key={key}: {payload}")
        value = float(payload[key])
        scores[key] = max(0.0, min(1.0, value))
    scores["rm_reward_avg"] = sum(scores[k] for k in SCORE_KEYS) / len(SCORE_KEYS)
    return scores


def score_one(
    video_path: str | Path,
    audio_path: str | Path | None = None,
    *,
    endpoint: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    retries: int | None = None,
) -> dict[str, Any]:
    endpoint = endpoint or os.environ.get("QWEN3OMNI_RM_ENDPOINT", "http://127.0.0.1:8010/v1/chat/completions")
    model = model or os.environ.get("QWEN3OMNI_RM_MODEL", "reward_model")
    api_key = api_key if api_key is not None else os.environ.get("QWEN3OMNI_RM_API_KEY", "")
    timeout = float(timeout if timeout is not None else os.environ.get("QWEN3OMNI_RM_TIMEOUT", 600))
    max_tokens = int(max_tokens if max_tokens is not None else os.environ.get("QWEN3OMNI_RM_MAX_TOKENS", 4096))
    temperature = float(temperature if temperature is not None else os.environ.get("QWEN3OMNI_RM_TEMPERATURE", 0.0))
    retries = int(retries if retries is not None else os.environ.get("QWEN3OMNI_RM_RETRIES", 3))

    audio, tmp_audio = find_or_extract_audio(video_path, audio_path)
    try:
        payload = {
            "model": model,
            "messages": build_messages(video_path, audio),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        last_error = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
                if not response.ok:
                    detail = response.text.strip()
                    if len(detail) > 2000:
                        detail = detail[:2000] + "..."
                    raise RuntimeError(
                        f"reward model HTTP {response.status_code} from {endpoint}: {detail}"
                    )
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                scores = parse_rm_scores(content)
                return {
                    "scores": scores,
                    "content": content,
                    "video_path": str(video_path),
                    "audio_path": str(audio),
                    "endpoint": endpoint,
                    "model": model,
                }
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"reward model request failed after {retries + 1} attempts: {last_error}") from last_error
    finally:
        if tmp_audio:
            try:
                os.remove(audio)
            except OSError:
                pass


def score_batch(video_paths: list[str], audio_paths: list[str | None] | None = None, **kwargs) -> list[dict[str, Any]]:
    if audio_paths is None:
        audio_paths = [None] * len(video_paths)
    if len(video_paths) != len(audio_paths):
        raise ValueError(f"video/audio length mismatch: {len(video_paths)} vs {len(audio_paths)}")
    return [score_one(video, audio, **kwargs) for video, audio in zip(video_paths, audio_paths)]
