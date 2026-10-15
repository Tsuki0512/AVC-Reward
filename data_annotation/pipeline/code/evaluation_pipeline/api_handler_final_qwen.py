import os
import re
import json
import time
import base64
import requests
from typing import Optional, Tuple, List, Union

try:
    import cv2
except Exception:
    cv2 = None

from .detailed_logger import get_logger

VLLM_API_URL = os.getenv("QWEN_VLLM_API_URL", "")
QWEN_VLLM_MODEL = os.getenv("QWEN_VLLM_MODEL", "")

HEADERS = {
    "Content-Type": "application/json"
}

MAX_RETRIES = 15
RETRY_DELAY = 5

COMPRESSION_STRATEGIES = [
    (1.0, 95),
    (0.75, 85),
    (0.5, 75),
    (0.3, 60)
]

MAX_TOKENS = int(os.getenv("QWEN_MAX_TOKENS", "9192"))
TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.2"))

def _extract_frames_as_data_urls(
    video_path: str,
    target_fps: int = 4,
    max_frames: Optional[int] = None,
    resize_scale: float = 1.0,
    jpeg_quality: int = 95
) -> List[str]:
    data_urls: List[str] = []
    if cv2 is None:
        return data_urls

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return data_urls

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return data_urls

        duration = total_frames / fps
        target_count = int(duration * target_fps)
        if max_frames is not None:
            target_count = min(target_count, max_frames)

        for i in range(target_count):
            idx = int(round(i * fps / target_fps))
            if idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            if isinstance(resize_scale, float) and resize_scale < 1.0:
                width = int(frame.shape[1] * resize_scale)
                height = int(frame.shape[0] * resize_scale)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            ok, buf = cv2.imencode('.jpg', frame, encode_params)

            if not ok:
                continue
            b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
            data_urls.append(f"data:image/jpeg;base64,{b64}")

        cap.release()
        return data_urls
    except Exception as e:
        return []

def _build_vllm_messages(
    prompt: str,
    file_path: Optional[str],
    mime_type: Optional[str],
    resize_scale: Union[float, str] = 1.0,
    jpeg_quality: int = 95
):
    content_blocks = [{"type": "text", "text": prompt}]

    if file_path and os.path.exists(file_path) and mime_type:
        local_path = os.path.abspath(file_path)
        uri = f"file://{local_path}"

        if mime_type == "video/mp4":

            if resize_scale == "VIDEO_FILE":
                content_blocks.append({
                    "type": "video_url",
                    "video_url": {"url": uri}
                })

            else:
                frame_data_urls = _extract_frames_as_data_urls(
                    local_path,
                    target_fps=4,
                    resize_scale=resize_scale,
                    jpeg_quality=jpeg_quality
                )
                if frame_data_urls:
                    for data_url in frame_data_urls:
                        content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
                else:
                    pass

        elif mime_type in ("audio/mp3", "audio/wav"):
            content_blocks.append({"type": "audio_url", "audio_url": {"url": uri}})

        elif mime_type in ("image/jpeg", "image/png"):

            content_blocks.append({"type": "image_url", "image_url": {"url": uri}})

        else:
            pass

    messages = [{"role": "user", "content": content_blocks}]
    return messages

def _extract_text_from_response(data: dict) -> Optional[str]:
    try:
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content")
        if content is None: return None
        if isinstance(content, str): return content
        if isinstance(content, list):
            parts = [seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in content]
            return "".join(parts)
        return str(content)
    except Exception:
        return None

def _strip_think_section(text: str) -> str:
    if not text: return text
    try:
        cleaned = re.sub(r".*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()
    except re.error:
        return text.strip()

def call_llm_api(prompt: str, file_path: str = None, mime_type: str = None) -> Tuple[Optional[dict], Optional[dict]]:
    if not VLLM_API_URL or not QWEN_VLLM_MODEL:
        raise RuntimeError("Set QWEN_VLLM_API_URL and QWEN_VLLM_MODEL for the annotation vLLM service.")
    logger = get_logger()
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    if mime_type == "video/mp4":
        strategies = [("VIDEO_FILE", 0)] + COMPRESSION_STRATEGIES
    else:
        strategies = [(1.0, 95)]

    if file_path:
        pass

    for strat_idx, (scale, quality) in enumerate(strategies):

        if scale == "VIDEO_FILE":
            strategy_name = "Mode: Direct Video File (Server-side processing)"
        else:
            strategy_name = f"Mode: Client-side Frames (Scale={scale}, Q={quality})"

        if strat_idx > 0:
            pass

        messages = _build_vllm_messages(prompt, file_path, mime_type, resize_scale=scale, jpeg_quality=quality)

        payload = {
            "model": QWEN_VLLM_MODEL,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(VLLM_API_URL, headers=HEADERS, data=json.dumps(payload), timeout=300)

                if response.status_code == 200:
                    data = response.json()
                    response_text = _extract_text_from_response(data)

                    if response_text is None:
                        time.sleep(RETRY_DELAY)
                        continue


                    try:
                        usage = data.get("usage", {})
                        token_usage = {
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        }
                    except: pass

                    cleaned_text = _strip_think_section(response_text)
                    if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[7:]
                    if cleaned_text.startswith("```"): cleaned_text = cleaned_text[3:]
                    if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-3]
                    cleaned_text = cleaned_text.strip()

                    unanalyzable_keywords = ["impossible to analyze", "unable to analyze"]
                    if any(k in cleaned_text.lower() for k in unanalyzable_keywords):
                         return {"result": "unable_to_analyze"}, token_usage

                    try:
                        json_response = json.loads(cleaned_text)
                        if logger: logger.log_api_call(prompt=prompt, file_path=file_path, mime_type=mime_type, response=json_response, attempt=attempt+1, **token_usage)
                        return json_response, token_usage
                    except json.JSONDecodeError:

                        match = re.search(r"```json\s*(.*?)\s*```", cleaned_text, re.DOTALL)
                        if match:
                            try:
                                json_response = json.loads(match.group(1).strip())
                                return json_response, token_usage
                            except: pass

                        return {"result": "parse_error"}, token_usage

                elif response.status_code == 400:
                    err_text = response.text

                    if strat_idx < len(strategies) - 1:
                        break
                    else:
                        return None, None

                else:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    else:

                        if scale == "VIDEO_FILE" and strat_idx < len(strategies) - 1:
                            break
                        return None, None

            except Exception as e:
                time.sleep(RETRY_DELAY)

    return None, None
