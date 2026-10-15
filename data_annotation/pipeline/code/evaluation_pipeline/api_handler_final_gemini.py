import os
import json
import time
import base64
import requests

try:
    from .detailed_logger import get_logger
except ImportError:
    from detailed_logger import get_logger

MAX_RETRIES = 30
RETRY_DELAY = 5

GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT", "")

def _encode_file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _build_payload(prompt: str, file_path: str = None, mime_type: str = None) -> dict:
    payload = {"text": prompt}

    if file_path and mime_type:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        b64 = _encode_file_to_base64(file_path)
        mt = mime_type.lower()

        if mt.startswith("video/"):
            payload["video_data"] = b64
        elif mt.startswith("image/"):
            payload["img_data"] = b64
        elif mt.startswith("audio/"):

            payload["audio_data"] = b64
        else:

            payload["file_data"] = b64

        payload["mime_type"] = mime_type
        payload["file_name"] = os.path.basename(file_path)

    return payload

def call_llm_api(prompt: str, file_path: str = None, mime_type: str = None):
    if not GEMINI_ENDPOINT:
        raise RuntimeError("Set GEMINI_ENDPOINT to your authorized Gemini-compatible service endpoint.")
    logger = get_logger()

    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    if logger:
        logger.log_api_call(
            prompt=prompt,
            file_path=file_path,
            mime_type=mime_type,
            response=None,
            error=None,
            attempt=1,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )

    print("=" * 100)
    print(prompt)
    print("=" * 100)
    if file_path:
        pass

    for attempt in range(MAX_RETRIES):
        try:

            payload = _build_payload(prompt, file_path, mime_type)
            response = requests.post(GEMINI_ENDPOINT, json=payload, timeout=120)

            if response.status_code != 200:
                raise RuntimeError(
                    f"Unexpected HTTP status {response.status_code}: {response.text[:500]}"
                )

            try:
                response_json = response.json()
            except Exception as e:
                raise RuntimeError(f"Response is not valid JSON: {e}; raw response: {response.text[:500]}")

            try:
                usage = None
                if isinstance(response_json, dict):

                    usage = response_json.get("usage") or response_json.get("token_usage")
                    if usage is None:
                        usage = response_json.get("usageMetadata")

                if isinstance(usage, dict):

                    input_tokens = usage.get("input_tokens")
                    output_tokens = usage.get("output_tokens")
                    total_tokens = usage.get("total_tokens")

                    if input_tokens is None:
                        input_tokens = usage.get("promptTokenCount", 0)
                    if output_tokens is None:

                        output_tokens = usage.get("candidatesTokenCount", 0)
                    if total_tokens is None:
                        total_tokens = usage.get("totalTokenCount")

                    token_usage["input_tokens"] = input_tokens or 0
                    token_usage["output_tokens"] = output_tokens or 0
                    if isinstance(total_tokens, int) and total_tokens > 0:
                        token_usage["total_tokens"] = total_tokens
                    else:
                        token_usage["total_tokens"] = (
                            token_usage["input_tokens"] + token_usage["output_tokens"]
                        )
                else:
                    pass
            except Exception as token_error:
                pass

            raw_text = None
            parsed_json = None

            if isinstance(response_json, dict) and isinstance(response_json.get("text"), str):
                raw_text = response_json["text"].strip()

            if raw_text is None and isinstance(response_json, dict):
                candidates = response_json.get("candidates")
                if isinstance(candidates, list) and len(candidates) > 0:
                    collected_texts = []
                    for cand in candidates:
                        content = cand.get("content") if isinstance(cand, dict) else None
                        parts = content.get("parts") if isinstance(content, dict) else None
                        if isinstance(parts, list):
                            for p in parts:
                                t = p.get("text") if isinstance(p, dict) else None
                                if isinstance(t, str) and t.strip():
                                    collected_texts.append(t.strip())
                    if collected_texts:

                        raw_text = "\n".join(collected_texts).strip()

            if parsed_json is None and isinstance(response_json, dict) and raw_text is None:
                candidate = dict(response_json)

                for key in [
                    "usage",
                    "token_usage",
                    "usageMetadata",
                    "model",
                    "meta",
                    "mime_type",
                    "file_name",
                    "finishReason",
                    "responseId",
                    "modelVersion",
                ]:
                    candidate.pop(key, None)
                if candidate:
                    parsed_json = candidate

            if raw_text is None and parsed_json is None:
                raw_text = response.text.strip()

            if raw_text is not None:
                print("=" * 100)
                print(raw_text)
                print("=" * 100)

                cleaned_text = raw_text.strip()

                if "```json" in cleaned_text:
                    json_start = cleaned_text.find("```json") + 7
                    json_end = cleaned_text.find("```", json_start)
                    if json_end != -1:
                        cleaned_text = cleaned_text[json_start:json_end].strip()
                    else:
                        cleaned_text = cleaned_text[json_start:].strip()

                elif cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                    if cleaned_text.endswith("```"):
                        cleaned_text = cleaned_text[:-3]

                elif cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]

                cleaned_text = cleaned_text.strip()

                def _extract_first_json_fragment(text: str) -> str:

                    start = text.find("{")
                    if start == -1:
                        return text
                    depth = 0
                    end = start
                    for i, ch in enumerate(text[start:], start=start):
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                end = i
                                break
                    if end >= start:
                        return text[start : end + 1]
                    return text

                if cleaned_text and not cleaned_text.lstrip().startswith("{"):
                    cleaned_text = _extract_first_json_fragment(cleaned_text)

                unable_to_analyze_keywords = [
                    "did not find any discernible faces",
                    "impossible to analyze",
                    "cannot identify",
                    "unable to detect",
                    "no faces found",
                    "cannot determine",
                    "unable to analyze",
                    "no clear faces",
                    "cannot analyze",
                ]

                if any(keyword.lower() in cleaned_text.lower() for keyword in unable_to_analyze_keywords):

                    if "speaker_segments" in prompt.lower():
                        fallback_result = {"speaker_segments": []}
                    elif "is_present" in prompt.lower():
                        fallback_result = {"is_present": False, "confidence": 0.0}
                    elif "emotion" in prompt.lower():
                        fallback_result = {"emotion": "neutral", "confidence": 0.0}
                    else:
                        fallback_result = {"result": "unable_to_analyze", "confidence": 0.0}

                    print("=" * 100)
                    print(json.dumps(fallback_result, ensure_ascii=False, indent=2))
                    print("=" * 100)

                    if logger:
                        logger.log_api_call(
                            prompt, file_path, mime_type, fallback_result, None, attempt + 1,
                            token_usage.get("input_tokens", 0),
                            token_usage.get("output_tokens", 0),
                            token_usage.get("total_tokens", 0),
                        )
                    return fallback_result, token_usage

                parsed_json = json.loads(cleaned_text)

            if parsed_json is None and isinstance(response_json, dict):
                parsed_json = dict(response_json)
                for key in [
                    "usage",
                    "token_usage",
                    "usageMetadata",
                    "model",
                    "meta",
                    "mime_type",
                    "file_name",
                    "finishReason",
                    "responseId",
                    "modelVersion",
                ]:
                    parsed_json.pop(key, None)

            print("=" * 100)
            print(json.dumps(parsed_json, ensure_ascii=False, indent=2))
            print("=" * 100)

            if logger:
                logger.log_api_call(
                    prompt=prompt,
                    file_path=file_path,
                    mime_type=mime_type,
                    response=parsed_json,
                    error=None,
                    attempt=attempt + 1,
                    input_tokens=token_usage.get("input_tokens", 0),
                    output_tokens=token_usage.get("output_tokens", 0),
                    total_tokens=token_usage.get("total_tokens", 0),
                )

            return parsed_json, token_usage

        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing failed on attempt {attempt + 1}: {e}"
            if logger:
                logger.log_api_call(
                    prompt=prompt,
                    file_path=file_path,
                    mime_type=mime_type,
                    response=None,
                    error=error_msg,
                    attempt=attempt + 1,
                    input_tokens=token_usage.get("input_tokens", 0),
                    output_tokens=token_usage.get("output_tokens", 0),
                    total_tokens=token_usage.get("total_tokens", 0),
                )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

        except Exception as e:
            error_msg = f"Unexpected API error on attempt {attempt + 1}: {e}"
            if logger:
                logger.log_api_call(
                    prompt=prompt,
                    file_path=file_path,
                    mime_type=mime_type,
                    response=None,
                    error=error_msg,
                    attempt=attempt + 1,
                    input_tokens=token_usage.get("input_tokens", 0),
                    output_tokens=token_usage.get("output_tokens", 0),
                    total_tokens=token_usage.get("total_tokens", 0),
                )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    final_error = f"API call failed after {MAX_RETRIES} attempts."
    if logger:
        logger.log_error(final_error)
    return None, None
