"""Rewrite annotation labels into structured JSON targets."""

FULL_LABEL_MAPPING = {
    "macro_semantic": {
        "A": "[A] Alignment Good",
        "B": "[B] Narrative Conflict (Critical Story Mismatch)",
        "C": "[C] Emotion/Atmosphere Mismatch",
        "D": "[D] Scene/Environment Conflict",
        "E": "[E] Other Macro Misalignment"
    },
    "temporal_physical": {
        "A": "[A] Physical Match (Synchronized)",
        "B": "[B] Audio Missing (Visible but Silent)",
        "C": "[C] Audio Hallucination / Illogical Off-screen",
        "D": "[D] Timing Deviation",
        "E": "[E] Off-screen Logic Reasonable",
        "F": "[F] No such event",
        "G": "[G] Other Inconsistency",
        "H": "[H] Unable to judge due to Macro Conflict"
    },
    "temporal_speaker": {
        "A": "[A] Physical Match (Lip-sync)",
        "B": "[B] Audio Missing (Mouth moving but silent)",
        "C": "[C] Audio Hallucination (Voice heard, mouth closed)",
        "D": "[D] Timing Deviation (Start/End offset)",
        "E": "[E] Narrative - Voiceover (Reasonable)",
        "F": "[F] No such event",
        "G": "[G] Other Inconsistency",
        "H": "[H] Unable to judge due to Macro Conflict"
    },
    "temporal_music": {
        "A": "[A] Physical Match (Instrument/Dance)",
        "B": "[B] Audio Missing (Instrument/Dance silent)",
        "C": "[C] Audio Hallucination (Music heard, performer still)",
        "D": "[D] Physical Rhythm Mismatch (Off-beat)",
        "E": "[E] Narrative - BGM Reasonable",
        "F": "[F] Narrative - BGM Rhythm Mismatch",
        "G": "[G] No such event",
        "H": "[H] Other Inconsistency",
        "I": "[I] Unable to judge due to Macro Conflict"
    },
    "fine_grained": {
        "A": "[A] Perfect Match",
        "B": "[B] Physical - Material/Timbre Mismatch",
        "C": "[C] Physical - Energy/Force Mismatch",
        "D": "[D] Physical - Space/Distance Mismatch",
        "E": "[E] Morphological - Micro-state Mismatch",
        "F": "[F] Logical - Reaction Contradicts Causality",
        "G": "[G] Failed Temporal Verification",
        "H": "[H] Other Mismatch",
        "I": "[I] Unable to judge due to Macro Conflict"
    }
}


import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List

import requests

QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
QWEN_MODEL_NAME = os.environ.get("QWEN_MODEL_NAME", "")

FULL_LABEL_MAPPING: Dict[str, Dict[str, str]] = {
    "macro_semantic": {
        "A": "[A] Alignment Good",
        "B": "[B] Narrative Conflict (Critical Story Mismatch)",
        "C": "[C] Emotion/Atmosphere Mismatch",
        "D": "[D] Scene/Environment Conflict",
        "E": "[E] Other Macro Misalignment",
    },
    "temporal_physical": {
        "A": "[A] Physical Match (Synchronized)",
        "B": "[B] Audio Missing (Visible but Silent)",
        "C": "[C] Audio Hallucination / Illogical Off-screen",
        "D": "[D] Timing Deviation",
        "E": "[E] Off-screen Logic Reasonable",
        "F": "[F] No such event",
        "G": "[G] Other Inconsistency",
        "H": "[H] Unable to judge due to Macro Conflict",
    },
    "temporal_speaker": {
        "A": "[A] Physical Match (Lip-sync)",
        "B": "[B] Audio Missing (Mouth moving but silent)",
        "C": "[C] Audio Hallucination (Voice heard, mouth closed)",
        "D": "[D] Timing Deviation (Start/End offset)",
        "E": "[E] Narrative - Voiceover (Reasonable)",
        "F": "[F] No such event",
        "G": "[G] Other Inconsistency",
        "H": "[H] Unable to judge due to Macro Conflict",
    },
    "temporal_music": {
        "A": "[A] Physical Match (Instrument/Dance)",
        "B": "[B] Audio Missing (Instrument/Dance silent)",
        "C": "[C] Audio Hallucination (Music heard, performer still)",
        "D": "[D] Physical Rhythm Mismatch (Off-beat)",
        "E": "[E] Narrative - BGM Reasonable",
        "F": "[F] Narrative - BGM Rhythm Mismatch",
        "G": "[G] No such event",
        "H": "[H] Other Inconsistency",
        "I": "[I] Unable to judge due to Macro Conflict",
    },
    "fine_grained": {
        "A": "[A] Perfect Match",
        "B": "[B] Physical - Material/Timbre Mismatch",
        "C": "[C] Physical - Energy/Force Mismatch",
        "D": "[D] Physical - Space/Distance Mismatch",
        "E": "[E] Morphological - Micro-state Mismatch",
        "F": "[F] Logical - Reaction Contradicts Causality",
        "G": "[G] Failed Temporal Verification",
        "H": "[H] Other Mismatch",
        "I": "[I] Unable to judge due to Macro Conflict",
    },
}

LABEL_FIELDS = [
    "macro_semantic",
    "temporal_physical",
    "temporal_speaker",
    "temporal_music",
    "fine_grained",
]

CSV_FIELD_MAPPING: Dict[str, str] = {
    "macro_semantic": "metric_macro",
    "temporal_physical": "metric_temporal_other",
    "temporal_speaker": "metric_temporal_speaker",
    "temporal_music": "metric_temporal_music",
    "fine_grained": "metric_fine_grained",
}


def extract_json_from_text(text: str) -> str:
    text = text.strip()
    think_block = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    if think_block.strip():
        text = think_block.strip()
    json_code_block = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if json_code_block:
        return json_code_block.group(1)
    list_match = re.search(r"\{\s*\"macro_semantic_alignment\"[\s\S]*\}", text)
    if list_match:
        return list_match.group(0)
    return text


def load_prompt_text(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def parse_label_codes(raw: str) -> List[str]:
    if not raw:
        return []
    candidates: List[str] = []
    for token in re.split(r"[，,]", raw):
        token = token.strip()
        if not token:
            continue
        brackets = re.findall(r"\[([A-I])\]", token)
        if brackets:
            candidates.extend(brackets)
        elif len(token) == 1 and token.isalpha():
            candidates.append(token.upper())
        else:
            candidates.append(token[0].upper())
    seen = set()
    result: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def merge_unique_codes(code_lists: List[List[str]]) -> List[str]:
    merged = sorted({code for codes in code_lists for code in codes})
    return merged


def select_majority_codes(row: Dict[str, Any], base_csv_field: str) -> List[str]:
    votes: List[List[str]] = []
    for suffix in ["_1", "_2", "_3"]:
        key = f"{base_csv_field}{suffix}"
        if key not in row:
            continue
        raw_value = row.get(key, "")
        parsed = parse_label_codes(str(raw_value) if raw_value is not None else "")
        if parsed:
            votes.append(sorted(parsed))
    if not votes:
        return parse_label_codes(str(row.get(base_csv_field, "") or ""))
    counter = Counter(tuple(codes) for codes in votes)
    top_codes, top_count = counter.most_common(1)[0]
    if top_count >= 2:
        return list(top_codes)
    return merge_unique_codes(votes)


def extract_valid_json_object(text: str) -> Dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if validate_output_schema(obj):
            return obj
    return None


def parse_model_output_to_json(raw_text: str) -> Dict[str, Any]:
    cleaned = extract_json_from_text(raw_text)
    try:
        obj = json.loads(cleaned)
        if validate_output_schema(obj):
            return obj
    except json.JSONDecodeError:
        pass
    recovered = extract_valid_json_object(raw_text)
    if recovered is not None:
        return recovered
    recovered = extract_valid_json_object(cleaned)
    if recovered is not None:
        return recovered
    raise ValueError("Cannot parse valid target JSON from model output.")


def flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        joined_parts = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part and part["text"] is not None:
                    joined_parts.append(str(part["text"]))
                elif "content" in part and part["content"] is not None:
                    joined_parts.append(str(part["content"]))
                else:
                    joined_parts.append(str(part))
            else:
                joined_parts.append(str(part))
        return "\n".join(joined_parts)
    return str(content)


def build_clusters(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cluster_map: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        labels: Dict[str, List[str]] = {}
        for field in LABEL_FIELDS:
            csv_field = CSV_FIELD_MAPPING.get(field, field)
            labels[field] = select_majority_codes(row, csv_field)
        key_parts: List[str] = []
        for field in LABEL_FIELDS:
            codes = labels[field]
            key_parts.append(f"{field}=" + ",".join(sorted(codes)))
        key = "|".join(key_parts)
        if key not in cluster_map:
            cluster_map[key] = {"labels": labels, "rows": []}
        cluster_map[key]["rows"].append(row)
    clusters: List[Dict[str, Any]] = []
    for idx, (key, value) in enumerate(cluster_map.items()):
        clusters.append(
            {
                "cluster_id": idx,
                "key": key,
                "labels": value["labels"],
                "rows": value["rows"],
            }
        )
    return clusters


def build_label_description(labels: Dict[str, List[str]]) -> str:
    lines: List[str] = []
    lines.append("Input annotation labels for one video sample.")
    for field in LABEL_FIELDS:
        codes = labels.get(field, [])
        mapping = FULL_LABEL_MAPPING[field]
        human_texts: List[str] = []
        for c in codes:
            if c in mapping:
                human_texts.append(mapping[c])
            else:
                human_texts.append(c)
        joined_codes = ", ".join(codes) if codes else "None"
        joined_labels = "; ".join(human_texts) if human_texts else "None"
        lines.append(f"{field}: codes = [{joined_codes}]")
        lines.append(f"{field}: labels = {joined_labels}")
        lines.append("")
    return "\n".join(lines)


def build_full_prompt(base_prompt: str, labels: Dict[str, List[str]]) -> str:
    label_desc = build_label_description(labels)
    pieces = [
        base_prompt.strip(),
        "",
        "### PART 5: INPUT LABELS",
        label_desc,
        "",
        "Generate the JSON output for this single video only.",
    ]
    return "\n".join(pieces)


def validate_output_schema(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    required_keys = [
        "macro_semantic_alignment",
        "temporal_synchronicity",
        "fine_grained_event_correspondence",
        "conclusion",
    ]
    for k in required_keys:
        if k not in obj:
            return False
    if not isinstance(obj["conclusion"], str):
        return False
    for k in [
        "macro_semantic_alignment",
        "temporal_synchronicity",
        "fine_grained_event_correspondence",
    ]:
        v = obj.get(k)
        if not isinstance(v, (int, float)):
            return False
        if v < 0.0 or v > 1.0:
            return False
    return True


class MockAPIHandler:
    def __init__(self) -> None:
        self.sleep_seconds = 0.05

    def generate(self, prompt_text: str, labels: Dict[str, List[str]]) -> Dict[str, Any]:
        print("Mock handler invoked.")
        print("Prompt preview:")
        preview = prompt_text[:400]
        print(preview)
        macro_codes = ",".join(labels.get("macro_semantic", [])) or "None"
        temporal_codes: List[str] = []
        for f in ["temporal_physical", "temporal_speaker", "temporal_music"]:
            temporal_codes.extend(labels.get(f, []))
        temporal_str = ",".join(temporal_codes) or "None"
        fine_codes = ",".join(labels.get("fine_grained", [])) or "None"
        time.sleep(self.sleep_seconds)
        return {
            "macro_semantic_alignment": 0.5,
            "temporal_synchronicity": 0.5,
            "fine_grained_event_correspondence": 0.5,
            "conclusion": (
                "Mock output for pipeline testing. "
                f"macro_semantic codes={macro_codes}, "
                f"temporal codes={temporal_str}, "
                f"fine_grained codes={fine_codes}."
            ),
        }


class OnlineAPIHandler:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float = 0.2,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt_text: str, labels: Dict[str, List[str]]) -> Dict[str, Any]:
        messages = [{"role": "user", "content": prompt_text}]
        response = Generation.call(
            api_key=self.api_key,
            model=self.model_name,
            messages=messages,
            result_format="message",
            temperature=self.temperature,
        )
        if getattr(response, "status_code", None) != 200:
            code = getattr(response, "code", None)
            message = getattr(response, "message", None)
            raise RuntimeError(f"Online API error: {code}, {message}")
        content = response.output.choices[0].message.content
        content_str = flatten_message_content(content)
        obj = parse_model_output_to_json(content_str)
        if not validate_output_schema(obj):
            raise ValueError("Online API returned JSON with invalid schema.")
        return obj


class VLLMAPIHandler:
    def __init__(
        self,
        base_url: str | None = None,
        model_path: str | None = None,
        max_tokens: int = 24096,
        temperature: float = 0.7,
    ) -> None:
        self.base_url = base_url or os.environ.get("QWEN_VLLM_API_URL", "")
        self.model_path = model_path or os.environ.get("QWEN_VLLM_MODEL", "")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt_text: str, labels: Dict[str, List[str]]) -> Dict[str, Any]:
        if not self.base_url or not self.model_path:
            raise RuntimeError("Set QWEN_VLLM_API_URL and QWEN_VLLM_MODEL before using --mode vllm.")
        prompt_char_count = len(prompt_text)
        prompt_byte_count = len(prompt_text.encode("utf-8"))
        print(
            f"  vLLM request prompt_size: chars={prompt_char_count}, bytes={prompt_byte_count}, "
            f"max_tokens={self.max_tokens}"
        )
        body = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        }
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        resp = requests.post(self.base_url, json=body, timeout=600)
        if resp.status_code != 200:
            raise RuntimeError(
                f"vLLM HTTP error {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        usage = data.get("usage")
        if isinstance(usage, dict):
            print(
                "  vLLM usage: "
                f"prompt_tokens={usage.get('prompt_tokens')}, "
                f"completion_tokens={usage.get('completion_tokens')}, "
                f"total_tokens={usage.get('total_tokens')}"
            )
        else:
            print("  vLLM usage: unavailable")
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("vLLM response has no choices.")
        finish_reason = choices[0].get("finish_reason")
        print(f"  vLLM finish_reason: {finish_reason}")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        content_str = flatten_message_content(content)
        reasoning = message.get("reasoning_content")
        if reasoning:
            content_str = f"{flatten_message_content(reasoning)}\n{content_str}"
        obj = parse_model_output_to_json(content_str)
        if not validate_output_schema(obj):
            raise ValueError("vLLM API returned JSON with invalid schema.")
        return obj


def read_csv_rows(csv_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row["_row_index"] = idx + 1
            rows.append(row)
    return rows


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_pipeline(
    csv_path: str,
    prompt_path: str,
    output_dir: str,
    mode: str,
    max_retries: int,
) -> None:
    base_prompt = load_prompt_text(prompt_path)
    rows = read_csv_rows(csv_path)
    if not rows:
        raise RuntimeError("No rows loaded from CSV.")
    clusters = build_clusters(rows)
    print(f"Total rows: {len(rows)}")
    print(f"Total clusters: {len(clusters)}")
    if mode == "mock":
        handler = MockAPIHandler()
    elif mode == "online":
        handler = OnlineAPIHandler(api_key=QWEN_API_KEY, model_name=QWEN_MODEL_NAME)
    elif mode == "vllm":
        handler = VLLMAPIHandler()
    else:
        raise ValueError(f"Unknown mode {mode}")
    ensure_output_dir(output_dir)
    cluster_count = len(clusters)
    for idx, cluster in enumerate(clusters):
        labels = cluster["labels"]
        cluster_rows = cluster["rows"]
        print(
            f"Processing cluster {idx + 1}/{cluster_count}, "
            f"rows in cluster: {len(cluster_rows)}"
        )
        prompt_text = build_full_prompt(base_prompt, labels)
        last_err: Exception | None = None
        result_obj: Dict[str, Any] | None = None
        for attempt in range(1, max_retries + 1):
            try:
                print(f"  Attempt {attempt}/{max_retries} for this cluster.")
                result_obj = handler.generate(prompt_text, labels)
                if not validate_output_schema(result_obj):
                    raise ValueError("Returned JSON failed schema validation.")
                break
            except Exception as e:
                last_err = e
                print(f"  Error in attempt {attempt}: {e}")
                time.sleep(1.0)
        if result_obj is None:
            raise RuntimeError(
                f"Failed to obtain valid JSON for cluster {idx + 1}: {last_err}"
            )
        for row in cluster_rows:
            index_value = row.get("index")
            if index_value is None or index_value == "":
                index_value = row.get("video_id")
            if index_value is None or index_value == "":
                index_value = str(row.get("_row_index", ""))
            if not index_value:
                raise KeyError(
                    "CSV row missing usable identifier: expected 'index' or 'video_id'."
                )
            filename = f"{index_value}.json"
            out_path = os.path.join(output_dir, filename)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result_obj, f, ensure_ascii=False, indent=2)
        print(
            f"  Cluster {idx + 1} done. Written {len(cluster_rows)} files into {output_dir}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite annotation CSV into JSON targets using LLM."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to input CSV file containing labels.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Path to prompt.txt. Defaults to prompt.txt in script directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write JSON files. Defaults to annotation_rewrite in script directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mock", "online", "vllm"],
        default="vllm",
        help="API mode: mock, online (dashscope), or vllm.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries per cluster when calling the API.",
    )
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = args.prompt_path or os.path.abspath(os.path.join(base_dir, "..", "..", "prompts", "score_rewrite.txt"))
    output_dir = args.output_dir or os.path.abspath(os.path.join(base_dir, "..", "..", "outputs", "score_rewrite"))
    run_pipeline(
        csv_path=args.csv_path,
        prompt_path=prompt_path,
        output_dir=output_dir,
        mode=args.mode,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
