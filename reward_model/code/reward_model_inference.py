from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rm_reward_client import score_one


def load_records(path: Path, video_column: str, audio_column: str | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                row = json.loads(line)
                videos = row.get("videos") or []
                audios = row.get("audios") or []
                records.append({"id": row.get("id", str(i)), "video": videos[0] if videos else row.get(video_column), "audio": audios[0] if audios else None})
    else:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                records.append({"id": row.get("id", str(i)), "video": row.get(video_column), "audio": row.get(audio_column) if audio_column else None})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--video-column", default="video_path")
    parser.add_argument("--audio-column", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    records = load_records(Path(args.input), args.video_column, args.audio_column)
    base_dir = Path(args.base_dir).expanduser().resolve()
    if args.limit > 0:
        records = records[: args.limit]
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            result = {"id": record.get("id"), "video": record.get("video"), "audio": record.get("audio")}
            try:
                video_path = Path(record["video"]).expanduser()
                if not video_path.is_absolute():
                    video_path = base_dir / video_path
                audio_value = record.get("audio")
                audio_path = Path(audio_value).expanduser() if audio_value else None
                if audio_path is not None and not audio_path.is_absolute():
                    audio_path = base_dir / audio_path
                scored = score_one(video_path, audio_path, endpoint=args.endpoint, model=args.model)
                result.update(scored)
                result["ok"] = True
            except Exception as exc:
                result["ok"] = False
                result["error"] = repr(exc)
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
