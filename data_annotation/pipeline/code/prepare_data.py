import os
import sys
import json
import glob
import argparse
import subprocess
from pathlib import Path

try:
    from librosa_sed import process_audio_file
except Exception:
    process_audio_file = None

TASK_ROOT = Path(__file__).resolve().parents[1]
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

def ffmpeg_available():
    try:
        subprocess.run([FFMPEG_BIN, '-version'], capture_output=True, check=True)
        subprocess.run([FFPROBE_BIN, '-version'], capture_output=True, check=True)
        return True
    except Exception:
        return False

def get_video_duration(video_path: str):
    try:
        cmd = [
            FFPROBE_BIN, '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        return None

def compress_video(input_path: str, output_path: str, target_size_kb: int = 800, keep_audio: bool = True):
    try:
        duration = get_video_duration(input_path)
        if duration is None or duration <= 0:
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        target_size_bits = target_size_kb * 1024 * 8
        audio_bitrate_bps = 96 * 1000 if keep_audio else 0
        video_bitrate_bps = max(100000, int((target_size_bits - audio_bitrate_bps * duration) / duration))

        video_bitrate_k = max(100, int(video_bitrate_bps / 1000))

        cmd = [FFMPEG_BIN, '-i', input_path]
        cmd += ['-c:v', 'libx264', '-b:v', f'{video_bitrate_k}k', '-maxrate', f'{video_bitrate_k}k', '-bufsize', f'{video_bitrate_k * 2}k']
        if keep_audio:
            cmd += ['-c:a', 'aac', '-b:a', '96k', '-ac', '2']
        else:
            cmd += ['-an']
        cmd += ['-preset', 'medium', '-pix_fmt', 'yuv420p', '-profile:v', 'main', '-level', '4.0', '-movflags', '+faststart', '-f', 'mp4', '-y', output_path]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return False

        if not os.path.exists(output_path):
            return False

        return True
    except Exception as e:
        return False

def extract_audio(input_path: str, audio_output_path: str, audio_bitrate_k: int = 96):
    try:
        os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
        cmd = [
            FFMPEG_BIN, '-i', input_path,
            '-vn',
            '-c:a', 'libmp3lame',
            '-b:a', f'{audio_bitrate_k}k',
            '-ac', '2',
            '-y', audio_output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            if 'does not contain any stream' in (result.stderr or '') or 'Stream specifier' in (result.stderr or ''):
                return False
            return False
        if not os.path.exists(audio_output_path):
            return False
        return True
    except Exception as e:
        return False

def preprocess_video(video_path: str, output_dir: str, target_size_kb: int = 800):
    if not os.path.exists(video_path):
        return

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    video_output_path = os.path.join(output_dir, f"{base_name}_video.mp4")
    audio_output_path = os.path.join(output_dir, f"{base_name}_audio.mp3")
    sed_json_output_path = os.path.join(output_dir, f"{base_name}.json")

    try:

        if not ffmpeg_available():
            return

        compress_ok = compress_video(video_path, video_output_path, target_size_kb=target_size_kb, keep_audio=True)
        if not compress_ok:
            return

        audio_ok = extract_audio(video_path, audio_output_path, audio_bitrate_k=96)

        if audio_ok and process_audio_file is not None:
            sed_results = process_audio_file(audio_output_path)
            if sed_results is not None:
                with open(sed_json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(sed_results, f, indent=4, ensure_ascii=False)
            else:
                pass
        elif process_audio_file is None:
            pass
        else:
            pass


    except Exception as e:
        pass

def main(input_dir: str, output_dir: str, target_size_kb: int = 800):
    os.makedirs(output_dir, exist_ok=True)

    video_exts = ('.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(video_exts)]
    for filename in files:
        video_path = os.path.join(input_dir, filename)
        preprocess_video(video_path, output_dir, target_size_kb=target_size_kb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress videos, extract audio, and detect sound events")
    parser.add_argument("--input_dir", type=str, default=str(TASK_ROOT / "data" / "original_videos"), help="Directory containing source videos")
    parser.add_argument("--output_dir", type=str, default=str(TASK_ROOT / "data" / "processed_data"), help="Directory for processed files")
    parser.add_argument("--target_size_kb", type=int, default=800, help="Approximate target video size in KB")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, target_size_kb=args.target_size_kb)
