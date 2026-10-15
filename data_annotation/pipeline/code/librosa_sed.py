import os
import librosa as librosa
import numpy as np
import json
import argparse
from pprint import pprint

def find_adaptive_threshold(db_rms):
    if len(db_rms) < 2:
        return -80.0, True

    hist, bin_edges = np.histogram(db_rms, bins='auto')
    peak_location = np.argmax(hist)
    peak_db_value = bin_edges[peak_location]

    valley_location = -1
    for i in range(peak_location + 1, len(hist) - 1):
        if hist[i] < hist[i-1] and hist[i] < hist[i+1]:
            valley_location = i
            break

    used_heuristic = False
    if valley_location == -1:
        used_heuristic = True
        if peak_db_value < -50.0:
            threshold = peak_db_value + 10.0
        else:
            threshold = peak_db_value - 5.0
    else:
        threshold = bin_edges[valley_location]

    return threshold, used_heuristic

def calculate_adaptive_offset(db_rms_absolute):
    hist, bin_edges = np.histogram(db_rms_absolute, bins='auto')
    peak_location = np.argmax(hist)

    noise_peak_start = bin_edges[peak_location]
    noise_peak_end = bin_edges[peak_location + 1]

    noise_db_values = db_rms_absolute[
        (db_rms_absolute >= noise_peak_start) & (db_rms_absolute < noise_peak_end)
    ]

    if len(noise_db_values) < 2:
        return 5.0

    noise_std = np.std(noise_db_values)
    adaptive_offset = 3.0 * noise_std
    return np.clip(adaptive_offset, 3.0, 15.0)

def merge_intervals(intervals, padding_s=0.1):
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x['start'])
    merged = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current['start'] <= last['end'] + padding_s:
            last['end'] = max(last['end'], current['end'])
        else:
            merged.append(current)

    return merged

def process_audio_file(audio_file_path):

    SAMPLING_RATE = 16000
    frame_length = 2048
    hop_length = 512
    MERGE_GAP_S = 0.15

    try:
        wav, _ = librosa.load(audio_file_path, sr=SAMPLING_RATE)
        if len(wav) == 0:
            print(f"Warning: Audio file is empty, skipping: {audio_file_path}")
            return None
    except Exception as e:
        print(f"Error loading audio file {audio_file_path}: {e}")
        return None

    rms_global = librosa.feature.rms(y=wav, frame_length=frame_length, hop_length=hop_length)[0]
    if rms_global.size == 0 or np.all(rms_global == 0):
        print(f"Warning: Could not compute RMS for {audio_file_path}. Possibly a silent file.")

        return {f"method_{i+1}": [] for i in range(8)}

    results = {}

    db_rms_global = librosa.amplitude_to_db(rms_global, ref=np.max)
    global_threshold_db, _ = find_adaptive_threshold(db_rms_global)
    global_intervals = librosa.effects.split(wav, top_db=-global_threshold_db, frame_length=frame_length, hop_length=hop_length)
    global_timestamps = [{'start': round(s/SAMPLING_RATE, 2), 'end': round(e/SAMPLING_RATE, 2)} for s, e in global_intervals]
    results['method_1'] = global_timestamps

    results['method_5'] = merge_intervals(global_timestamps, padding_s=MERGE_GAP_S)

    window_size_s = 2.0
    hop_size_s = 1.0
    window_size_samples = int(window_size_s * SAMPLING_RATE)
    hop_size_samples = int(hop_size_s * SAMPLING_RATE)
    all_window_timestamps = []
    for i in range(0, len(wav), hop_size_samples):
        window_wav = wav[i : i + window_size_samples]
        if len(window_wav) < hop_length: continue
        rms_window = librosa.feature.rms(y=window_wav, frame_length=frame_length, hop_length=hop_length)[0]
        if rms_window.size == 0 or np.all(rms_window == 0): continue
        db_rms_window = librosa.amplitude_to_db(rms_window, ref=np.max)
        window_threshold_db, used_heuristic = find_adaptive_threshold(db_rms_window)
        if used_heuristic: window_threshold_db = global_threshold_db
        window_intervals_samples = librosa.effects.split(window_wav, top_db=-window_threshold_db, frame_length=frame_length, hop_length=hop_length)
        for start_sample, end_sample in window_intervals_samples:
            all_window_timestamps.append({'start': round((i + start_sample) / SAMPLING_RATE, 2), 'end': round((i + end_sample) / SAMPLING_RATE, 2)})
    method2_timestamps = merge_intervals(all_window_timestamps)
    results['method_2'] = method2_timestamps

    results['method_6'] = merge_intervals(method2_timestamps, padding_s=MERGE_GAP_S)

    OFFSET_FROM_MIN_DB = 5.0
    db_rms_absolute = librosa.amplitude_to_db(rms_global, ref=1.0)
    min_energy_db = np.min(db_rms_absolute)
    threshold_db_method3 = min_energy_db + OFFSET_FROM_MIN_DB
    frames = np.where(db_rms_absolute > threshold_db_method3)[0]
    if len(frames) > 0:
        split_points = np.where(np.diff(frames) > 1)[0] + 1
        frame_groups = np.split(frames, split_points)
        consecutive_frames = [(g[0], g[-1]) for g in frame_groups]
    else:
        consecutive_frames = []
    method3_timestamps = [{'start': round(librosa.frames_to_samples(s, hop_length=hop_length)/SAMPLING_RATE, 2), 'end': round(librosa.frames_to_samples(e, hop_length=hop_length)/SAMPLING_RATE, 2)} for s, e in consecutive_frames]
    results['method_3'] = method3_timestamps

    results['method_7'] = merge_intervals(method3_timestamps, padding_s=MERGE_GAP_S)

    adaptive_offset = calculate_adaptive_offset(db_rms_absolute)
    threshold_db_method4 = min_energy_db + adaptive_offset
    frames = np.where(db_rms_absolute > threshold_db_method4)[0]
    if len(frames) > 0:
        split_points = np.where(np.diff(frames) > 1)[0] + 1
        frame_groups = np.split(frames, split_points)
        consecutive_frames = [(g[0], g[-1]) for g in frame_groups]
    else:
        consecutive_frames = []
    method4_timestamps = [{'start': round(librosa.frames_to_samples(s, hop_length=hop_length)/SAMPLING_RATE, 2), 'end': round(librosa.frames_to_samples(e, hop_length=hop_length)/SAMPLING_RATE, 2)} for s, e in consecutive_frames]
    results['method_4'] = method4_timestamps

    results['method_8'] = merge_intervals(method4_timestamps, padding_s=MERGE_GAP_S)

    return results

def main(root_dir):
    print(f"Starting processing in root directory: {root_dir}")
    found_audio_files = False

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:

            is_audio_file = filename.lower().endswith(('_audio.wav', '_audio.mp3', '_audio.flac', '_audio.m4a'))

            if is_audio_file:
                found_audio_files = True
                audio_file_path = os.path.join(dirpath, filename)
                print(f"\nProcessing file: {audio_file_path}")

                timestamps_data = process_audio_file(audio_file_path)

                if timestamps_data is not None:

                    json_filename = os.path.splitext(filename)[0] + '.json'
                    json_file_path = os.path.join(dirpath, json_filename)

                    try:
                        with open(json_file_path, 'w', encoding='utf-8') as f:
                            json.dump(timestamps_data, f, indent=4, ensure_ascii=False)
                        print(f"  - Successfully processed '{filename}' -> '{os.path.basename(json_file_path)}'")
                    except Exception as e:
                        print(f"  - Error writing JSON for {filename}: {e}")

    if not found_audio_files:
        print("No audio files ending with '_audio.<ext>' found in the specified directory.")

    print("\nProcessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process audio files for sound event detection.")
    parser.add_argument("root_directory", type=str, help="The root directory containing subfolders to process.")

    args = parser.parse_args()

    if not os.path.isdir(args.root_directory):
        print(f"Error: The specified directory does not exist: {args.root_directory}")
    else:
        main(args.root_directory)
