import os
import json
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import List, Tuple, Dict, Optional
from . import level_1_audio, level_4_video2audio, level_5_scoring
from . import level_2_video_qwen, level_3_audio2video_qwen
from .detailed_logger import create_logger, get_logger

def get_audio_file_path(base_file_path: str) -> str:

    wav_path = f"{base_file_path}_audio.wav"
    if os.path.exists(wav_path):
        return wav_path

    mp3_path = f"{base_file_path}_audio.mp3"
    if os.path.exists(mp3_path):
        return mp3_path

    raise FileNotFoundError(f"Audio file not found: {wav_path} or {mp3_path}")

def update_token_stats(token_stats, level_name, token_usage):
    if token_usage and isinstance(token_usage, dict):

        if level_name in token_stats:
            token_stats[level_name]["input_tokens"] += token_usage.get("input_tokens", 0)
            token_stats[level_name]["output_tokens"] += token_usage.get("output_tokens", 0)
            token_stats[level_name]["total_tokens"] += token_usage.get("total_tokens", 0)
            token_stats[level_name]["api_calls"] += 1

            token_stats["total"]["input_tokens"] += token_usage.get("input_tokens", 0)
            token_stats["total"]["output_tokens"] += token_usage.get("output_tokens", 0)
            token_stats["total"]["total_tokens"] += token_usage.get("total_tokens", 0)
            token_stats["total"]["api_calls"] += 1

def run_full_pipeline_qwen(base_file_path, original_video_path):

    video_file_name = os.path.splitext(os.path.basename(original_video_path))[0]

    logger = create_logger(video_file_name, "reports_qwen")

    audio_path = get_audio_file_path(base_file_path)
    video_path = f"{base_file_path}_video.mp4"
    sed_json_path = f"{base_file_path}.json"


    all_results = {}

    token_stats = {
        "level_1_audio": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0},
        "level_2_video_qwen": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0},
        "level_3_audio2video_qwen": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0},
        "level_4_video2audio": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0},
        "level_5_scoring": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0},
        "total": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0}
    }



    level1_results, level1_token_usage = level_1_audio.analyze_audio_with_tokens(audio_path, sed_json_path)
    all_results["level_1_audio_analysis"] = level1_results

    if level1_token_usage:
        for usage in level1_token_usage:
            update_token_stats(token_stats, "level_1_audio", usage)


    audio_events_count = len(level1_results.get('acoustic_event_detection', {}).get('event_instances', []))


    level2_results, level2_token_usage = level_2_video_qwen.analyze_video_with_tokens(video_path)
    all_results["level_2_video_analysis"] = level2_results

    if level2_token_usage:
        for usage in level2_token_usage:
            update_token_stats(token_stats, "level_2_video_qwen", usage)


    video_events_count = len(level2_results.get('visual_event_detection', {}).get('event_instances', []))


    level3_results, level3_token_usage = level_3_audio2video_qwen.verify_audio_in_video_with_tokens(video_path, level1_results, level2_results)

    all_results["level_3_audio_driven_verification"] = {
        "audio_driven_verification_results": level3_results
    }

    if level3_token_usage:
        for usage in level3_token_usage:
            update_token_stats(token_stats, "level_3_audio2video_qwen", usage)


    audio_events_verified = sum(1 for result in level3_results if result and result.get("is_match", False))
    total_audio_events = len(level3_results)


    level4_results, level4_token_usage = level_4_video2audio.verify_video_in_audio_with_tokens(audio_path, level2_results)
    all_results["level_4_video_driven_verification"] = level4_results

    if level4_token_usage:
        for usage in level4_token_usage:
            update_token_stats(token_stats, "level_4_video2audio", usage)


    video_events_verified = sum(1 for result in level4_results.get("video_driven_verification_results", [])
                               if result and result.get("is_match", False))
    total_video_events = len(level4_results.get("video_driven_verification_results", []))


    final_report, level5_token_usage = level_5_scoring.generate_final_score_with_tokens(original_video_path, all_results)

    if level5_token_usage:
        for usage in level5_token_usage:
            update_token_stats(token_stats, "level_5_scoring", usage)


    full_report = {
        "final_summary_report": final_report,
        "detailed_pipeline_stages": all_results,
        "token_consumption_report": {
            "level_breakdown": {
                "level_1_audio": token_stats["level_1_audio"],
                "level_2_video_qwen": token_stats["level_2_video_qwen"],
                "level_3_audio2video_qwen": token_stats["level_3_audio2video_qwen"],
                "level_4_video2audio": token_stats["level_4_video2audio"],
                "level_5_scoring": token_stats["level_5_scoring"]
            },
            "total_consumption": token_stats["total"],
            "cost_analysis": {
                "total_api_calls": token_stats["total"]["api_calls"],
                "average_tokens_per_call": token_stats["total"]["total_tokens"] / max(token_stats["total"]["api_calls"], 1),
                "input_output_ratio": token_stats["total"]["input_tokens"] / max(token_stats["total"]["output_tokens"], 1)
            },
            "api_distribution": {
                "gemini_api_calls": token_stats["level_1_audio"]["api_calls"] + token_stats["level_4_video2audio"]["api_calls"] + token_stats["level_5_scoring"]["api_calls"],
                "qwen_api_calls": token_stats["level_2_video_qwen"]["api_calls"] + token_stats["level_3_audio2video_qwen"]["api_calls"]
            }
        }
    }


    final_score = final_report.get("overall_consistency_score", 0) if final_report else 0

    summary_stats = {
        "final_consistency_score": final_score,
        "audio_events_total": total_audio_events,
        "audio_events_verified": audio_events_verified,
        "video_events_total": total_video_events,
        "video_events_verified": video_events_verified,
        "verification_rates": {
            "audio_to_visual": audio_events_verified / max(total_audio_events, 1),
            "video_to_audio": video_events_verified / max(total_video_events, 1)
        },
        "token_consumption": token_stats["total"],
        "api_usage": {
            "gemini_calls": full_report["token_consumption_report"]["api_distribution"]["gemini_api_calls"],
            "qwen_calls": full_report["token_consumption_report"]["api_distribution"]["qwen_api_calls"]
        }
    }

    log_file_path = logger.finalize_log(summary_stats)

    print(f"\n{'='*60}")
    print(f"{'='*60}")
    print(f"{'='*60}")

    return full_report

def run_single_file_pipeline_qwen(args: Tuple[str, str, str]) -> Tuple[str, Optional[Dict], Optional[str]]:
    base_file_path, original_video_path, base_name = args

    try:

        logger = create_logger(base_name, "reports_qwen", process_safe=True)

        result = run_full_pipeline_qwen(base_file_path, original_video_path)
        return base_name, result, None

    except Exception as e:

        error_msg = f"Evaluation failed for {base_name}: {str(e)}"
        return base_name, None, error_msg

def run_parallel_pipeline_qwen(
    file_tasks: List[Tuple[str, str, str]],
    max_workers: Optional[int] = None,
    show_progress: bool = True
) -> Dict[str, Dict]:

    if not file_tasks:
        return {
            "successful_results": {},
            "failed_results": {},
            "summary_statistics": {
                "total_files": 0,
                "successful_count": 0,
                "failed_count": 0,
                "success_rate": 0.0,
                "total_processing_time": 0.0,
                "average_time_per_file": 0.0
            }
        }

    if max_workers is None:
        cpu_count = mp.cpu_count()

        max_workers = max(1, int(cpu_count * 0.75))
        if show_progress:
            pass

    total_files = len(file_tasks)
    successful_results = {}
    failed_results = {}
    start_time = time.time()

    if show_progress:
        print(f"{'='*60}")

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            future_to_task = {
                executor.submit(run_single_file_pipeline_qwen, task): task[2]
                for task in file_tasks
            }

            completed_count = 0
            for future in as_completed(future_to_task):
                base_name = future_to_task[future]
                completed_count += 1

                try:

                    file_name, result, error_msg = future.result()

                    if result is not None:

                        successful_results[file_name] = result
                        if show_progress:
                            final_score = result.get("final_summary_report", {}).get("overall_consistency_score", 0)
                    else:

                        failed_results[file_name] = error_msg
                        if show_progress:
                            pass

                except Exception as e:

                    error_msg = f"Failed to collect the task result: {str(e)}"
                    failed_results[base_name] = error_msg
                    if show_progress:
                        pass

                if show_progress and completed_count % 5 == 0:
                    elapsed_time = time.time() - start_time
                    avg_time = elapsed_time / completed_count
                    remaining_files = total_files - completed_count
                    estimated_remaining = avg_time * remaining_files

    except KeyboardInterrupt:
        if show_progress:
            pass

        return {
            "successful_results": successful_results,
            "failed_results": {**failed_results, "interrupted": "Processing interrupted by the user"},
            "summary_statistics": {
                "total_files": total_files,
                "successful_count": len(successful_results),
                "failed_count": len(failed_results) + 1,
                "success_rate": len(successful_results) / total_files,
                "total_processing_time": time.time() - start_time,
                "average_time_per_file": (time.time() - start_time) / max(completed_count, 1)
            }
        }

    except Exception as e:
        if show_progress:
            pass

        return {
            "successful_results": successful_results,
            "failed_results": {**failed_results, "system_error": f"System error: {str(e)}"},
            "summary_statistics": {
                "total_files": total_files,
                "successful_count": len(successful_results),
                "failed_count": len(failed_results) + 1,
                "success_rate": len(successful_results) / total_files,
                "total_processing_time": time.time() - start_time,
                "average_time_per_file": (time.time() - start_time) / max(completed_count, 1)
            }
        }

    total_processing_time = time.time() - start_time
    successful_count = len(successful_results)
    failed_count = len(failed_results)
    success_rate = successful_count / total_files if total_files > 0 else 0.0
    average_time_per_file = total_processing_time / total_files if total_files > 0 else 0.0

    summary_statistics = {
        "total_files": total_files,
        "successful_count": successful_count,
        "failed_count": failed_count,
        "success_rate": success_rate,
        "total_processing_time": total_processing_time,
        "average_time_per_file": average_time_per_file,
        "processing_speed": total_files / (total_processing_time / 60),
        "parallel_efficiency": total_processing_time / (average_time_per_file * max_workers) if average_time_per_file > 0 else 0.0,
        "version": "qwen"
    }

    if show_progress:
        print(f"\n{'='*60}")
        print(f"{'='*60}")

        if failed_results:
            for file_name, error_msg in failed_results.items():
                print(f"   {file_name}: {error_msg}")

        print(f"{'='*60}")

    return {
        "successful_results": successful_results,
        "failed_results": failed_results,
        "summary_statistics": summary_statistics
    }
