import os
import argparse
import json
import time
from evaluation_pipeline.main_pipeline_qwen import run_full_pipeline_qwen, run_parallel_pipeline_qwen

def save_report_immediately_qwen(file_name, result, reports_dir):
    try:

        report_path = os.path.join(reports_dir, f"{file_name}_final_report_qwen.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        final_score = result.get("final_summary_report", {}).get("overall_consistency_score", 0)
        return True

    except Exception as e:
        return False

def run_parallel_pipeline_qwen_with_immediate_save(
    file_tasks, reports_dir, max_workers=None, show_progress=True
):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

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

            from evaluation_pipeline.main_pipeline_qwen import run_single_file_pipeline_qwen
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
                        save_success = save_report_immediately_qwen(file_name, result, reports_dir)
                        if save_success:
                            if show_progress:
                                pass
                        else:
                            if show_progress:
                                pass
                    else:

                        failed_results[file_name] = error_msg or "Unknown error"
                        if show_progress:
                            pass

                except Exception as e:

                    failed_results[base_name] = str(e)
                    if show_progress:
                        pass

                if show_progress:
                    progress = (completed_count / total_files) * 100
                    elapsed_time = time.time() - start_time

                    if completed_count > 0:
                        avg_time_per_file = elapsed_time / completed_count
                        remaining_files = total_files - completed_count
                        estimated_remaining_time = avg_time_per_file * remaining_files


    except Exception as e:
        return {
            "successful_results": {},
            "failed_results": {f"parallel_error": str(e)},
            "summary_statistics": {
                "total_files": len(file_tasks),
                "successful_count": 0,
                "failed_count": len(file_tasks),
                "success_rate": 0.0,
                "total_processing_time": time.time() - start_time,
                "average_time_per_file": 0.0
            }
        }

    end_time = time.time()
    total_time = end_time - start_time
    successful_count = len(successful_results)
    failed_count = len(failed_results)
    success_rate = successful_count / total_files if total_files > 0 else 0
    avg_time_per_file = total_time / total_files if total_files > 0 else 0

    if show_progress:
        print(f"\n{'='*60}")

    return {
        "successful_results": successful_results,
        "failed_results": failed_results,
        "summary_statistics": {
            "total_files": total_files,
            "successful_count": successful_count,
            "failed_count": failed_count,
            "success_rate": success_rate,
            "total_processing_time": total_time,
            "average_time_per_file": avg_time_per_file,
            "processing_speed": successful_count/(total_time/60) if total_time > 0 else 0,
            "max_workers": max_workers
        }
    }

def main(processed_dir, original_dir, reports_dir, parallel=True, max_workers=None):
    os.makedirs(reports_dir, exist_ok=True)

    if parallel and max_workers:
        pass

    processed_files = os.listdir(processed_dir)

    base_names = sorted(list(set([
        f.replace('_video', '').replace('_audio', '').replace('.json', '')
        for f in processed_files if f.endswith('.json')
    ])))

    if not base_names:
        return


    file_tasks = []
    skipped_files = []

    for base_name in base_names:
        base_file_path = os.path.join(processed_dir, base_name)

        original_video_path = None
        for ext in ['.mp4', '.mov', '.avi', '.mkv']:
            potential_path = os.path.join(original_dir, base_name + ext)
            if os.path.exists(potential_path):
                original_video_path = potential_path
                break

        if not original_video_path:
            skipped_files.append(base_name)
            continue

        report_path = os.path.join(reports_dir, f"{base_name}_final_report_qwen.json")

        if os.path.exists(report_path):
            skipped_files.append(base_name)
            continue

        file_tasks.append((base_file_path, original_video_path, base_name))

    if not file_tasks:
        return

    if skipped_files:
        pass

    start_time = time.time()

    if parallel:

        results = run_parallel_pipeline_qwen_with_immediate_save(file_tasks, reports_dir, max_workers=max_workers, show_progress=True)

        successful_results = results["successful_results"]
        failed_results = results["failed_results"]

        stats = results["summary_statistics"]

        if successful_results:
            total_gemini_calls = 0
            total_qwen_calls = 0
            total_tokens = 0

            for result in successful_results.values():
                token_report = result.get("token_consumption_report", {})
                api_dist = token_report.get("api_distribution", {})
                total_gemini_calls += api_dist.get("gemini_api_calls", 0)
                total_qwen_calls += api_dist.get("qwen_api_calls", 0)
                total_tokens += token_report.get("total_consumption", {}).get("total_tokens", 0)


        if failed_results:
            for file_name, error_msg in failed_results.items():
                print(f"   {file_name}: {error_msg}")

    else:

        successful_count = 0
        failed_count = 0
        failed_files = {}
        total_gemini_calls = 0
        total_qwen_calls = 0
        total_tokens = 0

        for i, (base_file_path, original_video_path, base_name) in enumerate(file_tasks, 1):

            try:

                final_report = run_full_pipeline_qwen(base_file_path, original_video_path)

                report_path = os.path.join(reports_dir, f"{base_name}_final_report_qwen.json")
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(final_report, f, indent=4, ensure_ascii=False)

                final_score = final_report.get("final_summary_report", {}).get("overall_consistency_score", 0)
                successful_count += 1

                token_report = final_report.get("token_consumption_report", {})
                api_dist = token_report.get("api_distribution", {})
                total_gemini_calls += api_dist.get("gemini_api_calls", 0)
                total_qwen_calls += api_dist.get("qwen_api_calls", 0)
                total_tokens += token_report.get("total_consumption", {}).get("total_tokens", 0)

            except Exception as e:
                error_msg = f"Unrecoverable error while evaluating {base_name}: {e}"
                print(f"❌ {error_msg}")
                failed_files[base_name] = str(e)
                failed_count += 1
                continue

        total_time = time.time() - start_time

        if successful_count > 0:
            pass

        if failed_files:
            for file_name, error_msg in failed_files.items():
                print(f"   {file_name}: {error_msg}")

    print(f"\n{'='*60}")
    print(f"{'='*60}")

if __name__ == "__main__":
    task_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="Run the five-stage audio-video evaluation pipeline")
    parser.add_argument("--processed_dir", type=str, default=os.path.join(task_root, "data", "processed_data"),
                       help="Directory containing preprocessed files")
    parser.add_argument("--original_dir", type=str, default=os.path.join(task_root, "data", "original_videos"),
                       help="Directory containing source videos")
    parser.add_argument("--reports_dir", type=str, default=os.path.join(task_root, "data", "reports_qwen"),
                       help="Directory for evaluation reports (default: reports_qwen)")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Enable parallel processing")
    parser.add_argument("--serial", action="store_true",
                       help="Use serial processing instead of parallel mode")
    parser.add_argument("--max_workers", type=int, default=None,
                       help="Maximum worker processes; detected automatically when omitted")

    args = parser.parse_args()

    if args.serial:
        parallel_mode = False
    else:
        parallel_mode = args.parallel


    main(args.processed_dir, args.original_dir, args.reports_dir,
         parallel=parallel_mode, max_workers=args.max_workers)
