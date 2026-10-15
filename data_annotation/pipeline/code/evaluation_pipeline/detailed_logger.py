import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

class DetailedLogger:

    def __init__(self, video_file_name: str, base_dir: str = "reports", level: int = None):
        self.video_file_name = video_file_name
        self.base_dir = base_dir
        self.level = level

        if level is not None:

            base_name = video_file_name.replace('_pid' + str(os.getpid()), '') if '_pid' in video_file_name else video_file_name
            base_name = base_name.replace('.mp4', '').replace('.json', '')

            dir_base_name = os.path.basename(base_dir)
            parent_dir = os.path.dirname(base_dir)

            if parent_dir and dir_base_name not in ['pipeline_outputs', 'reports', 'reports_qwen', 'reports_test']:

                level_dir = base_dir
                self.log_file_path = os.path.join(level_dir, f"level_{level}_detailed_log.json")
            else:

                level_dir = os.path.join(base_dir, base_name)
                self.log_file_path = os.path.join(level_dir, f"level_{level}_detailed_log.json")

            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        else:

            self.log_file_path = os.path.join(base_dir, f"{video_file_name}_detailed_log.json")

            os.makedirs(base_dir, exist_ok=True)

        self.log_data = {
            "video_file": video_file_name,
            "analysis_start_time": datetime.now().isoformat(),
            "analysis_end_time": None,
            "pipeline_stages": {},
            "api_calls": [],
            "key_variables": {},
            "errors_and_warnings": [],
            "summary": {}
        }

        self.current_stage = None
        self.stage_start_time = None

    def start_stage(self, stage_name: str, description: str = ""):
        self.current_stage = stage_name
        self.stage_start_time = datetime.now().isoformat()

        self.log_data["pipeline_stages"][stage_name] = {
            "description": description,
            "start_time": self.stage_start_time,
            "end_time": None,
            "status": "in_progress",
            "details": {},
            "api_calls": [],
            "key_variables": {},
            "summary": ""
        }

        print(f"\n{'='*20} {description} {'='*20}")
        print(f"[STAGE START] {stage_name} - {self.stage_start_time}")

    def end_stage(self, status: str = "completed", summary: str = ""):
        if self.current_stage:
            end_time = datetime.now().isoformat()

            self.log_data["pipeline_stages"][self.current_stage]["end_time"] = end_time
            self.log_data["pipeline_stages"][self.current_stage]["status"] = status
            self.log_data["pipeline_stages"][self.current_stage]["summary"] = summary

            print(f"[STAGE END] {self.current_stage} - {end_time} - Status: {status}")
            if summary:
                print(f"[STAGE SUMMARY] {summary}")

    def log_key_variable(self, var_name: str, var_value: Any, description: str = ""):
        timestamp = datetime.now().isoformat()

        var_info = {
            "timestamp": timestamp,
            "value": var_value,
            "description": description
        }

        if self.current_stage not in self.log_data["key_variables"]:
            self.log_data["key_variables"][self.current_stage] = {}
        self.log_data["key_variables"][self.current_stage][var_name] = var_info

        if self.current_stage:
            self.log_data["pipeline_stages"][self.current_stage]["key_variables"][var_name] = var_info

        value_str = str(var_value)
        display_value = value_str[:100] + '...' if len(value_str) > 100 else value_str
        print(f"[KEY VAR] {var_name}: {display_value}")
        if description:
            pass

    def log_api_call(self, prompt: str, file_path: str = None, mime_type: str = None,
                     response: Any = None, error: str = None, attempt: int = 1,
                     input_tokens: int = None, output_tokens: int = None, total_tokens: int = None):
        timestamp = datetime.now().isoformat()

        api_call_info = {
            "timestamp": timestamp,
            "stage": self.current_stage,
            "attempt": attempt,
            "input": {
                "prompt": prompt,
                "file_path": file_path,
                "mime_type": mime_type
            },
            "output": {
                "response": response,
                "error": error,
                "success": error is None
            }
        }

        if any(token is not None for token in [input_tokens, output_tokens, total_tokens]):
            api_call_info["token_usage"] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }

        self.log_data["api_calls"].append(api_call_info)

        if self.current_stage:
            self.log_data["pipeline_stages"][self.current_stage]["api_calls"].append(api_call_info)

        if file_path:
            pass

        if any(token is not None for token in [input_tokens, output_tokens, total_tokens]):
            token_info = []
            if input_tokens is not None:
                token_info.append(f"input={input_tokens}")
            if output_tokens is not None:
                token_info.append(f"output={output_tokens}")
            if total_tokens is not None:
                token_info.append(f"total={total_tokens}")

        if error:
            pass
        elif response:
            pass

            if isinstance(response, dict) and any(key in str(response).lower() for key in ['mock', 'test', 'sample']):
                pass

    def log_warning(self, message: str, details: Dict = None):
        timestamp = datetime.now().isoformat()
        warning_info = {
            "timestamp": timestamp,
            "level": "WARNING",
            "stage": self.current_stage,
            "message": message,
            "details": details or {}
        }

        self.log_data["errors_and_warnings"].append(warning_info)
        print(f"[WARNING] {message}")

    def log_error(self, message: str, details: Dict = None):
        timestamp = datetime.now().isoformat()
        error_info = {
            "timestamp": timestamp,
            "level": "ERROR",
            "stage": self.current_stage,
            "message": message,
            "details": details or {}
        }

        self.log_data["errors_and_warnings"].append(error_info)
        print(f"[ERROR] {message}")

    def finalize_log(self, final_summary: Dict = None):
        self.log_data["analysis_end_time"] = datetime.now().isoformat()

        total_api_calls = len(self.log_data["api_calls"])
        successful_calls = sum(1 for call in self.log_data["api_calls"] if call["output"]["success"])
        failed_calls = total_api_calls - successful_calls
        total_errors = len([item for item in self.log_data["errors_and_warnings"] if item["level"] == "ERROR"])
        total_warnings = len([item for item in self.log_data["errors_and_warnings"] if item["level"] == "WARNING"])

        self.log_data["summary"] = {
            "analysis_status": "completed",
            "statistics": {
                "total_api_calls": total_api_calls,
                "successful_api_calls": successful_calls,
                "failed_api_calls": failed_calls,
                "total_stages": len(self.log_data["pipeline_stages"]),
                "total_errors": total_errors,
                "total_warnings": total_warnings
            }
        }

        if final_summary:
            self.log_data["summary"].update(final_summary)

        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)


        return self.log_file_path

_global_logger: Optional[DetailedLogger] = None

def get_logger() -> Optional[DetailedLogger]:
    return _global_logger

def set_logger(logger: DetailedLogger):
    global _global_logger
    _global_logger = logger

def create_logger(video_file_name: str, base_dir: str = "reports", process_safe: bool = False, level: int = None) -> DetailedLogger:

    if process_safe:
        import os
        process_id = os.getpid()

        base_name = video_file_name.replace('.mp4', '').replace('.json', '')
        video_file_name = f"{base_name}_pid{process_id}"

    logger = DetailedLogger(video_file_name, base_dir, level)
    set_logger(logger)
    return logger
