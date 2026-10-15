import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

class VariableManager:

    def __init__(self, base_dir: str = "pipeline_outputs"):
        self.base_dir = base_dir
        self.ensure_output_dir()

    def ensure_output_dir(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def get_variable_file_path(self, level: int) -> str:

        base_name = os.path.basename(self.base_dir)
        parent_dir = os.path.dirname(self.base_dir)

        if parent_dir and base_name not in ['pipeline_outputs', 'reports', 'reports_qwen', 'reports_test']:
            level_file_path = os.path.join(parent_dir, base_name, f"level_{level}_variables.json")

            if os.path.exists(level_file_path):
                return level_file_path

        return os.path.join(self.base_dir, f"level_{level}_variables.json")

    def get_log_file_path(self, level: int) -> str:
        return os.path.join(self.base_dir, f"level_{level}_log.txt")

    def save_variables(self, level: int, variables: dict) -> bool:
        try:
            if variables is None:
                return False
            if isinstance(variables, dict):
                if variables.get("error"):
                    return False
                if variables.get("result") == "unable_to_analyze":
                    return False

            base_name = os.path.basename(self.base_dir)
            parent_dir = os.path.dirname(self.base_dir)

            if parent_dir and base_name not in ['pipeline_outputs', 'reports', 'reports_qwen', 'reports_test']:
                file_path = os.path.join(parent_dir, base_name, f"level_{level}_variables.json")
            else:
                file_path = os.path.join(self.base_dir, f"level_{level}_variables.json")

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            data = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'variables': variables
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            return False

    def load_variables(self, level: int) -> Optional[Dict[str, Any]]:
        try:
            file_path = self.get_variable_file_path(level)

            if not os.path.exists(file_path):
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return data.get("variables", {})

        except Exception as e:
            return None

    def variable_file_exists(self, level: int) -> bool:
        file_path = self.get_variable_file_path(level)
        return os.path.exists(file_path)

    def get_missing_levels(self) -> List[int]:
        missing_levels = []
        for level in range(1, 6):
            if not self.variable_file_exists(level):
                missing_levels.append(level)
        return missing_levels

    def get_completed_levels(self) -> List[int]:
        completed_levels = []
        for level in range(1, 6):
            if self.variable_file_exists(level):
                completed_levels.append(level)
        return completed_levels

    def clear_level_variables(self, level):
        try:
            variable_file = self.get_variable_file_path(level)
            log_file = self.get_log_file_path(level)

            if os.path.exists(variable_file):
                os.remove(variable_file)

            if os.path.exists(log_file):
                os.remove(log_file)

            return True

        except Exception as e:
            return False

    def clear_all_variables(self):
        success = True
        for level in range(1, 6):
            if not self.clear_level_variables(level):
                success = False
        return success

    def clear_level_data(self, level: int):
        var_file = self.get_variable_file_path(level)
        log_file = self.get_log_file_path(level)

        for file_path in [var_file, log_file]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    pass

    def clear_all_data(self):
        for level in range(1, 6):
            self.clear_level_data(level)

    def get_status_summary(self) -> Dict[str, Any]:
        completed_levels = self.get_completed_levels()
        missing_levels = self.get_missing_levels()

        return {
            "total_levels": 5,
            "completed_levels": completed_levels,
            "missing_levels": missing_levels,
            "completion_rate": len(completed_levels) / 5 * 100,
            "next_level_to_run": missing_levels[0] if missing_levels else None
        }

    def setup_level_logger(self, level):
        logger_name = f"Level_{level}"
        logger = logging.getLogger(logger_name)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        logger.setLevel(logging.INFO)

        log_file = self.get_log_file_path(level)

        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.propagate = False

        return logger

class MockDataGenerator:

    @staticmethod
    def get_mock_level_1_result() -> Dict[str, Any]:
        return {
            "validated_layers_analysis": {
                "human_voice": {
                    "detected": True,
                    "confidence": 0.95,
                    "speakers": ["Speaker_1", "Speaker_2"],
                    "segments": [
                        {"speaker": "Speaker_1", "start": 0.0, "end": 5.2, "text": "Mock speech content 1"},
                        {"speaker": "Speaker_2", "start": 5.5, "end": 10.1, "text": "Mock speech content 2"}
                    ]
                },
                "background_music": {
                    "detected": True,
                    "style": "ambient",
                    "instruments": ["piano", "strings"],
                    "mood": "calm"
                },
                "other_sounds": {
                    "detected": True,
                    "events": ["footsteps", "door_close"]
                }
            },
            "macro_analysis": {
                "scene_hypothesis": "Indoor conversation scene",
                "speaker_emotions": {
                    "Speaker_1": "neutral",
                    "Speaker_2": "positive"
                }
            },
            "acoustic_event_detection": {
                "events": [
                    {"type": "speech", "start": 0.0, "end": 5.2, "confidence": 0.95},
                    {"type": "music", "start": 0.0, "end": 15.0, "confidence": 0.8},
                    {"type": "footsteps", "start": 12.0, "end": 13.5, "confidence": 0.7}
                ]
            }
        }

    @staticmethod
    def get_mock_level_2_result() -> Dict[str, Any]:
        return {
            "macro_analysis": {
                "scene_overview": "Mock indoor scene with two people talking",
                "key_events": ["Person talking", "Person gesturing", "Person standing up"],
                "people_and_actions": ["Two people engaged in conversation"]
            },
            "people_analysis": {
                "people_analysis": [
                    {
                        "person_id": "P1",
                        "dominant_emotion": "neutral",
                        "appears_speaking": True,
                        "emotional_changes": "No significant changes"
                    },
                    {
                        "person_id": "P2",
                        "dominant_emotion": "positive",
                        "appears_speaking": True,
                        "emotional_changes": "Becomes more animated when speaking"
                    }
                ]
            },
            "visual_event_detection": {
                "event_instances": [
                    {
                        "start_time": 0.0,
                        "end_time": 5.2,
                        "event_description": "Person talking and gesturing",
                        "confidence": 0.9,
                        "visual_details": {
                            "actions": ["talking", "hand_gesturing"],
                            "objects": ["person", "chair"],
                            "scene_context": "Indoor conversation"
                        }
                    },
                    {
                        "start_time": 8.5,
                        "end_time": 10.0,
                        "event_description": "Person standing up from chair",
                        "confidence": 0.85,
                        "visual_details": {
                            "actions": ["standing", "moving"],
                            "objects": ["person", "chair"],
                            "scene_context": "Person changing position"
                        }
                    },
                    {
                        "start_time": 10.5,
                        "end_time": 15.0,
                        "event_description": "Person walking and continuing conversation",
                        "confidence": 0.8,
                        "visual_details": {
                            "actions": ["walking", "talking"],
                            "objects": ["person"],
                            "scene_context": "Mobile conversation"
                        }
                    }
                ]
            }
        }

    @staticmethod
    def get_mock_level_3_result() -> Dict[str, Any]:
        return {
            "audio_visual_consistency": {
                "lip_sync_analysis": {
                    "overall_score": 0.85,
                    "segments": [
                        {"start": 0.0, "end": 5.2, "score": 0.9, "speaker": "Speaker_1"},
                        {"start": 5.5, "end": 10.1, "score": 0.8, "speaker": "Speaker_2"}
                    ]
                },
                "scene_audio_match": {
                    "score": 0.9,
                    "analysis": "Audio content matches visual scene well"
                }
            }
        }

    @staticmethod
    def get_mock_level_4_result() -> Dict[str, Any]:
        return {
            "visual_audio_consistency": {
                "action_sound_match": {
                    "score": 0.88,
                    "matches": [
                        {"action": "talking", "sound": "speech", "match_score": 0.95},
                        {"action": "walking", "sound": "footsteps", "match_score": 0.8}
                    ]
                },
                "ambient_consistency": {
                    "score": 0.85,
                    "analysis": "Visual environment matches audio ambience"
                }
            }
        }

    @staticmethod
    def get_mock_level_5_result() -> Dict[str, Any]:
        return {
            "comprehensive_report": {
                "evaluation_metadata": {
                    "video_path": "mock_video.mp4",
                    "evaluation_method": "Three-stage scoring system",
                    "evaluation_timestamp": "{\"timestamp\": \"mock\"}"
                },
                "key_variables_summary": {
                    "audio_macro_overview": {
                        "scene_hypothesis": "Indoor conversation",
                        "speaker_emotions": {"speaker_1": "calm", "speaker_2": "friendly"},
                        "background_music": {"detected": True, "type": "light music"},
                        "other_sounds": {"detected": True, "events": ["keyboard typing", "cup impact"]},
                        "total_speakers": 2
                    },
                    "video_macro_overview": {
                        "scene_description": "Modern office with two people talking",
                        "speaker_emotions": {"speaker_1": "focused", "speaker_2": "relaxed"},
                        "visual_atmosphere": "bright and professional",
                        "camera_movements": ["static shot", "subtle zoom"],
                        "lighting_conditions": "daylight and indoor lighting"
                    },
                    "audio_event_matching": {
                        "match_rate": 0.85,
                        "matched_events": ["speech", "background music"],
                        "unmatched_events": ["slight echo"]
                    },
                    "video_event_matching": {
                        "match_rate": 0.88,
                        "matched_events": ["lip synchronization", "hand gestures"],
                        "unmatched_events": ["unnatural blink rate"]
                    }
                },
                "assessment_results": {
                    "orthogonal_dimension_scores": {
                        "macro_semantic_alignment": {"score": 0.87},
                        "temporal_synchronicity": {"score": 0.85},
                        "fine_grained_event_correspondence": {"score": 0.89}
                    },
                    "bad_case_labels": ["minor background-effect mismatch"],
                    "analysis_process_review": {
                        "evaluation_methodology": "Three-dimensional orthogonal evaluation",
                        "key_observations": ["overall consistency is good", "details are well aligned"],
                        "critical_findings": ["audio-video synchronization is strong"],
                        "consistency_assessment": "high audio-video consistency",
                        "recommendations": ["further refine background effects"]
                    }
                },
                "final_scores": {
                    "macro_semantic_alignment": 0.87,
                    "temporal_synchronicity": 0.85,
                    "fine_grained_event_correspondence": 0.89,
                    "overall_consistency_score": 0.87
                },
                "quality_assessment": {"level": "Good", "description": "Audio and video are generally consistent."}
            },
            "final_report": {
                "executive_summary": {
                    "overall_score": 0.87,
                    "quality_level": "Good",
                    "primary_strengths": ["strong lip synchronization", "clear audio quality"],
                    "primary_weaknesses": ["minor background-effect issue"],
                    "recommendation": "Overall quality is good; refine the background effects."
                },
                "key_variables": {
                    "audio_macro_overview": {
                        "scene_hypothesis": "Indoor conversation",
                        "speaker_emotions": {"speaker_1": "calm", "speaker_2": "friendly"},
                        "total_speakers": 2
                    },
                    "video_macro_overview": {
                        "scene_description": "Modern office with two people talking",
                        "speaker_emotions": {"speaker_1": "focused", "speaker_2": "relaxed"}
                    }
                },
                "model_assessment": {
                    "orthogonal_dimension_scores": {
                        "macro_semantic_alignment": {"score": 0.87},
                        "temporal_synchronicity": {"score": 0.85},
                        "fine_grained_event_correspondence": {"score": 0.89}
                    }
                },
                "final_score_breakdown": {
                    "score": 0.87,
                    "interpretation": "Good audio-video consistency",
                    "confidence_level": "High"
                }
            },
            "overall_consistency_score": 0.87
        }
