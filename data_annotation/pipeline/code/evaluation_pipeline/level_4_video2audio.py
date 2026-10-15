import os
from .api_handler_final_gemini import call_llm_api
from .variable_manager import VariableManager, MockDataGenerator

def get_audio_mime_type(audio_path):
    if audio_path.lower().endswith('.wav'):
        return 'audio/wav'
    elif audio_path.lower().endswith('.mp3'):
        return 'audio/mp3'
    else:
        return 'audio/mp3'

def verify_timbre_match(audio_path, video_event_desc, start_time, end_time):

    audio_mime_type = get_audio_mime_type(audio_path)

    prompt_timbre_match = f"""
    Analyze audio physical properties in {start_time:.2f}s–{end_time:.2f}s for visual action "{video_event_desc}".

    Focus on measurable characteristics: frequency spectrum, amplitude dynamics, physical generation mechanism, material properties.

    Return JSON:
    {{
    "video_event": "{video_event_desc}",
    "physical_compatibility": true/false,
    "confidence_score": 0.0-1.0,
    "reasoning": "..."
    }}"""

    try:
        result, token_usage = call_llm_api(prompt_timbre_match, audio_path, audio_mime_type)
        return result, token_usage
    except Exception as e:
        return None, None

def verify_temporal_match(audio_path, video_event_desc, start_time, end_time):

    audio_mime_type = get_audio_mime_type(audio_path)

    prompt_temporal_match = f"""You are an audio-video synchronization expert. Analyze if the audio timing synchronizes with visual event "{video_event_desc}" in time range {start_time:.2f}s to {end_time:.2f}s.

    Check:
    1. When audio events occur
    2. Duration consistency with visual action
    3. Timing alignment

    Return JSON:
    {{
    "video_event": "{video_event_desc}",
    "temporal_analysis": {{
        "observed_audio_timing": "...",
        "duration_consistency": "...",
        "temporal_synchronization": true/false,
        "confidence_score": 0.0-1.0
    }},
    "reasoning": "..."
    }}"""

    try:
        result, token_usage = call_llm_api(prompt_temporal_match, audio_path, audio_mime_type)
        return result, token_usage
    except Exception as e:
        return None, None

def verify_video_in_audio(audio_path, video_analysis_results):

    total_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def accumulate_tokens(token_usage):
        if token_usage:
            total_token_usage["input_tokens"] += token_usage.get("input_tokens", 0)
            total_token_usage["output_tokens"] += token_usage.get("output_tokens", 0)
            total_token_usage["total_tokens"] += token_usage.get("total_tokens", 0)

    verification_results = {
        "audio_file": audio_path,
        "video_driven_verification_results": [],
        "overall_consistency": {
            "total_events": 0,
            "verified_events": 0,
            "consistency_score": 0.0
        }
    }

    if 'visual_events' in video_analysis_results:
        video_events = video_analysis_results.get('visual_events', [])
    elif 'visual_event_detection' in video_analysis_results:
        video_events = video_analysis_results.get('visual_event_detection', {}).get('event_instances', [])
    else:
        video_events = []

    verification_results["overall_consistency"]["total_events"] = len(video_events)

    for event in video_events:

        event_desc = event.get('event_description') or event.get('objective_description', '')

        try:
            start_time = float(event.get('start_time', 0)) if event.get('start_time') is not None else 0.0
            end_time = float(event.get('end_time', 0)) if event.get('end_time') is not None else 0.0
        except (ValueError, TypeError) as e:
            continue


        existence_result, existence_token_usage = check_audio_existence(audio_path, event_desc, start_time, end_time)
        accumulate_tokens(existence_token_usage)

        event_verification = {
            "video_event_to_verify": {
                "description": event_desc,
                "start_time": start_time,
                "end_time": end_time
            },
            "existence_check": existence_result,
            "physical_match": None,
            "is_match": False,
            "reasoning": ""
        }

        is_reasonable_silence = existence_result and existence_result.get('reasonable_silence', False)
        audio_evidence_exists = existence_result and existence_result.get('audio_evidence_exists', False)

        if is_reasonable_silence:
            event_verification["is_match"] = True
            event_verification["reasoning"] = "Naturally silent action; verification passed"
            verification_results["overall_consistency"]["verified_events"] += 1

        elif audio_evidence_exists:

            timbre_result, timbre_token_usage = verify_timbre_match(audio_path, event_desc, start_time, end_time)
            accumulate_tokens(timbre_token_usage)

            temporal_result, temporal_token_usage = verify_temporal_match(audio_path, event_desc, start_time, end_time)
            accumulate_tokens(temporal_token_usage)

            event_verification["physical_match"] = {
                "timbre_match": timbre_result,
                "temporal_match": temporal_result
            }

            timbre_pass = timbre_result and timbre_result.get('physical_compatibility', False)
            temporal_pass = temporal_result and temporal_result.get('temporal_analysis', {}).get('temporal_synchronization', False)

            event_verification["is_match"] = timbre_pass and temporal_pass

            if event_verification["is_match"]:
                verification_results["overall_consistency"]["verified_events"] += 1
                event_verification["reasoning"] = "Physical consistency verification passed"
            else:
                event_verification["reasoning"] = "Physical consistency verification failed"
        else:

            event_verification["reasoning"] = "Existence check failed: the action should produce sound, but no audio evidence was found"

        verification_results["video_driven_verification_results"].append(event_verification)

    total = verification_results["overall_consistency"]["total_events"]
    verified = verification_results["overall_consistency"]["verified_events"]
    verification_results["overall_consistency"]["consistency_score"] = verified / total if total > 0 else 0.0

    return verification_results, total_token_usage

def check_audio_existence(audio_path, video_event_desc, start_time, end_time):

    audio_mime_type = get_audio_mime_type(audio_path)

    prompt_existence_check = f"""You are an expert in audio-visual analysis. Analyze the visual action "{video_event_desc}" in time range {start_time:.2f}s to {end_time:.2f}s and determine:

1. Whether this visual action should naturally produce sound
2. Whether corresponding audio evidence exists in the audio

Consider these categories:
- SILENT ACTIONS (naturally silent): static scenes, gentle hand waving, facial expressions, slow movements, reading, thinking, looking around, etc.
- SOUND-PRODUCING ACTIONS: speaking, singing, clapping, footsteps, object impacts, musical instruments, mechanical sounds, etc.

For silent actions, absence of audio evidence is REASONABLE and should be marked as "reasonable_silence".
For sound-producing actions, absence of audio evidence indicates a mismatch.

Return JSON:
{{
    "action_type_description": "{video_event_desc}",
    "action_category": "silent_action" or "sound_producing_action",
    "audio_evidence_exists": true/false,
    "reasonable_silence": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation of the categorization and audio analysis"
}}"""

    try:
        result, token_usage = call_llm_api(prompt_existence_check, audio_path, audio_mime_type)
        return result, token_usage
    except Exception as e:
        return None, None

def verify_video_in_audio_with_tokens(audio_path, video_analysis_results, use_mock=False, var_manager=None):

    if var_manager is None:
        var_manager = VariableManager()

    logger = var_manager.setup_level_logger(4)

    if var_manager.variable_file_exists(4):
        verification_results = var_manager.load_variables(4)
        if verification_results:
            return verification_results, []


    if use_mock:
        verification_results = MockDataGenerator.get_mock_level_4_result()
        token_usage_list = []
    else:
        verification_results, total_token_usage = verify_video_in_audio(audio_path, video_analysis_results)
        token_usage_list = [total_token_usage] if total_token_usage else []

    if var_manager.save_variables(4, verification_results):
        pass
    else:
        pass

    return verification_results, token_usage_list

def run_level_4_standalone(audio_path, video_analysis_results=None, use_mock=False, output_dir="pipeline_outputs"):
    var_manager = VariableManager(output_dir)
    logger = var_manager.setup_level_logger(4)

    logger.info("=" * 60)
    logger.info("=" * 60)

    try:

        if video_analysis_results is None:
            video_analysis_results = var_manager.load_variables(2)
            if video_analysis_results is None:
                raise ValueError("Level 2 video analysis is unavailable; run Level 2 first or provide its result")

        result, token_usage = verify_video_in_audio_with_tokens(
            audio_path, video_analysis_results,
            use_mock=use_mock, var_manager=var_manager
        )

        if token_usage:
            total_tokens = sum(usage.get('total_tokens', 0) for usage in token_usage)

        return result

    except Exception as e:
        raise
