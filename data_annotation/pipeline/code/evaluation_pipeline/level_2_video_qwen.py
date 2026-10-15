from .api_handler_final_qwen import call_llm_api
import json

def analyze_video(video_path):

    total_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def accumulate_tokens(token_usage):
        if token_usage:
            total_token_usage["input_tokens"] += token_usage.get("input_tokens", 0)
            total_token_usage["output_tokens"] += token_usage.get("output_tokens", 0)
            total_token_usage["total_tokens"] += token_usage.get("total_tokens", 0)

    all_event_instances = []

    prompt_macro = """
    Analyze this video and provide a concise overview of:
    1. Scene setting and environment
    2. Key events or activities happening
    3. People present and their general actions

    Return in JSON format:
    {
        "scene_overview": "Brief description of the overall scene and setting",
        "key_events": ["List of main events or activities observed"],
        "people_and_actions": ["Description of people present and their general actions"]
    }
    """
    macro_result, token_usage = call_llm_api(prompt_macro, video_path, 'video/mp4')
    accumulate_tokens(token_usage)

    prompt_emotion = """
    Analyze the people in this video and identify:
    1. Each person's general emotional state
    2. Whether they appear to be speaking
    3. Any significant emotional changes

    Return in JSON format:
    {
        "people_analysis": [
            {
                "person_id": "P1, P2, etc.",
                "dominant_emotion": "Main emotion observed",
                "appears_speaking": true/false,
                "emotional_changes": "Brief description of any notable changes"
            }
        ]
    }
    """
    emotion_result, token_usage = call_llm_api(prompt_emotion, video_path, 'video/mp4')
    accumulate_tokens(token_usage)

    prompt_identify = """
    Identify visual events that could produce audio. Focus ONLY on events that likely generate sound.

    AUDIO-PRODUCING EVENTS:
    - Speaking, talking, conversation
    - Musical instruments being played
    - Object interactions (hitting, dropping, clapping)
    - Environmental sounds (footsteps, doors, machinery)

    IGNORE silent visual events like: facial expressions, silent gestures, background movements

    Return in JSON format:
    {
        "visual_events": [
            {
                "event_description": "Brief description of the audio-producing action",
                "event_type": "speech|music|interaction|environment"
            }
        ]
    }
    """
    events_result, token_usage = call_llm_api(prompt_identify, video_path, 'video/mp4')
    accumulate_tokens(token_usage)

    if events_result:
        filtered_events = []
        if isinstance(events_result, dict) and 'visual_events' in events_result and isinstance(events_result['visual_events'], list):
            filtered_events = events_result['visual_events']
        elif isinstance(events_result, dict) and ('event_description' in events_result and 'event_type' in events_result):
            filtered_events = [events_result]

        for event in filtered_events:
            event_desc = event.get('event_description')
            if not event_desc:
                continue
            prompt_validate = f'''
            Does this audio-producing event exist in the video: "{event_desc}"?

            Return JSON:
            {{
                "is_present": true/false,
                "confidence": 0.0-1.0
            }}'''
            validation, token_usage = call_llm_api(prompt_validate, video_path, 'video/mp4')
            accumulate_tokens(token_usage)

            proceed_detail = False
            val_conf = 0.0
            if validation and isinstance(validation, dict):
                val_conf = float(validation.get('confidence', 0) or 0)
                if validation.get('is_present') and val_conf >= 0.6:
                    proceed_detail = True
            if not proceed_detail:
                etype = event.get('event_type')
                if etype in ('music', 'speech'):
                    if not validation or val_conf >= 0.5:
                        proceed_detail = True

            if proceed_detail:
                prompt_detail = f'''Find all occurrences of this audio-producing event: "{event_desc}"

                Provide timestamps and audio-relevant details.

                Important: The video is provided as an ordered sequence of image frames sampled at 2 FPS (one frame every 0.5 seconds). Calculate "start_time" and "end_time" based on this sampling rate. If an occurrence spans N frames, approximate duration as N * 0.5 seconds.

                Return JSON:
                {{
                    "event_instances": [
                        {{
                            "start_time": float,
                            "end_time": float,
                            "objective_description": "{event_desc}",
                            "visual_details": {{
                                "audio_source": "What/who is producing the sound",
                                "spatial_location": "Location in frame",
                                "audio_intensity": "Subtle|Moderate|Prominent"
                            }},
                            "confidence": float
                        }}
                    ]
                }}'''
                detail_result, token_usage = call_llm_api(prompt_detail, video_path, 'video/mp4')
                accumulate_tokens(token_usage)

                if detail_result and 'event_instances' in detail_result:
                    for instance in detail_result['event_instances']:
                        if 'confidence' not in instance:
                            instance['confidence'] = validation.get('confidence', 0.6) if validation else 0.6
                        all_event_instances.append(instance)
                else:
                    fallback_instance = {
                        "start_time": 0.0,
                        "end_time": 0.0,
                        "objective_description": event_desc,
                        "visual_details": {
                            "audio_source": "",
                            "spatial_location": "",
                            "audio_intensity": "Subtle"
                        },
                        "confidence": val_conf if val_conf > 0 else 0.6
                    }
                    all_event_instances.append(fallback_instance)

    analysis_result = {
        "macro_analysis": macro_result,
        "people_analysis": emotion_result,
        "visual_event_detection": {"event_instances": all_event_instances}
    }

    return analysis_result, total_token_usage

def analyze_video_with_tokens(video_path, use_mock=False, var_manager=None):

    if var_manager is None:
        from .variable_manager import VariableManager
        var_manager = VariableManager()

    logger = var_manager.setup_level_logger(2)

    if var_manager.variable_file_exists(2):
        analysis_result = var_manager.load_variables(2)
        if analysis_result:
            return analysis_result, []


    if use_mock:
        from .variable_manager import MockDataGenerator
        analysis_result = MockDataGenerator.get_mock_level_2_result()
        token_usage_list = []
    else:
        analysis_result, total_token_usage = analyze_video(video_path)
        token_usage_list = [total_token_usage] if total_token_usage else []

    if var_manager.save_variables(2, analysis_result):
        pass
    else:
        pass

    return analysis_result, token_usage_list

def run_level_2_standalone(video_path, use_mock=False, output_dir="pipeline_outputs"):
    from .variable_manager import VariableManager

    var_manager = VariableManager(output_dir)
    logger = var_manager.setup_level_logger(2)

    logger.info("=" * 60)
    logger.info("=" * 60)

    try:
        result, token_usage = analyze_video_with_tokens(
            video_path, use_mock=use_mock, var_manager=var_manager
        )

        if token_usage:
            total_tokens = sum(usage.get('total_tokens', 0) for usage in token_usage)

        return result

    except Exception as e:
        raise
