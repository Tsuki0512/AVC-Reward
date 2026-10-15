import os
import cv2

from .api_handler_final_gemini import call_llm_api

def get_first_frame(video_path: str, output_image_path: str):
    try:

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.imwrite(output_image_path, frame_rgb)
            cap.release()
            return output_image_path
        cap.release()
        return None
    except Exception as e:
        return None

def verify_timbre_match(video_path, audio_event_desc, start_time, end_time):
    prompt_timbre_match = f"""
    You are an acoustics expert. Analyze if the visual actions in the time window {start_time:.2f}s–{end_time:.2f}s can produce the timbre of "{audio_event_desc}" detected during this same period. Focus only on what happens visually within this exact interval.

    Check:
    1. Materials involved in visual actions
    2. How these materials would generate sound
    3. Whether the expected timbre matches the audio event

    Return JSON:
    {{
    "audio_event": "{audio_event_desc}",
    "timbre_analysis": {{
        "observed_materials": "...",
        "sound_generation_method": "...",
        "timbre_compatibility": true/false,
        "confidence_score": 0.0-1.0
    }},
    "reasoning": "..."
    }}"""

    try:
        result, token_usage = call_llm_api(prompt_timbre_match, video_path, 'video/mp4')
        return result, token_usage
    except Exception as e:
        return None, None

def verify_temporal_match(video_path, audio_event_desc, start_time, end_time):

    prompt_temporal_match = f"""You are an audio-video synchronization expert. Analyze if the visual actions synchronize with "{audio_event_desc}" in time range {start_time:.2f}s to {end_time:.2f}s.

    Check:
    1. When visual actions occur
    2. Duration consistency with expected audio
    3. Timing alignment with audio generation

    Return JSON:
    {{
    "audio_event": "{audio_event_desc}",
    "temporal_analysis": {{
        "observed_action_timing": "...",
        "duration_consistency": "...",
        "temporal_synchronization": true/false,
        "confidence_score": 0.0-1.0
    }},
    "reasoning": "..."
    }}"""

    try:
        result, token_usage = call_llm_api(prompt_temporal_match, video_path, 'video/mp4')
        return result, token_usage
    except Exception as e:
        return None, None

def verify_audio_in_video(video_path, audio_analysis_results, video_analysis_results=None):

    total_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def accumulate_tokens(token_usage):
        if token_usage:
            total_token_usage["input_tokens"] += token_usage.get("input_tokens", 0)
            total_token_usage["output_tokens"] += token_usage.get("output_tokens", 0)
            total_token_usage["total_tokens"] += token_usage.get("total_tokens", 0)

    results = []

    audio_events = audio_analysis_results.get("acoustic_event_detection", {}).get("event_instances", [])

    unique_id = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    temp_img_path = os.path.join(video_dir, f"temp_first_frame_{unique_id}.jpg")

    for event in audio_events:

        try:
            start = float(event['start_time']) if event['start_time'] is not None else 0.0
            end = float(event['end_time']) if event['end_time'] is not None else 0.0
        except (ValueError, TypeError) as e:
            continue

        if 'objective_description' in event:
            desc = event['objective_description']
        elif 'texture' in event and event['texture']:

            desc = f"{event.get('layer_type', 'Audio Event')}: {event['texture']}"
        elif 'layer_type' in event:
            desc = event['layer_type']
        else:
            continue


        event_result = {
            "audio_event_to_verify": {
                "start_time": start,
                "end_time": end,
                "description": desc
            },
            "visual_evidence_found": False,
            "evidence_verification_details": [],
            "is_match": False,
            "reasoning": "",
            "emotion_consistency": None
        }

        event_duration = end - start
        is_transient_event = event_duration <= 2.0

        if is_transient_event:


            prompt_transient_check = f"""You are an audio-video synchronization expert. Check if there's visual evidence for "{desc}" during {start:.2f}s to {end:.2f}s.

            This is a transient event (duration: {event_duration:.2f}s). Look for any visual activity that could generate this audio.

            Return JSON:
            {{
            "audio_event": "{desc}",
            "time_window": "{start:.2f}s - {end:.2f}s",
            "visual_evidence_found": true/false,
            "observed_visual_activities": "...",
            "confidence": 0.0-1.0,
            "reasoning": "..."
            }}"""

            transient_result, token_usage = call_llm_api(prompt_transient_check, video_path, 'video/mp4')
            accumulate_tokens(token_usage)

            if transient_result:

                evidence_detail = {
                    "evidence_description": f"Visual evidence for transient event '{desc}' in time window {start:.2f}s-{end:.2f}s",
                    "existence_check": {
                        "is_present": transient_result.get('visual_evidence_found', False),
                        "confidence": transient_result.get('confidence', 0.0),
                        "observed_activities": transient_result.get('observed_visual_activities', ''),
                        "evidence_quality": transient_result.get('evidence_quality', '')
                    },
                    "physical_match": None
                }

                evidence_details = [evidence_detail]
                existence_check_passed = transient_result.get('visual_evidence_found', False)

                if existence_check_passed:
                    pass
                else:
                    pass
            else:
                event_result["reasoning"] = "Transient-event analysis API call failed"
                results.append(event_result)
                continue
        else:


            first_frame_path = get_first_frame(video_path, temp_img_path)
            if not first_frame_path:
                event_result["reasoning"] = "Unable to extract the first video frame for scene analysis"
                results.append(event_result)
                continue

            prompt_get_evidence = f"""You are a visual environment analyst. Based on the first frame scene, identify the main visual evidence that would convincingly prove the audio event "{desc}" is credible.

            Guidelines:
            - ONLY refer to objects that are clearly visible in the first-frame scene – do **NOT** invent unseen objects
            - Keep the evidence list minimal:
              • If the audio event can be generated by a single object/action, list **at most one** object
              • For compound events that truly require interaction (e.g. "striking a drum with sticks"), list **no more than two** interacting objects
            - Avoid redundant or overlapping evidence phrases; do not repeat similar motions for the same object
            - Each evidence description must be a short clause of an observable **dynamic** process (movement, vibration, collision, deformation…)
            - Do not add background or irrelevant context

            Example (simple event): For "dog barking", visual evidence: "object:dog, necessary_visual_evidence:dog’s mouth opening and closing while barking"
            Example (compound event): For "playing piano", visual evidence includes:
              "object:fingers, necessary_visual_evidence:hands moving up and down on keys" and
              "object:piano, necessary_visual_evidence:piano keys being pressed"

            Identify the minimal set of objects and their key visual evidence needed to confirm this audio event.

            Return JSON: {{
                "interacting_objects": [
                    {{
                        "object": "...",
                        "necessary_visual_evidence": [
                            "...",
                            "..."
                        ]
                    }},
                    {{
                        "object": "...",
                        "necessary_visual_evidence": [
                            "...",
                            "..."
                        ]
                    }}
                    ...
                ]
            }}"""

            evidence_result, token_usage = call_llm_api(prompt_get_evidence, first_frame_path, 'image/jpeg')
            accumulate_tokens(token_usage)

            if not (evidence_result and 'interacting_objects' in evidence_result and evidence_result['interacting_objects']):
                event_result["reasoning"] = "Unable to determine the visual evidence required for this audio event"
                results.append(event_result)
                continue

            required_evidence_list = evidence_result['interacting_objects']

            evidence_details = []
            existence_check_passed = True

            for evidence in required_evidence_list:
                for evidence_desc in evidence.get('necessary_visual_evidence', []):

                    prompt_existence_check = f"""
                    You are a meticulous and extremely objective fact checker. Your task is to analyze only the direct visual evidence contained in the video during the time window {start:.2f}s to {end:.2f}s. Please watch the following video carefully and clearly answer whether the visual action of "{evidence_desc}" is clear and visible specifically during this time period ({start:.2f}s-{end:.2f}s). Your judgment must be strictly limited to the physical displacement of the visual action subject itself within this exact time window, and you must completely ignore all other relevant visual information outside this timeframe. It is strictly prohibited to make any form of reasoning based on logic, such as "if a is moving, then b must also be moving". If the action cannot be clearly observed in the video during this specific time window due to issues such as angle or clarity, it should be determined that there is no direct visual evidence.

                    Please return in JSON format without any other text: {{ "evidence_description": "{evidence_desc}", "time_window": "{start:.2f}s-{end:.2f}s", "is_present": true/false, "confidence": ... }}"""

                    existence_result, token_usage = call_llm_api(prompt_existence_check, video_path, 'video/mp4')
                    accumulate_tokens(token_usage)

                    evidence_detail = {
                        "evidence_description": evidence_desc,
                        "existence_check": {
                            "is_present": False,
                            "confidence": 0.0
                        },
                        "physical_match": None
                    }

                    if existence_result:
                        evidence_detail["existence_check"]["is_present"] = existence_result.get('is_present', False)
                        evidence_detail["existence_check"]["confidence"] = existence_result.get('confidence', 0.0)

                    evidence_details.append(evidence_detail)

                    if not evidence_detail["existence_check"]["is_present"]:
                        existence_check_passed = False
                    else:
                        pass

        is_event_validated = False
        if existence_check_passed:

            timbre_match_result, timbre_token_usage = verify_timbre_match(video_path, desc, start, end)
            accumulate_tokens(timbre_token_usage)

            temporal_match_result, temporal_token_usage = verify_temporal_match(video_path, desc, start, end)
            accumulate_tokens(temporal_token_usage)

            if timbre_match_result and temporal_match_result:

                for evidence_detail in evidence_details:
                    evidence_detail["physical_match"] = {
                        "timbre_analysis": timbre_match_result,
                        "temporal_analysis": temporal_match_result,
                        "overall_physical_compatibility": (
                            timbre_match_result.get("timbre_analysis", {}).get("timbre_compatibility", False) and
                            temporal_match_result.get("temporal_analysis", {}).get("temporal_synchronization", False)
                        ),
                        "confidence_score": min(
                            timbre_match_result.get("timbre_analysis", {}).get("confidence_score", 0.0),
                            temporal_match_result.get("temporal_analysis", {}).get("confidence_score", 0.0)
                        ),
                        "reasoning": f"Timbre match: {timbre_match_result.get('reasoning', '')}; temporal match: {temporal_match_result.get('reasoning', '')}"
                    }

                is_event_validated = (
                    timbre_match_result.get("timbre_analysis", {}).get("timbre_compatibility", False) and
                    temporal_match_result.get("temporal_analysis", {}).get("temporal_synchronization", False)
                )

                if is_event_validated:
                    pass
                else:
                    pass
            else:
                pass
        else:
            pass

        emotion_consistency_result = None
        if "Human Speech" in desc or "speech" in desc.lower() or "voice" in desc.lower():

            video_emotions = []
            if video_analysis_results and 'people_analysis' in video_analysis_results:
                people_data = video_analysis_results['people_analysis']
                if people_data and 'people_analysis' in people_data:
                    for person in people_data['people_analysis']:
                        if person.get('appears_speaking', False):
                            video_emotions.append(person.get('dominant_emotion', 'neutral'))

            audio_emotion = 'neutral'
            speaker_id = event.get('speaker_id')
            if speaker_id and audio_analysis_results:

                macro_analysis = audio_analysis_results.get('macro_analysis', {})
                speaker_emotions = macro_analysis.get('speaker_emotions', [])
                for speaker_emotion in speaker_emotions:
                    if speaker_emotion.get('speaker_id') == speaker_id:
                        audio_emotion = speaker_emotion.get('emotion', 'neutral')
                        break
                else:
                    pass
            else:
                pass

            prompt_emotion_consistency = f"""You are an emotion analysis expert. Analyze the consistency between audio emotion and visual emotion of the speaker in the video during {start:.2f}s to {end:.2f}s.

            Audio emotion detected: {audio_emotion}
            Video emotions observed: {video_emotions if video_emotions else 'Not available'}

            Return JSON:
            {{
            "audio_emotion": "{audio_emotion}",
            "video_emotions": {video_emotions if video_emotions else []},
            "emotion_consistency": {{
                "is_consistent": true/false,
                "consistency_score": 0.0-1.0,
                "reasoning": "..."
            }}
            }}"""

            emotion_consistency_result, emotion_token_usage = call_llm_api(prompt_emotion_consistency, video_path, 'video/mp4')
            accumulate_tokens(emotion_token_usage)

            if emotion_consistency_result:
                pass
            else:
                pass

        event_result.update({
            "visual_evidence_found": existence_check_passed,
            "evidence_verification_details": evidence_details,
            "is_match": is_event_validated,
            "reasoning": f"Existence check: {'passed' if existence_check_passed else 'failed'}; physical consistency: {'passed' if is_event_validated else 'failed'}",
            "emotion_consistency": emotion_consistency_result
        })

        results.append(event_result)

        if os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except:
                pass

    return {"audio_driven_verification_results": results}, total_token_usage

def verify_audio_in_video_with_tokens(video_path, audio_analysis_results, video_analysis_results=None, use_mock=False, var_manager=None):

    if var_manager is None:
        from .variable_manager import VariableManager
        var_manager = VariableManager()

    logger = var_manager.setup_level_logger(3)

    if var_manager.variable_file_exists(3):
        verification_results = var_manager.load_variables(3)
        if verification_results:
            return verification_results, []


    if use_mock:
        from .variable_manager import MockDataGenerator
        verification_results = MockDataGenerator.get_mock_level_3_result()
        token_usage_list = []
    else:
        verification_results, total_token_usage = verify_audio_in_video(video_path, audio_analysis_results, video_analysis_results)
        token_usage_list = [total_token_usage] if total_token_usage else []

    if var_manager.save_variables(3, verification_results):
        pass
    else:
        pass

    return verification_results, token_usage_list

def run_level_3_standalone(video_path, audio_analysis_results, video_analysis_results=None, use_mock=False, output_dir="pipeline_outputs"):
    from .variable_manager import VariableManager

    var_manager = VariableManager(output_dir)
    logger = var_manager.setup_level_logger(3)

    logger.info("=" * 60)
    logger.info("=" * 60)

    try:

        if audio_analysis_results is None:
            audio_analysis_results = var_manager.load_variables(1)
            if audio_analysis_results is None:
                raise ValueError("Level 1 audio analysis is unavailable; run Level 1 first or provide its result")

        if video_analysis_results is None:
            video_analysis_results = var_manager.load_variables(2)

        result, token_usage = verify_audio_in_video_with_tokens(
            video_path, audio_analysis_results, video_analysis_results,
            use_mock=use_mock, var_manager=var_manager
        )

        if token_usage:
            total_tokens = sum(usage.get('total_tokens', 0) for usage in token_usage)

        return result

    except Exception as e:
        raise
