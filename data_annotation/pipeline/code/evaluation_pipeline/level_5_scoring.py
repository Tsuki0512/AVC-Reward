import json
try:
    from .api_handler_final_gemini import call_llm_api
    from .variable_manager import VariableManager, MockDataGenerator
except ImportError:

    from api_handler import call_llm_api
    from variable_manager import VariableManager, MockDataGenerator

def generate_final_score(original_video_path, all_previous_results):
    result, _ = generate_final_score_with_tokens(original_video_path, all_previous_results)
    return result

def generate_final_score_with_tokens(original_video_path, all_previous_results, use_mock=False, var_manager=None):

    if var_manager is None:
        var_manager = VariableManager()

    logger = var_manager.setup_level_logger(5)

    logger.info("=" * 60)
    logger.info("=" * 60)

    if use_mock:
        final_result = MockDataGenerator.get_mock_level_5_result()
        token_usage_list = []

        if var_manager.save_variables(5, final_result):
            pass
        else:
            pass

        return final_result, token_usage_list

    token_usage_list = []

    def accumulate_tokens(token_usage):
        if token_usage:
            token_usage_list.append(token_usage)

    assessment_result, assessment_token_usage = perform_single_assessment_with_tokens(
        original_video_path, key_variables)
    accumulate_tokens(assessment_token_usage)

    final_result = calculate_final_score_and_report(
        original_video_path, key_variables, assessment_result, var_manager)

    if var_manager.save_variables(5, final_result):
        pass
    else:
        pass

    return final_result, token_usage_list

def aggregate_key_variables(var_manager, logger):

    level_1_data = var_manager.load_variables(1)
    level_2_data = var_manager.load_variables(2)
    level_3_data = var_manager.load_variables(3)
    level_4_data = var_manager.load_variables(4)

    missing_levels = []
    if not level_1_data:
        missing_levels.append(1)
    if not level_2_data:
        missing_levels.append(2)
    if not level_3_data:
        missing_levels.append(3)
    if not level_4_data:
        missing_levels.append(4)

    if missing_levels:
        pass

    audio_macro_overview = extract_audio_macro_overview(level_1_data, logger)

    video_macro_overview = extract_video_macro_overview(level_2_data, logger)

    audio_event_matching = extract_audio_event_matching(level_3_data, logger)

    video_event_matching = extract_video_event_matching(level_4_data, logger)

    key_variables = {
        "audio_macro_overview": audio_macro_overview,
        "video_macro_overview": video_macro_overview,
        "audio_event_matching": audio_event_matching,
        "video_event_matching": video_event_matching,
        "data_completeness": {
            "available_levels": [i for i in range(1, 5) if i not in missing_levels],
            "missing_levels": missing_levels
        }
    }

    print_key_variables_summary(key_variables, logger)

    return key_variables

def extract_audio_macro_overview(level_1_data, logger):
    if not level_1_data:
        return {
            "scene_hypothesis": "Unknown scene",
            "speaker_emotions": {},
            "background_music": {"detected": False},
            "other_sounds": {"detected": False, "events": []},
            "total_speakers": 0
        }

    macro_analysis = level_1_data.get("macro_analysis", {})
    validated_layers = level_1_data.get("validated_layers_analysis", [])

    background_music = {"detected": False}
    human_voice_detected = False
    other_sounds = {"detected": False, "events": []}

    for layer in validated_layers:
        layer_type = layer.get("layer_type", "").lower()
        if "music" in layer_type:
            background_music = {"detected": True, "genre": layer.get("genre", "Unknown")}
        elif "voice" in layer_type or "speech" in layer_type:
            human_voice_detected = True
        elif layer_type not in ["music", "voice", "speech"]:
            other_sounds["detected"] = True
            other_sounds["events"].append(layer_type)

    scene_hypotheses = macro_analysis.get("scene_hypotheses", [])
    scene_hypothesis = scene_hypotheses[0].get("scene", "Unknown scene") if scene_hypotheses else "Unknown scene"

    return {
        "scene_hypothesis": scene_hypothesis,
        "speaker_emotions": macro_analysis.get("speaker_emotions", []),
        "background_music": background_music,
        "other_sounds": other_sounds,
        "total_speakers": len(macro_analysis.get("speaker_emotions", []))
    }

def extract_video_macro_overview(level_2_data, logger):
    if not level_2_data:
        return {
            "scene_overview": "Unknown scene",
            "people_emotions": {},
            "key_events": [],
            "total_people": 0,
            "people_analysis": []
        }

    macro_analysis = level_2_data.get("macro_analysis", {})
    people_analysis = level_2_data.get("people_analysis", {}).get("people_analysis", [])

    people_emotions = {}
    for person in people_analysis:
        person_id = person.get("person_id", "Unknown")
        emotion = person.get("dominant_emotion", "unknown")
        people_emotions[person_id] = emotion

    return {
        "scene_overview": macro_analysis.get("scene_overview", "Unknown scene"),
        "people_emotions": people_emotions,
        "key_events": macro_analysis.get("key_events", []),
        "total_people": len(people_analysis),

        "people_analysis": people_analysis
    }

def extract_audio_event_matching(level_3_data, logger):
    if not level_3_data:
        return {
            "matched_events": [],
            "unmatched_events": [],
            "match_rate": 0.0,
            "total_audio_events": 0,
            "verification_results": []
        }

    verification_results = level_3_data.get("audio_driven_verification_results", [])

    matched_events = []
    unmatched_events = []

    for result in verification_results:
        if result.get("is_match", False):
            matched_events.append({
                "audio_event": result.get("audio_event_to_verify", {}),
                "confidence": 1.0 if result.get("is_match", False) else 0.0
            })
        else:
            unmatched_events.append({
                "audio_event": result.get("audio_event_to_verify", {}),
                "reason": result.get("reasoning", "Unknown reason")
            })

    total_events = len(verification_results)
    match_rate = len(matched_events) / total_events if total_events > 0 else 0.0

    return {
        "matched_events": matched_events,
        "unmatched_events": unmatched_events,
        "match_rate": match_rate,
        "total_audio_events": total_events,

        "verification_results": verification_results
    }

def extract_video_event_matching(level_4_data, logger):
    if not level_4_data:
        return {
            "matched_events": [],
            "unmatched_events": [],
            "match_rate": 0.0,
            "total_video_events": 0
        }

    verification_results = level_4_data.get("video_driven_verification_results", [])

    matched_events = []
    unmatched_events = []

    for result in verification_results:
        if result.get("is_match", False):
            matched_events.append({
                "video_event": result.get("video_event_to_verify", {}),
                "confidence": 1.0 if result.get("is_match", False) else 0.0
            })
        else:
            unmatched_events.append({
                "video_event": result.get("video_event_to_verify", {}),
                "reason": result.get("reasoning", "Unknown reason")
            })

    total_events = len(verification_results)
    match_rate = len(matched_events) / total_events if total_events > 0 else 0.0

    return {
        "matched_events": matched_events,
        "unmatched_events": unmatched_events,
        "match_rate": match_rate,
        "total_video_events": total_events
    }

def print_key_variables_summary(key_variables, logger):

    audio_overview = key_variables["audio_macro_overview"]

    video_overview = key_variables["video_macro_overview"]

    audio_matching = key_variables["audio_event_matching"]

    video_matching = key_variables["video_event_matching"]

    completeness = key_variables["data_completeness"]
    if completeness['missing_levels']:
        pass

def perform_single_assessment(original_video_path, key_variables):
    result, _ = perform_single_assessment_with_tokens(original_video_path, key_variables)
    return result

def perform_single_assessment_with_tokens(original_video_path, key_variables):

    prompt = f"""
You are an expert audio-video consistency evaluator. Based on the provided key variables from previous analysis levels AND by carefully re-examining the original audio-video file, evaluate the consistency across three orthogonal dimensions.

CRITICAL: Your evaluation should be based on BOTH the provided analysis variables AND your direct observation of the original audio-video content. Re-watch and re-listen to the file carefully.

KEY VARIABLES FROM PREVIOUS ANALYSIS:
{json.dumps(key_variables, indent=2, ensure_ascii=False)}

MANDATORY PRE-CHECK: Off-Screen Voice-over Causal Chain Analysis
Before scoring the three dimensions, perform the following:
1) Traverse `audio_event_matching.verification_results` and select events with `visual_evidence_found: false` (off-screen audio events).
2) Determine significance for each off-screen event:
   - If `audio_event_to_verify.intensity` exists and is "High" or "Medium"; OR
   - Otherwise, use `audio_event_to_verify.description` and direct listening to decide. Consider triggers such as explosion, alarm, bang, scream, gunshot, crash, or siren.
   Significant off-screen sounds should reasonably induce a visual reaction in 2D slices.
3) Cross-validate visual reactions for significant off-screen events:
   - Query `video_macro_overview.key_events` and `video_macro_overview.people_analysis` (especially `emotional_changes`).
   - Check around `start_time` (±2 seconds) for a logically matching visual reaction, such as fear, camera shake, surprise, environmental disturbance, or an emotional change.
4) Identify an off-screen causal mismatch:
   - If a significant off-screen audio event (e.g., explosion sound) occurs at t=X but no matching visual reaction is found in [X-2s, X+2s], mark it as "Off-Screen Causal Mismatch".

Impact on output:
   - Scoring: Lower `macro_semantic_alignment.score` whenever Off-Screen Causal Mismatch is detected (deduct 0.1–0.3 per mismatch depending on severity, capped total deduction at ~0.5).
   - Reasoning: Mention this finding in `macro_semantic_alignment.reasoning` with its timestamp and a brief description. Explain that a salient off-screen sound lacks a logical visual reaction.
   - Bad Case Labels: Add `Scene atmosphere mismatch` and/or `Fine-grained event mismatch` when this mismatch occurs.

Please evaluate the following three orthogonal dimensions:

1. Macro-Semantic Alignment:
   - Evaluate semantic harmony between audio and visual content rather than exact label matching
   - Speaker emotion compatibility (audio tone and visual expression working together naturally)
   - Scene atmosphere coordination (background sounds fitting the visual environment)
   - Background sound effects appropriateness (sounds complementing rather than contradicting visuals)
   - Voice-over Appropriateness: If human voice is present but no speaker is visible (e.g., total_people is 0), assess if it's a reasonable voice-over for the scene (e.g., narration, documentary style). If so, this should be considered a positive alignment.

2. Temporal Synchronicity:
   - Do audio-video events occur simultaneously at specific time points?
   - Are identical audio-video events temporally consistent?
   - Speaker temporal matching (lip-sync, gesture timing)
   - Rhythmic matching (music and other continuous events)

3. Fine-grained Event Correspondence:
   - Do audio-video events at specific time points correspond reasonably?
   - Detailed event-level matching quality

After scoring each dimension, identify applicable bad case labels (0-8+ labels possible):
- Speaker emotion mismatch
- Scene atmosphere mismatch
- Background sound effects mismatch
- Inappropriate voice-over
- Audio-video event temporal mismatch
- Speaker temporal mismatch
- Rhythmic mismatch
- Fine-grained event mismatch

Finally, provide a highly readable analysis process review.

JSON Response:
{{
  "orthogonal_dimension_scores": {{
    "macro_semantic_alignment": {{
      "score": 0.0-1.0,
      "reasoning": "Brief macro-semantic alignment reasoning"
    }},
    "temporal_synchronicity": {{
      "score": 0.0-1.0,
      "reasoning": "Brief temporal synchronicity reasoning"
    }},
    "fine_grained_event_correspondence": {{
      "score": 0.0-1.0,
      "reasoning": "Brief fine-grained correspondence reasoning"
    }}
  }},
  "bad_case_labels": [
    // List of applicable bad case labels from the 7 categories
  ],
  "analysis_process_review": {{
    "evaluation_methodology": "Description of evaluation approach",
    "key_observations": [
      "Important observation 1",
      "Important observation 2"
    ],
    "critical_findings": [
      "Critical finding 1",
      "Critical finding 2"
    ],
    "consistency_assessment": "Overall consistency evaluation summary",
  }}
}}
"""

    result, token_usage = call_llm_api(prompt, original_video_path, 'video/mp4')

    if result:
        scores = result.get('orthogonal_dimension_scores', {})
        macro_score = scores.get('macro_semantic_alignment', {}).get('score', 0.0)
        temporal_score = scores.get('temporal_synchronicity', {}).get('score', 0.0)
        event_score = scores.get('fine_grained_event_correspondence', {}).get('score', 0.0)


        bad_cases = result.get('bad_case_labels', [])
        if bad_cases:
            pass
        else:
            pass

    return result, token_usage

def calculate_final_score_and_report(original_video_path, key_variables, assessment_result, var_manager):

    scores = assessment_result.get('orthogonal_dimension_scores', {})
    macro_score = scores.get('macro_semantic_alignment', {}).get('score', 0.0)
    temporal_score = scores.get('temporal_synchronicity', {}).get('score', 0.0)
    event_score = scores.get('fine_grained_event_correspondence', {}).get('score', 0.0)

    audio_match_rate = key_variables['audio_event_matching']['match_rate']
    video_match_rate = key_variables['video_event_matching']['match_rate']

    base_match_constraint = (audio_match_rate + video_match_rate) / 2.0

    if base_match_constraint < 0.3:

        weights = {'macro': 0.2, 'temporal': 0.3, 'event': 0.5}
    elif base_match_constraint < 0.6:

        weights = {'macro': 0.3, 'temporal': 0.35, 'event': 0.35}
    else:

        weights = {'macro': 0.5, 'temporal': 0.25, 'event': 0.25}

    weighted_score = (macro_score * weights['macro'] +
                     temporal_score * weights['temporal'] +
                     event_score * weights['event'])

    constraint_factor = 0.5 + 0.5 * base_match_constraint
    final_score = weighted_score * constraint_factor

    if "Inappropriate voice-over" in assessment_result.get('bad_case_labels', []):
        final_score *= 0.8


    report = generate_comprehensive_report(
        original_video_path, key_variables, assessment_result,
        final_score, weights, base_match_constraint)

    final_report = generate_final_report(
        key_variables, assessment_result, final_score)

    final_result = {
        "comprehensive_report": report,
        "final_report": final_report,
        "overall_consistency_score": final_score
    }

    return final_result

def generate_comprehensive_report(original_video_path, key_variables, assessment_result,
                                final_score, weights, base_match_constraint):
    scores = assessment_result.get('orthogonal_dimension_scores', {})

    return {
        "evaluation_metadata": {
            "video_path": original_video_path,
            "evaluation_method": "Three-stage scoring system",
            "evaluation_timestamp": json.dumps({"timestamp": "current"})
        },
        "key_variables_summary": key_variables,
        "assessment_results": assessment_result,
        "scoring_methodology": {
            "dimension_weights": weights,
            "base_match_constraint": base_match_constraint,
            "constraint_reasoning": "Use the mean match rate as a soft constraint with a strength between 0.5 and 1.0",
            "weight_adaptation": "Adapt dimension weights to the baseline match quality",
            "scoring_improvements": "Use the mean match rate to avoid over-penalizing a single dimension"
        },
        "final_scores": {
            "macro_semantic_alignment": scores.get('macro_semantic_alignment', {}).get('score', 0.0),
            "temporal_synchronicity": scores.get('temporal_synchronicity', {}).get('score', 0.0),
            "fine_grained_event_correspondence": scores.get('fine_grained_event_correspondence', {}).get('score', 0.0),
            "overall_consistency_score": final_score
        },
        "quality_assessment": get_quality_assessment(final_score),
        "bad_case_analysis": {
            "detected_issues": assessment_result.get('bad_case_labels', []),
            "issue_count": len(assessment_result.get('bad_case_labels', []))
        }
    }

def generate_final_report(key_variables, assessment_result, final_score):
    return {
        "executive_summary": {
            "overall_score": final_score,
            "quality_level": get_quality_level(final_score),
            "primary_strengths": extract_primary_strengths(assessment_result),
            "primary_weaknesses": extract_primary_weaknesses(assessment_result),
            "recommendation": get_overall_recommendation(final_score, assessment_result)
        },
        "key_variables": key_variables,
        "model_assessment": assessment_result,
        "final_score_breakdown": {
            "score": final_score,
            "interpretation": get_score_interpretation(final_score),
            "confidence_level": assess_confidence_level(key_variables, assessment_result)
        }
    }

def get_quality_assessment(score):
    if score >= 0.9:
        return {"level": "Excellent", "description": "The audio and video are highly consistent."}
    elif score >= 0.7:
        return {"level": "Good", "description": "The audio and video are generally consistent."}
    elif score >= 0.5:
        return {"level": "Fair", "description": "The clip is partially consistent and has noticeable issues."}
    elif score >= 0.3:
        return {"level": "Poor", "description": "The clip has substantial audio-video inconsistencies."}
    else:
        return {"level": "Very poor", "description": "The clip is severely inconsistent."}

def get_quality_level(score):
    if score >= 0.9:
        return "Excellent"
    elif score >= 0.7:
        return "Good"
    elif score >= 0.5:
        return "Fair"
    elif score >= 0.3:
        return "Poor"
    else:
        return "Very poor"

def extract_primary_strengths(assessment_result):
    scores = assessment_result.get('orthogonal_dimension_scores', {})
    strengths = []

    macro_score = scores.get('macro_semantic_alignment', {}).get('score', 0.0)
    temporal_score = scores.get('temporal_synchronicity', {}).get('score', 0.0)
    event_score = scores.get('fine_grained_event_correspondence', {}).get('score', 0.0)

    if macro_score >= 0.8:
        strengths.append("strong macro-semantic alignment")
    if temporal_score >= 0.8:
        strengths.append("strong temporal synchronicity")
    if event_score >= 0.8:
        strengths.append("strong event correspondence")

    return strengths if strengths else ["overall quality needs improvement"]

def extract_primary_weaknesses(assessment_result):
    bad_cases = assessment_result.get('bad_case_labels', [])
    scores = assessment_result.get('orthogonal_dimension_scores', {})

    weaknesses = []

    if bad_cases:
        weaknesses.extend([f"Detected issue: {case}" for case in bad_cases[:3]])

    macro_score = scores.get('macro_semantic_alignment', {}).get('score', 0.0)
    temporal_score = scores.get('temporal_synchronicity', {}).get('score', 0.0)
    event_score = scores.get('fine_grained_event_correspondence', {}).get('score', 0.0)

    if macro_score < 0.5:
        weaknesses.append("weak macro-semantic alignment")
    if temporal_score < 0.5:
        weaknesses.append("weak temporal synchronicity")
    if event_score < 0.5:
        weaknesses.append("weak event correspondence")

    return weaknesses if weaknesses else ["no major weakness identified"]

def get_overall_recommendation(final_score, assessment_result):
    if final_score >= 0.8:
        return "Maintain the current generation quality."
    elif final_score >= 0.6:
        return "Refine the localized inconsistencies."
    elif final_score >= 0.4:
        return "Prioritize the main inconsistency sources."
    else:
        return "Rework the audio-video alignment across the clip."

def get_score_interpretation(score):
    return {
        "numerical_score": score,
        "percentage": f"{score * 100:.1f}%",
        "description": get_quality_assessment(score)["description"]
    }

def assess_confidence_level(key_variables, assessment_result):

    completeness = key_variables["data_completeness"]
    audio_match_rate = key_variables["audio_event_matching"]["match_rate"]
    video_match_rate = key_variables["video_event_matching"]["match_rate"]

    data_completeness_factor = len(completeness["available_levels"]) / 4.0
    match_rate_factor = (audio_match_rate + video_match_rate) / 2.0

    confidence = (data_completeness_factor * 0.4 + match_rate_factor * 0.6)

    if confidence >= 0.8:
        return {"level": "High", "score": confidence, "description": "Most required evidence is available."}
    elif confidence >= 0.6:
        return {"level": "Medium", "score": confidence, "description": "Some evidence is incomplete."}
    else:
        return {"level": "Low", "score": confidence, "description": "Important evidence is missing."}
