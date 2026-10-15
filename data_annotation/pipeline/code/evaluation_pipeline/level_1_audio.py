from .api_handler_final_gemini import call_llm_api
from .variable_manager import VariableManager, MockDataGenerator
import json
import os

def get_audio_mime_type(audio_path):
    _, ext = os.path.splitext(audio_path.lower())
    if ext == '.wav':
        return 'audio/wav'
    elif ext in ['.mp3', '.mpeg']:
        return 'audio/mp3'
    else:

        return 'audio/mp3'

def analyze_audio(audio_path, sed_json_path):

    audio_mime_type = get_audio_mime_type(audio_path)

    total_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def accumulate_tokens(token_usage):
        if token_usage:
            total_token_usage["input_tokens"] += token_usage.get("input_tokens", 0)
            total_token_usage["output_tokens"] += token_usage.get("output_tokens", 0)
            total_token_usage["total_tokens"] += token_usage.get("total_tokens", 0)

    all_event_instances = []

    with open(sed_json_path, 'r', encoding='utf-8') as f:
        sed_data = json.load(f)

    speech_hints = sed_data.get('method_8', [])
    music_hints = sed_data.get('method_8', [])
    event_hints = sed_data.get('method_1', [])

    def filter_sed_hints(hints, min_confidence=0.3):
        if not hints:
            return []
        return [h for h in hints if isinstance(h, dict) and h.get('confidence', 0) >= min_confidence]

    speech_hints = filter_sed_hints(speech_hints)
    music_hints = filter_sed_hints(music_hints)
    event_hints = filter_sed_hints(event_hints)

    prompt_identify = """You are a professional audio layer analyst. Please listen carefully to the audio and identify all potential acoustic layers present (Human Speech, (Background) Music, and Other Sound Events - which refers to any audio events other than human speech and background music). Return in JSON format: {"potential_layers": ["Human Speech", "(Background) Music", "Other Sound Events"]}"""
    layers_result, token_usage = call_llm_api(prompt_identify, audio_path, audio_mime_type)
    accumulate_tokens(token_usage)

    validated_layers_analysis = []
    has_other_sound_events = False

    if layers_result and 'potential_layers' in layers_result:
        for layer in layers_result['potential_layers']:
            if layer == 'Other Sound Events':
                has_other_sound_events = True

                hs_caption = ""
                hs_analysis_ctx = next((l for l in validated_layers_analysis if isinstance(l, dict) and l.get('layer_type') == 'Human Speech'), None)
                if hs_analysis_ctx:
                    hs_caption = hs_analysis_ctx.get('global_transcription', "")

                prompt_identify_events = f"""You are an experienced scene analyst. Focus on audio events that are perceptible to an average human listener and carry semantic relevance or clear physical causes. Note that 'Other Sound Events' refers to any audio events other than human speech and background music.
Avoid micro-fragmented, low-amplitude high-frequency sustained noise that lacks human-perceived meaning (e.g., imperceptible hiss).
Return in JSON format, brainstorming objective physical descriptions only (no semantic labels):
e.g., {{ "potential_event_types": {{ "transient": ["short, high-frequency tap"], "continuous": ["rustling friction sound"] }} }}
"""
                event_types_result, token_usage = call_llm_api(prompt_identify_events, audio_path, audio_mime_type)
                accumulate_tokens(token_usage)

                if event_types_result and 'potential_event_types' in event_types_result:
                    potential_events = event_types_result['potential_event_types']
                    identified_event_descriptions = [desc for desc_list in potential_events.values() for desc in desc_list]
                else:
                    identified_event_descriptions = []

            if layer == 'Human Speech':

                prompt_analyze_hs = """You are an audio diarization expert. Please:
1) Identify distinct speakers and assign stable IDs (S1, S2, S3, ...).
2) Provide global language and an optional global transcription.
3) For each speaker, provide characteristics (e.g., gender-like timbre, pitch range, speaking style) without guessing identity.

Return JSON:
{
  "layer_type": "Human Speech",
  "language": "...",
  "global_transcription": "...",
  "speakers": [
    {"speaker_id": "S1", "characteristics": {"timbre": "...", "pitch_range": "...", "speaking_style": "..."}},
    {"speaker_id": "S2", "characteristics": {"timbre": "..."}}
  ]
}
"""
                analysis_result, token_usage = call_llm_api(prompt_analyze_hs, audio_path, audio_mime_type)
                accumulate_tokens(token_usage)
            elif layer == '(Background) Music':

                prompt_analyze_music = """You are a versatile audio content analysis expert. Please analyze the '(Background) Music' layer for genre, mood, and instrumentation. Return JSON: {"layer_type":"(Background) Music", "genre": "...", "mood": "...", "instrumentation": ["..."]}"""
                analysis_result, token_usage = call_llm_api(prompt_analyze_music, audio_path, audio_mime_type)
                accumulate_tokens(token_usage)
            else:

                prompt_analyze_other = """You are a versatile audio content analysis expert. Please analyze the 'Other Sound Events' layer (which refers to any audio events other than human speech and background music) and summarize dominant sounds and notable acoustic features. Return JSON: {"layer_type": "Other Sound Events", "dominant_sounds": "..." }"""
                analysis_result, token_usage = call_llm_api(prompt_analyze_other, audio_path, audio_mime_type)
                accumulate_tokens(token_usage)

            if analysis_result and isinstance(analysis_result, dict):
                validated_layers_analysis.append(analysis_result)
            else:
                pass

    prompt_macro = f"""You are a lead scene constructor. Please synthesize all the verified acoustic layer analysis reports below, listen to the audio once more, and make judgments on:
1. Overall scene context and setting
2. Individual speaker emotions (not overall mood, but specific emotions of each speaker)

Verified acoustic layer analysis reports: {json.dumps(validated_layers_analysis, ensure_ascii=False)}

Return JSON format:
{{
    "scene_hypotheses": [
        {{
            "scene": "...",
            "confidence": 0.0-1.0,
            "reasoning": "..."
        }}
    ],
    "speaker_emotions": [
        {{
            "speaker_id": "S1",
            "emotion": "...",
            "confidence": 0.0-1.0,
            "emotional_indicators": "..."
        }}
    ]
}}"""
    macro_result, token_usage = call_llm_api(prompt_macro, audio_path, audio_mime_type)
    accumulate_tokens(token_usage)

    present_layers = [l.get('layer_type') for l in validated_layers_analysis if isinstance(l, dict)]

    cross_layer_warning = """
IMPORTANT: This audio may contain multiple overlapping acoustic layers (Human Speech, (Background) Music, Other Sound Events).
Please carefully distinguish between different layer types when generating timestamps:
- Human Speech: Focus only on spoken words and vocal sounds
- (Background) Music: Focus only on musical instruments and melodies
- Other Sound Events: Focus only on non-speech, non-music environmental sounds
Be cautious not to confuse sounds from different layers or include timestamps from other layers in your analysis.
"""

    if 'Human Speech' in present_layers:

        prompt_speech_timing = f"""You are an audio analysis expert specializing in speech analysis.
{cross_layer_warning}

Please analyze ONLY the Human Speech layer and:
1) Identify distinct speakers and assign stable IDs (S1, S2, S3, ...).
2) Provide time-aligned speaker segments with transcriptions.
3) Use SED reference data as timing hints: {json.dumps(speech_hints, ensure_ascii=False)}

Return JSON:
{{
  "speaker_segments": [
    {{"speaker_id": "S1", "start_time": ..., "end_time": ..., "transcription": "..."}},
    {{"speaker_id": "S2", "start_time": ..., "end_time": ..., "transcription": "..."}}
  ]
}}
"""
        speech_timing_result, token_usage = call_llm_api(prompt_speech_timing, audio_path, audio_mime_type)
        accumulate_tokens(token_usage)

        if speech_timing_result and 'speaker_segments' in speech_timing_result:

            speaker_characteristics = {}
            for layer_analysis in validated_layers_analysis:
                if isinstance(layer_analysis, dict) and layer_analysis.get('layer_type') == 'Human Speech':
                    speakers = layer_analysis.get('speakers', [])
                    for speaker in speakers:
                        speaker_id = speaker.get('speaker_id')
                        if speaker_id:
                            speaker_characteristics[speaker_id] = speaker.get('characteristics', {})

            for seg in speech_timing_result['speaker_segments']:

                def parse_timestamp(timestamp):
                    if timestamp is None:
                        return 0.0

                    if isinstance(timestamp, (int, float)):
                        return float(timestamp)

                    if isinstance(timestamp, str):

                        if ':' in timestamp:
                            try:
                                parts = timestamp.split(':')
                                if len(parts) == 2:
                                    minutes = float(parts[0])
                                    seconds = float(parts[1])
                                    return minutes * 60 + seconds
                            except ValueError:
                                pass

                        try:
                            return float(timestamp)
                        except ValueError:
                            pass

                    return 0.0

                try:
                    start_time = parse_timestamp(seg.get('start_time'))
                    end_time = parse_timestamp(seg.get('end_time'))
                except Exception as e:
                    start_time = 0.0
                    end_time = 0.0

                speaker_id = seg.get('speaker_id')
                transcription = seg.get('transcription', "")

                characteristics = speaker_characteristics.get(speaker_id, {})
                timbre = characteristics.get('timbre', 'Unknown')

                content_summary = transcription[:15] + "..." if len(transcription) > 15 else transcription

                objective_desc = f"{timbre} voice ({speaker_id}): {content_summary}"

                event = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "layer_type": "Human Speech",
                    "speaker_id": speaker_id,
                    "transcription": transcription,
                    "objective_description": objective_desc,
                    "confidence": 0.8,
                    "intensity": "medium"
                }
                all_event_instances.append(event)

    if '(Background) Music' in present_layers:

        prompt_music_timing = f"""You are an audio analysis expert specializing in music analysis.
{cross_layer_warning}

Please analyze ONLY the (Background) Music layer and locate music segments in the audio.
Use SED reference data as timing hints: {json.dumps(music_hints, ensure_ascii=False)}

Return JSON:
{{
  "music_segments": [
    {{ "start_time": ..., "end_time": ..., "intensity": "medium", "dominant_frequency": "...", "texture": "...", "musical_characteristics": "...", "objective_description": "..." }}
  ]
}}

For objective_description, provide a concise 2-3 word description based on the musical characteristics (e.g., "Piano Classical Music", "Violin Romantic Music", "Orchestral Dramatic Music").
IMPORTANT: Always include an "intensity" field with values like "medium", "high", or "low" for each music segment."""

        music_timing_result, token_usage = call_llm_api(prompt_music_timing, audio_path, audio_mime_type)
        accumulate_tokens(token_usage)
        if music_timing_result and 'music_segments' in music_timing_result:

            music_layer_analysis = None
            for layer_analysis in validated_layers_analysis:
                if isinstance(layer_analysis, dict) and layer_analysis.get('layer_type') == '(Background) Music':
                    music_layer_analysis = layer_analysis
                    break

            for seg in music_timing_result['music_segments']:

                try:
                    start_time = float(seg.get('start_time', 0)) if seg.get('start_time') is not None else 0.0
                    end_time = float(seg.get('end_time', 0)) if seg.get('end_time') is not None else 0.0
                except (ValueError, TypeError):
                    continue

                base_desc = seg.get('objective_description', "Background Music")

                if music_layer_analysis:
                    genre = music_layer_analysis.get('genre', '')
                    mood = music_layer_analysis.get('mood', [])
                    instrumentation = music_layer_analysis.get('instrumentation', [])

                    desc_parts = []
                    if genre:
                        desc_parts.append(genre)
                    if mood and isinstance(mood, list) and len(mood) > 0:
                        desc_parts.append(mood[0])
                    if instrumentation and isinstance(instrumentation, list) and len(instrumentation) > 0:
                        desc_parts.append(instrumentation[0])

                    if desc_parts:
                        enhanced_desc = " ".join(desc_parts[:2])
                    else:
                        enhanced_desc = base_desc
                else:
                    enhanced_desc = base_desc

                music_intensity = seg.get('intensity')

                if music_intensity is None or str(music_intensity).strip().lower() in ('very low', 'low', '', 'none'):
                    music_intensity = 'medium'

                event = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "layer_type": "(Background) Music",
                    "intensity": music_intensity,
                    "dominant_frequency": seg.get('dominant_frequency'),
                    "texture": seg.get('texture'),
                    "musical_characteristics": seg.get('musical_characteristics'),
                    "objective_description": enhanced_desc
                }
                all_event_instances.append(event)

    if has_other_sound_events and 'identified_event_descriptions' in locals():
        for desc in identified_event_descriptions:

            prompt_validate_event = f"""You are a meticulous fact-checker specializing in sound event detection.
Please analyze ONLY the Other Sound Events layer (excluding speech and music) and check whether a sound event with the objective description '{desc}' exists.
Return JSON: {{ "event_type_description": "{desc}", "is_present": true/false, "confidence": ... }}"""

            event_validation, token_usage = call_llm_api(prompt_validate_event, audio_path, audio_mime_type)
            accumulate_tokens(token_usage)

            if event_validation and event_validation.get('is_present'):

                prompt_event_timing = f"""You are a forensic-level audio analyst specializing in sound event detection.
{cross_layer_warning}
You have confirmed that a sound with the objective description '{desc}' exists in the Other Sound Events layer.
Generate robust, human-perception-aligned event instances:
- Classify intensity as one of: Very low, Low, Medium, High.
- Be extremely careful not to include speech or music sounds.
Reference timestamps (optional hints): {json.dumps(event_hints, ensure_ascii=False)}
Return JSON:
{{
  "event_instances": [
    {{ "start_time": ..., "end_time": ..., "objective_description": "{desc}", "intensity": "...", "dominant_frequency": "...", "texture": "..." }}
  ]
}}
"""
                event_timing_result, token_usage = call_llm_api(prompt_event_timing, audio_path, audio_mime_type)
                accumulate_tokens(token_usage)

                if event_timing_result and 'event_instances' in event_timing_result:
                    all_event_instances.extend(event_timing_result['event_instances'])

    filtered_event_instances = []
    for ev in all_event_instances:
        intensity = str(ev.get('intensity', '')).strip().lower()

        if intensity in ('very low', 'low'):
            continue
        filtered_event_instances.append(ev)

    analysis_result = {
        "validated_layers_analysis": validated_layers_analysis,
        "macro_analysis": macro_result,
        "acoustic_event_detection": {"event_instances": filtered_event_instances}
    }

    return analysis_result, total_token_usage

def analyze_audio_with_tokens(audio_path, sed_json_path, use_mock=False, var_manager=None):

    if var_manager is None:
        var_manager = VariableManager()

    logger = var_manager.setup_level_logger(1)

    if var_manager.variable_file_exists(1):
        analysis_result = var_manager.load_variables(1)
        if analysis_result:
            return analysis_result, []


    if use_mock:
        analysis_result = MockDataGenerator.get_mock_level_1_result()
        token_usage_list = []
    else:
        analysis_result, total_token_usage = analyze_audio(audio_path, sed_json_path)
        token_usage_list = [total_token_usage] if total_token_usage else []

    if var_manager.save_variables(1, analysis_result):
        pass
    else:
        pass

    return analysis_result, token_usage_list

def run_level_1_standalone(audio_path, sed_json_path, use_mock=False, output_dir="pipeline_outputs"):
    var_manager = VariableManager(output_dir)
    logger = var_manager.setup_level_logger(1)

    logger.info("=" * 60)
    logger.info("=" * 60)

    try:
        result, token_usage = analyze_audio_with_tokens(
            audio_path, sed_json_path, use_mock=use_mock, var_manager=var_manager
        )

        if token_usage:
            total_tokens = sum(usage.get('total_tokens', 0) for usage in token_usage)

        return result

    except Exception as e:
        raise
