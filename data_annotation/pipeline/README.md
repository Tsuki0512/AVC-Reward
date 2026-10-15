# Five-stage data annotation pipeline

This code implements five stages: blind audio analysis, deaf video analysis, audio-to-video verification, video-to-audio verification, and final world-construction scoring. `prepare_data.py` compresses videos, extracts audio, and can create sound-event hints. `run_batch_evaluation_qwen.py` runs the stages over prepared media. `prompts/data_filter.txt` contains four binary screening questions; `prompts/score_rewrite.txt` maps annotation labels to the three AVC scores and a conclusion. `code/score_rewrite.py` applies that rewrite prompt to annotation CSV files.

Create the environment from the repository root:

```bash
conda env create -f data_annotation/environment.yml
conda activate pe
```

Prepare local input media and choose output locations:

```bash
python data_annotation/pipeline/code/prepare_data.py \
  --input_dir /path/to/source_videos \
  --output_dir /path/to/processed_videos \
  --target_size_kb 800

python data_annotation/pipeline/code/run_batch_evaluation_qwen.py \
  --processed_dir /path/to/processed_videos \
  --original_dir /path/to/source_videos \
  --reports_dir /path/to/reports \
  --max_workers 4
```

The handlers require service configuration from the environment. Set `QWEN_VLLM_API_URL` to the authorized OpenAI-compatible chat-completions endpoint and `QWEN_VLLM_MODEL` to its served model name before running the Qwen pipeline. The alternate Gemini handler requires an authorized compatible bridge configured as `GEMINI_ENDPOINT`; this code does not configure or provide that service. API keys, where required by the selected service, must be supplied through that service's approved environment variables.

The separate score rewrite tool accepts an annotation CSV and uses the supplied rewrite prompt by default:

```bash
export QWEN_VLLM_API_URL=http://127.0.0.1:8901/v1/chat/completions
export QWEN_VLLM_MODEL=your-served-model-name
python data_annotation/pipeline/code/score_rewrite.py \
  --csv_path /path/to/labels.csv \
  --mode vllm \
  --output_dir /path/to/rewritten_targets
```

For the `online` rewrite mode, provide `DASHSCOPE_API_KEY` and `QWEN_MODEL_NAME` through the environment. Use `--mode mock` to exercise prompt and output handling without an external service. Input labels and generated annotation outputs are not included in Git.
