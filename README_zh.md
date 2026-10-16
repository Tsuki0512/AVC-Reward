<div align="center">

# AVC-Reward: Event-Centric Audio-Video Consistency Evaluation

[**English**](README.md) | [**简体中文**](README_zh.md)

<p align="center">
  <a href="https://huggingface.co/Tsuki0512/AVC-Reward"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging_Face-FFD21E" alt="Hugging Face 奖励模型"></a>
  <a href="https://arxiv.org/abs/2610.22321"><img src="https://img.shields.io/badge/arXiv-2610.22321-B31B1B?logo=arxiv&logoColor=white" alt="arXiv:2610.22321"></a>
  <a href="https://tsuki0512.github.io/AVC-Reward.github.io/"><img src="https://img.shields.io/badge/Project-Page-2563EB?logo=googlechrome&logoColor=white" alt="项目主页"></a>
</p>

</div>

## 📰 最新动态（News）

- **已发布** — AVC-Reward 奖励模型权重和推理代码。
- **已发布** — AVC-Reward LTX-2 LoRA 权重、合并脚本和音视频联合推理代码。
- **已发布** — AVC-Benchmark 数据与评分代码。
- **已发布** — 五阶段数据标注代码。
- **已发布** — 论文已发布于 [arXiv:2610.22321](https://arxiv.org/abs/2610.22321)。

### 👉 TODO

- [ ] 后续开源奖励模型与生成模型的训练代码。
- [ ] 完成数据授权与隐私审核后，后续开源训练数据。

---

## 🏗️ 项目简介（Brief Introduction）

### 为什么采用事件中心评估？

现代联合音视频生成带来了多事件组合、多层音频混合和跨模态非对称关系，使现有的片段级指标和预定义关系难以适用。我们提出一种事件中心框架，分别拆解单个事件、区分音频来源角色，并跨模态验证依赖事件的关系，从而形成由粗到细的三层一致性指标。

<p align="center">
  <img src="fig/main-fig-cropped.png" alt="AVC-Reward event-centric audio-video consistency evaluation overview" width="100%">
</p>

- **全局叙事和谐度**（`S_mac`）：衡量整个片段中的叙事内容、实体身份、环境和情感氛围是否一致。
- **事件级时间同步**（`S_temp`）：依据事件相关的跨模态关系，评估事件是否存在，以及起始、结束和持续时间是否对齐。
- **细粒度属性对齐**（`S_attr`）：评估时间关系已验证的事件之间，在物理属性、形态和因果响应上的兼容性。

`S_AVC = (S_mac + S_temp + S_attr) / 3`

---

## 📦 下载（Downloads）

| 资源 | 说明 | 下载 |
| :--- | :--- | :--- |
| **奖励模型** | AVC-Reward Qwen3-Omni 奖励模型权重 | [🤗 AVC-Reward](https://huggingface.co/Tsuki0512/AVC-Reward) |
| **生成 LoRA** | AVC-Reward LTX-2 LoRA | [🤗 LTX-2-AVCReward-LoRA](https://huggingface.co/Tsuki0512/LTX-2-AVCReward-LoRA) |
| **AVC-Benchmark** | 音视频基准数据集 | [🤗 AVC-Benchmark](https://huggingface.co/datasets/Tsuki0512/AVC-Benchmark) |

模型权重和基准音视频由外部地址托管，不存放在本仓库中。

---

## ⚡ 奖励模型推理（Reward Model Inference）

### 环境与权重

请在仓库根目录创建环境。配置中包含 Qwen3-Omni/vLLM 运行时，以及未提供独立音频文件时从视频提取音轨所需的 `ffmpeg`。以下命令假设已安装 Conda 且 `conda` 命令已在 `PATH` 中；否则请先按现有安装路径将 Conda 的 `bin` 目录加入 `PATH`。


```bash
export WORK_ROOT="${WORK_ROOT:-$HOME/avc_reward_runtime}"
mkdir -p "$WORK_ROOT"/{envs,checkpoints,media/tmp,outputs,logs,cache,ltx-runtime}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create --prefix "$WORK_ROOT/envs/avc-reward" --file reward_model/environment.inference.yml
conda activate "$WORK_ROOT/envs/avc-reward"
hf download Tsuki0512/AVC-Reward --local-dir "$WORK_ROOT/checkpoints/AVC-Reward"
```

### 启动 vLLM

服务端会读取本地媒体文件。将 `ALLOWED_LOCAL_MEDIA_PATH` 设置为清单媒体所在目录；默认只监听 `127.0.0.1`。

```bash
export WORK_ROOT="${WORK_ROOT:-$HOME/avc_reward_runtime}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK_ROOT/envs/avc-reward"
export CUDA_VISIBLE_DEVICES=0
export MODEL_DIR="$WORK_ROOT/checkpoints/AVC-Reward"
export HOST=127.0.0.1
export PORT=8901
export TENSOR_PARALLEL_SIZE=1
export GPU_MEMORY_UTILIZATION=0.85
export MAX_MODEL_LEN=32768
export ALLOWED_LOCAL_MEDIA_PATH="$WORK_ROOT/media"
export TMPDIR="$WORK_ROOT/media/tmp"
export LOG_PATH="$WORK_ROOT/logs/reward_vllm.log"
bash reward_model/code/start_reward_vllm.sh
```

启动脚本还可通过环境变量配置 `SERVED_MODEL_NAME`、`GPU_MEMORY_UTILIZATION`、`MAX_MODEL_LEN`、`VLLM_BIN` 和 `LOG_PATH`。服务模型名默认为 `reward_model`。

单卡服务命令：

```bash
export CUDA_VISIBLE_DEVICES=0
export TENSOR_PARALLEL_SIZE=1
export MAX_MODEL_LEN=32768
bash reward_model/code/start_reward_vllm.sh
```

### 对清单批量评分

客户端支持 CSV 或 JSONL。JSONL 使用现有的 `videos`/`audios` 清单字段；可选的独立音频会覆盖从视频中提取的音轨。

```jsonl
{"id":"clip-001","videos":["/absolute/path/to/media/clip-001.mp4"]}
{"id":"clip-002","videos":["/absolute/path/to/media/clip-002.mp4"],"audios":["/absolute/path/to/media/clip-002.wav"]}
```

```bash
export WORK_ROOT="${WORK_ROOT:-$HOME/avc_reward_runtime}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK_ROOT/envs/avc-reward"
export TMPDIR="$WORK_ROOT/media/tmp"
mkdir -p "$WORK_ROOT/media/tmp" "$WORK_ROOT/outputs/reward"
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i 'testsrc2=size=640x360:rate=10:duration=3' \
  -f lavfi -i 'sine=frequency=440:sample_rate=48000:duration=3' \
  -c:v mpeg4 -q:v 5 -pix_fmt yuv420p -c:a aac -shortest \
  "$WORK_ROOT/media/sample_with_audio.mp4"
python -c 'import json,os; p=os.path.join(os.environ["WORK_ROOT"],"media/sample_with_audio.mp4"); open(os.path.join(os.environ["WORK_ROOT"],"media/smoke.jsonl"),"w").write(json.dumps({"id":"avc-smoke-001","videos":[p]})+"\n")'
python reward_model/code/reward_model_inference.py \
  --input "$WORK_ROOT/media/smoke.jsonl" \
  --output-jsonl "$WORK_ROOT/outputs/reward/scores.jsonl" \
  --base-dir "$WORK_ROOT/media" \
  --endpoint http://127.0.0.1:8901/v1/chat/completions \
  --model reward_model
```

CSV 默认读取 `video_path` 列；若有独立音频列，添加 `--audio-column audio_path`。其他选项包括 `--video-column`、`--limit`、`--base-dir`、`--endpoint` 和 `--model`。

客户端使用原有五层系统提示，并分别发送 `video_url` 和 `audio_url`。这遵循 [Qwen3-Omni vLLM 服务说明](https://github.com/QwenLM/Qwen3-Omni/blob/main/README.md#vllm-serve-usage)中对该 API 分别传入音频和视频的用法。成功结果包含 `macro_semantic_alignment`、`temporal_synchronicity`、`fine_grained_event_correspondence` 三项得分，及其算术平均值 `rm_reward_avg`，同时保留原始模型回复和请求信息。失败行通过 `ok: false` 和错误信息报告，不会伪造默认分数。

---

## 🎬 生成模型推理（Generation Model Inference）

### 环境与 LTX-2 运行时代码

在仓库根目录创建独立生成环境，然后获取固定版本的 LTX-2 源码。推理入口使用上游 `TI2VidOneStagePipeline`，并分别配置视频与音频 guider。

```bash
export WORK_ROOT="${WORK_ROOT:-$HOME/avc_reward_runtime}"
export CUDA_VISIBLE_DEVICES=0
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
unset PIP_EXTRA_INDEX_URL || true
export PIP_CACHE_DIR="$WORK_ROOT/cache/pip"
export TMPDIR="$WORK_ROOT/media/tmp"
mkdir -p "$WORK_ROOT"/{envs,checkpoints,media/tmp,outputs,logs,cache,ltx-runtime}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create --prefix "$WORK_ROOT/envs/avc-generation" --file generation_model/environment.inference.yml
conda activate "$WORK_ROOT/envs/avc-generation"
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0
python -m pip install -r generation_model/requirements.inference.txt
export LTX_V2_ROOT="$WORK_ROOT/ltx-runtime/LTX-2"
bash generation_model/code/install_ltx_runtime.sh
git -C "$LTX_V2_ROOT" rev-parse HEAD
```

上游源码固定为 [`d887bbd1e045b316d0a855f8d421209ecfc97ed1`](https://github.com/Lightricks/LTX-2/tree/d887bbd1e045b316d0a855f8d421209ecfc97ed1)。使用运行时代码或模型前，请阅读上游的 [LTX-2 Community License](https://github.com/Lightricks/LTX-2/blob/d887bbd1e045b316d0a855f8d421209ecfc97ed1/LICENSE)。

### 下载基座、Gemma 和 AVC-Reward LoRA

AVC-Reward LoRA 对应 LTX-2 dev 基座，请搭配 `ltx-2-19b-dev.safetensors` 和对应的 Gemma 3 文本编码器。下载 Gemma 可能需要先在 Hugging Face 接受模型条款。

```bash
export WORK_ROOT="${WORK_ROOT:-$HOME/avc_reward_runtime}"
mkdir -p "$WORK_ROOT"/{checkpoints,cache/huggingface}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK_ROOT/envs/avc-generation"
export HF_HOME="$WORK_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
hf auth login
hf download Lightricks/LTX-2 ltx-2-19b-dev.safetensors --local-dir "$WORK_ROOT/checkpoints/LTX-2"
hf download google/gemma-3-12b-it --local-dir "$WORK_ROOT/checkpoints/gemma-3-12b-it"
hf download Tsuki0512/LTX-2-AVCReward-LoRA --local-dir "$WORK_ROOT/checkpoints/LTX-2-AVCReward-LoRA"
```

### 合并与生成

```bash
python generation_model/code/merge_lora.py \
  --checkpoint-path "$WORK_ROOT/checkpoints/LTX-2/ltx-2-19b-dev.safetensors" \
  --lora-dir "$WORK_ROOT/checkpoints/LTX-2-AVCReward-LoRA" \
  --output-path "$WORK_ROOT/checkpoints/ltx2_avc_reward_merged.safetensors" \
  --dtype bf16 \
  --strict

CUDA_VISIBLE_DEVICES=0 python generation_model/code/inference.py \
  --model_path "$WORK_ROOT/checkpoints/ltx2_avc_reward_merged.safetensors" \
  --gemma_path "$WORK_ROOT/checkpoints/gemma-3-12b-it" \
  --prompt "A chef chops vegetables while a phone rings off-screen." \
  --num_frames 121 \
  --video_guidance_scale 3 \
  --audio_guidance_scale 7 \
  --stg_scale 1 \
  --seed 42 \
  --dtype bf16 \
  --output_dir "$WORK_ROOT/outputs/generation"
```

推理会保存包含音轨的 MP4 和独立 WAV。基础选项包括 `--model_path`、`--gemma_path`、`--prompt`、`--negative_prompt`、`--num_frames`、`--height`、`--width`、`--frame_rate`、`--num_inference_steps`、`--seed`、`--output_dir` 和 `--dtype`。Guider 选项如下：

| 参数 | 含义 |
|---|---|
| `--video_guidance_scale` / `--audio_guidance_scale` | 设置视频 / 音频 CFG scale |
| `--stg_scale` | 同时设置两个模态的 STG scale |
| `--video_stg_scale` / `--audio_stg_scale` | 分别设置视频 / 音频 STG scale |
| `--video_rescale_scale` / `--audio_rescale_scale` | 设置视频 / 音频 CFG rescale |
| `--a2v_guidance_scale` | 视频 guider 中的 audio-to-video modality scale |
| `--v2a_guidance_scale` | 音频 guider 中的 video-to-audio modality scale |
| `--video_skip_step` / `--audio_skip_step` | 设置视频 / 音频 guider 的 skip step |
| `--video_stg_blocks` / `--audio_stg_blocks` | 设置视频 / 音频 STG block 索引，可传一个或多个整数 |

模态专属 STG scale 优先于 `--stg_scale`；未指定专属值时使用通用值；两者均未指定时使用该模态的 checkpoint 版本预设。未覆盖的 CFG、rescale、modality、skip step 和 STG blocks 保留各自模态预设。`detect_params()` 根据识别到的 checkpoint 版本选择预设；版本未知时回退到 LTX-2 默认值。当前 pipeline 只支持 `--dtype bf16`。合并脚本选项：`--checkpoint-path`、`--lora-dir`、`--output-path`、`--dtype` 和 `--strict`。

基座、Gemma 编码器、LoRA 和合并权重分别从对应链接下载。

---

## 📊 Benchmark 评测（Benchmark Evaluation）

Hugging Face 数据集托管音视频媒体。本仓库的本地 CSV 是独立的 706 行评分表，供现有评测脚本读取；脚本不会下载媒体或运行模型推理。

```bash
export WORK_ROOT="${WORK_ROOT:-$HOME/avc_reward_runtime}"
source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p "$WORK_ROOT"/{envs,outputs/benchmark}
conda create -y --prefix "$WORK_ROOT/envs/avc-benchmark" python=3.11 pip
conda activate "$WORK_ROOT/envs/avc-benchmark"
python -m pip install -r benchmark/requirements.txt
python -m pip check
python benchmark/scripts/benchmark_eval.py \
  --csv benchmark/data/benchmark_all_data_all_models.csv \
  --output_dir "$WORK_ROOT/outputs/benchmark" \
  --deltas 0 0.05 0.1 0.15 0.2 0.25 0.3 0.5
```

CSV 包含人工目标列 `human_macro`、`human_temporal`、`human_fine_grained`，标量基线列，以及 `MODEL__macro`、`MODEL__temporal`、`MODEL__fine_grained` 格式的分组模型分数。脚本输出 `benchmark_metrics.csv` 和 `benchmark_metrics_summary.json`，包含五参数 Logistic 映射后的 PLCC、SRCC、Pairwise 和 Delta-Pairwise。示例指标文件位于 `benchmark/metrics/`。

---

五阶段数据标注流程和提示词的环境配置及命令见 [`data_annotation/pipeline/README.md`](data_annotation/pipeline/README.md)。

---

## 🤝 致谢（Acknowledgements）

[OmniNFT](https://github.com/zghhui/OmniNFT) · [LTX-2](https://github.com/Lightricks/LTX-2) · [JavisDiT](https://github.com/JavisVerse/JavisDiT) · [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)

## 🖊️ 引用（Citation）

[arXiv:2610.22321](https://arxiv.org/abs/2610.22321)。

## ⚖️ 许可（License）

仅限研究用途。模型权重、基准媒体和上游代码分别遵循各自条款。生成运行时请遵守上游 [LTX-2 Community License](https://github.com/Lightricks/LTX-2/blob/d887bbd1e045b316d0a855f8d421209ecfc97ed1/LICENSE)。
