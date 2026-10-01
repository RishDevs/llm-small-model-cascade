# LLM → Small Model Cascade

A text classifier that answers most requests with a small, fast model and passes only the hard ones to a 7B LLM. Everything in it is built from open-source models and libraries and runs on free compute.

> **Status (2026-10-01): planned. No code exists yet.** This document describes the intended design. Commands and file paths under *Running the pipeline* show the planned interface. Section 9 lists the decisions that are still open.

---

## Contents

1. [What the project is](#1-what-the-project-is)
2. [Targets and how they are measured](#2-targets-and-how-they-are-measured)
3. [Dataset](#3-dataset)
4. [Architecture](#4-architecture)
5. [Components](#5-components)
6. [Repository layout](#6-repository-layout)
7. [Setup (free)](#7-setup-free)
8. [Running the pipeline](#8-running-the-pipeline)
9. [Open design decisions](#9-open-design-decisions)
10. [Zero-cost bill of materials](#10-zero-cost-bill-of-materials)
11. [Known limitations and risks](#11-known-limitations-and-risks)

---

## 1. What the project is

**Problem.** Sending every classification request to a large LLM is slow (hundreds of ms to seconds per call) and costly (GPU time or API fees). Most inputs are easy, and a much smaller model could classify them correctly.

**Approach.** Use a three-step cascade:

1. **Teacher labeling.** A 7B instruction-tuned LLM labels the training texts. This is knowledge distillation from pseudo-labels.
2. **Student training.** DistilBERT (66M parameters) is fine-tuned on the LLM's labels so it imitates the teacher and runs in milliseconds on a CPU.
3. **Confidence-gated routing.** At inference time, the student predicts a label and a calibrated confidence. If the confidence is at or above a threshold τ, the student's answer is returned. Below τ, the request is escalated to the LLM.

**Task.** 77-way intent classification of online-banking customer queries, using [Banking77](#3-dataset).

**Out of scope.** Multi-label classification, languages other than English, and online learning from production traffic.

---

## 2. Targets and how they are measured

These are goals, not results. No results exist yet.

| Target | Value | Precise definition |
|---|---|---|
| Fewer LLM calls | ≈ 85% | **Coverage** = the fraction of test inputs the student answers without escalation. An 85% reduction means coverage is 0.85 at the chosen τ. This counts inference-time calls only. Labeling the training set is a separate, one-time cost of about 13k LLM calls. |
| Match LLM accuracy | cascade ≥ LLM − δ | Accuracy on the Banking77 **gold** test labels. The cascade's accuracy is compared with LLM-only accuracy on the same 3,080 test texts. The tolerance δ is an open decision ([§9.6](#96-accuracy-tolerance-δ-and-threshold-selection)). |
| Student latency | ≈ 8 ms on CPU | p50 wall-clock time for one forward pass at batch size 1, max sequence length 64, measured on a stated CPU with a stated thread count. The protocol is in [§5.6](#56-latency-benchmark). |
| Calibration error | ≈ 50% lower | Expected Calibration Error (ECE, 15 equal-width bins) of the student's max-softmax confidence on the test set, measured before and after calibration. |

---

## 3. Dataset

**Banking77**: [`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) on the Hugging Face Hub, licensed CC BY 4.0.

| Split | Size | Use in this project |
|---|---|---|
| train | 10,003 | Labeled by the LLM, then split as shown below. The gold labels are used only for few-shot examples, if that option is chosen, and for diagnostics. |
| test | 3,080 | Final evaluation against gold labels. The LLM also labels these texts once so that escalation can be simulated offline. |

The original train split is divided with a stratified split (seed 42):

| Subset | Share | ≈ Size | Purpose |
|---|---|---|---|
| `student_train` | 72% | 7,202 | Fine-tune the student on LLM labels |
| `student_val` | 8% | 800 | Choose the best epoch (early stopping) |
| `calib` | 20% | 2,001 | Fit the temperature T and select the threshold τ |

All 13,083 texts (train + test) are sent to the LLM **once**, and the outputs are cached. After that, every evaluation runs offline without calling the LLM.

> If `load_dataset("PolyAI/banking77")` fails on your `datasets` version, the same CSVs are published in PolyAI's [`task-specific-datasets`](https://github.com/PolyAI-LDN/task-specific-datasets) GitHub repo. `data.py` should support both sources.

---

## 4. Architecture

### Offline (build time)

```
Banking77 texts ──► LLM teacher (7B, constrained output) ──► labels cache (Parquet)
                                                                    │
                    ┌───────────────────────────────────────────────┤
                    ▼                                               ▼
     student_train + student_val                                 calib
                    │                                               │
                    ▼                                               ▼
     fine-tune DistilBERT ──► student checkpoint ──► fit temperature T, choose τ
                                        │
                                        ▼
                     export to ONNX (+ optional int8 quantization)
```

### Online (request time)

```
text ──► tokenizer ──► student (ONNX Runtime, CPU)
                          │
                 logits / T ─► softmax ─► conf = max prob
                          │
            conf ≥ τ ? ───┴─── yes ──► return student label  (source = "student")
                          │
                          no
                          ▼
              LLM via OpenAI-compatible endpoint ──► return LLM label (source = "llm")
```

Average request latency is the student latency plus (1 − coverage) × LLM latency.

---

## 5. Components

### 5.1 LLM client (`llm_client.py`)

- A single client that uses the `openai` Python package (an MIT-licensed client library) pointed at a configurable `base_url`. Ollama, the llama.cpp server, and vLLM all expose OpenAI-compatible `/v1/chat/completions` endpoints, and so do most free hosted APIs. Switching backends therefore changes only the config, not the code. No paid OpenAI service is involved.
- **Constrained output.** The response is forced into a JSON object `{"label": <one of 77 names>}` using the backend's structured-output feature: Ollama `format` with a JSON schema enum, llama.cpp grammars or JSON schema, or vLLM guided decoding. Every output is then a valid label. Without this feature, free-text outputs must be matched to label names, and some cannot be matched at all.
- `temperature = 0` for reproducibility.

### 5.2 Teacher labeling (`label.py`)

- The prompt contains the task description, the 77 intent names, optional few-shot examples ([§9.3](#93-prompting-strategy)), and the query.
- **Cache.** Each result is appended to `data/llm_labels.parquet`, keyed by `(text_id, model_id, prompt_hash)`. Re-runs skip rows already in the cache, so an interrupted run (for example, a free GPU session timing out) resumes where it stopped.
- **Logged per row:** label, raw response, latency, and an invalid-output flag.
- **Logged in summary:** LLM accuracy against gold on train and test, the invalid-output rate, and the per-class confusion matrix.

### 5.3 Student training (`train_student.py`)

- Model: `distilbert-base-uncased` with a 77-way classification head (Hugging Face `AutoModelForSequenceClassification`).
- Targets: **LLM labels** (hard labels). Gold labels are never used for training.
- Starting hyperparameters: max length 64, batch size 32, learning rate 5e-5, 4–5 epochs, weight decay 0.01, linear warmup of 6%. The checkpoint with the best `student_val` accuracy (against LLM labels) is kept.
- Runs on Apple-Silicon MPS, a free Colab/Kaggle GPU, or CPU (slowest). With about 7k short texts, training takes minutes on MPS or a T4. This estimate should be checked on the first run.

### 5.4 Calibration (`calibrate.py`)

- **Why.** Fine-tuned transformers are usually overconfident. A raw softmax value of 0.95 might be correct only 85% of the time. The threshold τ is meaningful only if confidence tracks actual accuracy.
- **Default method: temperature scaling.** Learn one scalar T > 0 by minimizing negative log-likelihood on `calib`. At inference, use `softmax(logits / T)`. This does not change the argmax, so accuracy stays the same and only the confidence changes. Other methods are listed in [§9.5](#95-calibration-method).
- **Metrics:** ECE (15 bins), NLL, Brier score, and a reliability diagram (`reports/reliability.png`), each reported before and after calibration.

### 5.5 Cascade and threshold (`cascade.py`, `evaluate.py`)

- **Confidence score:** the maximum calibrated probability.
- **Threshold selection on `calib`:** sweep τ over all distinct confidence values. For each τ, compute the cascade accuracy, using the student's prediction when conf ≥ τ and the cached LLM label otherwise, and the coverage. Pick the τ with the **highest coverage** whose cascade accuracy is ≥ LLM accuracy − δ.
- **Final test report (`reports/metrics.json`):**
  - LLM-only accuracy, student-only accuracy, and cascade accuracy (all against gold), plus macro-F1
  - coverage and LLM-call reduction at τ
  - ECE before and after calibration
  - risk–coverage curve (`reports/risk_coverage.png`)
  - student–LLM agreement rate

### 5.6 Latency benchmark

Planned protocol:

- **Runtime:** ONNX Runtime CPU execution provider, with PyTorch eager reported alongside for comparison.
- **Input:** batch size 1, inputs padded or truncated to 64 tokens, using real test-set texts.
- **Timing:** 50 warm-up runs, then 1,000 timed runs. Report p50 and p95.
- **Threads:** report results at 1 thread and at 4 intra-op threads.
- **Reported separately:**
  - model forward pass only
  - tokenization plus forward pass
- **Hardware:** record the CPU model with every result. An M4 Pro and a 2-vCPU cloud VM will produce very different numbers.

### 5.7 Export (`export_onnx.py`)

- Export the calibrated student to ONNX (`torch.onnx.export` or Hugging Face Optimum).
- Optionally apply ONNX Runtime dynamic int8 quantization ([§9.7](#97-student-inference-runtime)).
- Check that the exported model's predictions match PyTorch on the full test set, and report the agreement rate.
- T and τ are stored next to the model in `artifacts/cascade.json`.

### 5.8 Service (`serve.py`, optional)

- A FastAPI app with `POST /classify` that takes `{"text": "..."}` and returns `{"label", "confidence", "source": "student"|"llm", "latency_ms"}`.
- `GET /health`.
- Runs locally with `uvicorn`.

---

## 6. Repository layout

The layout below is planned.

```
.
├── pyproject.toml
├── configs/
│   └── default.yaml          # model ids, LLM base_url, splits, hyperparams, δ
├── src/cascade/
│   ├── data.py               # load Banking77, stratified splits
│   ├── llm_client.py         # OpenAI-compatible client, constrained output
│   ├── label.py              # teacher labeling with resumable cache
│   ├── train_student.py      # DistilBERT fine-tuning
│   ├── calibrate.py          # temperature scaling, ECE/NLL/Brier
│   ├── cascade.py            # routing logic, threshold sweep
│   ├── evaluate.py           # test-set report and plots
│   ├── export_onnx.py        # ONNX export (+ int8)
│   ├── bench.py              # latency benchmark
│   └── serve.py              # FastAPI service
├── tests/                    # unit tests: ECE, threshold sweep, routing, cache
├── data/                     # gitignored: raw data, llm_labels.parquet
├── artifacts/                # gitignored: checkpoints, ONNX model, cascade.json
├── reports/                  # metrics.json, plots
└── docs/
    └── PROJECT.md
```

Example `configs/default.yaml`:

```yaml
seed: 42
dataset: PolyAI/banking77
splits: {student_train: 0.72, student_val: 0.08, calib: 0.20}

llm:
  base_url: http://localhost:11434/v1   # Ollama; change for other backends
  api_key: none                         # local servers ignore it
  model: qwen2.5:7b-instruct
  few_shot_per_class: 0                 # 0 = zero-shot
  temperature: 0

student:
  model: distilbert-base-uncased
  max_length: 64
  batch_size: 32
  lr: 5.0e-5
  epochs: 5

cascade:
  calibration: temperature
  ece_bins: 15
  accuracy_tolerance: 0.01              # δ, see §9.6
```

---

## 7. Setup (free)

**Prerequisites:**

- macOS, Linux, or Windows (WSL)
- About 10 GB of free disk for a quantized 7B model, or none if you use a cloud backend
- 16 GB or more of RAM to run the 7B model locally

### 7.1 Python environment

Use Python 3.12. Wheels for PyTorch and ONNX Runtime are reliably available for it, while newer interpreters such as 3.14 can lag. [`uv`](https://github.com/astral-sh/uv) installs it without changing your system Python.

```bash
brew install uv
```

```bash
uv venv --python 3.12
```

```bash
source .venv/bin/activate
```

```bash
uv pip install -e ".[dev]"
```

Planned dependencies, all open source:

- `torch`
- `transformers`
- `datasets`
- `accelerate`
- `scikit-learn`
- `numpy`
- `pandas`
- `pyarrow`
- `pyyaml`
- `openai` (client only)
- `onnx`
- `onnxruntime`
- `matplotlib`
- `fastapi`
- `uvicorn`
- `pytest` (dev only)

### 7.2 LLM backend

Pick one option ([§9.1](#91-where-the-llm-runs)). Option A is shown here because it needs no account:

```bash
brew install ollama
```

```bash
ollama serve
```

```bash
ollama pull qwen2.5:7b-instruct
```

The OpenAI-compatible endpoint is now at `http://localhost:11434/v1`, which matches the default config.

---

## 8. Running the pipeline

Each step reads `configs/default.yaml` and writes to `data/`, `artifacts/`, or `reports/`. All steps are idempotent.

| # | Step | Planned command | Output |
|---|---|---|---|
| 1 | Prepare splits | `python -m cascade.data` | `data/splits.parquet` |
| 2 | Teacher labeling | `python -m cascade.label` | `data/llm_labels.parquet` |
| 3 | Train student | `python -m cascade.train_student` | `artifacts/student/` |
| 4 | Calibrate + choose τ | `python -m cascade.calibrate` | `artifacts/cascade.json` |
| 5 | Export | `python -m cascade.export_onnx` | `artifacts/student.onnx` |
| 6 | Evaluate | `python -m cascade.evaluate` | `reports/metrics.json`, plots |
| 7 | Benchmark | `python -m cascade.bench` | `reports/latency.json` |
| 8 | Serve (optional) | `uvicorn cascade.serve:app` | HTTP API on :8000 |

**Tip.** Before labeling everything, run step 2 with `--limit 200`. Check the LLM's accuracy and invalid-output rate, and measure its throughput. Use the throughput to estimate total labeling time.

---

## 9. Open design decisions

Each choice below is still open. The free options and their tradeoffs are listed for you to decide. Where a default appears in the config above, it is marked as a suggestion.

### 9.1 Where the LLM runs

This is your highest-impact choice. The same backend serves both the one-time labeling of about 13k texts and the escalated calls at request time.

| Option | How | Pros | Cons |
|---|---|---|---|
| **A. Local (Ollama / llama.cpp)** on your M4 Pro, 24 GB | 4-bit quantized 7B model (about 4.7 GB) | No account, rate limit, or session timeout. Data stays local. The same setup works for the live demo. | Slowest option. Expect roughly 0.3–1 s per query, so about 1–4 h for all 13k; measure this with `--limit 200`. 4-bit quantization can lower accuracy slightly compared with fp16. |
| **B. Free cloud GPU (Kaggle or Colab)** with vLLM | Kaggle offers a weekly GPU quota (T4×2 or P100); Colab's free T4 has variable availability | Batched vLLM inference labels everything in well under an hour. Can run fp16 or AWQ weights. | Sessions time out (Kaggle about 12 h; Colab unpredictable), so the resumable cache is required. Not suitable for serving live escalations, so you would need A or C as well for the demo. Requires an account. |
| **C. Free hosted API** (e.g. Groq, Hugging Face Inference Providers free credits) | OpenAI-compatible URL and a free key | No local setup, and very fast | Rate limits and daily caps change and may not cover 13k calls in a day. Hosted models are often 8B (e.g. Llama 3.1 8B), not exactly 7B. Data is sent to a third party. Free terms can be withdrawn. |

A common combination is **B for bulk labeling and A for the live cascade**. If you do this, use the same model family in both places. The quantized local model may not match the fp16 labeler exactly, so report the agreement between them.

### 9.2 Teacher model

| Model | License | Notes |
|---|---|---|
| Qwen2.5-7B-Instruct | Apache 2.0 | Strong at instruction-following and JSON output. Suggested default. |
| Mistral-7B-Instruct-v0.3 | Apache 2.0 | Widely used baseline. Somewhat weaker at strict formatting. |
| Llama-3.1-8B-Instruct | Llama 3.1 Community License | 8B, not 7B. Common on free hosted APIs. |

Newer open 7–8B instruction-tuned models also work. Pin the exact model ID and quantization in the config, because changing either changes every label.

### 9.3 Prompting strategy

- **Zero-shot:** only the label names. This is the simplest option and uses no gold labels. With 77 fine-grained intents, accuracy is likely to be noticeably lower.
- **k-shot per class:** k gold examples per intent are taken from the train split. This usually raises LLM accuracy a lot, but the prompt grows to several thousand tokens, which slows down local inference. These examples come from gold labels, so record that fact in the write-up.
- **Retrieved few-shot:** a free local embedding model finds the most similar labeled examples for each query. This gives the best accuracy per token, but it adds a dependency and another component to explain.

### 9.4 What the student is calibrated against

- **LLM labels on `calib`, the suggested option.** Confidence then means "probability the LLM would give the same answer". This matches the cascade's goal of matching the LLM, and it needs no human labels at any stage.
- **Gold labels on `calib`.** Confidence means "probability of being correct". This is closer to how ECE is reported on the test set, but it assumes a human-labeled calibration set, which a real deployment may not have.
- **Either way:** report test-set ECE against both gold and LLM labels so the choice is transparent.

### 9.5 Calibration method

| Method | Parameters | Tradeoff |
|---|---|---|
| Temperature scaling | 1 | Hard to overfit, preserves accuracy, and is easy to explain. Cannot fix class-specific miscalibration. Suggested default. |
| Vector / matrix scaling | 77 / 77² + 77 | More flexible, but can overfit 2k calibration examples split across 77 classes. |
| Histogram binning / isotonic regression on the max-probability | Non-parametric | Can reach a lower ECE, but needs more data and gives step-shaped confidences. |

### 9.6 Accuracy tolerance δ and threshold selection

- **δ = 0, strict.** Under this rule, "matching the LLM" means the cascade's accuracy is ≥ the LLM's. Coverage will be lower.
- **δ = 1 percentage point, the config default.** This is a common practical tolerance and allows higher coverage.
- **Rule of thumb.** The tighter δ is, the further coverage falls below 85%. Report the full risk–coverage curve so readers can see the tradeoff for themselves.
- **Fixed-coverage alternative.** Set τ to hit exactly 85% coverage on `calib`, then report whatever accuracy results.

### 9.7 Student inference runtime

| Runtime | Expected speed | Tradeoff |
|---|---|---|
| PyTorch eager fp32 | Slowest | No conversion step |
| ONNX Runtime fp32 | Faster | Requires an export step. Predictions should be identical, which should be verified. |
| ONNX Runtime dynamic int8 | Fastest and about 4× smaller | Can cost a small amount of accuracy and shift calibration. If chosen, fit T *after* quantization. |

Whether 8 ms is reached depends on the CPU. Report it for each machine you test.

### 9.8 Demo hosting (optional)

- **None or local only:** run the FastAPI app with uvicorn on your machine. This has no limitations.
- **Hugging Face Spaces, free CPU tier:** gives you a public URL, and the student runs well on CPU. A 7B LLM is too slow on the free CPU hardware, though. Escalations would have to go to a free hosted API (option C), or the demo would show "would escalate" instead of calling the LLM. Free Spaces also sleep when idle.

---

## 10. Zero-cost bill of materials

| Need | Free choice | Tradeoff compared with a paid option |
|---|---|---|
| Teacher LLM | Open 7B weights (Qwen2.5 / Mistral) | Lower accuracy than frontier API models |
| LLM compute | Local Mac, Kaggle/Colab free GPU, or free API tier | Slower, or capped by quotas and timeouts (§9.1) |
| Student training | MPS on Mac or a free GPU | Fine at this scale |
| Data | Banking77 (CC BY 4.0) | Public benchmark, not production traffic |
| Experiment tracking | JSON reports + git | No dashboard. MLflow can run locally for free if wanted. |
| Serving | FastAPI + uvicorn locally, or a free HF Space | No uptime guarantee, and Spaces sleep when idle |
| Storage | Local disk, with `data/` and `artifacts/` gitignored | Not backed up. Hugging Face Hub (free) can host the model if needed. |

---

## 11. Known limitations and risks

- **The student inherits the teacher's errors.** It is trained on LLM labels, so systematic LLM mistakes are learned as if they were correct. The student may even match the LLM's wrong answers with high confidence, and the cascade then keeps those errors. Measuring against gold labels makes this visible.
- **The coverage target is not guaranteed.** Reaching 85% depends on how often the student agrees with the LLM and on δ. If the LLM is weak on Banking77, gold accuracy is low for both models, and the comparison means less.
- **Calibration can shift.** It is fit on in-distribution `calib` data, and inputs unlike Banking77 can break it.
- **Latency figures depend on hardware.** Always report them with the CPU model and thread count.
- **Labeling cost is not counted.** The "85% fewer LLM calls" figure excludes the one-time cost of labeling about 13k texts. Report that cost separately.
- **Local and fp16 teachers may differ.** If labeling and serving use different quantizations of the LLM, the escalation answers may differ slightly from the training labels.
