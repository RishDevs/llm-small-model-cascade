# Project reference

This page is the technical reference for the cascade: data, pipeline stages, configuration, outputs, and the free compute it runs on. For setup commands, see the [README](../README.md).

## Contents

1. [Data](#1-data)
2. [Pipeline](#2-pipeline)
3. [Configuration](#3-configuration)
4. [Outputs](#4-outputs)
5. [Metrics and how they're defined](#5-metrics-and-how-theyre-defined)
6. [Zero-cost choices and their tradeoffs](#6-zero-cost-choices-and-their-tradeoffs)
7. [Design choices still open](#7-design-choices-still-open)

---

## 1. Data

CLINC150, `plus` variant, loaded from the Hugging Face Hub as `clinc/clinc_oos`.

| Split | Utterances | Out-of-scope (`oos`) | Used for |
|---|---|---|---|
| train | 15,250 | 250 (1.6%) | Teacher labels → student training |
| validation | 3,100 | 100 (3.2%) | Fitting the temperature, picking the threshold |
| test | 5,500 | 1,000 (18.2%) | Final scoring against gold |

There are 151 labels: 150 intents plus `oos`. The test split has far more out-of-scope traffic than train or validation. That difference is deliberate in the dataset, and it matters for how routing behaves on test (see [§5](#5-metrics-and-how-theyre-defined)).

## 2. Pipeline

| Stage | Module | In | Out |
|---|---|---|---|
| Teacher labeling | `cascade.teacher` | CLINC150 text | `data/teacher/{train,validation,test}.jsonl` |
| Student training | `cascade.train` | train text + teacher labels | `artifacts/student/` (HF checkpoint, tokenizer, `labels.json`) |
| Export + INT8 | `cascade.export` | student checkpoint | `student.onnx` (FP32, ~257 MB), `student.int8.onnx` (~65 MB) |
| Calibrate + route + score | `cascade.evaluate` | INT8 model, teacher labels, gold | `artifacts/routing.json`, `reports/*` |
| Latency | `cascade.bench` | ONNX models | `reports/latency.json` |
| Serving | `cascade.serve` | ONNX model, `routing.json`, LLM endpoint | HTTP API |

**Teacher labels.** These are labels only, not probabilities. Each cached row is `{"idx", "raw", "label", "secs"}`:

- `label` is `null` when the raw output wasn't a valid label.
- Rows that failed at the network level aren't written, so a rerun retries them.

**Validation labels.** The validation split is labeled by the teacher too. Calibration and threshold search are fitted against teacher labels, so neither needs human labels.

**Test labels.** The test split is labeled by the teacher once. Escalation can then be simulated offline, because "ask the LLM" means reading the cached teacher label.

## 3. Configuration

All settings live in `configs/default.yaml`, which runs the Qwen2.5-7B teacher. `configs/qwen3.8.yaml` is the main run: it uses the Qwen3.8 teacher and writes to `data/teacher_qwen3.8`, `artifacts/qwen3.8` and `reports/qwen3.8`. Every command accepts `--config path/to/other.yaml`.

| Key | Default | Meaning |
|---|---|---|
| `teacher.backend` | `vllm` | `vllm` (offline batch on GPU) or `http` (any OpenAI-compatible server) |
| `teacher.model` | `Qwen/Qwen2.5-7B-Instruct-AWQ` | Model for the vLLM backend |
| `teacher.api_extra` | `{}` | Extra request fields. `configs/qwen3.8.yaml` uses `{reasoning_effort: "none"}` to turn thinking off |
| `teacher.api_model` | `qwen2.5:7b-instruct` | Model name for the HTTP backend and live escalation |
| `teacher.base_url` | `http://localhost:11434/v1` | HTTP endpoint (Ollama by default) |
| `teacher.chunk_size` | 2048 | Utterances per vLLM batch between cache writes |
| `student.base` | `distilbert-base-uncased` | Student checkpoint |
| `student.max_length` | 64 | Token limit; CLINC utterances are much shorter |
| `student.epochs`, `lr`, `batch_size` | 4, 5e-5, 32 | Fine-tuning settings |
| `routing.max_disagreement` | 0.01 | Allowed share of validation traffic where the student answers and disagrees with the teacher |
| `routing.ece_bins` | 15 | Bins for ECE |
| `bench.threads` | [1, 4] | ONNX Runtime intra-op threads to test |

## 4. Outputs

| File | Contents |
|---|---|
| `reports/teacher_constrained.json` | Per split: labeled count, invalid rate, accuracy vs gold, OOS recall, wall-clock minutes |
| `reports/teacher_free.json` | Same fields, for the free-form prompting run |
| `reports/train.json` | Device, number of examples, per-epoch loss and validation agreement |
| `reports/export.json` | Model sizes; argmax agreement of FP32 and INT8 ONNX with PyTorch |
| `reports/metrics.json` | Everything `evaluate` measures |
| `reports/RESULTS.md` | The same numbers as a readable table |
| `reports/reliability.png` | Reliability diagram before and after temperature scaling |
| `reports/coverage.png` | Cascade accuracy as student coverage goes from 0% to 100% |
| `reports/latency.json` | p50, p95 and mean in ms, per model and thread count, with the CPU name |

## 5. Metrics and how they're defined

| Metric | Definition |
|---|---|
| Coverage / LLM calls avoided | Share of test utterances where calibrated confidence ≥ τ |
| Accuracy | Prediction matches the gold label (`oos` included as a class) |
| Gap vs teacher | Cascade accuracy minus teacher-only accuracy, in percentage points |
| OOS recall | Share of gold-`oos` utterances predicted as `oos` |
| OOS recall gain | Cascade OOS recall minus student-only OOS recall |
| ECE | Expected calibration error with 15 equal-width confidence bins. Reported against teacher labels (what the router uses) and against gold labels |
| Latency | Batch-1 wall-clock time over 1,000 test utterances after 50 warm-up runs, reported two ways: model-only, and tokenization through to the routing decision |

**Coverage on test vs validation.** Test has 18% out-of-scope utterances against 3% in validation, so coverage can differ between them. In the main run it was 71.7% on validation and 68.0% on test.

## 6. Zero-cost choices and their tradeoffs

| Need | Free choice | Tradeoff |
|---|---|---|
| Teacher | Qwen3.8 27.8B, 4-bit, in Ollama (main run). Qwen2.5-7B for comparison | The 27.8B model needs about 24 GB and about 16 h to label everything on a laptop. The 7B model is fast but 17 points less accurate |
| Labeling compute | Kaggle/Colab free T4 | Session limits, which the resumable cache handles |
| Live escalation | Ollama on your own machine | Slower than a hosted endpoint. Ollama's Q4 build isn't byte-identical to the AWQ labeler |
| Student training | MPS, free GPU, or CPU | None at this size |
| Serving | FastAPI + uvicorn | No auth, rate limiting or autoscaling |
| Tracking | JSON reports + git | No dashboard |

## 7. Design choices still open

- **Disagreement budget.** `routing.max_disagreement` sets the tradeoff. Raising it means more coverage and a larger possible accuracy gap. `reports/coverage.png` shows the whole curve, so the choice can be made from data.
- **Calibration target.** Temperature is fitted against teacher labels, so confidence means "chance the teacher would agree". Fitting against gold validation labels is a one-line change, but it assumes you have human labels at deployment time.
- **INT8 vs FP32 student.** INT8 is faster and 4× smaller, and it flips a small share of predictions; `reports/export.json` gives the exact share. To score the FP32 model instead, run `python -m cascade.evaluate --fp32`.
