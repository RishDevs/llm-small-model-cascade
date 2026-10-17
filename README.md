# LLM → Small Model Cascade

Sending every classification request to a large LLM is slow and expensive. Most requests are easy, though. This project distills an open LLM into a small INT8 DistilBERT model that runs in milliseconds on a CPU. Each request is routed on temperature-scaled confidence: the small model answers when it is sure, and the request goes to the LLM when it isn't.

Task: intent classification on [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos), which has 150 intents plus an out-of-scope class.

Everything runs for free: open models, open-source libraries, and your own machine or a free Kaggle/Colab T4.

## Results

Results are on the CLINC150 test split (5,500 utterances, 18% out-of-scope). The student is DistilBERT, INT8, on ONNX Runtime. Both teachers are zero-shot and ran locally through Ollama. Full numbers are in [`reports/qwen3.8/`](reports/qwen3.8/) for the main run and [`reports/`](reports/) for the 7B run.

**Main run: teacher Qwen3.8 (27.8B, 4-bit)**

| | Teacher only | Student only | Cascade |
|---|---|---|---|
| Accuracy | 89.09% | 85.69% | **89.27%** |
| Out-of-scope recall | 88.8% | 79.3% | **89.6%** |
| LLM calls | 100% | 0% | **32.0%** |

- **68% fewer LLM calls**, with cascade accuracy +0.2 pp over the teacher alone. The threshold was picked on validation under a 1% teacher-disagreement budget.
- **Out-of-scope recall** is +10.3 pp over the student alone. Out-of-scope messages are 30% of escalations, against 18% of traffic.
- **Student latency:** 1.9 ms p50 and 3.1 ms p95, at batch size 1 on one CPU thread (Apple M4 Pro, INT8). FP32 runs at 3.6 ms.
- **INT8:** 257 MB → 65 MB, with 98.3% prediction agreement with the FP32 PyTorch model.
- **Labeling:** all 23,850 utterances took about 16 hours on a laptop, at zero cost, with 0 invalid outputs.
- **Calibration tradeoff:** the temperature is chosen to minimize validation log-loss against the teacher's labels, not ECE. The two objectives disagreed here. The student was already well calibrated (ECE 0.021). T = 0.94 lowered test log-loss (0.4493 → 0.4478) but raised ECE to 0.027. I kept it because routing is what the system is for: with the threshold re-picked on validation, it raised coverage from 67.3% to 68.0% at the same 89.27% accuracy. See [docs/PROJECT.md §6](docs/PROJECT.md#6-calibration-tradeoff).

**Teacher comparison: same pipeline, same 1% budget**

| Teacher | Teacher accuracy | Cascade accuracy | LLM calls avoided | Labeling time |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 72.4% | 72.6% | 45.0% | ~100 min |
| Qwen3.8 (27.8B) | 89.1% | 89.3% | **68.0%** | ~16 h |

A teacher the student can copy more reliably lets more traffic skip the LLM. With Qwen2.5-7B, constrained decoding also raised teacher accuracy from 67.8% to 74.6%, compared with free-form prompting on the same 2,000 utterances.

## How it works

```
                 offline                                           online
┌────────────────────────────────────────────┐        ┌──────────────────────────────────┐
│ CLINC150 text                              │        │ request                          │
│   └─► Qwen LLM (Ollama/vLLM, constrained   │        │   └─► INT8 DistilBERT (ONNX, CPU)│
│       to 151 labels) ─► teacher labels     │        │         └─► softmax(logits / T)  │
│   └─► fine-tune DistilBERT on them         │        │   conf ≥ τ ─► student's label    │
│   └─► export ONNX ─► INT8                  │        │   conf < τ ─► ask the LLM         │
│   └─► fit T, pick τ on validation          │        └──────────────────────────────────┘
└────────────────────────────────────────────┘
```

1. **Teacher labels.** An open Qwen model labels every utterance: Qwen3.8 (27.8B) in the main run, Qwen2.5-7B in the comparison. Decoding is constrained to the label set, so every answer is a valid label.
2. **Student.** `distilbert-base-uncased` is fine-tuned on the teacher's labels. Gold labels are used only for scoring.
3. **INT8.** The student is exported to ONNX and dynamically quantized to INT8, which makes it about 4× smaller.
4. **Calibration.** Temperature scaling is fitted on the validation split.
5. **Routing threshold.** τ is the threshold that lets the student answer as much as possible, while it disagrees with the teacher on at most 1% of validation traffic.
6. **Evaluation.** Everything is scored on the test split against gold labels: accuracy, out-of-scope recall, LLM calls avoided, ECE and latency.

## Layout

```
cascade/
  config.py        config loading, seeding
  data.py          CLINC150 loading, teacher-label cache I/O
  prompts.py       teacher prompt, output parsing
  teacher.py       labeling with vLLM or any OpenAI-compatible server
  llm.py           OpenAI-compatible client (local labeling and live escalation)
  train.py         DistilBERT fine-tuning
  export.py        ONNX export, INT8 quantization, parity check
  student.py       ONNX Runtime wrapper
  calibration.py   temperature scaling, ECE / NLL / Brier
  routing.py       threshold search and routing
  evaluate.py      calibrate, pick threshold, score on test
  bench.py         batch-1 CPU latency
  serve.py         FastAPI service
configs/default.yaml
tests/
docs/
```

## Setup

You need Python 3.10–3.13. These steps use [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/RishDevs/llm-small-model-cascade.git
cd llm-small-model-cascade
make setup
source .venv/bin/activate
make test
```

## 1. Teacher labels

Pick one option.

**Free T4 on Kaggle (fast).** Create a notebook, set Accelerator to *GPU T4* and turn Internet on, then run:

```
!git clone https://github.com/RishDevs/llm-small-model-cascade.git
%cd llm-small-model-cascade
!pip install -q vllm
!python -m cascade.teacher --backend vllm
```

Download `data/teacher/*.jsonl` from the notebook's output and put the files in `data/teacher/` in your local clone.

The labeler appends results in chunks and skips anything already labeled. If the session dies, rerun the same command and it picks up where it stopped. Colab works the same way.

**Locally with Ollama (no account).**

```bash
ollama pull qwen2.5:7b-instruct
make label-local
```

For the main run's teacher, use the Qwen3.8 config. It needs about 24 GB of memory, and labeling takes about 16 hours on an M4 Pro:

```bash
make label-local CONFIG=configs/qwen3.8.yaml
make pipeline CONFIG=configs/qwen3.8.yaml
```

**Optional: measure free-form prompting.** This labels 2,000 training utterances without constrained decoding. The invalid-output rate goes to `reports/teacher_free.json`, so you can compare it with the constrained run.

```bash
make label-free
```

## 2. Train, export, evaluate, benchmark

```bash
make pipeline
```

This runs four steps in order:

| Step | Writes |
|---|---|
| `train` | `artifacts/student/`, `reports/train.json` |
| `export` | `artifacts/student.onnx`, `artifacts/student.int8.onnx`, `reports/export.json` |
| `evaluate` | `artifacts/routing.json`, `reports/metrics.json`, `reports/RESULTS.md`, `reports/reliability.png`, `reports/coverage.png` |
| `bench` | `reports/latency.json` |

On an Apple-silicon Mac, training uses MPS. On a CUDA machine it uses the GPU, and otherwise it falls back to CPU.

## 3. Serve

```bash
make serve
curl -s localhost:8000/classify -H 'content-type: application/json' -d '{"text": "how much is in my checking account"}'
```

The response looks like this:

```json
{"label": "balance", "confidence": 0.98, "source": "student", "latency_ms": 2.1}
```

Escalations go to the OpenAI-compatible endpoint set by `teacher.base_url` in the config. The default is Ollama at `localhost:11434`. You can override it with these environment variables:

- `CASCADE_LLM_URL` and `CASCADE_LLM_MODEL` point escalations at a different endpoint or model.
- `CASCADE_NO_LLM=1` runs the student alone. Low-confidence requests then come back with `"source": "student_fallback"`.

## Docs

- [docs/PROJECT.md](docs/PROJECT.md): design reference and configuration
- [docs/SIGNIFICANCE.md](docs/SIGNIFICANCE.md): what this is and why it matters
- [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md): code walkthrough, file by file

## License

The code is licensed under [MIT](LICENSE). CLINC150 is licensed separately under CC BY 3.0.
