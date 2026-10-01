# LLM → Small Model Cascade

Calling a large LLM for every simple classification is slow and expensive. This project trains a small model to handle the easy cases and sends only the hard ones to the LLM.

- **Teacher:** an open 7B instruction-tuned LLM labels the training data.
- **Student:** DistilBERT is fine-tuned on those labels and runs fast on a CPU.
- **Router:** calibrated confidence scores (temperature scaling) decide when the student answers and when the request is escalated to the LLM.

Task: 77-way intent classification on [Banking77](https://huggingface.co/datasets/PolyAI/banking77).

Everything runs at zero cost: open-source models and libraries, local or free-tier compute, and no paid services.

> **Status:** in development. Results will be added once the evaluation is complete.

## How it works

```
text ──► DistilBERT student ──► calibrated confidence
                                   │
                    conf ≥ τ ──────┼────── conf < τ
                       │                      │
               student's label         escalate to 7B LLM
```

1. The 7B LLM labels all Banking77 texts once. Its outputs are constrained to valid intent names and cached.
2. DistilBERT is trained on the LLM's labels. The human-labeled gold test set is used only for evaluation.
3. Temperature scaling, fitted on a held-out split, calibrates the student's confidence.
4. The threshold τ is set to the highest level of student coverage that still keeps the cascade's accuracy matched to the LLM's.
5. The student is exported to ONNX for fast CPU inference.

## Goals

| Goal | Target |
|---|---|
| Reduction in LLM calls | ~85%, while matching the LLM's accuracy |
| Student latency (CPU, batch size 1) | ~8 ms |
| Calibration error (ECE) | ~50% lower after calibration |

## Stack

- **ML:** Python 3.12, PyTorch, Hugging Face Transformers / Datasets, scikit-learn, ONNX Runtime
- **LLM serving:** any OpenAI-compatible endpoint, such as Ollama, llama.cpp or vLLM
- **API:** FastAPI (optional)

## Documentation

See [docs/PROJECT.md](docs/PROJECT.md) for the full design, setup steps and open design decisions.

## License

The Banking77 dataset is licensed under CC BY 4.0.
