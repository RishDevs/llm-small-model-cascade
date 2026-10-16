import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class Student:
    def __init__(self, model_dir, onnx_path, max_length=64, threads=None):
        model_dir = Path(model_dir)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.labels = json.loads((model_dir / "labels.json").read_text())
        self.max_length = max_length
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    def encode(self, texts):
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="np")
        return {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }

    def forward(self, feeds):
        return self.session.run(["logits"], feeds)[0]

    def logits(self, texts, batch_size=256):
        parts = [self.forward(self.encode(texts[i:i + batch_size])) for i in range(0, len(texts), batch_size)]
        return np.concatenate(parts).astype(np.float64)
