import json

import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from cascade.config import base_parser, load_config
from cascade.data import load_clinc, split_arrays
from cascade.student import Student


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def to_onnx(student_dir, path):
    model = AutoModelForSequenceClassification.from_pretrained(student_dir).eval()
    tok = AutoTokenizer.from_pretrained(student_dir)
    sample = tok(["what is my checking account balance", "hi"], padding=True, return_tensors="pt")
    batch, seq = torch.export.Dim("batch"), torch.export.Dim("seq")
    torch.onnx.export(
        LogitsOnly(model).eval(),
        (sample["input_ids"], sample["attention_mask"]),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
        opset_version=18,
        external_data=False,
        dynamo=True,
    )
    graph = onnx.load(str(path))
    del graph.graph.value_info[:]
    onnx.save(graph, str(path))


@torch.no_grad()
def torch_argmax(student_dir, texts, max_length):
    model = AutoModelForSequenceClassification.from_pretrained(student_dir).eval()
    tok = AutoTokenizer.from_pretrained(student_dir)
    preds = []
    for i in range(0, len(texts), 256):
        enc = tok(texts[i:i + 256], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        preds.append(model(**enc).logits.argmax(-1).numpy())
    return np.concatenate(preds)


def main():
    args = base_parser("Export the student to ONNX and quantize it to INT8").parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    student_dir = paths["student_dir"]
    max_length = cfg["student"]["max_length"]

    to_onnx(student_dir, paths["onnx_fp32"])
    quantize_dynamic(str(paths["onnx_fp32"]), str(paths["onnx_int8"]), weight_type=QuantType.QInt8)

    ds, _ = load_clinc(cfg)
    texts, _ = split_arrays(ds, "validation")
    ref = torch_argmax(student_dir, texts, max_length)
    report = {"checked_on": f"validation ({len(texts)} utterances)"}
    for name in ("onnx_fp32", "onnx_int8"):
        pred = Student(student_dir, paths[name], max_length).logits(texts).argmax(1)
        report[name] = {
            "size_mb": round(paths[name].stat().st_size / 2**20, 1),
            "argmax_match_vs_torch": round(float((pred == ref).mean()), 4),
        }
    print(json.dumps(report, indent=2))
    paths["reports"].mkdir(parents=True, exist_ok=True)
    (paths["reports"] / "export.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
