import json
from pathlib import Path

from datasets import load_dataset

SPLITS = ("train", "validation", "test")
OOS = "oos"


def load_clinc(cfg):
    ds = load_dataset(cfg["data"]["dataset"], cfg["data"]["subset"])
    labels = ds["train"].features["intent"].names
    return ds, labels


def split_arrays(ds, split):
    return list(ds[split]["text"]), list(ds[split]["intent"])


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def teacher_labels(teacher_dir, split, labels, n):
    index = {name: i for i, name in enumerate(labels)}
    out = [-1] * n
    for row in read_jsonl(Path(teacher_dir) / f"{split}.jsonl"):
        if row["label"] is not None:
            out[row["idx"]] = index[row["label"]]
    return out
