import json
import math
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from cascade.config import base_parser, load_config, seed_everything
from cascade.data import load_clinc, split_arrays, teacher_labels


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(tok, texts, targets, max_length, batch_size, shuffle):
    pairs = list(zip(texts, targets))

    def collate(chunk):
        enc = tok([t for t, _ in chunk], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in chunk])
        return enc

    return DataLoader(pairs, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def agreement(model, loader, device):
    model.eval()
    hits = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch.pop("labels")
        pred = model(**batch).logits.argmax(-1)
        hits += (pred == y).sum().item()
        total += len(y)
    return hits / total


def main():
    parser = base_parser("Fine-tune the DistilBERT student on teacher labels")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    scfg = cfg["student"]
    seed_everything(cfg["seed"])
    ds, labels = load_clinc(cfg)
    teacher_dir = cfg["paths"]["teacher_dir"]

    texts, _ = split_arrays(ds, "train")
    targets = teacher_labels(teacher_dir, "train", labels, len(texts))
    keep = [i for i, y in enumerate(targets) if y >= 0]
    if not keep:
        raise SystemExit(f"no teacher labels in {teacher_dir}/train.jsonl - run cascade.teacher first")
    print(f"training on {len(keep)}/{len(texts)} teacher-labelled utterances")
    train_texts = [texts[i] for i in keep]
    train_y = [targets[i] for i in keep]

    val_texts, _ = split_arrays(ds, "validation")
    val_y = teacher_labels(teacher_dir, "validation", labels, len(val_texts))
    val_keep = [i for i, y in enumerate(val_y) if y >= 0]

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(scfg["base"])
    model = AutoModelForSequenceClassification.from_pretrained(
        scfg["base"],
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={name: i for i, name in enumerate(labels)},
    ).to(device)

    train_loader = batches(tok, train_texts, train_y, scfg["max_length"], scfg["batch_size"], True)
    val_loader = batches(
        tok, [val_texts[i] for i in val_keep], [val_y[i] for i in val_keep], scfg["max_length"], 128, False
    )

    epochs = args.epochs or scfg["epochs"]
    steps = epochs * len(train_loader)
    params = list(model.parameters())
    groups = [
        {"params": [p for p in params if p.ndim > 1], "weight_decay": scfg["weight_decay"]},
        {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=scfg["lr"])
    sched = get_linear_schedule_with_warmup(opt, math.ceil(scfg["warmup"] * steps), steps)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += loss.item()
        val_agree = agreement(model, val_loader, device) if val_keep else None
        history.append({"epoch": epoch, "loss": running / len(train_loader), "val_agreement": val_agree})
        print(f"epoch {epoch}: loss {history[-1]['loss']:.4f}  val agreement {val_agree}  ({time.time() - t0:.0f}s)")

    out = cfg["paths"]["student_dir"]
    out.mkdir(parents=True, exist_ok=True)
    model.cpu().save_pretrained(out)
    tok.save_pretrained(out)
    (out / "labels.json").write_text(json.dumps(labels))
    reports = cfg["paths"]["reports"]
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "train.json").write_text(
        json.dumps({"device": str(device), "examples": len(keep), "history": history}, indent=2)
    )


if __name__ == "__main__":
    main()
