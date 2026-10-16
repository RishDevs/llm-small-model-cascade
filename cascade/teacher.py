import json
import time
from pathlib import Path

from cascade.config import base_parser, load_config
from cascade.data import OOS, SPLITS, append_jsonl, load_clinc, read_jsonl, split_arrays
from cascade.prompts import messages, parse_label


def choice_constraint(choices):
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return {"structured_outputs": StructuredOutputsParams(choice=choices)}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return {"guided_decoding": GuidedDecodingParams(choice=choices)}


class VLLMLabeler:
    def __init__(self, tcfg, labels, constrained, seed):
        from vllm import LLM, SamplingParams

        self.labels = list(labels)
        self.llm = LLM(
            model=tcfg["model"],
            dtype="half",
            max_model_len=tcfg["max_model_len"],
            gpu_memory_utilization=tcfg["gpu_memory_utilization"],
            enable_prefix_caching=True,
            seed=seed,
        )
        self.tok = self.llm.get_tokenizer()
        extra = choice_constraint(self.labels) if constrained else {}
        self.params = SamplingParams(temperature=0, max_tokens=24, **extra)

    def __call__(self, texts):
        prompts = [
            self.tok.apply_chat_template(messages(t, self.labels), tokenize=False, add_generation_prompt=True)
            for t in texts
        ]
        outs = self.llm.generate(prompts, self.params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


def make_labeler(backend, cfg, labels, constrained):
    tcfg = cfg["teacher"]
    if backend == "vllm":
        return VLLMLabeler(tcfg, labels, constrained, cfg["seed"])
    from cascade.llm import ChatLabeler

    return ChatLabeler(
        tcfg["base_url"],
        tcfg["api_model"],
        labels,
        constrained=constrained,
        timeout=tcfg["timeout"],
        workers=tcfg["workers"],
        extra_body=tcfg.get("api_extra"),
    )


def summarize(rows, gold, labels, n):
    rows = [r for r in rows if r["idx"] < n]
    oos = labels.index(OOS)
    index = {name: i for i, name in enumerate(labels)}
    pred = {r["idx"]: index.get(r["label"], -1) for r in rows}
    hits = sum(pred[i] == gold[i] for i in pred)
    oos_ids = [i for i in pred if gold[i] == oos]
    return {
        "labeled": len(rows),
        "of": n,
        "invalid_rate": round(sum(r["label"] is None for r in rows) / max(len(rows), 1), 4),
        "accuracy": round(hits / max(len(rows), 1), 4),
        "oos_recall": round(sum(pred[i] == oos for i in oos_ids) / len(oos_ids), 4) if oos_ids else None,
        "minutes": round(sum(r["secs"] for r in rows) / 60, 2),
    }


def main():
    parser = base_parser("Label CLINC150 with the teacher LLM")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--mode", choices=["constrained", "free"], default="constrained")
    parser.add_argument("--backend", choices=["vllm", "http"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out")
    args = parser.parse_args()

    cfg = load_config(args.config)
    backend = args.backend or cfg["teacher"]["backend"]
    out_dir = cfg["paths"]["teacher_dir"]
    if args.out:
        out_dir = Path(args.out)
    elif args.mode == "free":
        out_dir = out_dir.with_name(out_dir.name + "_free")

    ds, labels = load_clinc(cfg)
    labelset = set(labels)
    labeler = None
    chunk = cfg["teacher"]["chunk_size"] if backend == "vllm" else cfg["teacher"]["workers"] * 16

    summary = {}
    for split in args.splits:
        texts, gold = split_arrays(ds, split)
        n = min(args.limit or len(texts), len(texts))
        path = out_dir / f"{split}.jsonl"
        done = {r["idx"] for r in read_jsonl(path)}
        todo = [i for i in range(n) if i not in done]
        print(f"{split}: {len(done)} cached, {len(todo)} to label")

        for start in range(0, len(todo), chunk):
            if labeler is None:
                labeler = make_labeler(backend, cfg, labels, args.mode == "constrained")
            ids = todo[start:start + chunk]
            t0 = time.perf_counter()
            raws = labeler([texts[i] for i in ids])
            per_item = (time.perf_counter() - t0) / len(ids)
            rows = [
                {"idx": i, "raw": raw, "label": parse_label(raw, labelset), "secs": per_item}
                for i, raw in zip(ids, raws)
                if raw is not None
            ]
            append_jsonl(path, rows)
            print(f"  {split} {start + len(ids)}/{len(todo)}  {per_item * 1000:.0f} ms/item")

        summary[split] = summarize(read_jsonl(path), gold, labels, n)
        print(f"  {split}: {summary[split]}")

    report_path = cfg["paths"]["reports"] / f"teacher_{args.mode}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(report_path.read_text()) if report_path.exists() else {}
    existing.update({"backend": backend, "model": cfg["teacher"]["model" if backend == "vllm" else "api_model"]})
    existing.setdefault("splits", {}).update(summary)
    report_path.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()
