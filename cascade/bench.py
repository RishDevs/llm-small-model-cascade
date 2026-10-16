import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from cascade.calibration import softmax
from cascade.config import base_parser, load_config
from cascade.data import load_clinc, split_arrays
from cascade.student import Student


def cpu_name():
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def stats(ms):
    ms = np.asarray(ms)
    return {"p50": round(float(np.percentile(ms, 50)), 2), "p95": round(float(np.percentile(ms, 95)), 2), "mean": round(float(ms.mean()), 2)}


def time_one(student, texts, warmup, temperature, threshold):
    feeds = [student.encode([t]) for t in texts]
    for f in feeds[:warmup]:
        student.forward(f)

    model_ms = []
    for f in feeds:
        t0 = time.perf_counter()
        student.forward(f)
        model_ms.append((time.perf_counter() - t0) * 1000)

    full_ms = []
    for text in texts:
        t0 = time.perf_counter()
        probs = softmax(student.forward(student.encode([text])), temperature)[0]
        _ = probs.max() >= threshold
        full_ms.append((time.perf_counter() - t0) * 1000)

    return {"model_only": stats(model_ms), "tokenize_to_decision": stats(full_ms)}


def main():
    args = base_parser("Measure batch-1 CPU latency of the student").parse_args()
    cfg = load_config(args.config)
    paths, bcfg = cfg["paths"], cfg["bench"]

    ds, _ = load_clinc(cfg)
    texts, _ = split_arrays(ds, "test")
    texts = [texts[i % len(texts)] for i in range(bcfg["runs"])]

    temperature, threshold = 1.0, 1.0
    if paths["routing"].exists():
        routing = json.loads(paths["routing"].read_text())
        temperature = routing["temperature"]
        threshold = routing["threshold"] if routing["threshold"] is not None else float("inf")

    report = {"cpu": cpu_name(), "batch_size": 1, "runs": bcfg["runs"], "results": {}}
    for name in ("onnx_int8", "onnx_fp32"):
        for threads in bcfg["threads"]:
            student = Student(paths["student_dir"], paths[name], cfg["student"]["max_length"], threads=threads)
            key = f"{name.removeprefix('onnx_')}_{threads}t"
            report["results"][key] = time_one(student, texts, bcfg["warmup"], temperature, threshold)
            print(key, report["results"][key])

    paths["reports"].mkdir(parents=True, exist_ok=True)
    (paths["reports"] / "latency.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
