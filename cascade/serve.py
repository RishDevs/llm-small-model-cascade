import json
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from cascade.calibration import softmax
from cascade.config import load_config
from cascade.llm import ChatLabeler
from cascade.student import Student


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def build_app(config=None, student=None, teacher=None, routing=None):
    cfg = load_config(config or os.environ.get("CASCADE_CONFIG"))
    paths = cfg["paths"]

    if routing is None:
        routing = json.loads(paths["routing"].read_text())
    temperature = routing["temperature"]
    threshold = routing["threshold"] if routing["threshold"] is not None else float("inf")

    if student is None:
        threads = int(os.environ.get("CASCADE_THREADS", "1"))
        student = Student(paths["student_dir"], paths["student_dir"].parent / routing["onnx"], cfg["student"]["max_length"], threads)

    if teacher is None and not os.environ.get("CASCADE_NO_LLM"):
        tcfg = cfg["teacher"]
        teacher = ChatLabeler(
            os.environ.get("CASCADE_LLM_URL", tcfg["base_url"]),
            os.environ.get("CASCADE_LLM_MODEL", tcfg["api_model"]),
            student.labels,
            timeout=tcfg["timeout"],
            extra_body=tcfg.get("api_extra"),
        )

    app = FastAPI(title="llm-small-model-cascade")

    @app.get("/health")
    def health():
        return {"ok": True, "temperature": temperature, "threshold": routing["threshold"], "llm": teacher is not None}

    @app.post("/classify")
    def classify(q: Query):
        t0 = time.perf_counter()
        probs = softmax(student.forward(student.encode([q.text])), temperature)[0]
        top = int(probs.argmax())
        conf = float(probs[top])
        label, source = student.labels[top], "student"
        if conf < threshold:
            answer = teacher.label(q.text) if teacher else None
            if answer is None:
                source = "student_fallback"
            else:
                label, source = answer, "llm"
        return {
            "label": label,
            "confidence": round(conf, 4),
            "source": source,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    return app
