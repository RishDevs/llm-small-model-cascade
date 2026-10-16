import numpy as np
from fastapi.testclient import TestClient

from cascade.serve import build_app


class FakeStudent:
    labels = ["balance", "oos", "transfer"]

    def __init__(self, logits):
        self._logits = np.array([logits], dtype=np.float64)

    def encode(self, texts):
        return {}

    def forward(self, feeds):
        return self._logits


class FakeTeacher:
    def __init__(self, answer):
        self.answer = answer

    def label(self, text):
        return self.answer


ROUTING = {"temperature": 1.0, "threshold": 0.8, "onnx": "unused.onnx"}


def client(logits, answer="transfer"):
    app = build_app(student=FakeStudent(logits), teacher=FakeTeacher(answer), routing=ROUTING)
    return TestClient(app)


def test_confident_student_answers():
    body = client([8.0, 0.0, 0.0]).post("/classify", json={"text": "balance please"}).json()
    assert body["source"] == "student" and body["label"] == "balance"


def test_unsure_student_escalates():
    body = client([1.0, 0.9, 0.8]).post("/classify", json={"text": "hmm"}).json()
    assert body["source"] == "llm" and body["label"] == "transfer"


def test_llm_down_falls_back_to_student():
    body = client([1.0, 0.9, 0.8], answer=None).post("/classify", json={"text": "hmm"}).json()
    assert body["source"] == "student_fallback" and body["label"] == "balance"


def test_empty_text_rejected():
    assert client([8.0, 0.0, 0.0]).post("/classify", json={"text": ""}).status_code == 422
