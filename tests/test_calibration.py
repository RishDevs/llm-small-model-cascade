import numpy as np

from cascade.calibration import ece, fit_temperature, softmax


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]]))
    assert np.allclose(p.sum(1), 1.0)
    assert np.isfinite(p).all()


def test_temperature_keeps_argmax():
    logits = np.random.default_rng(0).normal(size=(50, 7))
    assert (softmax(logits, 0.3).argmax(1) == softmax(logits, 4.0).argmax(1)).all()


def test_fit_recovers_known_temperature():
    rng = np.random.default_rng(1)
    logits = rng.normal(scale=4.0, size=(20000, 10))
    true_t = 2.5
    p = softmax(logits, true_t)
    y = np.array([rng.choice(10, p=row) for row in p])
    assert abs(fit_temperature(logits, y) - true_t) < 0.15


def test_ece_zero_when_confidence_matches_accuracy():
    probs = np.tile([0.75, 0.25], (400, 1))
    y = np.array([0] * 300 + [1] * 100)
    assert ece(probs, y) < 1e-9


def test_ece_overconfident():
    probs = np.tile([0.99, 0.01], (100, 1))
    y = np.array([0] * 50 + [1] * 50)
    assert abs(ece(probs, y) - 0.49) < 1e-9
