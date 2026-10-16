import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(probs, y):
    picked = probs[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, None))))


def brier(probs, y):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def fit_temperature(logits, y):
    res = minimize_scalar(lambda log_t: nll(softmax(logits, np.exp(log_t)), y), bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def reliability(probs, y, bins=15):
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            rows.append((float(conf[in_bin].mean()), float(correct[in_bin].mean()), int(in_bin.sum())))
    return rows


def ece(probs, y, bins=15):
    n = len(y)
    return float(sum(count / n * abs(acc - conf) for conf, acc, count in reliability(probs, y, bins)))


def summary(probs, y, bins=15):
    return {"ece": round(ece(probs, y, bins), 4), "nll": round(nll(probs, y), 4), "brier": round(brier(probs, y), 4)}
