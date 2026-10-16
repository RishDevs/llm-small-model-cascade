import numpy as np


def frontier(conf, student, teacher):
    order = np.argsort(-conf, kind="stable")
    ranked = conf[order]
    misses = np.cumsum(student[order] != teacher[order])
    ends = np.flatnonzero(np.r_[ranked[1:] < ranked[:-1], True])
    n = len(conf)
    return ranked[ends], (ends + 1) / n, misses[ends] / n


def pick_threshold(conf, student, teacher, max_disagreement):
    thresholds, coverage, disagreement = frontier(conf, student, teacher)
    ok = np.flatnonzero(disagreement <= max_disagreement)
    if len(ok) == 0:
        return float("inf")
    return float(thresholds[ok[-1]])


def route(student, conf, teacher, threshold):
    accept = conf >= threshold
    return np.where(accept, student, teacher), accept


def accuracy_curve(conf, student, teacher, gold):
    order = np.argsort(-conf, kind="stable")
    ranked = conf[order]
    s_hit = np.r_[0, np.cumsum(student[order] == gold[order])]
    t_hit = np.r_[0, np.cumsum(teacher[order] == gold[order])]
    ends = np.flatnonzero(np.r_[ranked[1:] < ranked[:-1], True]) + 1
    ks = np.r_[0, ends]
    n = len(conf)
    acc = (s_hit[ks] + t_hit[-1] - t_hit[ks]) / n
    return ks / n, acc
