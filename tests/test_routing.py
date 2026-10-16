import numpy as np

from cascade.routing import accuracy_curve, frontier, pick_threshold, route


def test_frontier_groups_ties():
    conf = np.array([0.9, 0.9, 0.5, 0.2])
    student = np.array([1, 2, 3, 4])
    teacher = np.array([1, 0, 3, 0])
    thr, cov, dis = frontier(conf, student, teacher)
    assert thr.tolist() == [0.9, 0.5, 0.2]
    assert cov.tolist() == [0.5, 0.75, 1.0]
    assert dis.tolist() == [0.25, 0.25, 0.5]


def test_pick_threshold_respects_budget():
    conf = np.array([0.99, 0.95, 0.9, 0.6, 0.4])
    student = np.array([0, 1, 2, 3, 4])
    teacher = np.array([0, 1, 2, 9, 9])
    assert pick_threshold(conf, student, teacher, 0.0) == 0.9
    assert pick_threshold(conf, student, teacher, 0.2) == 0.6


def test_pick_threshold_escalates_everything_when_nothing_fits():
    conf = np.array([0.9, 0.8])
    assert pick_threshold(conf, np.array([1, 1]), np.array([0, 0]), 0.0) == float("inf")


def test_route_uses_teacher_below_threshold():
    final, accepted = route(np.array([5, 6, 7]), np.array([0.95, 0.5, 0.8]), np.array([1, 2, 3]), 0.8)
    assert final.tolist() == [5, 2, 7]
    assert accepted.tolist() == [True, False, True]


def test_accuracy_curve_endpoints():
    conf = np.array([0.9, 0.7, 0.3])
    student = np.array([0, 1, 9])
    teacher = np.array([0, 9, 2])
    gold = np.array([0, 1, 2])
    coverage, acc = accuracy_curve(conf, student, teacher, gold)
    assert coverage[0] == 0 and coverage[-1] == 1
    assert np.isclose(acc[0], 2 / 3)
    assert np.isclose(acc[2], 1.0)
    assert np.isclose(acc[-1], 2 / 3)
