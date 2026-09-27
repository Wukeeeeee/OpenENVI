"""Unit tests for classification accuracy assessment."""

import numpy as np
import pytest
from core.algorithms.accuracy import compute_confusion_matrix


def test_confusion_matrix_perfect():
    c = np.array([[0, 1], [2, 0]], dtype=np.int32)
    r = np.array([[0, 1], [2, 0]], dtype=np.int32)

    res = compute_confusion_matrix(c, r)
    assert res["overall_accuracy"] == 100.0
    assert np.isclose(res["kappa"], 1.0)
    assert res["matrix"].shape == (3, 3)
    assert np.all(res["matrix"] == np.diag(np.diag(res["matrix"])))


def test_confusion_matrix_partial():
    c = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    r = np.array([0, 0, 1, 0, 2, 2], dtype=np.int32)

    res = compute_confusion_matrix(c, r)
    assert np.isclose(res["overall_accuracy"], 5 / 6 * 100)
    assert np.isclose(res["kappa"], 0.75)
    assert "Overall Accuracy" in res["report"]
    assert "Kappa Coefficient" in res["report"]
