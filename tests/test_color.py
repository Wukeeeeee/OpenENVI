"""Unit tests for color space transformation."""

import numpy as np
import pytest
from core.algorithms.color import hsv_to_rgb, rgb_to_grayscale, rgb_to_hsv


def test_rgb_to_hsv_and_back():
    r = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    g = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    b = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    hsv, names = rgb_to_hsv(r, g, b, normalize_inputs=False)
    assert hsv.shape == (2, 2, 3)
    assert names == ["Hue", "Saturation", "Value"]

    # Red pixel [0, 0] -> Hue=0, Sat=1, Val=1
    assert np.isclose(hsv[0, 0, 0], 0.0)
    assert np.isclose(hsv[0, 0, 1], 1.0)
    assert np.isclose(hsv[0, 0, 2], 1.0)

    # Convert back to RGB
    rgb_back, rgb_names = hsv_to_rgb(hsv[..., 0], hsv[..., 1], hsv[..., 2])
    assert rgb_back.shape == (2, 2, 3)
    assert rgb_names == ["Red", "Green", "Blue"]
    assert np.allclose(rgb_back[..., 0], r)
    assert np.allclose(rgb_back[..., 1], g)
    assert np.allclose(rgb_back[..., 2], b)


def test_rgb_to_grayscale():
    r = np.ones((5, 5), dtype=np.float32)
    g = np.ones((5, 5), dtype=np.float32)
    b = np.ones((5, 5), dtype=np.float32)

    gray, names = rgb_to_grayscale(r, g, b)
    assert gray.shape == (5, 5, 1)
    assert np.allclose(gray, 1.0)
