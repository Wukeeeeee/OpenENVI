"""OpenENVI Spectral Indices and Band Math Engine.

Provides standard remote sensing vegetation, water, and soil indices,
along with a safe mathematical expression evaluator for ENVI Band Math.
"""

import ast
from typing import Dict, Optional
import numpy as np


def calculate_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Calculate Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)."""
    denom = nir.astype(np.float32) + red.astype(np.float32) + eps
    numer = nir.astype(np.float32) - red.astype(np.float32)
    ndvi = numer / denom
    return np.clip(ndvi, -1.0, 1.0)


def calculate_ndwi(green: np.ndarray, nir: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Calculate Normalized Difference Water Index: (Green - NIR) / (Green + NIR)."""
    denom = green.astype(np.float32) + nir.astype(np.float32) + eps
    numer = green.astype(np.float32) - nir.astype(np.float32)
    ndwi = numer / denom
    return np.clip(ndwi, -1.0, 1.0)


def calculate_evi(
    nir: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    g: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    l: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Calculate Enhanced Vegetation Index: G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)."""
    numer = nir.astype(np.float32) - red.astype(np.float32)
    denom = (
        nir.astype(np.float32)
        + c1 * red.astype(np.float32)
        - c2 * blue.astype(np.float32)
        + l
        + eps
    )
    evi = g * (numer / denom)
    return np.clip(evi, -1.0, 1.0)


def calculate_savi(
    nir: np.ndarray,
    red: np.ndarray,
    l: float = 0.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Calculate Soil-Adjusted Vegetation Index: ((NIR - Red) / (NIR + Red + L)) * (1 + L)."""
    numer = nir.astype(np.float32) - red.astype(np.float32)
    denom = nir.astype(np.float32) + red.astype(np.float32) + l + eps
    savi = (numer / denom) * (1.0 + l)
    return np.clip(savi, -1.0, 1.0)


def calculate_nbr(nir: np.ndarray, swir2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Calculate Normalized Burn Ratio: (NIR - SWIR2) / (NIR + SWIR2)."""
    denom = nir.astype(np.float32) + swir2.astype(np.float32) + eps
    numer = nir.astype(np.float32) - swir2.astype(np.float32)
    nbr = numer / denom
    return np.clip(nbr, -1.0, 1.0)


# Allowed operators and functions for safe AST expression evaluation
_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=np.float32), where=b != 0),
    ast.Pow: lambda a, b: np.power(a, b),
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}

_ALLOWED_FUNCS = {
    "exp": np.exp,
    "log": lambda x: np.log(np.maximum(x, 1e-6)),
    "sqrt": lambda x: np.sqrt(np.maximum(x, 0.0)),
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "min": np.minimum,
    "max": np.maximum,
}


def _eval_node(node: ast.AST, variables: Dict[str, np.ndarray]) -> np.ndarray:
    """Recursively evaluate an AST expression tree with variable mapping."""
    if isinstance(node, ast.Constant):
        return float(node.value)

    if isinstance(node, ast.Name):
        var_name = node.id.lower()
        if var_name in variables:
            return variables[var_name]
        raise ValueError(f"Unknown variable in expression: {node.id}")

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        op_type = type(node.op)
        if op_type in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, variables)
        op_type = type(node.op)
        if op_type in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported")
        func_name = node.func.id.lower()
        if func_name in _ALLOWED_FUNCS:
            args = [_eval_node(arg, variables) for arg in node.args]
            return _ALLOWED_FUNCS[func_name](*args)
        raise ValueError(f"Unsupported function in Band Math: {node.func.id}")

    raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")


def evaluate_band_math(
    expression: str,
    band_variables: Dict[str, np.ndarray],
) -> np.ndarray:
    """Safely evaluate an algebraic Band Math formula (e.g. '(b4 - b3) / (b4 + b3)').

    Args:
        expression: Mathematical expression string.
        band_variables: Dictionary mapping variable names ('b1', 'b2', 'b3', etc.) to 2D numpy arrays.

    Returns:
        2D numpy array containing evaluated result.
    """
    # Normalize variable names in lookup table to lowercase
    norm_vars = {k.lower(): v.astype(np.float32) for k, v in band_variables.items()}

    # Parse expression into AST
    tree = ast.parse(expression.strip(), mode="eval")
    result = _eval_node(tree.body, norm_vars)

    if isinstance(result, (int, float)):
        # If expression returned a scalar, broadcast to shape of first variable
        first_arr = next(iter(band_variables.values()))
        result = np.full_like(first_arr, result, dtype=np.float32)

    return result.astype(np.float32)
