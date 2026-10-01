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


def _safe_div(a, b):
    """Safely divide two arrays or array and scalar with zero and NaN protection."""
    a_arr = np.asanyarray(a, dtype=np.float32)
    b_arr = np.asanyarray(b, dtype=np.float32)
    out_shape = np.broadcast_shapes(a_arr.shape, b_arr.shape)
    out = np.zeros(out_shape, dtype=np.float32)
    mask = (b_arr != 0) & (~np.isnan(b_arr))
    return np.divide(a_arr, b_arr, out=out, where=mask)


# Allowed operators and functions for safe AST expression evaluation
_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: _safe_div,
    ast.Pow: lambda a, b: np.power(a, b),
    ast.BitAnd: lambda a, b: np.logical_and(a, b).astype(np.float32),
    ast.BitOr: lambda a, b: np.logical_or(a, b).astype(np.float32),
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
    ast.Invert: lambda a: np.logical_not(a).astype(np.float32),
}

_ALLOWED_COMPARE = {
    ast.Lt: lambda a, b: (a < b).astype(np.float32),
    ast.LtE: lambda a, b: (a <= b).astype(np.float32),
    ast.Gt: lambda a, b: (a > b).astype(np.float32),
    ast.GtE: lambda a, b: (a >= b).astype(np.float32),
    ast.Eq: lambda a, b: (a == b).astype(np.float32),
    ast.NotEq: lambda a, b: (a != b).astype(np.float32),
}

def _safe_min(*args):
    if not args:
        raise ValueError("min() requires at least 1 argument")
    if len(args) == 1:
        return np.nanmin(args[0])
    res = args[0]
    for a in args[1:]:
        res = np.minimum(res, a)
    return res


def _safe_max(*args):
    if not args:
        raise ValueError("max() requires at least 1 argument")
    if len(args) == 1:
        return np.nanmax(args[0])
    res = args[0]
    for a in args[1:]:
        res = np.maximum(res, a)
    return res


def _cast_int(
    x: np.ndarray,
    dtype: type,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> np.ndarray:
    """Cast to an integer type while keeping nodata as NaN.

    A direct float-to-int cast leaves NaN undefined and NumPy resolves it to
    INT_MIN instead of raising. np.clip does not help either: NaN is neither
    below nor above a bound, so it passes straight through and the cast then
    invents a value. Either way a nodata pixel silently becomes plausible data,
    and under byte/uint it becomes a black pixel indistinguishable from a real 0.
    """
    arr = np.asanyarray(x, dtype=np.float64)
    finite = np.isfinite(arr)
    if lo is not None:
        arr = np.where(finite, np.clip(arr, lo, hi), arr)
    out = np.zeros(arr.shape, dtype=np.float64)
    if finite.any():
        with np.errstate(invalid="ignore"):
            out[finite] = arr[finite].astype(dtype)
    return np.where(finite, out, np.nan)


_ALLOWED_FUNCS = {
    "exp": np.exp,
    "log": lambda x: np.log(np.maximum(x, 1e-7)),
    "ln": lambda x: np.log(np.maximum(x, 1e-7)),
    "alog": lambda x: np.log(np.maximum(x, 1e-7)),
    "log10": lambda x: np.log10(np.maximum(x, 1e-7)),
    "alog10": lambda x: np.log10(np.maximum(x, 1e-7)),
    "sqrt": lambda x: np.sqrt(np.maximum(x, 0.0)),
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": lambda x: np.arcsin(np.clip(x, -1.0, 1.0)),
    "acos": lambda x: np.arccos(np.clip(x, -1.0, 1.0)),
    "atan": np.arctan,
    "atan2": lambda y, x: np.arctan2(y, x),
    "min": _safe_min,
    "max": _safe_max,
    "mean": lambda x: np.nanmean(x),
    "std": lambda x: np.nanstd(x),
    "median": lambda x: np.nanmedian(x),
    "sum": lambda x: np.nansum(x),
    "clip": lambda x, a_min, a_max: np.clip(x, a_min, a_max),
    "where": lambda cond, x, y: np.where(cond != 0, x, y),
    "sign": np.sign,
    "round": np.round,
    "floor": np.floor,
    "ceil": np.ceil,
    # ENVI / IDL type casting & conversion functions
    "float": lambda x: np.asanyarray(x, dtype=np.float32),
    "double": lambda x: np.asanyarray(x, dtype=np.float64),
    "fix": lambda x: _cast_int(x, np.int32),
    "int": lambda x: _cast_int(x, np.int32),
    "long": lambda x: _cast_int(x, np.int32),
    "byte": lambda x: _cast_int(x, np.uint8, 0, 255),
    "uint": lambda x: _cast_int(x, np.uint16, 0, 65535),
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

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, variables)
        result = None
        current_left = left
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, variables)
            op_type = type(op)
            if op_type in _ALLOWED_COMPARE:
                cmp_res = _ALLOWED_COMPARE[op_type](current_left, right)
                result = cmp_res if result is None else (result * cmp_res)
                current_left = right
            else:
                raise ValueError(f"Unsupported comparison operator: {op_type.__name__}")
        return result

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            res = _eval_node(node.values[0], variables)
            for val in node.values[1:]:
                res = np.logical_and(res, _eval_node(val, variables)).astype(np.float32)
            return res
        elif isinstance(node.op, ast.Or):
            res = _eval_node(node.values[0], variables)
            for val in node.values[1:]:
                res = np.logical_or(res, _eval_node(val, variables)).astype(np.float32)
            return res

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

    if np.ndim(result) == 0:
        # If expression returned a scalar, broadcast to shape of first variable
        first_arr = next(iter(band_variables.values()))
        result = np.full_like(first_arr, result, dtype=np.float32)

    return result.astype(np.float32)
