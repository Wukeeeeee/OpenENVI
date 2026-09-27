"""OpenENVI Classification Accuracy Assessment & Confusion Matrix Engine.

Calculates:
- Confusion / Error Matrix (k x k)
- Overall Accuracy (OA, %)
- Kappa Coefficient (Cohen's Kappa)
- Producer's Accuracy (PA, %) & Omission Error (OE, %)
- User's Accuracy (UA, %) & Commission Error (CE, %)
- Generates formatted ENVI-style accuracy text and CSV reports.
"""

from typing import Any, Dict, List, Optional
import numpy as np


def compute_confusion_matrix(
    classified: np.ndarray,
    reference: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    nodata_val: Optional[int] = None,
) -> Dict[str, Any]:
    """Calculate confusion matrix and accuracy metrics between classified map and reference.

    Args:
        classified: 2D integer array of predicted class indices.
        reference: 2D integer array of ground truth reference classes.
        class_names: Optional mapping from class index to human-readable name.
        nodata_val: Optional nodata value to ignore.

    Returns:
        Dictionary containing:
            - 'matrix': 2D integer ndarray (rows=Reference, cols=Classified)
            - 'classes': List of unique class labels
            - 'overall_accuracy': Percentage in [0, 100]
            - 'kappa': Cohen's kappa coefficient [-1, 1]
            - 'producers_accuracy': Dict[class_id, float %]
            - 'users_accuracy': Dict[class_id, float %]
            - 'report': Formatted ASCII text report
    """
    if classified.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: classified {classified.shape} vs reference {reference.shape}"
        )

    mask = np.isfinite(classified) & np.isfinite(reference)
    if nodata_val is not None:
        mask &= (classified != nodata_val) & (reference != nodata_val)

    c = classified[mask].astype(np.int64)
    r = reference[mask].astype(np.int64)

    if len(c) == 0:
        raise ValueError("No valid overlapping pixels between classified map and reference.")

    unique_classes = np.sort(np.unique(np.concatenate([c, r])))
    k = len(unique_classes)
    cls_map = {cls_val: idx for idx, cls_val in enumerate(unique_classes)}

    matrix = np.zeros((k, k), dtype=np.int64)
    for ref_val, pred_val in zip(r, c):
        matrix[cls_map[ref_val], cls_map[pred_val]] += 1

    total_pixels = int(np.sum(matrix))
    correct_pixels = int(np.trace(matrix))
    oa = (correct_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    # Kappa calculation
    po = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    row_sums = np.sum(matrix, axis=1)  # Reference totals
    col_sums = np.sum(matrix, axis=0)  # Classified totals
    pe = float(np.sum(row_sums * col_sums)) / (total_pixels ** 2) if total_pixels > 0 else 0.0
    kappa = float((po - pe) / (1.0 - pe)) if (1.0 - pe) != 0 else 1.0

    # Per-class Producer's & User's accuracy
    pa_dict = {}
    ua_dict = {}
    for idx, cls_val in enumerate(unique_classes):
        diag = matrix[idx, idx]
        r_sum = row_sums[idx]
        c_sum = col_sums[idx]

        pa_dict[int(cls_val)] = (diag / r_sum) * 100.0 if r_sum > 0 else 0.0
        ua_dict[int(cls_val)] = (diag / c_sum) * 100.0 if c_sum > 0 else 0.0

    # Generate ASCII report
    lines = [
        "===========================================================",
        "        OpenENVI Classification Accuracy Assessment        ",
        "===========================================================",
        f"Overall Accuracy: {oa:.2f}% ({correct_pixels}/{total_pixels} pixels)",
        f"Kappa Coefficient: {kappa:.4f}",
        "",
        "--------------------- Confusion Matrix ---------------------",
    ]

    header_cols = ["Class"] + [f"C_{cls}" for cls in unique_classes] + ["Total", "PA (%)"]
    lines.append(f"{'Ref \\ Pred':<12}" + "".join(f"{col:>10}" for col in header_cols[1:]))

    for idx, cls_val in enumerate(unique_classes):
        name = class_names.get(cls_val, f"Class {cls_val}") if class_names else f"Class {cls_val}"
        row_vals = "".join(f"{matrix[idx, j]:>10}" for j in range(k))
        lines.append(f"{name:<12}{row_vals}{row_sums[idx]:>10}{pa_dict[cls_val]:>9.2f}%")

    lines.append("-----------------------------------------------------------")
    col_totals = "".join(f"{col_sums[j]:>10}" for j in range(k))
    lines.append(f"{'Total':<12}{col_totals}{total_pixels:>10}")
    ua_vals = "".join(f"{ua_dict[cls_val]:>9.2f}%" for cls_val in unique_classes)
    lines.append(f"{'UA (%)':<12}{ua_vals}")
    lines.append("===========================================================")

    report_text = "\n".join(lines)

    return {
        "matrix": matrix,
        "classes": unique_classes.tolist(),
        "overall_accuracy": oa,
        "kappa": kappa,
        "producers_accuracy": pa_dict,
        "users_accuracy": ua_dict,
        "report": report_text,
    }
