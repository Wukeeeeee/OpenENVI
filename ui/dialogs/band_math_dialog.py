"""OpenENVI Band Math Dialog.

Replicates ENVI's signature Band Math interface, allowing users to enter
custom algebraic expressions, select from previous presets, and map variables
to dataset bands with full geospatial metadata preservation.
"""

import re
from typing import Dict, List, Optional
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from core.algorithms.indices import _ALLOWED_FUNCS, evaluate_band_math
from core.i18n import tr
from core.models import RasterLayer

# Standard ENVI Band Math Presets
STANDARD_EXPRESSIONS = [
    ("(float(b4) - float(b3)) / (float(b4) + float(b3))", "Normalized Difference (NDVI: NIR-Red)"),
    ("(float(b2) - float(b4)) / (float(b2) + float(b4))", "Normalized Difference (NDWI: Green-NIR)"),
    ("float(b1) / (float(b2) + 1e-6)", "Simple Ratio (b1 / b2)"),
    ("(float(b1) > 0.0) * float(b1)", "Threshold Mask (b1 > 0)"),
    ("sqrt(float(b1) * float(b2))", "Geometric Mean"),
    ("0.299 * float(b1) + 0.587 * float(b2) + 0.114 * float(b3)", "Luminance Composite (Rec. 601)"),
]

RESERVED_WORDS = set(_ALLOWED_FUNCS.keys()) | {
    "and", "or", "not", "true", "false", "none", "pi", "e",
}


class BandMathDialog(QDialog):
    """Dialog for configuring and executing custom Band Math expressions."""

    result_generated = Signal(str, np.ndarray, object)  # layer_name, 2D array, parent_meta

    def __init__(self, layer: RasterLayer, reader, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self.setWindowTitle(f"{tr('dialog.band_math.title')} - {layer.name}")
        self.resize(560, 460)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # 1. Presets / Previous Expressions
        h_presets = QHBoxLayout()
        h_presets.addWidget(QLabel(tr("dialog.band_math.previous_expr") if tr("dialog.band_math.previous_expr") != "dialog.band_math.previous_expr" else "Presets / History:"))
        self.combo_presets = QComboBox()
        self.combo_presets.addItem("-- Select Expression Template --", "")
        for expr, label in STANDARD_EXPRESSIONS:
            self.combo_presets.addItem(f"{label}: {expr}", expr)
        self.combo_presets.currentIndexChanged.connect(self._on_preset_selected)
        h_presets.addWidget(self.combo_presets, stretch=1)
        layout.addLayout(h_presets)

        # 2. Expression Input Section
        lbl_expr = QLabel(tr("dialog.band_math.expr"))
        layout.addWidget(lbl_expr)

        self.edit_expr = QLineEdit("(float(b4) - float(b3)) / (float(b4) + float(b3))")
        self.edit_expr.setPlaceholderText(tr("dialog.band_math.expr_tip"))
        self.edit_expr.textChanged.connect(self._on_expr_changed)
        layout.addWidget(self.edit_expr)

        # Quick Insert Helper Buttons
        h_quick = QHBoxLayout()
        for label, snippet in [("float()", "float()"), ("b1", "b1"), ("b2", "b2"), ("+", " + "), ("-", " - "), ("*", " * "), ("/", " / "), (">", " > ")]:
            btn = QPushButton(label)
            btn.setMaximumWidth(60)
            btn.clicked.connect(lambda _, s=snippet: self._insert_snippet(s))
            h_quick.addWidget(btn)
        h_quick.addStretch()
        layout.addLayout(h_quick)

        # 3. Variable Mapping Group
        self.group_vars = QGroupBox(tr("dialog.band_math.assign"))
        self.layout_vars = QFormLayout(self.group_vars)
        layout.addWidget(self.group_vars, stretch=1)

        # Variable to QComboBox mappings
        self._var_combos: Dict[str, QComboBox] = {}

        # 4. Button Row
        btn_box = QHBoxLayout()
        self.btn_execute = QPushButton(tr("dialog.band_math.btn_compute"))
        self.btn_execute.setStyleSheet("background-color: #238636; color: white; font-weight: bold; padding: 6px;")
        self.btn_execute.clicked.connect(self._execute)
        btn_box.addWidget(self.btn_execute)

        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(self.btn_cancel)

        layout.addLayout(btn_box)

        # Populate initial variables
        self._on_expr_changed(self.edit_expr.text())

    def _insert_snippet(self, text: str):
        self.edit_expr.insert(text)
        self.edit_expr.setFocus()

    def _on_preset_selected(self, index: int):
        expr = self.combo_presets.currentData()
        if expr:
            self.edit_expr.setText(expr)

    def _on_expr_changed(self, text: str) -> None:
        """Parse variables from expression (excluding reserved function names)."""
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text)
        var_names = sorted(list({t for t in tokens if t.lower() not in RESERVED_WORDS}))

        # Clear existing mappings
        while self.layout_vars.count():
            item = self.layout_vars.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._var_combos.clear()

        # Build dropdown for each variable
        meta = self.layer.metadata
        band_labels = [b.display_name() for b in meta.band_details]
        if not band_labels:
            band_labels = [f"Band {i + 1}" for i in range(meta.bands)]

        for var in var_names:
            combo = QComboBox()
            combo.addItems(band_labels)

            # Auto-match variable to band:
            # 1. By index: b1 -> 0, b2 -> 1
            matched = False
            match = re.search(r"\d+", var)
            if match:
                idx = int(match.group(0)) - 1
                if 0 <= idx < len(band_labels):
                    combo.setCurrentIndex(idx)
                    matched = True

            # 2. By semantic name if not matched by number
            if not matched:
                vl = var.lower()
                for b_idx, b_info in enumerate(meta.band_details):
                    if b_info.name and vl in b_info.name.lower():
                        combo.setCurrentIndex(b_idx)
                        matched = True
                        break

            self.layout_vars.addRow(QLabel(f"<b>Variable '{var}':</b>"), combo)
            self._var_combos[var.lower()] = combo

    def _execute(self) -> None:
        """Evaluate expression and emit resulting layer with metadata."""
        expr = self.edit_expr.text().strip()
        if not expr:
            QMessageBox.warning(
                self,
                tr("band_math.err_invalid_expr_title"),
                tr("band_math.err_invalid_expr_msg"),
            )
            return

        try:
            # Read mapped bands
            band_vars = {}
            for var, combo in self._var_combos.items():
                band_idx = combo.currentIndex()
                band_data = self.reader.read_band(band_idx)
                band_vars[var] = band_data

            result = evaluate_band_math(expr, band_vars)
            layer_name = f"BandMath: {expr[:25]}"
            self.result_generated.emit(layer_name, result, self.layer.metadata)
            self.accept()
        except Exception as e:
            QMessageBox.critical(
                self,
                tr("band_math.err_compute_title"),
                f"{tr('band_math.err_compute_msg')}\n{e}",
            )
