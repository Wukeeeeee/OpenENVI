"""OpenENVI Band Math Dialog.

Replicates ENVI's signature Band Math interface, allowing users to enter
custom algebraic expressions and map variables to dataset bands.
"""

import re
from typing import Dict, Optional
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

from core.algorithms.indices import evaluate_band_math
from core.i18n import tr
from core.models import RasterLayer


class BandMathDialog(QDialog):
    """Dialog for configuring and executing custom Band Math expressions."""

    result_generated = Signal(str, np.ndarray)  # layer_name, 2D array

    def __init__(self, layer: RasterLayer, reader, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self.setWindowTitle(tr("dialog.band_math.title"))
        self.resize(500, 380)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Expression Input Section
        lbl_expr = QLabel(tr("dialog.band_math.expr"))
        layout.addWidget(lbl_expr)

        self.edit_expr = QLineEdit("(b4 - b3) / (b4 + b3)")
        self.edit_expr.setPlaceholderText(tr("dialog.band_math.expr_tip"))
        self.edit_expr.textChanged.connect(self._on_expr_changed)
        layout.addWidget(self.edit_expr)

        # Variable Mapping Group
        self.group_vars = QGroupBox(tr("dialog.band_math.assign"))
        self.layout_vars = QFormLayout(self.group_vars)
        layout.addWidget(self.group_vars, stretch=1)

        # Variable to QComboBox mappings
        self._var_combos: Dict[str, QComboBox] = {}

        # Button Row
        btn_box = QHBoxLayout()
        self.btn_execute = QPushButton(tr("dialog.band_math.btn_compute"))
        self.btn_execute.setStyleSheet("background-color: #238636; color: white; font-weight: bold;")
        self.btn_execute.clicked.connect(self._execute)
        btn_box.addWidget(self.btn_execute)

        self.btn_cancel = QPushButton(tr("dialog.btn_cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(self.btn_cancel)

        layout.addLayout(btn_box)

        # Populate initial variables
        self._on_expr_changed(self.edit_expr.text())

    def _on_expr_changed(self, text: str) -> None:
        """Parse variables like b1, b2, float... from expression."""
        # Find all b1, b2, ... or B1, B2 ... variables
        var_names = sorted(list(set(re.findall(r"\b[bB]\d+\b", text))))

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
            # Default auto-select based on index
            match = re.search(r"\d+", var)
            if match:
                idx = int(match.group(0)) - 1
                if 0 <= idx < len(band_labels):
                    combo.setCurrentIndex(idx)
            self.layout_vars.addRow(QLabel(f"Variable '{var}':"), combo)
            self._var_combos[var.lower()] = combo

    def _execute(self) -> None:
        """Evaluate expression and emit resulting layer."""
        expr = self.edit_expr.text().strip()
        if not expr:
            QMessageBox.warning(self, "Invalid Expression", "Please enter a Band Math expression.")
            return

        try:
            # Read mapped bands
            band_vars = {}
            for var, combo in self._var_combos.items():
                band_idx = combo.currentIndex()
                band_data = self.reader.read_band(band_idx)
                band_vars[var] = band_data

            result = evaluate_band_math(expr, band_vars)
            layer_name = f"BandMath: {expr[:20]}"
            self.result_generated.emit(layer_name, result)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Computation Error", f"Failed to compute Band Math:\n{e}")
