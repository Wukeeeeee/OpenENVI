"""OpenENVI Data Manager Dock (Available Bands List).

Faithfully replicates ENVI's signature Available Bands List / Data Manager,
allowing users to inspect dataset bands, configure RGB/Grayscale assignments,
and load them into the main viewport.
"""

import os
from typing import List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDockWidget,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.events import event_bus
from core.models import BandInfo, RasterLayer


class DataManagerDock(QDockWidget):
    """Dock widget hosting the Available Bands / Data Manager panel."""

    load_grayscale_requested = Signal(str, int)  # layer_id, band_idx
    load_rgb_requested = Signal(str, int, int, int)  # layer_id, r, g, b
    close_file_requested = Signal(str)  # layer_id

    def __init__(self, parent=None):
        super().__init__("Data Manager", parent)
        self.setObjectName("DataManagerDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Available Datasets & Bands Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree, stretch=1)

        # Mode Selection: Gray Scale vs RGB Color
        mode_box = QHBoxLayout()
        self.rb_gray = QRadioButton("Gray Scale")
        self.rb_rgb = QRadioButton("RGB Color")
        self.rb_gray.setChecked(True)

        self.btn_group = QButtonGroup(self)
        self.btn_group.addButton(self.rb_gray)
        self.btn_group.addButton(self.rb_rgb)
        self.btn_group.buttonToggled.connect(self._on_mode_toggled)

        mode_box.addWidget(self.rb_gray)
        mode_box.addWidget(self.rb_rgb)
        mode_box.addStretch()
        layout.addLayout(mode_box)

        # RGB Assignment Box
        self.rgb_box = QGroupBox("RGB Assignment")
        rgb_layout = QVBoxLayout(self.rgb_box)
        rgb_layout.setContentsMargins(6, 6, 6, 6)
        rgb_layout.setSpacing(4)

        # Hint instruction
        self.lbl_rgb_hint = QLabel("Click R/G/B slot to switch target, then click a band to assign")
        self.lbl_rgb_hint.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.lbl_rgb_hint.setWordWrap(True)
        rgb_layout.addWidget(self.lbl_rgb_hint)

        # 3 Channel Slot Buttons
        slots_layout = QHBoxLayout()
        slots_layout.setSpacing(4)

        self.btn_slot_r = QPushButton("R: --")
        self.btn_slot_r.setCursor(Qt.PointingHandCursor)
        self.btn_slot_r.clicked.connect(lambda: self._set_active_rgb_slot(0))
        slots_layout.addWidget(self.btn_slot_r)

        self.btn_slot_g = QPushButton("G: --")
        self.btn_slot_g.setCursor(Qt.PointingHandCursor)
        self.btn_slot_g.clicked.connect(lambda: self._set_active_rgb_slot(1))
        slots_layout.addWidget(self.btn_slot_g)

        self.btn_slot_b = QPushButton("B: --")
        self.btn_slot_b.setCursor(Qt.PointingHandCursor)
        self.btn_slot_b.clicked.connect(lambda: self._set_active_rgb_slot(2))
        slots_layout.addWidget(self.btn_slot_b)

        rgb_layout.addLayout(slots_layout)
        layout.addWidget(self.rgb_box)
        self.rgb_box.setVisible(False)

        # Load & Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Band")
        self.btn_load.setStyleSheet("background-color: #238636; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_load.clicked.connect(self._on_load_clicked)
        btn_layout.addWidget(self.btn_load)

        self.btn_close_file = QPushButton("Close File")
        self.btn_close_file.clicked.connect(self._on_close_file_clicked)
        btn_layout.addWidget(self.btn_close_file)

        layout.addLayout(btn_layout)
        self.setWidget(container)

        # Internal state for RGB assignment
        self._selected_layer_id: Optional[str] = None
        self._selected_band_idx: Optional[int] = None
        self._active_rgb_slot: int = 0  # 0: R, 1: G, 2: B
        self._rgb_bands: List[Optional[int]] = [None, None, None]
        self._rgb_band_names: List[str] = ["", "", ""]

        self._update_rgb_slot_ui()

        # Retranslate on creation and on language change
        self.retranslate_ui()
        from core.i18n import i18n
        i18n.language_changed.connect(lambda _: self.retranslate_ui())

    def retranslate_ui(self) -> None:
        """Update texts based on active language."""
        from core.i18n import tr
        self.setWindowTitle(tr("dock.data_manager"))
        self.rb_gray.setText(tr("data_manager.rb_gray"))
        self.rb_rgb.setText(tr("data_manager.rb_rgb"))
        self.rgb_box.setTitle(tr("data_manager.rgb_group"))
        self.lbl_rgb_hint.setText(tr("data_manager.hint_rgb_target"))
        self.btn_load.setText(
            tr("data_manager.btn_load_rgb") if self.rb_rgb.isChecked() else tr("data_manager.btn_load_band")
        )
        self.btn_close_file.setText(tr("data_manager.btn_close_file"))
        self._update_rgb_slot_ui()

    def _set_active_rgb_slot(self, slot_idx: int) -> None:
        """Set the active slot (0=R, 1=G, 2=B) that receives the next clicked band."""
        self._active_rgb_slot = slot_idx % 3
        self._update_rgb_slot_ui()

    def _update_rgb_slot_ui(self) -> None:
        """Refresh buttons text, active highlight borders, and tooltips."""
        slots = [
            (self.btn_slot_r, 0, "R", "#ff7b72", "#3b1b1a"),
            (self.btn_slot_g, 1, "G", "#7ee787", "#132b1a"),
            (self.btn_slot_b, 2, "B", "#79c0ff", "#13253b"),
        ]

        for btn, idx, prefix, border_col, bg_col in slots:
            band_idx = self._rgb_bands[idx]
            band_desc = self._rgb_band_names[idx]
            is_active = (self._active_rgb_slot == idx)

            if band_idx is not None:
                btn.setText(f"{prefix}: Band {band_idx + 1}")
                btn.setToolTip(f"{prefix} Channel -> {band_desc if band_desc else f'Band {band_idx + 1}'}")
            else:
                btn.setText(f"{prefix}: --")
                btn.setToolTip(f"{prefix} Channel (Unassigned)")

            if is_active:
                btn.setStyleSheet(
                    f"QPushButton {{ border: 2px solid {border_col}; background-color: {bg_col}; "
                    f"color: {border_col}; font-weight: bold; border-radius: 3px; padding: 4px; }}"
                )
            else:
                btn.setStyleSheet(
                    f"QPushButton {{ border: 1px solid #444c56; background-color: #21262d; "
                    f"color: #adbac7; border-radius: 3px; padding: 4px; }} "
                    f"QPushButton:hover {{ border-color: {border_col}; }}"
                )

    def add_dataset(self, layer: RasterLayer) -> None:
        """Add a dataset and its bands to the Data Manager tree."""
        file_item = QTreeWidgetItem(self.tree, [f"File: {layer.name}"])
        file_item.setData(0, Qt.UserRole, ("file", layer.layer_id, -1))
        file_item.setExpanded(True)

        bands = layer.metadata.band_details
        if bands:
            for b in bands:
                band_item = QTreeWidgetItem(file_item, [b.display_name()])
                band_item.setData(0, Qt.UserRole, ("band", layer.layer_id, b.index))
        else:
            for i in range(layer.metadata.bands):
                band_item = QTreeWidgetItem(file_item, [f"Band {i + 1}"])
                band_item.setData(0, Qt.UserRole, ("band", layer.layer_id, i))

        self._selected_layer_id = layer.layer_id

        # Intelligent default RGB band pre-selection
        r, g, b = None, None, None
        if layer.metadata.default_bands and len(layer.metadata.default_bands) == 3:
            r, g, b = layer.metadata.default_bands
        elif layer.metadata.bands >= 4:
            # Landsat 8/9 standard natural color: Red (Band 4), Green (Band 3), Blue (Band 2)
            r, g, b = (3, 2, 1)
        elif layer.metadata.bands == 3:
            r, g, b = (0, 1, 2)

        if r is not None and g is not None and b is not None:
            self._rgb_bands = [r, g, b]
            if bands:
                self._rgb_band_names = [
                    bands[r].display_name() if r < len(bands) else f"Band {r + 1}",
                    bands[g].display_name() if g < len(bands) else f"Band {g + 1}",
                    bands[b].display_name() if b < len(bands) else f"Band {b + 1}",
                ]
            else:
                self._rgb_band_names = [f"Band {r + 1}", f"Band {g + 1}", f"Band {b + 1}"]
            self._update_rgb_slot_ui()

    def remove_dataset(self, layer_id: str) -> None:
        """Remove a dataset item and its bands from the Data Manager tree."""
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            child = root.child(i)
            data = child.data(0, Qt.UserRole)
            if data and data[1] == layer_id:
                root.removeChild(child)
                break
        if self._selected_layer_id == layer_id:
            self._selected_layer_id = None
            self._selected_band_idx = None
            self._rgb_bands = [None, None, None]
            self._rgb_band_names = ["", "", ""]
            self._update_rgb_slot_ui()

    def _on_mode_toggled(self) -> None:
        """Toggle between Grayscale and RGB mode."""
        from core.i18n import tr
        is_rgb = self.rb_rgb.isChecked()
        self.rgb_box.setVisible(is_rgb)
        self.btn_load.setText(
            tr("data_manager.btn_load_rgb") if is_rgb else tr("data_manager.btn_load_band")
        )
        self._set_active_rgb_slot(0)

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle single-click on tree item."""
        data = item.data(0, Qt.UserRole)
        if not data:
            return

        item_type, layer_id, band_idx = data
        self._selected_layer_id = layer_id

        if item_type != "band":
            return

        self._selected_band_idx = band_idx

        if self.rb_rgb.isChecked():
            # Assign to current active slot and advance
            self._rgb_bands[self._active_rgb_slot] = band_idx
            self._rgb_band_names[self._active_rgb_slot] = item.text(0)
            self._set_active_rgb_slot((self._active_rgb_slot + 1) % 3)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle double-click to directly load band."""
        data = item.data(0, Qt.UserRole)
        if not data:
            return

        item_type, layer_id, band_idx = data
        if item_type != "band":
            return

        self._selected_layer_id = layer_id
        self._selected_band_idx = band_idx

        if self.rb_gray.isChecked():
            self._on_load_clicked()
        else:
            # In RGB mode, double-click assigns to active slot and advances
            self._rgb_bands[self._active_rgb_slot] = band_idx
            self._rgb_band_names[self._active_rgb_slot] = item.text(0)
            self._set_active_rgb_slot((self._active_rgb_slot + 1) % 3)

    def _on_load_clicked(self) -> None:
        """Trigger loading of selected band(s)."""
        from core.i18n import tr
        if not self._selected_layer_id:
            event_bus.status_message.emit(tr("data_manager.msg_select_band"), 3000)
            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                QMessageBox.warning(self, tr("dock.data_manager"), tr("data_manager.msg_select_band"))
            return

        if self.rb_gray.isChecked():
            if self._selected_band_idx is not None:
                self.load_grayscale_requested.emit(self._selected_layer_id, self._selected_band_idx)
                event_bus.status_message.emit(
                    f"{tr('data_manager.msg_loading_gray')} {self._selected_band_idx + 1}", 2000
                )
            else:
                event_bus.status_message.emit(tr("data_manager.msg_select_band"), 3000)
                if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                    QMessageBox.warning(self, tr("dock.data_manager"), tr("data_manager.msg_select_band"))
        else:
            r, g, b = self._rgb_bands
            if r is not None and g is not None and b is not None:
                self.load_rgb_requested.emit(self._selected_layer_id, r, g, b)
                event_bus.status_message.emit(
                    f"{tr('data_manager.msg_loading_rgb')} (R:{r+1}, G:{g+1}, B:{b+1})", 2000
                )
            else:
                # Find first missing slot and activate it
                for idx, slot_val in enumerate(self._rgb_bands):
                    if slot_val is None:
                        self._set_active_rgb_slot(idx)
                        break
                event_bus.status_message.emit(tr("data_manager.msg_assign_rgb"), 3000)
                if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                    QMessageBox.warning(self, tr("data_manager.rgb_group"), tr("data_manager.msg_assign_rgb"))

    def _on_close_file_clicked(self) -> None:
        """Close selected dataset in tree and notify MainWindow to release layer."""
        from core.i18n import tr
        current = self.tree.currentItem()
        if not current:
            return
        parent = current.parent()
        target = parent if parent is not None else current
        data = target.data(0, Qt.UserRole)
        if data and len(data) >= 2:
            layer_id = data[1]
            self.close_file_requested.emit(layer_id)
            event_bus.status_message.emit(tr("data_manager.msg_file_closed"), 2000)
