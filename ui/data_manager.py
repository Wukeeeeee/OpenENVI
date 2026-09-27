"""OpenENVI Data Manager Dock (Available Bands List).

Faithfully replicates ENVI's signature Available Bands List / Data Manager,
allowing users to inspect dataset bands, configure RGB/Grayscale assignments,
and load them into the main viewport.
"""

from typing import List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDockWidget,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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
        self.tree.itemClicked.connect(self._on_band_clicked)
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
        rgb_layout = QGridLayout(self.rgb_box)
        rgb_layout.setContentsMargins(6, 6, 6, 6)
        rgb_layout.setSpacing(4)

        self.lbl_r = QLabel("R: --")
        self.lbl_r.setStyleSheet("color: #ff7b72; font-weight: bold;")
        self.lbl_g = QLabel("G: --")
        self.lbl_g.setStyleSheet("color: #7ee787; font-weight: bold;")
        self.lbl_b = QLabel("B: --")
        self.lbl_b.setStyleSheet("color: #79c0ff; font-weight: bold;")

        rgb_layout.addWidget(self.lbl_r, 0, 0)
        rgb_layout.addWidget(self.lbl_g, 0, 1)
        rgb_layout.addWidget(self.lbl_b, 0, 2)
        layout.addWidget(self.rgb_box)
        self.rgb_box.setVisible(False)

        # Load Buttons
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Band")
        self.btn_load.setStyleSheet("background-color: #238636; color: #ffffff; font-weight: bold;")
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
        self._rgb_bands = [None, None, None]  # [R, G, B]
        self._next_rgb_slot = 0  # 0: R, 1: G, 2: B

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
        self.btn_load.setText(
            tr("data_manager.btn_load_rgb") if self.rb_rgb.isChecked() else tr("data_manager.btn_load_band")
        )
        self.btn_close_file.setText(tr("data_manager.btn_close_file"))

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

    def remove_dataset(self, layer_id: str) -> None:
        """Remove a dataset item and its bands from the Data Manager tree."""
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            child = root.child(i)
            data = child.data(0, Qt.UserRole)
            if data and data[1] == layer_id:
                root.removeChild(child)
                break

    def _on_mode_toggled(self) -> None:
        """Toggle between Grayscale and RGB mode."""
        is_rgb = self.rb_rgb.isChecked()
        self.rgb_box.setVisible(is_rgb)
        self.btn_load.setText("Load RGB" if is_rgb else "Load Band")
        self._next_rgb_slot = 0

    def _on_band_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle band click to assign grayscale or RGB."""
        data = item.data(0, Qt.UserRole)
        if not data:
            return

        item_type, layer_id, band_idx = data
        if item_type != "band":
            return

        self._selected_layer_id = layer_id
        self._selected_band_idx = band_idx

        if self.rb_rgb.isChecked():
            # Assign cyclically to R, G, B
            self._rgb_bands[self._next_rgb_slot] = band_idx
            if self._next_rgb_slot == 0:
                self.lbl_r.setText(f"R: Band {band_idx + 1}")
                self._next_rgb_slot = 1
            elif self._next_rgb_slot == 1:
                self.lbl_g.setText(f"G: Band {band_idx + 1}")
                self._next_rgb_slot = 2
            else:
                self.lbl_b.setText(f"B: Band {band_idx + 1}")
                self._next_rgb_slot = 0

    def _on_load_clicked(self) -> None:
        """Trigger loading of selected band(s)."""
        if not self._selected_layer_id:
            event_bus.status_message.emit("Please select a band first", 3000)
            return

        if self.rb_gray.isChecked():
            if self._selected_band_idx is not None:
                self.load_grayscale_requested.emit(self._selected_layer_id, self._selected_band_idx)
                event_bus.status_message.emit(
                    f"Loading Grayscale: Band {self._selected_band_idx + 1}", 2000
                )
        else:
            r, g, b = self._rgb_bands
            if r is not None and g is not None and b is not None:
                self.load_rgb_requested.emit(self._selected_layer_id, r, g, b)
                event_bus.status_message.emit(
                    f"Loading RGB: (R:{r+1}, G:{g+1}, B:{b+1})", 2000
                )
            else:
                event_bus.status_message.emit("Please assign all R, G, and B bands", 3000)

    def _on_close_file_clicked(self) -> None:
        """Close selected dataset in tree."""
        current = self.tree.currentItem()
        if not current:
            return
        parent = current.parent()
        target = parent if parent is not None else current
        index = self.tree.indexOfTopLevelItem(target)
        if index >= 0:
            self.tree.takeTopLevelItem(index)
            event_bus.status_message.emit("Dataset closed", 2000)
