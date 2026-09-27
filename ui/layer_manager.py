"""OpenENVI Layer Manager Dock.

Provides layer organization, visibility toggling, layer ordering, and active layer selection.
"""

from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.events import event_bus
from core.models import RasterLayer


class LayerManagerDock(QDockWidget):
    """Dock widget hosting active image layers in the project workspace."""

    layer_visibility_changed = Signal(str, bool)
    layer_removed = Signal(str)
    export_layer_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__("Layer Manager", parent)
        self.setObjectName("LayerManagerDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Layer Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Layer Name", "Type"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.tree)

        # Action Buttons Toolbar
        btn_bar = QHBoxLayout()
        btn_bar.setContentsMargins(0, 0, 0, 0)
        btn_bar.setSpacing(4)

        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(self.remove_selected_layer)
        btn_bar.addWidget(self.btn_remove, stretch=2)

        self.btn_move_up = QPushButton("Up")
        self.btn_move_up.clicked.connect(self._on_move_up)
        btn_bar.addWidget(self.btn_move_up, stretch=1)

        self.btn_move_down = QPushButton("Down")
        self.btn_move_down.clicked.connect(self._on_move_down)
        btn_bar.addWidget(self.btn_move_down, stretch=1)

        layout.addLayout(btn_bar)
        self.setWidget(container)

        # Retranslate on creation and on language change
        self.retranslate_ui()
        from core.i18n import i18n
        i18n.language_changed.connect(lambda _: self.retranslate_ui())

    def retranslate_ui(self) -> None:
        """Update texts and headers based on active language."""
        from core.i18n import tr
        self.setWindowTitle(tr("dock.layer_manager"))
        self.tree.setHeaderLabels([
            tr("layer_manager.col_name"),
            tr("layer_manager.col_type"),
        ])
        self.btn_remove.setText(tr("layer_manager.btn_remove"))
        self.btn_move_up.setText(tr("layer_manager.btn_up"))
        self.btn_move_down.setText(tr("layer_manager.btn_down"))

    def add_layer(self, layer: RasterLayer) -> None:
        """Add a new raster layer to the manager."""
        display_type = "RGB" if layer.display_mode == "rgb" else "Gray"
        item = QTreeWidgetItem(self.tree, [layer.name, display_type])
        item.setCheckState(0, Qt.Checked if layer.is_visible else Qt.Unchecked)
        item.setData(0, Qt.UserRole, layer.layer_id)
        self.tree.setCurrentItem(item)

    def update_layer_display_mode(self, layer_id: str, display_mode: str) -> None:
        """Update display type string (RGB or Gray) for the given layer."""
        display_type = "RGB" if display_mode == "rgb" else "Gray"
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == layer_id:
                item.setText(1, display_type)
                break

    def remove_selected_layer(self) -> Optional[str]:
        """Remove currently selected layer item and emit layer_removed."""
        current = self.tree.currentItem()
        if current:
            layer_id = current.data(0, Qt.UserRole)
            index = self.tree.indexOfTopLevelItem(current)
            self.tree.takeTopLevelItem(index)
            if layer_id:
                self.layer_removed.emit(layer_id)
            event_bus.status_message.emit("Layer removed", 2000)
            return layer_id
        return None

    def remove_layer_by_id(self, layer_id: str) -> None:
        """Remove a layer item matching layer_id from the tree."""
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == layer_id:
                self.tree.takeTopLevelItem(i)
                break

    def _on_move_up(self) -> None:
        """Move selected layer up in layer order."""
        current = self.tree.currentItem()
        if not current:
            return
        index = self.tree.indexOfTopLevelItem(current)
        if index > 0:
            item = self.tree.takeTopLevelItem(index)
            self.tree.insertTopLevelItem(index - 1, item)
            self.tree.setCurrentItem(item)

    def _on_move_down(self) -> None:
        """Move selected layer down in layer order."""
        current = self.tree.currentItem()
        if not current:
            return
        index = self.tree.indexOfTopLevelItem(current)
        if index < self.tree.topLevelItemCount() - 1:
            item = self.tree.takeTopLevelItem(index)
            self.tree.insertTopLevelItem(index + 1, item)
            self.tree.setCurrentItem(item)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle layer check state changes."""
        if column == 0:
            layer_id = item.data(0, Qt.UserRole)
            is_visible = item.checkState(0) == Qt.Checked
            if layer_id:
                self.layer_visibility_changed.emit(layer_id, is_visible)

    def _on_selection_changed(self) -> None:
        """Handle active layer change."""
        current = self.tree.currentItem()
        if current:
            layer_id = current.data(0, Qt.UserRole)
            if layer_id:
                event_bus.layer_changed.emit(layer_id)

    def _show_context_menu(self, pos) -> None:
        """Show context menu on layer tree item."""
        item = self.tree.itemAt(pos)
        if not item:
            return

        layer_id = item.data(0, Qt.UserRole)
        if not layer_id:
            return

        from PySide6.QtWidgets import QMenu
        from core.i18n import tr

        menu = QMenu(self)
        act_export = menu.addAction(tr("layer_manager.ctx_export"))
        act_remove = menu.addAction(tr("layer_manager.btn_remove"))

        action = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if action == act_export:
            self.export_layer_requested.emit(layer_id)
        elif action == act_remove:
            self.remove_selected_layer()

