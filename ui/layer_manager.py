"""OpenENVI Layer Manager Dock.

Provides layer organization, visibility toggling, layer ordering, and active layer selection.
"""

from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QHeaderView,
    QMenu,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.events import event_bus
from core.models import RasterLayer


class LayerManagerDock(QDockWidget):
    """Dock widget hosting active image layers and ROIs in the project workspace."""

    layer_visibility_changed = Signal(str, bool)
    layer_removed = Signal(str)
    export_layer_requested = Signal(str)
    roi_tool_requested = Signal(str)  # layer_id
    roi_visibility_changed = Signal(str, str, bool)  # layer_id, roi_id, is_visible
    roi_removed = Signal(str, str)  # layer_id, roi_id

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
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.tree)

        # Action Buttons Toolbar
        btn_bar = QHBoxLayout()
        btn_bar.setContentsMargins(0, 0, 0, 0)
        btn_bar.setSpacing(4)

        from core.i18n import tr
        self.btn_remove = QPushButton(tr("layer_manager.btn_remove"))
        self.btn_remove.clicked.connect(self.remove_selected_layer)
        btn_bar.addWidget(self.btn_remove, stretch=2)

        self.btn_move_up = QPushButton(tr("layer_manager.btn_up"))
        self.btn_move_up.clicked.connect(self._on_move_up)
        btn_bar.addWidget(self.btn_move_up, stretch=1)

        self.btn_move_down = QPushButton(tr("layer_manager.btn_down"))
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

        # Retranslate display types of existing items
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            mode = item.data(1, Qt.UserRole)
            if mode:
                item.setText(1, tr("layer_manager.type_rgb") if mode == "rgb" else tr("layer_manager.type_gray"))

    def add_layer(self, layer: RasterLayer) -> None:
        """Add a new raster layer to the manager."""
        from core.i18n import tr
        display_type = tr("layer_manager.type_rgb") if layer.display_mode == "rgb" else tr("layer_manager.type_gray")
        item = QTreeWidgetItem(self.tree, [layer.name, display_type])
        item.setCheckState(0, Qt.Checked if layer.is_visible else Qt.Unchecked)
        item.setData(0, Qt.UserRole, layer.layer_id)
        item.setData(0, Qt.UserRole + 1, "layer")
        item.setData(1, Qt.UserRole, layer.display_mode)
        self.tree.setCurrentItem(item)

        # Sync any existing ROIs on layer
        self.sync_layer_rois(layer)

    def sync_layer_rois(self, layer: RasterLayer) -> None:
        """Synchronize child ROI items under the specified raster layer in tree."""
        target_item = None
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == layer.layer_id:
                target_item = item
                break

        if not target_item:
            return

        # Block signals during sync to avoid triggering itemChanged cascades
        self.tree.blockSignals(True)
        target_item.takeChildren()

        if hasattr(layer, "rois") and layer.rois:
            for roi in layer.rois:
                poly_label = f"{roi.num_polygons} P"
                child = QTreeWidgetItem(target_item, [roi.name, poly_label])
                child.setData(0, Qt.UserRole, (layer.layer_id, roi.roi_id))
                child.setData(0, Qt.UserRole + 1, "roi")
                child.setCheckState(0, Qt.Checked if getattr(roi, "is_visible", True) else Qt.Unchecked)

                pixmap = QPixmap(12, 12)
                pixmap.fill(QColor(roi.color))
                child.setIcon(0, QIcon(pixmap))

            target_item.setExpanded(True)

        self.tree.blockSignals(False)

    def update_layer_display_mode(self, layer_id: str, display_mode: str) -> None:
        """Update display type string (RGB or Gray) for the given layer."""
        from core.i18n import tr
        display_type = tr("layer_manager.type_rgb") if display_mode == "rgb" else tr("layer_manager.type_gray")
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == layer_id:
                item.setText(1, display_type)
                item.setData(1, Qt.UserRole, display_mode)
                break

    def _delete_selected_roi(self, item: QTreeWidgetItem) -> None:
        """Remove a specific ROI tree item and emit roi_removed signal."""
        from core.i18n import tr
        data = item.data(0, Qt.UserRole)
        if data:
            layer_id, roi_id = data
            parent = item.parent()
            if parent:
                parent.removeChild(item)
            self.roi_removed.emit(layer_id, roi_id)
            event_bus.status_message.emit(tr("status.roi_removed"), 2000)

    def remove_selected_layer(self) -> Optional[str]:
        """Remove currently selected layer or ROI item."""
        from core.i18n import tr
        current = self.tree.currentItem()
        if not current:
            return None

        item_type = current.data(0, Qt.UserRole + 1)
        if item_type == "roi":
            layer_id, roi_id = current.data(0, Qt.UserRole)
            parent = current.parent()
            if parent:
                parent.removeChild(current)
            self.roi_removed.emit(layer_id, roi_id)
            event_bus.status_message.emit(tr("status.roi_removed"), 2000)
            return roi_id

        layer_id = current.data(0, Qt.UserRole)
        index = self.tree.indexOfTopLevelItem(current)
        self.tree.takeTopLevelItem(index)
        if layer_id:
            self.layer_removed.emit(layer_id)
        event_bus.status_message.emit(tr("status.layer_removed"), 2000)
        return layer_id

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
        """Handle layer or ROI check state changes."""
        if column == 0:
            item_type = item.data(0, Qt.UserRole + 1)
            is_visible = item.checkState(0) == Qt.Checked
            if item_type == "roi":
                data = item.data(0, Qt.UserRole)
                if data and isinstance(data, (tuple, list)):
                    layer_id, roi_id = data
                    self.roi_visibility_changed.emit(layer_id, roi_id, is_visible)
            else:
                layer_id = item.data(0, Qt.UserRole)
                if layer_id and isinstance(layer_id, str):
                    self.layer_visibility_changed.emit(layer_id, is_visible)

    def _on_selection_changed(self) -> None:
        """Handle active layer change."""
        current = self.tree.currentItem()
        if current:
            item_type = current.data(0, Qt.UserRole + 1)
            if item_type == "roi":
                data = current.data(0, Qt.UserRole)
                if data:
                    layer_id = data[0]
                    event_bus.layer_changed.emit(layer_id)
            else:
                layer_id = current.data(0, Qt.UserRole)
                if layer_id and isinstance(layer_id, str):
                    event_bus.layer_changed.emit(layer_id)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle double-clicking on an item to open ROI tool."""
        item_type = item.data(0, Qt.UserRole + 1)
        if item_type == "roi":
            data = item.data(0, Qt.UserRole)
            if data:
                self.roi_tool_requested.emit(data[0])
        elif item_type == "layer":
            layer_id = item.data(0, Qt.UserRole)
            if layer_id and isinstance(layer_id, str):
                self.roi_tool_requested.emit(layer_id)

    def _show_context_menu(self, pos) -> None:
        """Show context menu on layer tree item."""
        item = self.tree.itemAt(pos)
        if not item:
            return

        from core.i18n import tr
        menu = QMenu(self)
        item_type = item.data(0, Qt.UserRole + 1)

        if item_type == "roi":
            data = item.data(0, Qt.UserRole)
            if not data:
                return
            layer_id, roi_id = data
            act_roi = menu.addAction(tr("layer_manager.ctx_roi_tool"))
            act_del = menu.addAction(tr("layer_manager.ctx_delete_roi"))
            action = menu.exec(self.tree.viewport().mapToGlobal(pos))
            if action == act_roi:
                self.roi_tool_requested.emit(layer_id)
            elif action == act_del:
                self._delete_selected_roi(item)
        else:
            layer_id = item.data(0, Qt.UserRole)
            if not layer_id or not isinstance(layer_id, str):
                return
            act_roi = menu.addAction(tr("layer_manager.ctx_roi_tool"))
            act_export = menu.addAction(tr("layer_manager.ctx_export"))
            act_remove = menu.addAction(tr("layer_manager.btn_remove"))

            action = menu.exec(self.tree.viewport().mapToGlobal(pos))
            if action == act_roi:
                self.roi_tool_requested.emit(layer_id)
            elif action == act_export:
                self.export_layer_requested.emit(layer_id)
            elif action == act_remove:
                self.remove_selected_layer()


