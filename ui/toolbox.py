"""OpenENVI Toolbox Dock.

Provides an ENVI-style hierarchical tree of remote sensing and spectral algorithms.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QDockWidget,
    QLineEdit,
    QStyle,
    QStyledItemDelegate,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.events import event_bus

# Set of tool identifiers that are fully implemented and available
IMPLEMENTED_TOOLS = {
    # Basic Tools
    "band_math",
    "roi",
    "export",
    "stats",
    "stacking",
    "resize",
    # Transforms
    "pansharpen",
    "radiometry",
    "pca",
    "mnf",
    "ica",
    "color",
    # Classification
    "kmeans",
    "isodata",
    "maxlik",
    "sam",
    "svm",
    "accuracy",
    # Spectral
    "sid",
    "sff",
    "continuum",
    # Indices
    "ndvi",
    "ndwi",
    "evi",
    "savi",
    "nbr",
}


class StrikethroughDelegate(QStyledItemDelegate):
    """Custom delegate to draw a distinct red strikethrough line on unimplemented tools."""

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        is_implemented = index.data(Qt.UserRole + 1)
        if is_implemented is False:
            text = index.data(Qt.DisplayRole)
            if text:
                text_rect = option.widget.style().subElementRect(
                    QStyle.SubElement.SE_ItemViewItemText, option, option.widget
                )
                if text_rect.isValid() and text_rect.width() > 0:
                    fm = option.fontMetrics
                    tw = fm.horizontalAdvance(text)
                    y_mid = text_rect.center().y()
                    x_start = text_rect.left()
                    x_end = min(text_rect.right() - 2, x_start + tw)
                    if x_end > x_start:
                        painter.save()
                        painter.setRenderHint(QPainter.Antialiasing, True)
                        painter.setPen(QPen(QColor("#e74c3c"), 1.8))
                        painter.drawLine(x_start, y_mid, x_end, y_mid)
                        painter.restore()


class ToolboxDock(QDockWidget):
    """Dock widget hosting the ENVI Toolbox processing tree."""

    tool_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__("Toolbox", parent)
        self.setObjectName("ToolboxDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Quick Search Filter
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Filter tools...")
        self.search_bar.textChanged.connect(self._filter_tools)
        layout.addWidget(self.search_bar)

        # Algorithm Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setItemDelegate(StrikethroughDelegate(self.tree))
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree)

        self.setWidget(container)
        self._populate_toolbox()

        # Retranslate on creation and on language change
        self.retranslate_ui()
        from core.i18n import i18n
        i18n.language_changed.connect(lambda _: self.retranslate_ui())

    def retranslate_ui(self) -> None:
        """Update texts based on active language."""
        from core.i18n import tr
        self.setWindowTitle(tr("dock.toolbox"))
        self.search_bar.setPlaceholderText(tr("toolbox.search_placeholder"))
        self._populate_toolbox()

    def _populate_toolbox(self) -> None:
        """Populate hierarchical ENVI processing tools."""
        from core.i18n import tr
        self.tree.clear()

        tools_hierarchy = {
            tr("toolbox.cat_basic"): [
                ("band_math", tr("toolbox.tool_band_math")),
                ("roi", tr("toolbox.tool_roi")),
                ("export", tr("toolbox.tool_export")),
                ("resize", tr("toolbox.tool_resize")),
                ("stacking", tr("toolbox.tool_stacking")),
                ("mosaic", tr("toolbox.tool_mosaic")),
                ("stats", tr("toolbox.tool_stats")),
            ],
            tr("toolbox.cat_transforms"): [
                ("pansharpen", tr("toolbox.tool_pansharpen")),
                ("radiometry", tr("toolbox.tool_radiometry")),
                ("pca", tr("toolbox.tool_pca")),
                ("mnf", tr("toolbox.tool_mnf")),
                ("ica", tr("toolbox.tool_ica")),
                ("color", tr("toolbox.tool_color")),
            ],
            tr("toolbox.cat_spectral"): [
                ("sam", tr("toolbox.tool_sam")),
                ("sid", tr("toolbox.tool_sid")),
                ("sff", tr("toolbox.tool_sff")),
                ("spectral_lib", tr("toolbox.tool_spectral_lib")),
                ("continuum", tr("toolbox.tool_continuum")),
            ],
            tr("toolbox.cat_classification"): [
                ("kmeans", tr("toolbox.tool_kmeans")),
                ("isodata", tr("toolbox.tool_isodata")),
                ("maxlik", tr("toolbox.tool_maxlik")),
                ("svm", tr("toolbox.tool_svm")),
                ("accuracy", tr("toolbox.tool_accuracy")),
            ],
            tr("toolbox.cat_indices"): [
                ("ndvi", tr("toolbox.tool_ndvi")),
                ("ndwi", tr("toolbox.tool_ndwi")),
                ("evi", tr("toolbox.tool_evi")),
                ("savi", tr("toolbox.tool_savi")),
                ("nbr", tr("toolbox.tool_nbr")),
            ],
        }

        for category, tool_list in tools_hierarchy.items():
            category_item = QTreeWidgetItem(self.tree, [category])
            category_item.setExpanded(True)
            for tool_id, display_name in tool_list:
                item = QTreeWidgetItem(category_item, [display_name])
                item.setData(0, Qt.UserRole, tool_id)
                is_implemented = tool_id in IMPLEMENTED_TOOLS
                item.setData(0, Qt.UserRole + 1, is_implemented)
                if not is_implemented:
                    font = item.font(0)
                    font.setStrikeOut(True)
                    item.setFont(0, font)
                    item.setForeground(0, QBrush(QColor("#8e9297")))
                    item.setToolTip(0, tr("toolbox.not_implemented"))
                else:
                    item.setForeground(0, QBrush(QColor("#e8eaed")))

    def _filter_tools(self, text: str) -> None:
        """Filter tree items by query text."""
        query = text.strip().lower()
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            cat_item = root.child(i)
            cat_matched = False
            for j in range(cat_item.childCount()):
                child = cat_item.child(j)
                match = query in child.text(0).lower()
                child.setHidden(not match and bool(query))
                if match:
                    cat_matched = True
            cat_item.setHidden(not cat_matched and bool(query))

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle tool selection."""
        tool_name = item.data(0, Qt.UserRole)
        is_implemented = item.data(0, Qt.UserRole + 1)
        if not tool_name:
            return

        if is_implemented is False:
            from core.i18n import tr
            msg = f"{item.text(0)}: {tr('toolbox.msg_not_implemented')}"
            event_bus.status_message.emit(f"{msg}", 3000)
            return

        self.tool_selected.emit(tool_name)
        from core.i18n import tr
        event_bus.status_message.emit(tr("toolbox.msg_tool_selected").format(name=tool_name), 3000)
