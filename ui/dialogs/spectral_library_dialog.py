"""OpenENVI Spectral Library Viewer Dialog.

Mirrors ENVI's Spectral Library Viewer: browse the built-in reference spectra,
overlay any selection against the scene's mean spectrum, and rank the whole
library by spectral angle so the closest material candidates appear first.
"""

from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from core.algorithms.spectral_library import (
    LIBRARY_WAVELENGTHS,
    LibrarySpectrum,
    get_categories,
    get_spectral_library,
    mean_spectrum,
    match_library,
)
from core.i18n import tr
from core.io.base import BaseRasterReader
from core.models import RasterLayer

# One colour per category so a multi-material scene stays legible on one plot.
CATEGORY_COLORS = {
    "Vegetation": "#2ecc71",
    "Soil": "#e67e22",
    "Rock": "#9b59b6",
    "Water": "#3498db",
    "Snow/Ice": "#ecf0f1",
}
FALLBACK_COLOR = "#f1c40f"
SCENE_COLOR = "#e74c3c"


def layer_wavelengths(reader: BaseRasterReader, bands: int) -> np.ndarray:
    """Wavelength axis for a layer, falling back to band indices when absent.

    Band indices are spaced evenly across the library range so that an image
    with no header wavelengths still produces a comparable profile instead of
    failing to match anything.
    """
    details = reader.metadata.band_details or []
    if len(details) >= bands:
        found = np.array([d.wavelength for d in details[:bands]], dtype=np.float64)
        # None arrives from readers that find no wavelength tag, and NaN from a
        # malformed tag; neither is usable as an axis.
        if np.all(np.isfinite(found)):
            return found
    return np.linspace(
        float(LIBRARY_WAVELENGTHS[0]),
        float(LIBRARY_WAVELENGTHS[-1]),
        bands,
    )


def scene_mean_spectrum(reader: BaseRasterReader) -> Tuple[np.ndarray, np.ndarray, int]:
    """Mean reflectance curve of the whole scene.

    Returns the (wavelengths, reflectance, pixel_count) triple used for both
    plotting and library matching.
    """
    bands = reader.metadata.bands
    cube = np.stack([reader.read_band(b) for b in range(bands)], axis=0)
    wl = layer_wavelengths(reader, bands)
    curve, count = mean_spectrum(cube, wl)
    return wl, curve, count


class SpectralLibraryDialog(QDialog):
    """Browse, plot and spectrally match the built-in reference library."""

    def __init__(
        self,
        layer: Optional[RasterLayer] = None,
        reader: Optional[BaseRasterReader] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.layer = layer
        self.reader = reader
        self._entries: List[LibrarySpectrum] = get_spectral_library()
        self._by_name = {e.name: e for e in self._entries}
        # Set before any signal fires, because populating the browser plots.
        self._scene_curve = None

        self.setWindowTitle(tr("spectral_lib.title"))
        self.resize(1080, 700)

        self._init_ui()
        self._populate_table()
        self._load_scene()

    # ------------------------------------------------------------------ UI

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)

        # --- Left: the library browser
        left = QWidget()
        l_left = QVBoxLayout(left)

        self.cmb_category = QComboBox()
        self.cmb_category.addItem(tr("spectral_lib.all_categories"), None)
        for name in get_categories():
            self.cmb_category.addItem(name, name)
        self.cmb_category.currentIndexChanged.connect(self._populate_table)
        l_left.addWidget(self.cmb_category)

        self.table_library = QTableWidget(0, 4)
        self.table_library.setHorizontalHeaderLabels([
            tr("spectral_lib.col_name"),
            tr("spectral_lib.col_category"),
            tr("spectral_lib.col_peak"),
            tr("spectral_lib.col_mean"),
        ])
        self.table_library.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_library.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_library.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_library.itemSelectionChanged.connect(self._update_plot)
        l_left.addWidget(self.table_library)

        self.lbl_library = QLabel("")
        self.lbl_library.setWordWrap(True)
        l_left.addWidget(self.lbl_library)

        splitter.addWidget(left)

        # --- Right: profile plot
        right = QWidget()
        l_right = QVBoxLayout(right)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#1e1e1e")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        self.plot_widget.setLabel("bottom", tr("spectral_profile.axis_x_wavelength"))
        self.plot_widget.setLabel("left", tr("spectral_profile.axis_y"))
        self.lbl_scene = QLabel("")
        self.lbl_scene.setWordWrap(True)
        l_right.addWidget(self.plot_widget)
        l_right.addWidget(self.lbl_scene)

        splitter.addWidget(right)
        splitter.setSizes([420, 660])
        layout.addWidget(splitter, stretch=1)

        # --- Matching controls
        grp_match = QGroupBox(tr("spectral_lib.grp_match"))
        form = QFormLayout(grp_match)

        self.chk_scene = QCheckBox(tr("spectral_lib.chk_scene"))
        self.chk_scene.setChecked(True)
        self.chk_scene.setToolTip(tr("spectral_lib.tip_scene"))
        self.chk_scene.stateChanged.connect(self._update_plot)
        form.addRow("", self.chk_scene)

        self.spin_max_angle = QDoubleSpinBox()
        self.spin_max_angle.setRange(0.0, 180.0)
        self.spin_max_angle.setDecimals(2)
        self.spin_max_angle.setValue(25.0)
        self.spin_max_angle.setSuffix(" deg")
        self.spin_max_angle.setToolTip(tr("spectral_lib.tip_max_angle"))
        self.spin_max_angle.valueChanged.connect(self._run_match)
        form.addRow(tr("spectral_lib.lbl_max_angle"), self.spin_max_angle)

        layout.addWidget(grp_match)

        self.table_matches = QTableWidget(0, 3)
        self.table_matches.setHorizontalHeaderLabels([
            tr("spectral_lib.col_name"),
            tr("spectral_lib.col_category"),
            tr("spectral_lib.col_angle"),
        ])
        self.table_matches.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_matches.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_matches.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_matches.itemSelectionChanged.connect(self._show_match_curve)
        layout.addWidget(self.table_matches, stretch=1)

        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        self.btn_close = QPushButton(tr("dialog.btn_close"))
        self.btn_close.clicked.connect(self.accept)
        btn_bar.addWidget(self.btn_close)
        layout.addLayout(btn_bar)

    # ------------------------------------------------------------- library

    def _populate_table(self, *_):
        """Refill the browser for the chosen category, keeping names stable."""
        category = self.cmb_category.currentData()
        visible = [e for e in self._entries if category is None or e.category == category]
        self.table_library.setRowCount(0)
        for row, entry in enumerate(visible):
            self.table_library.insertRow(row)
            self.table_library.setItem(row, 0, QTableWidgetItem(entry.name))
            self.table_library.setItem(row, 1, QTableWidgetItem(entry.category))
            self.table_library.setItem(row, 2, QTableWidgetItem(f"{entry.peak_wavelength:.0f}"))
            self.table_library.setItem(
                row, 3, QTableWidgetItem(f"{entry.mean_reflectance:.3f}")
            )
        self.lbl_library.setText(
            tr("spectral_lib.count").format(n=len(visible), total=len(self._entries))
        )
        if visible:
            self.table_library.selectRow(0)
        self._update_plot()

    def selected_spectra(self) -> List[LibrarySpectrum]:
        """Library entries highlighted in the browser."""
        rows = sorted({i.row() for i in self.table_library.selectedIndexes()})
        out = []
        for row in rows:
            item = self.table_library.item(row, 0)
            if item is not None and item.text() in self._by_name:
                out.append(self._by_name[item.text()])
        return out

    def _color_for(self, entry: LibrarySpectrum) -> str:
        return CATEGORY_COLORS.get(entry.category, FALLBACK_COLOR)

    # --------------------------------------------------------------- scene

    def _load_scene(self):
        """Read the scene's mean spectrum once; failure is not fatal."""
        if self.layer is None or self.reader is None:
            self.lbl_scene.setText(tr("spectral_lib.no_scene"))
            self.chk_scene.setEnabled(False)
            return
        try:
            wl, curve, count = scene_mean_spectrum(self.reader)
        except Exception as exc:
            self.lbl_scene.setText(tr("spectral_lib.scene_error").format(err=exc))
            self.chk_scene.setEnabled(False)
            return
        self._scene_curve = (wl, curve)
        self.lbl_scene.setText(
            tr("spectral_lib.scene_info").format(
                name=self.layer.name, bands=self.reader.metadata.bands, pixels=count
            )
        )
        self._run_match()

    # --------------------------------------------------------------- plot

    def _update_plot(self, *_):
        """Redraw the profile plot from the current browser selection."""
        plot = self.plot_widget
        plot.clear()

        if self.chk_scene.isChecked() and self._scene_curve is not None:
            wl, curve = self._scene_curve
            plot.plot(
                wl, curve,
                pen=pg.mkPen(SCENE_COLOR, width=2.5),
                name=tr("spectral_lib.scene_curve"),
            )

        for entry in self.selected_spectra():
            plot.plot(
                entry.wavelengths, entry.reflectance,
                pen=pg.mkPen(self._color_for(entry), width=2),
                name=entry.name,
            )

        if plot.listDataItems():
            plot.enableAutoRange()

    def _show_match_curve(self, *_):
        """Overlay a match-table row's curve when the user picks it."""
        if self._scene_curve is None:
            return
        rows = sorted({i.row() for i in self.table_matches.selectedIndexes()})
        if not rows:
            return
        item = self.table_matches.item(rows[0], 0)
        if item is None or item.text() not in self._by_name:
            return
        entry = self._by_name[item.text()]
        plot = self.plot_widget
        plot.plot(
            entry.wavelengths, entry.reflectance,
            pen=pg.mkPen(self._color_for(entry), width=2.5, style=Qt.DashLine),
            name=entry.name,
        )

    # -------------------------------------------------------------- match

    def _run_match(self, *_):
        """Rank the library against the scene spectrum by spectral angle."""
        self.table_matches.setRowCount(0)
        if self._scene_curve is None:
            return
        wl, curve = self._scene_curve
        try:
            results = match_library(wl, curve, max_angle=self.spin_max_angle.value())
        except Exception as exc:
            QMessageBox.warning(self, tr("dialog.error"), str(exc))
            return

        for row, (entry, angle) in enumerate(results):
            self.table_matches.insertRow(row)
            self.table_matches.setItem(row, 0, QTableWidgetItem(entry.name))
            self.table_matches.setItem(row, 1, QTableWidgetItem(entry.category))
            self.table_matches.setItem(row, 2, QTableWidgetItem(f"{angle:.2f}"))
        if results:
            self.table_matches.selectRow(0)