"""OpenENVI Spectral Profile Dock.

Provides interactive Z-Profile plotting for hyperspectral and multispectral pixel curves.
"""

from typing import List, Optional
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.events import event_bus


class SpectralProfileDock(QDockWidget):
    """Dock widget hosting the interactive Z-Profile plot canvas."""

    def __init__(self, parent=None):
        super().__init__("Spectral Profile (Z-Profile)", parent)
        self.setObjectName("SpectralProfileDock")
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Plot Controls Header
        header = QHBoxLayout()
        header.setContentsMargins(2, 2, 2, 2)
        self._lbl_info = QLabel("Position: (-, -) | Spectrum: No pixel selected")
        self._lbl_info.setStyleSheet("color: #8e9297; font-weight: 500;")
        header.addWidget(self._lbl_info, stretch=1)

        # X-Axis Mode Selector (Band Index vs Wavelength)
        self._lbl_mode = QLabel("X-Axis: ")
        self._lbl_mode.setStyleSheet("color: #8e9297; font-weight: 500;")
        header.addWidget(self._lbl_mode)

        self._combo_x_axis = QComboBox()
        self._combo_x_axis.addItem("Wavelength (nm)", "wavelength")
        self._combo_x_axis.addItem("Band Number", "band")
        self._combo_x_axis.currentIndexChanged.connect(self._on_axis_mode_changed)
        header.addWidget(self._combo_x_axis)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.clicked.connect(self.clear_spectrum)
        header.addWidget(self._btn_clear)
        layout.addLayout(header)

        # Configure PyQtGraph Dark Theme
        pg.setConfigOption("background", "#1a1c1e")
        pg.setConfigOption("foreground", "#c9d1d9")
        pg.setConfigOption("antialias", True)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        bottom_axis = self.plot_widget.getPlotItem().getAxis("bottom")
        bottom_axis.enableAutoSIPrefix(False)
        bottom_axis.setTextPen("#8e9297")
        self.plot_widget.getPlotItem().getAxis("left").setTextPen("#8e9297")
        self.plot_widget.setLabel("bottom", "Wavelength (nm)")
        self.plot_widget.setLabel("left", "Value / Reflectance")

        # Plot curve item
        self._curve = self.plot_widget.plot(
            pen=pg.mkPen(color="#58a6ff", width=1.5),
            symbol="o",
            symbolSize=4,
            symbolBrush="#58a6ff",
            symbolPen=None,
        )

        layout.addWidget(self.plot_widget)
        self.setWidget(container)

        # Connect to event bus
        event_bus.pixel_clicked.connect(self._on_pixel_clicked)

        # Cached active spectrum data for dynamic axis replotting
        self._current_values: Optional[np.ndarray] = None
        self._current_wavelengths: Optional[np.ndarray] = None
        self._current_x: int = 0
        self._current_y: int = 0
        self._last_stats: Optional[dict] = None

        # Retranslate on creation and on language change
        self.retranslate_ui()
        from core.i18n import i18n
        i18n.language_changed.connect(lambda _: self.retranslate_ui())

    def retranslate_ui(self) -> None:
        """Update texts based on active language."""
        from core.i18n import tr
        self.setWindowTitle(tr("dock.spectral_profile"))
        self._btn_clear.setText(tr("spectral_profile.btn_clear"))
        self._lbl_mode.setText(tr("spectral_profile.lbl_mode"))

        cur_idx = self._combo_x_axis.currentIndex()
        self._combo_x_axis.blockSignals(True)
        self._combo_x_axis.setItemText(0, tr("spectral_profile.mode_wavelength"))
        self._combo_x_axis.setItemText(1, tr("spectral_profile.mode_band"))
        self._combo_x_axis.setCurrentIndex(cur_idx if cur_idx >= 0 else 0)
        self._combo_x_axis.blockSignals(False)

        self.plot_widget.setLabel("left", tr("spectral_profile.axis_y"))
        if self._combo_x_axis.currentData() == "band":
            self.plot_widget.setLabel("bottom", tr("spectral_profile.axis_x_band"))
        else:
            self.plot_widget.setLabel("bottom", f"{tr('spectral_profile.axis_x_wavelength')} (nm)")

        if self._last_stats is None:
            self._lbl_info.setText(tr("spectral_profile.info_idle"))
        else:
            self._lbl_info.setText(tr("spectral_profile.info_stats").format(**self._last_stats))

    def _on_axis_mode_changed(self, index: int) -> None:
        """Handle user changing X-axis mode between Wavelength and Band Number."""
        from core.i18n import tr
        mode = self._combo_x_axis.currentData()
        if mode == "band":
            self.plot_widget.setLabel("bottom", tr("spectral_profile.axis_x_band"))
        else:
            self.plot_widget.setLabel("bottom", f"{tr('spectral_profile.axis_x_wavelength')} (nm)")
        self._replot_current()

    def _replot_current(self) -> None:
        """Replot active spectrum using selected X-axis mode."""
        if self._current_values is None or len(self._current_values) == 0:
            return

        values = self._current_values
        wavelengths = self._current_wavelengths
        mode = self._combo_x_axis.currentData()

        if mode == "wavelength" and wavelengths is not None and len(wavelengths) == len(values):
            # Sort monotonically by wavelength so lines never cross or loop backwards
            sort_idx = np.argsort(wavelengths)
            x_data = wavelengths[sort_idx]
            y_data = values[sort_idx]
        else:
            x_data = np.arange(1, len(values) + 1)
            y_data = values

        self._curve.setData(x_data, y_data)

    @Slot(int, int)
    def _on_pixel_clicked(self, x: int, y: int) -> None:
        """Handle pixel click from raster canvas."""
        from core.i18n import tr
        self._lbl_info.setText(tr("spectral_profile.info_loaded").format(x=x, y=y))
        event_bus.profile_requested.emit(x, y)

    def set_spectrum(
        self,
        values: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
        x: int = 0,
        y: int = 0,
    ) -> None:
        """Update the spectrum curve data.

        Args:
            values: 1D array of reflectance or radiance DN values across bands.
            wavelengths: Optional 1D array of band wavelengths.
            x: Source pixel X coordinate.
            y: Source pixel Y coordinate.
        """
        if len(values) == 0:
            self.clear_spectrum()
            return

        self._current_values = np.asarray(values, dtype=np.float64)
        self._current_wavelengths = (
            np.asarray(wavelengths, dtype=np.float64) if wavelengths is not None else None
        )
        self._current_x = x
        self._current_y = y

        # Auto-switch combo to Band Number if no wavelengths are present
        if wavelengths is None or len(wavelengths) != len(values):
            if self._combo_x_axis.currentData() == "wavelength":
                self._combo_x_axis.blockSignals(True)
                self._combo_x_axis.setCurrentIndex(1)  # band
                self._combo_x_axis.blockSignals(False)
        self._replot_current()

        from core.i18n import tr
        finite = values[np.isfinite(values)]
        min_v = float(np.min(finite)) if len(finite) > 0 else 0.0
        max_v = float(np.max(finite)) if len(finite) > 0 else 0.0
        self._last_stats = {
            "x": x,
            "y": y,
            "bands": len(values),
            "min_val": min_v,
            "max_val": max_v,
        }
        self._lbl_info.setText(tr("spectral_profile.info_stats").format(**self._last_stats))

    def add_spectrum_overlay(
        self,
        values: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
        name: str = "ROI Spectrum",
        color: str = "#2ecc71",
    ) -> None:
        """Add an overlay spectral curve (e.g. for ROI mean spectrum)."""
        if len(values) == 0:
            return

        mode = self._combo_x_axis.currentData()
        if mode == "wavelength" and wavelengths is not None and len(wavelengths) == len(values):
            sort_idx = np.argsort(wavelengths)
            x_data = wavelengths[sort_idx]
            y_data = values[sort_idx]
        else:
            x_data = np.arange(1, len(values) + 1)
            y_data = values

        self.plot_widget.plot(
            x=x_data,
            y=y_data,
            name=name,
            pen=pg.mkPen(color=color, width=2.0),
            symbol="s",
            symbolSize=4,
            symbolBrush=color,
        )

    def clear_spectrum(self) -> None:
        """Clear the current spectrum curves."""
        self._current_values = None
        self._current_wavelengths = None
        self._last_stats = None
        self.plot_widget.clear()
        self._curve = self.plot_widget.plot(
            pen=pg.mkPen(color="#58a6ff", width=1.5),
            symbol="o",
            symbolSize=4,
            symbolBrush="#58a6ff",
            symbolPen=None,
        )
        from core.i18n import tr
        self._lbl_info.setText(tr("spectral_profile.cleared"))
