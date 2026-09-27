"""OpenENVI Spectral Profile Dock.

Provides interactive Z-Profile plotting for hyperspectral and multispectral pixel curves.
"""

from typing import List, Optional
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
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
        self.plot_widget.setLabel("bottom", "Band / Wavelength", units="nm")
        self.plot_widget.setLabel("left", "Value / Reflectance")
        self.plot_widget.getPlotItem().getAxis("bottom").setTextPen("#8e9297")
        self.plot_widget.getPlotItem().getAxis("left").setTextPen("#8e9297")

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

        # Retranslate on creation and on language change
        self.retranslate_ui()
        from core.i18n import i18n
        i18n.language_changed.connect(lambda _: self.retranslate_ui())

    def retranslate_ui(self) -> None:
        """Update texts based on active language."""
        from core.i18n import tr
        self.setWindowTitle(tr("dock.spectral_profile"))
        self._btn_clear.setText(tr("spectral_profile.btn_clear"))
        self.plot_widget.setLabel("left", tr("spectral_profile.axis_y"))
        self.plot_widget.setLabel("bottom", tr("spectral_profile.axis_x_wavelength"), units="nm")

    @Slot(int, int)
    def _on_pixel_clicked(self, x: int, y: int) -> None:
        """Handle pixel click from raster canvas."""
        self._lbl_info.setText(f"Position: ({x}, {y}) | Spectrum Loaded")
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

        if wavelengths is not None and len(wavelengths) == len(values):
            x_data = wavelengths
            self.plot_widget.setLabel("bottom", "Wavelength", units="nm")
        else:
            x_data = np.arange(1, len(values) + 1)
            self.plot_widget.setLabel("bottom", "Band Number")

        self._curve.setData(x_data, values)
        self._lbl_info.setText(
            f"Position: ({x}, {y}) | Bands: {len(values)} | Min: {np.min(values):.3f} | Max: {np.max(values):.3f}"
        )

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

        if wavelengths is not None and len(wavelengths) == len(values):
            x_data = wavelengths
        else:
            x_data = np.arange(1, len(values) + 1)

        self.plot_widget.plot(
            x=x_data,
            y=values,
            name=name,
            pen=pg.mkPen(color=color, width=2.0),
            symbol="s",
            symbolSize=4,
            symbolBrush=color,
        )

    def clear_spectrum(self) -> None:
        """Clear the current spectrum curves."""
        self.plot_widget.clear()
        self._curve = self.plot_widget.plot(
            pen=pg.mkPen(color="#58a6ff", width=1.5),
            symbol="o",
            symbolSize=4,
            symbolBrush="#58a6ff",
            symbolPen=None,
        )
        self._lbl_info.setText("Position: (-, -) | Spectrum: Cleared")
