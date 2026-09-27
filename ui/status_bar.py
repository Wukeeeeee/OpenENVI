"""OpenENVI Status Bar.

Displays real-time pixel coordinates (File X/Y), geospatial coordinates (Lat/Lon),
spectral band values, and system operational messages.
"""

from typing import List, Optional
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QLabel, QStatusBar

from core.events import event_bus


class OpenENVIStatusBar(QStatusBar):
    """Custom high-density status bar for remote sensing feedback."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("OpenENVIStatusBar")

        # Status & Message Label (left-aligned)
        from core.i18n import tr
        self._lbl_status = QLabel(tr("app.ready"))
        self._lbl_status.setStyleSheet("color: #8e9297; font-weight: 500;")
        self.addWidget(self._lbl_status, stretch=1)

        # File coordinate indicator (X, Y)
        self._lbl_file_coords = QLabel("File: --, --")
        self._lbl_file_coords.setStyleSheet("color: #c9d1d9; padding: 0 8px; border-left: 1px solid #36393e;")
        self.addPermanentWidget(self._lbl_file_coords)

        # Geospatial coordinate indicator (Lat, Lon)
        self._lbl_geo_coords = QLabel("Geo: --, --")
        self._lbl_geo_coords.setStyleSheet("color: #c9d1d9; padding: 0 8px; border-left: 1px solid #36393e;")
        self.addPermanentWidget(self._lbl_geo_coords)

        # Multi-band DN value indicator
        self._lbl_pixel_values = QLabel("Value: --")
        self._lbl_pixel_values.setStyleSheet("color: #58a6ff; padding: 0 8px; border-left: 1px solid #36393e;")
        self.addPermanentWidget(self._lbl_pixel_values)

        # Connect to event bus signals
        self._connect_signals()

        # Retranslate on creation and on language change
        self.retranslate_ui()
        from core.i18n import i18n
        i18n.language_changed.connect(lambda _: self.retranslate_ui())

    def retranslate_ui(self) -> None:
        """Update static status bar labels based on active language."""
        from core.i18n import tr
        if self._lbl_status.text() in ("Ready", "就绪"):
            self._lbl_status.setText(tr("app.ready"))
        if "--, --" in self._lbl_file_coords.text():
            self._lbl_file_coords.setText(tr("status.file_coords"))
        if "--, --" in self._lbl_geo_coords.text():
            self._lbl_geo_coords.setText(tr("status.geo_coords"))
        if self._lbl_pixel_values.text() in ("Value: --", "像元灰度值: --"):
            self._lbl_pixel_values.setText(tr("status.pixel_value"))

    def clear(self) -> None:
        """Reset coordinates and pixel values to initial state."""
        from core.i18n import tr
        self._lbl_file_coords.setText(tr("status.file_coords"))
        self._lbl_geo_coords.setText(tr("status.geo_coords"))
        self._lbl_pixel_values.setText(tr("status.pixel_value"))
        self._lbl_status.setText(tr("app.ready"))

    def _connect_signals(self) -> None:
        """Connect status bar to global event bus."""
        event_bus.pixel_hovered.connect(self.update_hover_coords)
        event_bus.pixel_clicked.connect(self.update_clicked_coords)
        event_bus.status_message.connect(self.set_status)

    @Slot(int, int)
    def update_hover_coords(self, x: int, y: int) -> None:
        """Update file coordinates on cursor hover."""
        from core.i18n import tr
        prefix = tr("status.prefix_file")
        self._lbl_file_coords.setText(f"{prefix}: ({x}, {y})")

    @Slot(int, int)
    def update_clicked_coords(self, x: int, y: int) -> None:
        """Update probe location on click."""
        from core.i18n import tr
        prefix = tr("status.prefix_probe")
        self._lbl_file_coords.setText(f"{prefix}: ({x}, {y})")

    def update_geo_coords(
        self,
        geo_x: Optional[float],
        geo_y: Optional[float],
        crs_str: Optional[str] = None,
    ) -> None:
        """Update geospatial coordinates in status bar.

        Properly formats projected (X/Easting, Y/Northing in meters) vs
        geographic (Lon, Lat in degrees), and converts projected coords to WGS84 Lat/Lon.
        """
        from core.i18n import tr
        prefix = tr("status.prefix_geo")
        if geo_x is None or geo_y is None:
            self._lbl_geo_coords.setText(f"{prefix}: N/A")
            return

        is_geo = False
        lat_wgs: Optional[float] = None
        lon_wgs: Optional[float] = None

        if crs_str:
            try:
                from pyproj import CRS, Transformer
                crs = CRS.from_user_input(crs_str)
                if crs.is_geographic:
                    is_geo = True
                    lon_wgs, lat_wgs = float(geo_x), float(geo_y)
                else:
                    is_geo = False
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    t_lon, t_lat = transformer.transform(geo_x, geo_y)
                    lon_wgs, lat_wgs = float(t_lon), float(t_lat)
            except Exception:
                is_geo = abs(geo_x) <= 180.0 and abs(geo_y) <= 90.0
                if is_geo:
                    lon_wgs, lat_wgs = float(geo_x), float(geo_y)
        else:
            is_geo = abs(geo_x) <= 180.0 and abs(geo_y) <= 90.0
            if is_geo:
                lon_wgs, lat_wgs = float(geo_x), float(geo_y)

        if is_geo and lon_wgs is not None and lat_wgs is not None:
            ns = "N" if lat_wgs >= 0 else "S"
            ew = "E" if lon_wgs >= 0 else "W"
            self._lbl_geo_coords.setText(
                f"{prefix}: {abs(lon_wgs):.6f}°{ew}, {abs(lat_wgs):.6f}°{ns}"
            )
        elif lat_wgs is not None and lon_wgs is not None and (-90.0 <= lat_wgs <= 90.0 and -180.0 <= lon_wgs <= 180.0):
            ns = "N" if lat_wgs >= 0 else "S"
            ew = "E" if lon_wgs >= 0 else "W"
            self._lbl_geo_coords.setText(
                f"{prefix}: X: {geo_x:.2f}  Y: {geo_y:.2f} | {abs(lat_wgs):.4f}°{ns}, {abs(lon_wgs):.4f}°{ew}"
            )
        else:
            self._lbl_geo_coords.setText(
                f"{prefix}: X: {geo_x:.2f}  Y: {geo_y:.2f}"
            )

    def update_pixel_values(self, values: List[float]) -> None:
        """Update spectral band values display."""
        from core.i18n import i18n, tr
        prefix = tr("status.prefix_value")
        if not values:
            self._lbl_pixel_values.setText(f"{prefix}: --")
        elif len(values) == 1:
            self._lbl_pixel_values.setText(f"{prefix}: {values[0]:.4f}")
        elif len(values) <= 3:
            val_str = ", ".join(f"{v:.2f}" for v in values)
            self._lbl_pixel_values.setText(f"{prefix}: [{val_str}]")
        else:
            val_str = ", ".join(f"{v:.2f}" for v in values[:3])
            bands_label = "波段" if i18n.current_language == "zh" else "bands"
            self._lbl_pixel_values.setText(f"{prefix}: [{val_str}, ... ({len(values)} {bands_label})]")

    @Slot(str, int)
    def set_status(self, message: str, timeout: int = 0) -> None:
        """Set main status message."""
        self._lbl_status.setText(message)
        if timeout > 0:
            from PySide6.QtCore import QTimer
            from core.i18n import tr
            QTimer.singleShot(timeout, lambda: self._lbl_status.setText(tr("app.ready")))
