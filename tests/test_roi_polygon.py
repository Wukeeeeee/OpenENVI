"""Unit tests for ROI polygon drawing, mask extraction, and spectral calculation."""

import pytest
import numpy as np

from core.roi import ROI
from core.io.memory import MemoryRasterReader


def test_polygon_roi_mask():
    """Test generating a 2D boolean mask from polygon coordinates."""
    # Triangular polygon: (2, 2), (8, 2), (5, 8)
    poly_pts = [(2.0, 2.0), (8.0, 2.0), (5.0, 8.0)]
    roi = ROI(roi_id="poly_1", name="Triangle ROI", color="#ff0000", polygon_points=poly_pts)

    mask = roi.get_mask(height=10, width=10)
    assert mask.shape == (10, 10)
    assert mask.dtype == bool
    # Center of triangle (5, 4) should be True
    assert bool(mask[4, 5]) or bool(mask[3, 5])
    # Outside corner (0, 0) should be False
    assert not bool(mask[0, 0])
    assert not bool(mask[9, 9])


def test_polygon_roi_statistics_and_spectrum():
    """Test multi-band statistics and mean spectrum calculation from polygon ROI."""
    b1 = np.full((10, 10), 10.0, dtype=np.float32)
    b2 = np.full((10, 10), 20.0, dtype=np.float32)
    cube = np.stack([b1, b2], axis=0)
    reader = MemoryRasterReader(cube, name="Cube")

    # Square polygon from (2, 2) to (6, 6)
    pts = [(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0)]
    roi = ROI(roi_id="poly_sq", name="Square ROI", polygon_points=pts)

    stats = roi.calculate_statistics(b1)
    assert stats["count"] > 0
    np.testing.assert_allclose(stats["mean"], 10.0, rtol=1e-4)

    mean_spectrum = roi.calculate_mean_spectrum(reader)
    assert mean_spectrum is not None
    assert len(mean_spectrum) == 2
    np.testing.assert_allclose(mean_spectrum, [10.0, 20.0], rtol=1e-4)


def test_multi_polygon_roi():
    """Test single ROI containing multiple disjoint polygons."""
    poly1 = [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]
    poly2 = [(6.0, 6.0), (8.0, 6.0), (8.0, 8.0), (6.0, 8.0)]

    roi = ROI(roi_id="multi_1", name="Multi-poly ROI", color="#2ecc71")
    assert roi.num_polygons == 0

    roi.add_polygon(poly1)
    assert roi.num_polygons == 1
    roi.add_polygon(poly2)
    assert roi.num_polygons == 2

    mask = roi.get_mask(height=10, width=10)
    # Check pixels inside poly1
    assert bool(mask[2, 2])
    # Check pixels inside poly2
    assert bool(mask[7, 7])
    # Check pixels between the two polygons
    assert not bool(mask[4, 4])

    # Test undo last polygon
    assert roi.remove_last_polygon() is True
    assert roi.num_polygons == 1
    mask_after = roi.get_mask(height=10, width=10)
    assert bool(mask_after[2, 2])
    assert not bool(mask_after[7, 7])


def test_roi_dialog_operations(qapp):
    """Test ROIToolDialog multi-polygon, deletion, and class creation."""
    from ui.dialogs.roi_dialog import ROIToolDialog
    from core.models import RasterLayer, RasterMetadata

    data = np.zeros((20, 20), dtype=np.float32)
    reader = MemoryRasterReader(data, name="TestRaster")
    layer = RasterLayer(
        layer_id="l1",
        name="TestLayer",
        file_path="memory",
        metadata=reader.metadata,
    )

    dlg = ROIToolDialog(layer, reader)
    assert len(dlg.rois) == 1
    assert dlg.rois[0].name == "ROI #1"
    assert dlg.rois[0].num_polygons == 0

    # Add second class
    dlg._add_roi_class()
    assert len(dlg.rois) == 2
    assert dlg.rois[1].name == "ROI #2"

    # Add polygon to ROI #2
    poly = [(2.0, 2.0), (5.0, 2.0), (5.0, 5.0), (2.0, 5.0)]
    dlg._on_polygon_completed(poly)
    assert dlg.rois[1].num_polygons == 1

    # Delete selected ROI
    dlg.table.selectRow(1)
    dlg._delete_roi()
    assert len(dlg.rois) == 1
    assert dlg.rois[0].name == "ROI #1"

    dlg.close()


def test_roi_serialization():
    """Test ROI to_dict and from_dict roundtrip."""
    roi = ROI(
        roi_id="test_r1",
        name="Forest",
        color="#27ae60",
        polygon_points=[(1.0, 1.0), (4.0, 1.0), (4.0, 4.0)],
        is_visible=False,
    )
    d = roi.to_dict()
    assert d["roi_id"] == "test_r1"
    assert d["name"] == "Forest"
    assert d["color"] == "#27ae60"
    assert d["is_visible"] is False
    assert len(d["polygons"]) == 1

    restored = ROI.from_dict(d)
    assert restored.roi_id == "test_r1"
    assert restored.name == "Forest"
    assert restored.color == "#27ae60"
    assert restored.is_visible is False
    assert restored.num_polygons == 1


def test_roi_layer_persistence(qapp):
    """Test that ROIs persist in layer and reload in subsequent ROIToolDialog sessions."""
    from ui.dialogs.roi_dialog import ROIToolDialog
    from core.models import RasterLayer

    data = np.zeros((10, 10, 3), dtype=np.float32)
    reader = MemoryRasterReader(data, name="PersistentRaster")
    layer = RasterLayer(
        layer_id="pers_1",
        name="PersistentLayer",
        file_path="memory",
        metadata=reader.metadata,
    )

    # First session: add 2 ROIs with polygons
    dlg1 = ROIToolDialog(layer, reader)
    dlg1._add_roi_class()
    dlg1._on_polygon_completed([(1.0, 1.0), (3.0, 1.0), (3.0, 3.0)])
    dlg1.close()

    assert len(layer.rois) == 2
    assert layer.rois[1].num_polygons == 1

    # Second session: open again, ensure ROIs and polygons remain
    dlg2 = ROIToolDialog(layer, reader)
    assert len(dlg2.rois) == 2
    assert dlg2.rois[1].num_polygons == 1
    dlg2.close()


def test_layer_manager_roi_integration(qapp):
    """Test LayerManagerDock tree hierarchy, visibility toggle, and deletion for ROIs."""
    from ui.layer_manager import LayerManagerDock
    from core.models import RasterLayer

    data = np.zeros((10, 10), dtype=np.float32)
    reader = MemoryRasterReader(data, name="LMDockRaster")
    layer = RasterLayer(
        layer_id="lm_layer_1",
        name="LMLayer",
        file_path="memory",
        metadata=reader.metadata,
    )
    roi1 = ROI(roi_id="r1", name="Water", color="#3498db", polygon_points=[[(0, 0), (2, 0), (2, 2)]])
    layer.rois = [roi1]

    dock = LayerManagerDock()
    dock.add_layer(layer)

    # Check top level item
    assert dock.tree.topLevelItemCount() == 1
    layer_item = dock.tree.topLevelItem(0)
    assert layer_item.text(0) == "LMLayer"

    # Check child ROI item
    assert layer_item.childCount() == 1
    roi_item = layer_item.child(0)
    assert roi_item.text(0) == "Water"
    assert "1 P" in roi_item.text(1)

    # Test toggling ROI visibility via signal
    vis_events = []
    dock.roi_visibility_changed.connect(lambda lid, rid, v: vis_events.append((lid, rid, v)))
    from PySide6.QtCore import Qt
    roi_item.setCheckState(0, Qt.Unchecked)
    assert len(vis_events) == 1
    assert vis_events[0] == ("lm_layer_1", "r1", False)

    # Test deletion signal via delete context action
    del_events = []
    dock.roi_removed.connect(lambda lid, rid: del_events.append((lid, rid)))
    dock._delete_selected_roi(roi_item)
    assert len(del_events) == 1
    assert del_events[0] == ("lm_layer_1", "r1")


def test_roi_multiband_stats_dialog(qapp):
    """Test ROIMultiBandStatsDialog calculation and table population."""
    from ui.dialogs.roi_dialog import ROIMultiBandStatsDialog

    # 3 bands: band 1 all 5, band 2 all 10, band 3 all 15
    b1 = np.full((10, 10), 5.0, dtype=np.float32)
    b2 = np.full((10, 10), 10.0, dtype=np.float32)
    b3 = np.full((10, 10), 15.0, dtype=np.float32)
    cube = np.stack([b1, b2, b3], axis=0)
    reader = MemoryRasterReader(cube, name="StatsCube")

    roi = ROI(
        roi_id="stat_roi",
        name="StatsClass",
        polygon_points=[[(1, 1), (5, 1), (5, 5), (1, 5)]],
    )

    dlg = ROIMultiBandStatsDialog(roi, reader)
    assert dlg.table.rowCount() == 3
    # Check mean values
    mean_b1 = float(dlg.table.item(0, 3).text())
    mean_b2 = float(dlg.table.item(1, 3).text())
    mean_b3 = float(dlg.table.item(2, 3).text())
    assert abs(mean_b1 - 5.0) < 1e-4
    assert abs(mean_b2 - 10.0) < 1e-4
    assert abs(mean_b3 - 15.0) < 1e-4
    dlg.close()


def test_full_roi_workflow(qapp):
    """Test full ROI drawing, polygon addition, dock sync, and mean spectrum."""
    from app.main_window import OpenENVIMainWindow
    from PySide6.QtCore import Qt

    win = OpenENVIMainWindow()
    win.show()

    # Create dummy cube (3, 50, 50)
    data = np.full((3, 50, 50), 42.0, dtype=np.float32)
    layer = win.add_derived_layer("TestCube", data)
    assert layer.layer_id == win._active_layer_id

    # Launch ROI dialog
    win.show_roi_dialog()
    dlg = win._roi_dialog
    assert dlg is not None and dlg.isVisible()

    # Verify initial ROI
    assert len(dlg.rois) == 1
    assert dlg.rois[0].name == "ROI #1"

    # Start drawing polygon
    win.main_view.start_drawing_polygon(dlg.rois[0].color)
    assert win.main_view._is_drawing_roi is True

    # Left click adds 5 vertices
    for pt in [(10.0, 10.0), (20.0, 10.0), (25.0, 18.0), (15.0, 25.0), (8.0, 18.0)]:
        win.main_view._current_poly_pts.append(pt)
    assert len(win.main_view._current_poly_pts) == 5

    # Right click closes polygon
    win.main_view._finish_current_polygon()
    assert len(dlg.rois[0].polygons) == 1
    assert dlg.rois[0].num_polygons == 1

    # Add 2nd polygon to same ROI
    for pt in [(30.0, 30.0), (40.0, 30.0), (40.0, 40.0), (30.0, 40.0)]:
        win.main_view._current_poly_pts.append(pt)
    win.main_view._finish_current_polygon()
    assert dlg.rois[0].num_polygons == 2

    # Verify Layer Manager has child ROI item
    lm = win.dock_layer_manager
    top_item = lm.tree.topLevelItem(0)
    assert top_item.childCount() == 1
    roi_child = top_item.child(0)
    assert roi_child.text(0) == "ROI #1"
    assert "2 P" in roi_child.text(1)

    # Test Multi-Band Stats
    dlg.table.selectRow(0)
    dlg._compute_multiband_stats()

    # Test Projecting ROI Mean Spectrum
    win.dock_spectral_profile.hide()
    assert not win.dock_spectral_profile.isVisible()
    dlg._plot_mean_spectrum()
    assert win.dock_spectral_profile.isVisible()

    # Close ROI Dialog
    dlg.close()

    # Verify ROIs persist on layer
    assert len(layer.rois) == 1
    assert layer.rois[0].num_polygons == 2

    # Toggle ROI visibility in Layer Manager
    active_roi_child = top_item.child(0)
    active_roi_child.setCheckState(0, Qt.Unchecked)
    assert layer.rois[0].is_visible is False

    # Reopen ROI Dialog: verify all data persisted
    win.show_roi_dialog()
    dlg2 = win._roi_dialog
    assert len(dlg2.rois) == 1
    assert dlg2.rois[0].num_polygons == 2
    dlg2.close()
    win.close()



