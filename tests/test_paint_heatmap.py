from pathlib import Path
import unittest

import numpy as np

from colmap_reconstruction.paint_heatmap import (
    HeatmapPainter,
    apply_brush,
    default_output_path,
    opencv_jet_color,
)


class PaintHeatmapTests(unittest.TestCase):
    def test_apply_brush_blends_hot_cold_and_restore_colors(self):
        base = np.array([[100, 100, 100], [10, 20, 30]], dtype=np.float32)
        colors = base.copy()

        apply_brush(colors, [0], (200, 0, 0), 0.5)
        np.testing.assert_allclose(colors[0], [150, 50, 50])
        np.testing.assert_allclose(colors[1], base[1])

        apply_brush(colors, [0], (0, 0, 200), 1.0)
        np.testing.assert_allclose(colors[0], [0, 0, 200])

        apply_brush(colors, [0], base[[0]], 1.0)
        np.testing.assert_allclose(colors[0], base[0])

    def test_default_output_path_makes_a_copy_name(self):
        self.assertEqual(default_output_path(Path("scan.ply")), Path("scan_painted.ply"))

    def test_opencv_jet_color_runs_from_cold_blue_to_hot_red(self):
        self.assertEqual(opencv_jet_color(0.0), (0, 0, 128))
        self.assertEqual(opencv_jet_color(1.0), (128, 0, 0))
        middle = opencv_jet_color(0.5)
        self.assertGreater(middle[1], middle[0])
        self.assertGreater(middle[1], middle[2])

    def test_switching_between_paint_and_erase_does_not_reset_same_style(self):
        paint_style = object()

        class Interactor:
            def __init__(self):
                self.style = paint_style
                self.set_calls = 0

            def GetInteractorStyle(self):
                return self.style

            def SetInteractorStyle(self, style):
                self.style = style
                self.set_calls += 1

        painter = HeatmapPainter.__new__(HeatmapPainter)
        painter.interactor = Interactor()
        painter._paint_style = paint_style
        painter._navigate_style = object()
        painter._set_status = lambda message=None: None

        painter._set_mode("erase")

        self.assertEqual(painter.mode, "erase")
        self.assertEqual(painter.interactor.set_calls, 0)


if __name__ == "__main__":
    unittest.main()
