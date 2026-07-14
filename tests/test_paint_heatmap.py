from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from colmap_reconstruction.paint_heatmap import (
    HeatmapPainter,
    apply_brush,
    default_output_path,
    nearest_jet_score,
    opencv_jet_color,
)
from colmap_reconstruction.apply_heatmaps import write_colored_ply


class PaintHeatmapTests(unittest.TestCase):
    def _make_graph_painter(self):
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        ply = root / "pinned.ply"
        db = root / "graph.db"
        vertices = np.asarray(
            [[float(index), 0.0, 0.0] for index in range(6)], dtype=np.float64
        )
        colors = np.full((6, 3), 100, dtype=np.uint8)
        write_colored_ply(
            ply,
            vertices,
            [],
            colors,
            node_indices=np.asarray([0, 0, 1, 1, -1, -1], dtype=np.int32),
            pin_node_indices=np.asarray([-1, -1, -1, -1, 0, 1], dtype=np.int32),
            node_ids={0: "house_0", 1: "car_0"},
        )
        connection = sqlite3.connect(db)
        try:
            connection.execute(
                """
                CREATE TABLE nodes (
                    id TEXT PRIMARY KEY,
                    label TEXT,
                    score REAL,
                    color_r INTEGER,
                    color_g INTEGER,
                    color_b INTEGER
                )
                """
            )
            connection.executemany(
                "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ("house_0", "House", 10.0, 1, 2, 3),
                    ("car_0", "Car", 20.0, 4, 5, 6),
                ],
            )
            connection.commit()
        finally:
            connection.close()
        painter = HeatmapPainter(ply, graph_db=db)
        painter._set_status = lambda message=None: None
        return temp_dir, db, painter

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

    def test_graph_updates_use_median_visible_rgb_and_recolor_pins(self):
        temp_dir, db, painter = self._make_graph_painter()
        try:
            painter.colors[:4] = np.asarray(
                [[10, 100, 200], [30, 120, 220], [200, 50, 10], [220, 70, 30]],
                dtype=np.float32,
            )
            before = painter._update_graph_from_touched_vertices([0, 1, 2, 3])

            self.assertEqual(before["house_0"], (10.0, 1, 2, 3))
            connection = sqlite3.connect(db)
            try:
                rows = {
                    row[0]: row[1:]
                    for row in connection.execute(
                        "SELECT id, score, color_r, color_g, color_b FROM nodes"
                    )
                }
            finally:
                connection.close()
            self.assertEqual(rows["house_0"][1:], (20, 110, 210))
            self.assertAlmostEqual(rows["house_0"][0], nearest_jet_score((20, 110, 210)))
            self.assertEqual(rows["car_0"][1:], (210, 60, 20))
            np.testing.assert_array_equal(painter.base_colors[4], [20, 110, 210])
            np.testing.assert_array_equal(painter.base_colors[5], [210, 60, 20])

            painter._restore_graph_states(before)
            connection = sqlite3.connect(db)
            try:
                restored = connection.execute(
                    "SELECT score, color_r, color_g, color_b FROM nodes WHERE id='house_0'"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(restored, (10.0, 1, 2, 3))
        finally:
            painter.graph.close()
            temp_dir.cleanup()

    def test_brush_excludes_pin_marker_vertices(self):
        temp_dir, _, painter = self._make_graph_painter()
        try:
            self.assertEqual(painter._filter_brush_indices(range(6)), [0, 1, 2, 3])
        finally:
            painter.graph.close()
            temp_dir.cleanup()

    def test_pin_labels_use_database_names_at_pin_tips(self):
        temp_dir, _, painter = self._make_graph_painter()
        try:
            points, labels = painter._pin_label_data()

            self.assertEqual(labels, ["House", "Car"])
            np.testing.assert_array_equal(points, [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
            self.assertNotIn("house_0", labels)
            self.assertNotIn("car_0", labels)
        finally:
            painter.graph.close()
            temp_dir.cleanup()

    def test_undo_and_clear_restore_database_and_pin_state(self):
        temp_dir, db, painter = self._make_graph_painter()
        try:
            painter.colors[0:2] = [250, 10, 10]
            before = painter._update_graph_from_touched_vertices([0, 1])
            painter.undo_stack.append(
                {
                    "indices": [],
                    "before_alpha": [],
                    "before_colors": [],
                    "db_before": before,
                }
            )
            painter.undo()
            self.assertEqual(painter.graph.states["house_0"], (10.0, 1, 2, 3))
            np.testing.assert_array_equal(painter.base_colors[4], [1, 2, 3])

            painter.colors[0:2] = [10, 250, 10]
            painter._update_graph_from_touched_vertices([0, 1])
            edited_state = painter.graph.states["house_0"]
            painter.overlay_alpha[0] = 0.2
            painter.clear()
            self.assertEqual(painter.graph.states["house_0"], (10.0, 1, 2, 3))
            painter.undo()
            self.assertEqual(painter.graph.states["house_0"], edited_state)

            connection = sqlite3.connect(db)
            try:
                persisted = connection.execute(
                    "SELECT score, color_r, color_g, color_b FROM nodes WHERE id='house_0'"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(persisted, edited_state)
        finally:
            painter.graph.close()
            temp_dir.cleanup()

    def test_graph_mode_rejects_pins_missing_from_database(self):
        temp_dir, db, painter = self._make_graph_painter()
        painter.graph.close()
        try:
            connection = sqlite3.connect(db)
            try:
                connection.execute("DELETE FROM nodes WHERE id='car_0'")
                connection.commit()
            finally:
                connection.close()
            with self.assertRaisesRegex(ValueError, "missing from graph.db"):
                HeatmapPainter(painter.input_path, graph_db=db)
        finally:
            temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
