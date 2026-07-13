import tempfile
import unittest
from pathlib import Path

import numpy as np

from colmap_reconstruction.apply_heatmaps import (
    apply_heatmaps_as_vertex_colors,
    load_node_ownership,
    node_ownership_path,
    sample_node_ownership,
    write_colored_ply,
)
from colmap_reconstruction.object_pins import (
    read_colored_ply_mesh,
    read_labeled_ply_mesh,
    read_pinned_ply_mesh,
    place_nodes_from_vertex_associations,
    write_combined_pin_mesh,
)
from colmap_reconstruction.paint_heatmap import HeatmapPainter


class NodeAssociationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.vertices = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        self.faces = [[0, 1, 2]]
        self.colors = np.asarray(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_labeled_ply_round_trip_preserves_unicode_ids(self):
        path = self.root / "labeled.ply"
        write_colored_ply(
            path,
            self.vertices,
            self.faces,
            self.colors,
            node_indices=np.asarray([0, -1, 1], dtype=np.int32),
            node_ids={0: "buildings_0", 1: "café_1"},
        )

        vertices, faces, colors, node_indices, node_ids = read_labeled_ply_mesh(path)
        np.testing.assert_allclose(vertices, self.vertices)
        self.assertEqual(faces, self.faces)
        np.testing.assert_array_equal(colors, self.colors)
        np.testing.assert_array_equal(node_indices, [0, -1, 1])
        self.assertEqual(node_ids, {0: "buildings_0", 1: "café_1"})

    def test_ordinary_colored_ply_remains_readable(self):
        path = self.root / "plain.ply"
        write_colored_ply(path, self.vertices, self.faces, self.colors)

        vertices, faces, colors = read_colored_ply_mesh(path)
        np.testing.assert_allclose(vertices, self.vertices)
        self.assertEqual(faces, self.faces)
        np.testing.assert_array_equal(colors, self.colors)
        self.assertIsNone(read_labeled_ply_mesh(path)[3])

    def test_ownership_loading_and_resolution_mapping(self):
        path = self.root / "frame.nodes.npz"
        ownership = np.asarray(
            [["house_0", "house_1"], ["", "car_0"]], dtype="<U7"
        )
        np.savez_compressed(path, node_ids=ownership)

        loaded = load_node_ownership(path)
        sampled = sample_node_ownership(
            loaded,
            pixel_x=np.asarray([0, 3, 0, 3]),
            pixel_y=np.asarray([0, 0, 3, 3]),
            camera_width=4,
            camera_height=4,
        )
        self.assertEqual(sampled.tolist(), ["house_0", "house_1", "", "car_0"])
        self.assertEqual(node_ownership_path(self.root / "frame.png"), path)

    def test_malformed_ownership_files_are_rejected(self):
        missing_key = self.root / "missing.nodes.npz"
        np.savez_compressed(missing_key, labels=np.asarray([["house"]]))
        with self.assertRaisesRegex(ValueError, "missing 'node_ids'"):
            load_node_ownership(missing_key)

        wrong_shape = self.root / "shape.nodes.npz"
        np.savez_compressed(wrong_shape, node_ids=np.asarray(["house_0"]))
        with self.assertRaisesRegex(ValueError, "must be 2D"):
            load_node_ownership(wrong_shape)

    def test_pin_markers_are_unassigned_while_base_vertices_are_preserved(self):
        path = self.root / "pins.ply"
        write_combined_pin_mesh(
            path,
            self.vertices,
            self.faces,
            self.colors,
            pins=[
                {
                    "id": "house_0",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 1.0,
                    "color": (255, 255, 0),
                }
            ],
            marker_scale=0.1,
            node_indices=np.asarray([0, -1, 0], dtype=np.int32),
            node_ids={0: "house_0"},
        )

        vertices, _, _, node_indices, node_ids = read_labeled_ply_mesh(path)
        self.assertGreater(len(vertices), len(self.vertices))
        np.testing.assert_array_equal(node_indices[:3], [0, -1, 0])
        self.assertTrue(np.all(node_indices[3:] == -1))
        self.assertEqual(node_ids, {0: "house_0"})
        _, _, _, _, pin_node_indices, _ = read_pinned_ply_mesh(path)
        self.assertTrue(np.all(pin_node_indices[:3] == -1))
        self.assertTrue(np.all(pin_node_indices[3:] == 0))

    def test_painter_save_preserves_node_associations(self):
        source = self.root / "source.ply"
        output = self.root / "painted.ply"
        write_colored_ply(
            source,
            self.vertices,
            self.faces,
            self.colors,
            node_indices=np.asarray([0, -1, 0], dtype=np.int32),
            pin_node_indices=np.asarray([-1, 0, -1], dtype=np.int32),
            node_ids={0: "house_0"},
        )

        painter = HeatmapPainter(source)
        painter.save(output)

        _, _, _, node_indices, node_ids = read_labeled_ply_mesh(output)
        np.testing.assert_array_equal(node_indices, [0, -1, 0])
        self.assertEqual(node_ids, {0: "house_0"})
        self.assertEqual(read_pinned_ply_mesh(output)[4].tolist(), [-1, 0, -1])

    def test_associated_vertices_collapse_to_one_pin_per_node(self):
        vertices = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.1],
                [0.0, 0.1, 0.9],
                [10.0, 10.0, 2.0],
                [10.1, 10.0, 2.1],
            ]
        )
        nodes = [
            {"id": "house_0", "label": "house", "source_x": 999, "source_y": 999},
            {"id": "car_0", "label": "car", "source_x": 999, "source_y": 999},
            {"id": "missing_0", "label": "missing", "source_x": 0, "source_y": 0},
        ]
        pins, unmatched = place_nodes_from_vertex_associations(
            vertices,
            nodes,
            np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
            {0: "house_0", 1: "car_0"},
            up_offset=0.5,
        )

        self.assertEqual([pin["id"] for pin in pins], ["house_0", "car_0"])
        self.assertEqual(unmatched, ("missing_0",))
        self.assertAlmostEqual(pins[0]["z"], pins[0]["nearest_mesh_z"] + 0.5)
        self.assertLess(pins[0]["x"], 1.0)
        self.assertGreater(pins[1]["x"], 9.0)

    def _make_projection_fixture(self):
        from PIL import Image

        colmap_output = self.root / "colmap"
        sparse = colmap_output / "sparse"
        sparse.mkdir(parents=True)
        (sparse / "cameras.txt").write_text(
            "1 PINHOLE 4 4 1 1 1.5 1.5\n", encoding="utf-8"
        )
        (sparse / "images.txt").write_text(
            "1 1 0 0 0 0 0 0 1 first.png\n\n"
            "2 1 0 0 0 0 0 0 1 second.png\n\n",
            encoding="utf-8",
        )
        vertices = np.asarray(
            [[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, 0.2, 1.0]],
            dtype=np.float64,
        )
        write_colored_ply(
            colmap_output / "dense.ply",
            vertices,
            [[0, 1, 2]],
            np.full((3, 3), 180, dtype=np.uint8),
        )
        heatmaps = self.root / "heatmaps"
        heatmaps.mkdir()
        Image.fromarray(np.full((4, 4, 3), [255, 0, 0], dtype=np.uint8)).save(
            heatmaps / "first.png"
        )
        Image.fromarray(np.full((4, 4, 3), [0, 0, 255], dtype=np.uint8)).save(
            heatmaps / "second.png"
        )
        np.savez_compressed(
            heatmaps / "first.nodes.npz",
            node_ids=np.full((2, 2), "house_0", dtype="<U7"),
        )
        np.savez_compressed(
            heatmaps / "second.nodes.npz",
            node_ids=np.full((2, 2), "house_1", dtype="<U7"),
        )
        return colmap_output, heatmaps

    def test_projection_color_and_node_use_same_winning_image(self):
        colmap_output, heatmaps = self._make_projection_fixture()
        result = apply_heatmaps_as_vertex_colors(
            colmap_output,
            heatmaps,
            heatmap_only=True,
            heatmap_global_blur_fraction=0.0,
            heatmap_smooth_iterations=0,
        )

        _, output_path, assigned, total, associated, nodes = result
        _, _, colors, node_indices, node_ids = read_labeled_ply_mesh(output_path)
        self.assertEqual((assigned, total, associated, nodes), (3, 3, 3, 1))
        np.testing.assert_array_equal(colors, np.tile([0, 0, 255], (3, 1)))
        np.testing.assert_array_equal(node_indices, [0, 0, 0])
        self.assertEqual(node_ids, {0: "house_1"})

    def test_missing_winning_ownership_continues_unassigned(self):
        colmap_output, heatmaps = self._make_projection_fixture()
        (heatmaps / "second.nodes.npz").unlink()

        result = apply_heatmaps_as_vertex_colors(
            colmap_output,
            heatmaps,
            heatmap_only=True,
            heatmap_global_blur_fraction=0.0,
            heatmap_smooth_iterations=0,
        )

        self.assertEqual(result[4:], (0, 0))
        _, _, _, node_indices, node_ids = read_labeled_ply_mesh(result[1])
        np.testing.assert_array_equal(node_indices, [-1, -1, -1])
        self.assertEqual(node_ids, {})


if __name__ == "__main__":
    unittest.main()
