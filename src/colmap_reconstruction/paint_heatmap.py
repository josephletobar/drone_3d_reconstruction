#!/usr/bin/env python3
"""Interactive vertex-color painting for PLY meshes."""

import argparse
import sqlite3
import sys
from pathlib import Path

from .apply_heatmaps import write_colored_ply
from .live_graph import LiveKnowledgeGraph
from .object_pins import read_pinned_ply_mesh


DEFAULT_HEAT_VALUE = 0.75
DEFAULT_BRUSH_OPACITY = 0.18
DEFAULT_MAX_OVERLAY_OPACITY = 0.55
DEFAULT_BRUSH_RADIUS_FRACTION = 0.02


def desktop_work_area():
    """Return the Windows work area as left, top, width, height."""
    if sys.platform != "win32":
        return 0, 0, 1800, 900
    import ctypes

    class Rect(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    rect = Rect()
    if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
        return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    return 0, 0, 1800, 900


def apply_brush(colors, indices, color, opacity):
    """Blend one brush sample into vertex colors and return the changed colors."""
    import numpy as np

    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        return colors
    target = np.asarray(color, dtype=np.float32)
    colors[indices] = colors[indices] * (1.0 - opacity) + target * opacity
    return colors


def opencv_jet_color(value):
    """Return the RGB color at a 0..1 position in OpenCV's JET colormap."""
    value = max(0.0, min(1.0, float(value)))
    index = round(value * 255.0)
    x = index / 255.0
    red = max(0.0, min(1.0, 1.5 - abs(4.0 * x - 3.0)))
    green = max(0.0, min(1.0, 1.5 - abs(4.0 * x - 2.0)))
    blue = max(0.0, min(1.0, 1.5 - abs(4.0 * x - 1.0)))
    return tuple(round(channel * 255.0) for channel in (red, green, blue))


def nearest_jet_score(rgb):
    """Return the 0..100 score of the nearest OpenCV JET RGB entry."""
    score, _ = nearest_jet_entry(rgb)
    return score


def nearest_jet_entry(rgb):
    """Return the score and exact RGB of the nearest OpenCV JET entry."""
    import numpy as np

    palette = np.asarray(
        [opencv_jet_color(index / 255.0) for index in range(256)],
        dtype=np.float32,
    )
    color = np.asarray(rgb, dtype=np.float32)
    index = int(np.argmin(np.sum((palette - color) ** 2, axis=1)))
    return index / 255.0 * 100.0, tuple(int(value) for value in palette[index])


class GraphDatabaseSession:
    """Validated writable access to Priority Map base-node colors and scores."""

    REQUIRED_COLUMNS = {"id", "score", "color_r", "color_g", "color_b"}

    def __init__(self, path, required_node_ids):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Graph DB not found: {self.path}")
        self.connection = sqlite3.connect(self.path)
        columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(nodes)")
        }
        missing_columns = sorted(self.REQUIRED_COLUMNS - columns)
        if missing_columns:
            self.close()
            raise ValueError(
                "Graph DB nodes table is missing column(s): "
                + ", ".join(missing_columns)
            )
        self.states = self.read_states(required_node_ids)
        self.labels = self.read_labels(required_node_ids) if "label" in columns else {}
        missing_nodes = sorted(set(required_node_ids) - set(self.states))
        if missing_nodes:
            self.close()
            raise ValueError(
                "Pinned node ID(s) are missing from graph.db: "
                + ", ".join(missing_nodes)
            )

    def read_states(self, node_ids):
        node_ids = list(dict.fromkeys(node_ids))
        if not node_ids:
            return {}
        placeholders = ",".join("?" for _ in node_ids)
        rows = self.connection.execute(
            f"SELECT id, score, color_r, color_g, color_b "
            f"FROM nodes WHERE id IN ({placeholders})",
            node_ids,
        ).fetchall()
        return {
            str(node_id): (float(score), int(red), int(green), int(blue))
            for node_id, score, red, green, blue in rows
        }

    def read_labels(self, node_ids):
        """Return human-readable names without exposing node IDs as labels."""
        node_ids = list(dict.fromkeys(node_ids))
        if not node_ids:
            return {}
        placeholders = ",".join("?" for _ in node_ids)
        rows = self.connection.execute(
            f"SELECT id, label FROM nodes WHERE id IN ({placeholders})",
            node_ids,
        ).fetchall()
        return {
            str(node_id): str(label).strip()
            for node_id, label in rows
            if label is not None and str(label).strip()
        }

    def update_states(self, states):
        if not states:
            return
        with self.connection:
            self.connection.executemany(
                """
                UPDATE nodes
                SET score = ?, color_r = ?, color_g = ?, color_b = ?
                WHERE id = ?
                """,
                [(*state, node_id) for node_id, state in states.items()],
            )
        self.states.update(states)

    def close(self):
        if getattr(self, "connection", None) is not None:
            self.connection.close()
            self.connection = None


def default_output_path(input_path):
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}_painted{input_path.suffix}")


class HeatmapPainter:
    """PyVista/VTK desktop painter for one PLY mesh."""

    def __init__(
        self,
        input_path,
        output_path=None,
        graph_db=None,
        show_live_graph=False,
        heat_value=DEFAULT_HEAT_VALUE,
        brush_opacity=DEFAULT_BRUSH_OPACITY,
        max_overlay_opacity=DEFAULT_MAX_OVERLAY_OPACITY,
        brush_radius_fraction=DEFAULT_BRUSH_RADIUS_FRACTION,
    ):
        import numpy as np

        self.input_path = Path(input_path).resolve()
        if self.input_path.suffix.lower() != ".ply":
            raise ValueError("Input must be a .ply file")
        if not self.input_path.is_file():
            raise FileNotFoundError(f"PLY file not found: {self.input_path}")
        if not 0.0 < brush_opacity <= 1.0:
            raise ValueError("brush_opacity must be greater than 0 and at most 1")
        if brush_radius_fraction <= 0.0:
            raise ValueError("brush_radius_fraction must be greater than 0")
        if not 0.0 < max_overlay_opacity <= 1.0:
            raise ValueError("max_overlay_opacity must be greater than 0 and at most 1")

        (
            self.vertices,
            self.faces,
            loaded_colors,
            self.node_indices,
            self.pin_node_indices,
            self.node_ids,
        ) = read_pinned_ply_mesh(self.input_path)
        if not len(self.vertices):
            raise ValueError("PLY contains no vertices")
        self.node_id_to_index = {
            node_id: index for index, node_id in self.node_ids.items()
        }
        self.graph = None
        self.live_graph = None
        if show_live_graph and graph_db is None:
            raise ValueError("show_live_graph requires graph_db")
        self.opening_db_states = {}
        if graph_db is not None:
            if self.node_indices is None or self.pin_node_indices is None:
                raise ValueError(
                    "Graph synchronization requires a pinned PLY containing "
                    "node_index and pin_node_index properties"
                )
            pin_lookup_indices = {
                int(index) for index in self.pin_node_indices if int(index) >= 0
            }
            missing_lookup = sorted(pin_lookup_indices - set(self.node_ids))
            if missing_lookup:
                raise ValueError(f"Pin metadata references unknown indices: {missing_lookup}")
            pin_node_ids = [self.node_ids[index] for index in sorted(pin_lookup_indices)]
            if not pin_node_ids:
                raise ValueError("Graph synchronization requires at least one embedded pin")
            self.graph = GraphDatabaseSession(graph_db, pin_node_ids)
            if show_live_graph:
                self.live_graph = LiveKnowledgeGraph(graph_db)
            self.opening_db_states = dict(self.graph.states)
            for node_id, (_, red, green, blue) in self.graph.states.items():
                lookup_index = self.node_id_to_index[node_id]
                loaded_colors[self.pin_node_indices == lookup_index] = (red, green, blue)
        self.base_colors = loaded_colors.astype(np.float32)
        self.colors = self.base_colors.copy()
        self.overlay_colors = np.zeros_like(self.base_colors)
        self.overlay_alpha = np.zeros(len(self.vertices), dtype=np.float32)
        self.output_path = Path(output_path).resolve() if output_path else default_output_path(self.input_path)
        self.heat_value = max(0.0, min(1.0, float(heat_value)))
        self.brush_color = opencv_jet_color(self.heat_value)
        self.brush_opacity = float(brush_opacity)
        self.max_overlay_opacity = float(max_overlay_opacity)
        diagonal = float(np.linalg.norm(self.vertices.max(axis=0) - self.vertices.min(axis=0)))
        self.brush_radius = max(diagonal * float(brush_radius_fraction), 1e-9)
        self.mode = "navigate"
        self.undo_stack = []
        self._stroke_before = {}
        self._painting = False
        self._last_pick = None

    def _build_polydata(self):
        import numpy as np
        import pyvista as pv

        mesh = pv.PolyData(self.vertices)
        if self.faces:
            mesh.faces = np.asarray(
                [value for face in self.faces for value in (len(face), *face)],
                dtype=np.int64,
            )
        mesh.point_data["colors"] = np.clip(self.colors, 0, 255).astype(np.uint8)
        return mesh

    def _set_status(self, message=None):
        mode_label = {"paint": "PAINT", "erase": "ERASE"}.get(
            self.mode, "NAVIGATE"
        )
        text = message or (
            f"Mode: {mode_label} | heat: {self.heat_value:.2f} | "
            f"radius: {self.brush_radius:.4g} | "
            "P paint  X erase  N navigate  -/= size  U undo  C clear  S save"
        )
        self.plotter.add_text(text, name="status", position="lower_left", font_size=10)

    def _refresh_colors(self):
        import numpy as np

        alpha = self.overlay_alpha[:, None]
        self.colors[:] = self.base_colors * (1.0 - alpha) + self.overlay_colors * alpha
        if hasattr(self, "mesh"):
            self.mesh.point_data["colors"] = np.clip(self.colors, 0, 255).astype(np.uint8)
        if hasattr(self, "plotter"):
            self.plotter.render()

    def _filter_brush_indices(self, indices):
        """Exclude generated pin-marker geometry from surface painting."""
        if self.pin_node_indices is None:
            return list(indices)
        return [index for index in indices if self.pin_node_indices[index] < 0]

    def _begin_stroke(self):
        self._stroke_before = {}
        self._painting = True
        self._last_pick = None

    def _end_stroke(self):
        if self._stroke_before:
            indices = list(self._stroke_before)
            before_alpha = [self._stroke_before[index][0] for index in indices]
            before_colors = [self._stroke_before[index][1] for index in indices]
            try:
                db_before = self._update_graph_from_touched_vertices(indices)
            except Exception as error:
                self.overlay_alpha[indices] = before_alpha
                self.overlay_colors[indices] = before_colors
                self._refresh_colors()
                self._set_status(f"Database update failed; stroke reverted: {error}")
            else:
                self.undo_stack.append(
                    {
                        "indices": indices,
                        "before_alpha": before_alpha,
                        "before_colors": before_colors,
                        "db_before": db_before,
                    }
                )
        self._stroke_before = {}
        self._painting = False
        self._last_pick = None

    def _paint_at_cursor(self):
        import vtk

        x, y = self.interactor.GetEventPosition()
        if not self.picker.Pick(x, y, 0, self.mesh_renderer):
            return
        position = self.picker.GetPickPosition()
        if self._last_pick is not None:
            distance = sum((position[i] - self._last_pick[i]) ** 2 for i in range(3)) ** 0.5
            if distance < self.brush_radius * 0.15:
                return
        self._last_pick = position

        ids = vtk.vtkIdList()
        self.locator.FindPointsWithinRadius(self.brush_radius, position, ids)
        indices = [ids.GetId(index) for index in range(ids.GetNumberOfIds())]
        indices = self._filter_brush_indices(indices)
        for index in indices:
            self._stroke_before.setdefault(
                index, (float(self.overlay_alpha[index]), self.overlay_colors[index].copy())
            )
        if not indices:
            return

        import numpy as np

        index_array = np.asarray(indices, dtype=np.int64)
        distances = np.linalg.norm(self.vertices[index_array] - np.asarray(position), axis=1)
        normalized = np.clip(distances / self.brush_radius, 0.0, 1.0)
        # Smoothstep falloff has a flat center and feathered edge without a visible ring.
        falloff = 1.0 - normalized * normalized * (3.0 - 2.0 * normalized)
        amount = self.brush_opacity * falloff
        if self.mode == "erase":
            self.overlay_alpha[index_array] = np.maximum(
                0.0, self.overlay_alpha[index_array] - amount
            )
        else:
            old_alpha = self.overlay_alpha[index_array].copy()
            self.overlay_alpha[index_array] = np.minimum(
                self.max_overlay_opacity, old_alpha + amount
            )
            mix = np.where(old_alpha > 0.0, amount / (old_alpha + amount + 1e-12), 1.0)
            self.overlay_colors[index_array] = (
                self.overlay_colors[index_array] * (1.0 - mix[:, None])
                + np.asarray(self.brush_color, dtype=np.float32) * mix[:, None]
            )
        self._refresh_colors()

    def _pin_vertices_for_node(self, node_id):
        import numpy as np

        if self.pin_node_indices is None:
            return np.empty(0, dtype=np.int64)
        lookup_index = self.node_id_to_index.get(node_id)
        if lookup_index is None:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(self.pin_node_indices == lookup_index)

    def _pin_label_data(self):
        """Return outward pin-tip positions and their human-readable DB labels."""
        import numpy as np

        if self.graph is None or self.pin_node_indices is None:
            return np.empty((0, 3), dtype=np.float64), []
        points = []
        labels = []
        for lookup_index in sorted(self.node_ids):
            node_id = self.node_ids[lookup_index]
            label = self.graph.labels.get(node_id)
            if not label:
                continue
            pin_vertices = self._pin_vertices_for_node(node_id)
            if not len(pin_vertices):
                continue
            marker_points = self.vertices[pin_vertices]
            # Pins face the negative-Z side in this reconstruction pipeline.
            points.append(marker_points[np.argmin(marker_points[:, 2])])
            labels.append(label)
        if not points:
            return np.empty((0, 3), dtype=np.float64), []
        return np.asarray(points, dtype=np.float64), labels

    def _add_pin_labels(self):
        points, labels = self._pin_label_data()
        if not labels:
            return
        self._pin_label_actor = self.plotter.add_point_labels(
            points,
            labels,
            name="pin_labels",
            font_size=12,
            text_color="white",
            show_points=False,
            shape="rounded_rect",
            shape_color="black",
            shape_opacity=0.55,
            margin=5,
            always_visible=True,
            pickable=False,
        )

    def _set_pin_color(self, node_id, rgb):
        pin_vertices = self._pin_vertices_for_node(node_id)
        if len(pin_vertices):
            self.base_colors[pin_vertices] = rgb

    def _restore_graph_states(self, states):
        if not states or self.graph is None:
            return
        self.graph.update_states(states)
        for node_id, (_, red, green, blue) in states.items():
            self._set_pin_color(node_id, (red, green, blue))
        self._refresh_live_graph()

    def _update_graph_from_touched_vertices(self, touched_indices):
        """Update every pinned node represented by this stroke's base vertices."""
        import numpy as np

        if self.graph is None or self.node_indices is None:
            return {}
        touched = np.asarray(touched_indices, dtype=np.int64)
        touched_lookup_indices = {
            int(index) for index in self.node_indices[touched] if int(index) >= 0
        }
        updates = {}
        before = {}
        for lookup_index in sorted(touched_lookup_indices):
            node_id = self.node_ids.get(lookup_index)
            if node_id not in self.graph.states or not len(self._pin_vertices_for_node(node_id)):
                continue
            node_vertices = touched[self.node_indices[touched] == lookup_index]
            median_rgb = tuple(
                int(round(value))
                for value in np.median(self.colors[node_vertices], axis=0)
            )
            score, jet_rgb = nearest_jet_entry(median_rgb)
            before[node_id] = self.graph.states[node_id]
            updates[node_id] = (score, *jet_rgb)

        self.graph.update_states(updates)
        for node_id, (_, red, green, blue) in updates.items():
            self._set_pin_color(node_id, (red, green, blue))
        if updates:
            self._refresh_colors()
            self._refresh_live_graph()
        return before

    def _refresh_live_graph(self):
        """Refresh the neighboring native graph after a database change."""
        if self.live_graph is None:
            return
        self.live_graph.refresh_window()

    def _set_mode(self, mode):
        self.mode = mode
        if hasattr(self, "interactor"):
            style = self._navigate_style if mode == "navigate" else self._paint_style
            if self.interactor.GetInteractorStyle() is not style:
                self.interactor.SetInteractorStyle(style)
        self._set_status()

    def _resize_brush(self, factor):
        self.brush_radius = max(self.brush_radius * factor, 1e-9)
        self._set_status()

    def _set_heat_value(self, value):
        self.heat_value = max(0.0, min(1.0, float(value)))
        self.brush_color = opencv_jet_color(self.heat_value)
        self._set_status()

    def _add_heat_slider(self, vtk):
        """Add a slider thumb over a blue-to-red OpenCV JET gradient."""
        lookup_table = vtk.vtkLookupTable()
        lookup_table.SetNumberOfTableValues(256)
        lookup_table.SetRange(0.0, 1.0)
        for index in range(256):
            red, green, blue = opencv_jet_color(index / 255.0)
            lookup_table.SetTableValue(index, red / 255.0, green / 255.0, blue / 255.0, 1.0)
        lookup_table.Build()

        gradient = vtk.vtkScalarBarActor()
        gradient.SetLookupTable(lookup_table)
        gradient.SetOrientationToHorizontal()
        gradient.SetPosition(0.25, 0.88)
        gradient.SetWidth(0.5)
        gradient.SetHeight(0.045)
        gradient.SetNumberOfLabels(0)
        gradient.SetTitle("COLD                                      HOT")
        gradient.GetTitleTextProperty().SetFontSize(12)
        gradient.GetTitleTextProperty().SetColor(1.0, 1.0, 1.0)
        self.plotter.renderer.AddActor2D(gradient)
        self._heat_gradient_actor = gradient
        self._heat_lookup_table = lookup_table

        slider = self.plotter.add_slider_widget(
            self._set_heat_value,
            rng=(0.0, 1.0),
            value=self.heat_value,
            title="",
            pointa=(0.25, 0.90),
            pointb=(0.75, 0.90),
            interaction_event="always",
        )
        representation = slider.GetRepresentation()
        representation.GetTubeProperty().SetColor(1.0, 1.0, 1.0)
        representation.GetTubeProperty().SetOpacity(0.12)
        representation.GetSliderProperty().SetColor(1.0, 1.0, 1.0)
        representation.GetSliderProperty().SetOpacity(1.0)
        self._heat_slider = slider

    def undo(self):
        if not self.undo_stack:
            self._set_status("Nothing to undo")
            return
        entry = self.undo_stack.pop()
        indices = entry["indices"]
        if len(indices):
            self.overlay_alpha[indices] = entry["before_alpha"]
            self.overlay_colors[indices] = entry["before_colors"]
        self._restore_graph_states(entry["db_before"])
        self._refresh_colors()
        self._set_status()

    def clear(self):
        import numpy as np

        changed = np.flatnonzero(self.overlay_alpha > 0.0)
        db_before = {}
        if self.graph is not None:
            db_before = {
                node_id: state
                for node_id, state in self.graph.states.items()
                if state != self.opening_db_states[node_id]
            }
        if len(changed) or db_before:
            self.undo_stack.append(
                {
                    "indices": changed,
                    "before_alpha": self.overlay_alpha[changed].copy(),
                    "before_colors": self.overlay_colors[changed].copy(),
                    "db_before": db_before,
                }
            )
            self.overlay_alpha[:] = 0.0
            self._restore_graph_states(
                {node_id: self.opening_db_states[node_id] for node_id in db_before}
            )
            self._refresh_colors()
        self._set_status()

    def save_as(self):
        selected = None
        try:
            from tkinter import Tk, filedialog

            root = Tk()
            root.withdraw()
            selected = filedialog.asksaveasfilename(
                title="Save painted PLY copy",
                initialdir=str(self.output_path.parent),
                initialfile=self.output_path.name,
                defaultextension=".ply",
                filetypes=(("PLY mesh", "*.ply"),),
            )
            root.destroy()
        except Exception:
            selected = str(self.output_path)
        if not selected:
            self._set_status("Save cancelled")
            return
        path = self.save(selected)
        self._set_status(f"Saved painted copy: {path}")

    def save(self, path):
        """Save the current colors while preserving embedded node associations."""
        path = Path(path)
        write_colored_ply(
            path,
            self.vertices,
            self.faces,
            self.colors.clip(0, 255).astype("uint8"),
            node_indices=self.node_indices,
            node_ids=self.node_ids,
            pin_node_indices=self.pin_node_indices,
        )
        self.output_path = path
        return path

    def show(self):
        try:
            import pyvista as pv
            import vtk
        except ImportError as error:
            raise RuntimeError(
                "The painter requires PyVista and VTK. Reinstall this package or run "
                "`python -m pip install pyvista`."
            ) from error

        self.mesh = self._build_polydata()
        window_size = None
        window_position = None
        graph_geometry = None
        if self.live_graph is not None:
            left, top, width, height = desktop_work_area()
            mesh_width = width // 2
            window_size = (mesh_width, height)
            window_position = (left, top)
            graph_geometry = ((left + mesh_width, top), (width - mesh_width, height))
        self.plotter = pv.Plotter(
            title=f"Heatmap Painter - {self.input_path.name}",
            window_size=window_size,
        )
        if window_position is not None:
            self.plotter.render_window.SetPosition(*window_position)
        self.mesh_renderer = self.plotter.renderer
        self.plotter.add_mesh(self.mesh, scalars="colors", rgb=True, pickable=True)
        self._add_pin_labels()
        self.plotter.add_axes()
        self._set_status()
        if graph_geometry is not None:
            self.live_graph.open_window(*graph_geometry)
        self.interactor = self.plotter.iren.interactor
        self._navigate_style = vtk.vtkInteractorStyleTrackballCamera()
        self._navigate_style.SetDefaultRenderer(self.plotter.renderer)
        self._paint_style = vtk.vtkInteractorStyleUser()
        self._paint_style.SetDefaultRenderer(self.plotter.renderer)

        def begin_stroke(_caller, _event):
            self._begin_stroke()
            self._paint_at_cursor()

        def continue_stroke(_caller, _event):
            if self._painting:
                self._paint_at_cursor()

        def end_stroke(_caller, _event):
            if self._painting:
                self._end_stroke()

        self._paint_style.AddObserver("LeftButtonPressEvent", begin_stroke)
        self._paint_style.AddObserver("MouseMoveEvent", continue_stroke)
        self._paint_style.AddObserver("LeftButtonReleaseEvent", end_stroke)
        self.interactor.SetInteractorStyle(self._navigate_style)
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.0005)
        self.locator = vtk.vtkStaticPointLocator()
        self.locator.SetDataSet(self.mesh)
        self.locator.BuildLocator()

        self._add_heat_slider(vtk)
        self.plotter.add_key_event("p", lambda: self._set_mode("paint"))
        # VTK reserves E for Exit; X is the erase/remove key.
        self.plotter.add_key_event("x", lambda: self._set_mode("erase"))
        self.plotter.add_key_event("n", lambda: self._set_mode("navigate"))
        self.plotter.add_key_event("minus", lambda: self._resize_brush(0.8))
        self.plotter.add_key_event("equal", lambda: self._resize_brush(1.25))
        self.plotter.add_key_event("u", self.undo)
        self.plotter.add_key_event("c", self.clear)
        self.plotter.add_key_event("s", self.save_as)
        try:
            self.plotter.show()
        finally:
            if self.graph is not None:
                self.graph.close()
            if self.live_graph is not None:
                self.live_graph.close()


def paint_heatmap(input_path, graph_db=None, output_path=None, **kwargs):
    """Open an interactive painter for any PLY and return the painter session."""
    painter = HeatmapPainter(
        input_path,
        output_path=output_path,
        graph_db=graph_db,
        **kwargs,
    )
    painter.show()
    return painter


def main():
    parser = argparse.ArgumentParser(description="Paint a translucent heat overlay on any PLY mesh")
    parser.add_argument("ply", help="PLY mesh to open")
    parser.add_argument(
        "graph_db",
        nargs="?",
        help="Optional Priority Map graph.db for live pin/score synchronization",
    )
    parser.add_argument(
        "--live-graph",
        action="store_true",
        help="Show a NetworkX knowledge graph beside the mesh and refresh it after edits",
    )
    parser.add_argument("--output", help="Suggested Save As path; input is never overwritten automatically")
    parser.add_argument("--brush-opacity", type=float, default=DEFAULT_BRUSH_OPACITY)
    parser.add_argument(
        "--max-overlay-opacity", type=float, default=DEFAULT_MAX_OVERLAY_OPACITY,
        help="Maximum heat overlay opacity. Defaults to 0.55",
    )
    parser.add_argument("--brush-radius", type=float, default=DEFAULT_BRUSH_RADIUS_FRACTION,
                        help="Initial brush radius as a fraction of the mesh bounding-box diagonal")
    args = parser.parse_args()
    try:
        paint_heatmap(
            args.ply,
            graph_db=args.graph_db,
            show_live_graph=args.live_graph,
            output_path=args.output,
            brush_opacity=args.brush_opacity,
            max_overlay_opacity=args.max_overlay_opacity,
            brush_radius_fraction=args.brush_radius,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
