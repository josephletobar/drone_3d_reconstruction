#!/usr/bin/env python3
"""Interactive vertex-color painting for PLY meshes."""

import argparse
import sys
from pathlib import Path

from .apply_heatmaps import write_colored_ply
from .object_pins import read_colored_ply_mesh


DEFAULT_HEAT_VALUE = 0.75
DEFAULT_BRUSH_OPACITY = 0.18
DEFAULT_MAX_OVERLAY_OPACITY = 0.55
DEFAULT_BRUSH_RADIUS_FRACTION = 0.02


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


def default_output_path(input_path):
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}_painted{input_path.suffix}")


class HeatmapPainter:
    """PyVista/VTK desktop painter for one PLY mesh."""

    def __init__(
        self,
        input_path,
        output_path=None,
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

        self.vertices, self.faces, loaded_colors = read_colored_ply_mesh(self.input_path)
        if not len(self.vertices):
            raise ValueError("PLY contains no vertices")
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
        self.mesh.point_data["colors"] = np.clip(self.colors, 0, 255).astype(np.uint8)
        self.plotter.render()

    def _begin_stroke(self):
        self._stroke_before = {}
        self._painting = True
        self._last_pick = None

    def _end_stroke(self):
        if self._stroke_before:
            indices = list(self._stroke_before)
            before_alpha = [self._stroke_before[index][0] for index in indices]
            before_colors = [self._stroke_before[index][1] for index in indices]
            self.undo_stack.append((indices, before_alpha, before_colors))
        self._stroke_before = {}
        self._painting = False
        self._last_pick = None

    def _paint_at_cursor(self):
        import vtk

        x, y = self.interactor.GetEventPosition()
        if not self.picker.Pick(x, y, 0, self.plotter.renderer):
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
        indices, before_alpha, before_colors = self.undo_stack.pop()
        self.overlay_alpha[indices] = before_alpha
        self.overlay_colors[indices] = before_colors
        self._refresh_colors()
        self._set_status()

    def clear(self):
        import numpy as np

        changed = np.flatnonzero(self.overlay_alpha > 0.0)
        if len(changed):
            self.undo_stack.append(
                (changed, self.overlay_alpha[changed].copy(), self.overlay_colors[changed].copy())
            )
            self.overlay_alpha[:] = 0.0
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
        path = Path(selected)
        write_colored_ply(path, self.vertices, self.faces, self.colors.clip(0, 255).astype("uint8"))
        self.output_path = path
        self._set_status(f"Saved painted copy: {path}")

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
        self.plotter = pv.Plotter(title=f"Heatmap Painter - {self.input_path.name}")
        self.plotter.add_mesh(self.mesh, scalars="colors", rgb=True, pickable=True)
        self.plotter.add_axes()
        self._set_status()
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
        self.plotter.show()


def paint_heatmap(input_path, output_path=None, **kwargs):
    """Open an interactive painter for any PLY and return the painter session."""
    painter = HeatmapPainter(input_path, output_path=output_path, **kwargs)
    painter.show()
    return painter


def main():
    parser = argparse.ArgumentParser(description="Paint a translucent heat overlay on any PLY mesh")
    parser.add_argument("ply", help="PLY mesh to open")
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
