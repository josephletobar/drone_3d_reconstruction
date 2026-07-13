#!/usr/bin/env python3
import argparse
import csv
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from .apply_heatmaps import (
    find_geometry,
    parse_ply_header,
    read_ascii_scalar,
    read_binary_scalar,
    write_colored_ply,
)
from .orchestrate import ReconstructionResult


DEFAULT_MARKER_SCALE_FRACTION = 0.01
DEFAULT_PIN_UP_SCALE = 0.6


@dataclass(frozen=True)
class ObjectPinProjectionResult:
    """Paths and counts produced by graph-node-to-pin projection."""

    reconstruction: ReconstructionResult
    graph_db: Path
    output_dir: Path
    pins_csv_path: Path
    output_mesh_path: Path
    leveled_base_mesh_path: Path
    leveled_output_mesh_path: Path
    level_transform_path: Path
    base_mesh_path: Path
    matched_nodes: tuple[str, ...]
    unmatched_nodes: tuple[str, ...]
    pin_count: int


def _connect_read_only(db_path):
    db_path = Path(db_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Graph DB not found: {db_path}")
    return sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)


def _table_exists(cursor, table_name):
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def _table_columns(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def _require_columns(cursor, table_name, required):
    columns = _table_columns(cursor, table_name)
    missing = sorted(set(required) - columns)
    if missing:
        raise ValueError(
            f"{table_name!r} is missing required column(s): {', '.join(missing)}"
        )
    return columns


def _number(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _color_value(value, default):
    try:
        return max(0, min(255, int(value)))
    except (TypeError, ValueError):
        return default


def color_for_label(label):
    seed = sum((index + 1) * ord(char) for index, char in enumerate(label.lower()))
    return (
        80 + seed % 150,
        80 + (seed // 7) % 150,
        80 + (seed // 17) % 150,
    )


def load_graph_nodes(graph_db):
    """Load object nodes from graph.db's base nodes table only."""
    conn = _connect_read_only(graph_db)
    try:
        cursor = conn.cursor()
        if not _table_exists(cursor, "nodes"):
            raise ValueError("Graph DB does not contain a nodes table")

        columns = _require_columns(
            cursor,
            "nodes",
            ["id", "label", "score", "count", "geo_pos_x", "geo_pos_y"],
        )
        color_select = (
            "color_r, color_g, color_b"
            if {"color_r", "color_g", "color_b"}.issubset(columns)
            else "NULL, NULL, NULL"
        )
        cursor.execute(
            f"""
            SELECT id, label, score, count, geo_pos_x, geo_pos_y, {color_select}
            FROM nodes
            ORDER BY rowid
            """
        )

        nodes = []
        for node_id, label, score, count, x, y, color_r, color_g, color_b in cursor:
            label = str(label or "")
            fallback_color = color_for_label(label)
            nodes.append(
                {
                    "id": str(node_id),
                    "label": label,
                    "score": _number(score),
                    "count": int(_number(count, default=1)),
                    "source_x": _number(x),
                    "source_y": _number(y),
                    "color": (
                        _color_value(color_r, fallback_color[0]),
                        _color_value(color_g, fallback_color[1]),
                        _color_value(color_b, fallback_color[2]),
                    ),
                }
            )

        return nodes
    finally:
        conn.close()


def _normalized(vector, fallback):
    import numpy as np

    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def place_nodes_with_nearest_mesh_height(vertices, nodes, up_offset):
    import numpy as np

    xy = vertices[:, :2]
    pins = []
    for node in nodes:
        target = np.array([node["source_x"], node["source_y"]], dtype=np.float64)
        distances = np.linalg.norm(xy - target, axis=1)
        vertex_index = int(np.argmin(distances))
        snap_distance = float(distances[vertex_index])

        vertex = vertices[vertex_index]
        pin = dict(node)
        pin.update(
            {
                "x": float(node["source_x"]),
                "y": float(node["source_y"]),
                "z": float(vertex[2] + up_offset),
                "vertex_index": vertex_index,
                "nearest_mesh_x": float(vertex[0]),
                "nearest_mesh_y": float(vertex[1]),
                "nearest_mesh_z": float(vertex[2]),
                "nearest_mesh_distance": snap_distance,
                "pin_up_offset": float(up_offset),
            }
        )
        pins.append(pin)

    return pins


def place_nodes_from_vertex_associations(
    vertices,
    nodes,
    vertex_node_indices,
    node_ids,
    up_offset,
):
    """Place one pin per graph node near the median of its associated vertices."""
    import numpy as np

    if vertex_node_indices is None:
        return [], tuple(node["id"] for node in nodes)

    lookup_by_node_id = {node_id: index for index, node_id in node_ids.items()}
    pins = []
    unmatched = []
    for node in nodes:
        lookup_index = lookup_by_node_id.get(node["id"])
        if lookup_index is None:
            unmatched.append(node["id"])
            continue
        associated = np.flatnonzero(vertex_node_indices == lookup_index)
        if len(associated) == 0:
            unmatched.append(node["id"])
            continue

        associated_vertices = vertices[associated]
        center = np.median(associated_vertices, axis=0)
        local_index = int(
            np.argmin(np.linalg.norm(associated_vertices - center, axis=1))
        )
        vertex_index = int(associated[local_index])
        vertex = vertices[vertex_index]
        pin = dict(node)
        pin.update(
            {
                "x": float(vertex[0]),
                "y": float(vertex[1]),
                "z": float(vertex[2] + up_offset),
                "vertex_index": vertex_index,
                "nearest_mesh_x": float(vertex[0]),
                "nearest_mesh_y": float(vertex[1]),
                "nearest_mesh_z": float(vertex[2]),
                "nearest_mesh_distance": 0.0,
                "pin_up_offset": float(up_offset),
            }
        )
        pins.append(pin)

    return pins, tuple(unmatched)


def preferred_base_mesh(colmap_output, fallback_mesh_path):
    """Prefer a heatmapped vertex-color mesh when available."""
    colmap_output = Path(colmap_output)
    heatmapped = colmap_output / "heatmapped_vertex_colors.ply"
    if heatmapped.exists():
        return heatmapped
    return Path(fallback_mesh_path)


def read_pinned_ply_mesh(path):
    """Read geometry, colors, base-node associations, and pin ownership."""
    import numpy as np

    path = Path(path)
    with path.open("rb") as file:
        fmt, elements = parse_ply_header(file)
        vertices = []
        colors = []
        faces = []
        node_indices = []
        pin_node_indices = []
        node_ids = {}
        has_node_indices = False
        has_pin_node_indices = False

        for element in elements:
            name = element["name"]
            properties = element["properties"]
            if name == "vertex":
                has_node_indices = any(
                    prop[0] == "scalar" and prop[2] == "node_index"
                    for prop in properties
                )
                has_pin_node_indices = any(
                    prop[0] == "scalar" and prop[2] == "pin_node_index"
                    for prop in properties
                )
            for _ in range(element["count"]):
                row = {}
                if fmt == "ascii":
                    values = file.readline().decode("ascii").split()
                    value_index = 0
                    for prop in properties:
                        if prop[0] == "scalar":
                            _, value_type, prop_name = prop
                            row[prop_name] = read_ascii_scalar(values[value_index], value_type)
                            value_index += 1
                        else:
                            _, count_type, value_type, prop_name = prop
                            count = int(read_ascii_scalar(values[value_index], count_type))
                            value_index += 1
                            row[prop_name] = [
                                read_ascii_scalar(values[value_index + i], value_type)
                                for i in range(count)
                            ]
                            value_index += count
                else:
                    for prop in properties:
                        if prop[0] == "scalar":
                            _, value_type, prop_name = prop
                            row[prop_name] = read_binary_scalar(file, value_type)
                        else:
                            _, count_type, value_type, prop_name = prop
                            count = int(read_binary_scalar(file, count_type))
                            row[prop_name] = [
                                int(read_binary_scalar(file, value_type))
                                for _ in range(count)
                            ]

                if name == "vertex":
                    vertices.append((row["x"], row["y"], row["z"]))
                    red = row.get("red", row.get("diffuse_red", row.get("r")))
                    green = row.get("green", row.get("diffuse_green", row.get("g")))
                    blue = row.get("blue", row.get("diffuse_blue", row.get("b")))
                    colors.append(
                        (
                            _color_value(red, 180),
                            _color_value(green, 180),
                            _color_value(blue, 180),
                        )
                    )
                    node_indices.append(int(row.get("node_index", -1)))
                    pin_node_indices.append(int(row.get("pin_node_index", -1)))
                elif name == "face":
                    indices = row.get("vertex_indices", row.get("vertex_index"))
                    if indices:
                        faces.append(indices)
                elif name == "node_label":
                    index = int(row["index"])
                    if index in node_ids:
                        raise ValueError(f"Duplicate node lookup index {index} in {path}")
                    encoded = bytes(int(value) for value in row.get("node_id", []))
                    try:
                        node_ids[index] = encoded.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise ValueError(
                            f"Invalid UTF-8 node ID at lookup index {index} in {path}"
                        ) from error

    node_index_array = (
        np.asarray(node_indices, dtype=np.int32) if has_node_indices else None
    )
    if node_index_array is not None:
        referenced = {int(index) for index in node_index_array if int(index) >= 0}
        missing = sorted(referenced - set(node_ids))
        if missing:
            raise ValueError(f"PLY node lookup is missing index/indices {missing}: {path}")

    pin_node_index_array = (
        np.asarray(pin_node_indices, dtype=np.int32)
        if has_pin_node_indices
        else None
    )
    if pin_node_index_array is not None:
        referenced = {int(index) for index in pin_node_index_array if int(index) >= 0}
        missing = sorted(referenced - set(node_ids))
        if missing:
            raise ValueError(f"PLY pin lookup is missing index/indices {missing}: {path}")

    return (
        np.asarray(vertices, dtype=np.float64),
        faces,
        np.asarray(colors, dtype=np.uint8),
        node_index_array,
        pin_node_index_array,
        node_ids,
    )


def read_labeled_ply_mesh(path):
    """Read geometry, colors, and base-node associations, ignoring pin ownership."""
    vertices, faces, colors, node_indices, _, node_ids = read_pinned_ply_mesh(path)
    return vertices, faces, colors, node_indices, node_ids


def read_colored_ply_mesh(path):
    """Read PLY positions, faces, and colors while ignoring optional node IDs."""
    vertices, faces, colors, _, _ = read_labeled_ply_mesh(path)
    return vertices, faces, colors


def write_pins_csv(path, pins):
    fieldnames = [
        "id",
        "label",
        "score",
        "count",
        "source_x",
        "source_y",
        "x",
        "y",
        "z",
        "vertex_index",
        "nearest_mesh_x",
        "nearest_mesh_y",
        "nearest_mesh_z",
        "nearest_mesh_distance",
        "pin_up_offset",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for pin in pins:
            writer.writerow({field: pin[field] for field in fieldnames})


def default_marker_scale(vertices):
    import numpy as np

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    return max(diagonal * DEFAULT_MARKER_SCALE_FRACTION, 1.0)


def default_pin_up_offset(marker_scale):
    return max(marker_scale * DEFAULT_PIN_UP_SCALE, 0.5)


def marker_basis(normal):
    import numpy as np

    normal = _normalized(normal, (0.0, 0.0, 1.0))
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(normal, reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    tangent_a = _normalized(np.cross(reference, normal), (1.0, 0.0, 0.0))
    tangent_b = _normalized(np.cross(normal, tangent_a), (0.0, 1.0, 0.0))
    return normal, tangent_a, tangent_b


def add_pin_marker(vertices, faces, colors, anchor, normal, scale, color):
    import numpy as np

    base_index = len(vertices)
    anchor = np.asarray(anchor, dtype=np.float64)
    normal, tangent_a, tangent_b = marker_basis(normal)
    radius = scale * 0.35
    height = scale * 0.45
    marker_vertices = [
        anchor + normal * height,
        anchor + tangent_a * radius,
        anchor + tangent_b * radius,
        anchor - tangent_a * radius,
        anchor - tangent_b * radius,
        anchor - normal * height,
    ]
    vertices.extend(marker_vertices)
    colors.extend([color] * len(marker_vertices))
    faces.extend(
        [
            [base_index + 0, base_index + 1, base_index + 2],
            [base_index + 0, base_index + 2, base_index + 3],
            [base_index + 0, base_index + 3, base_index + 4],
            [base_index + 0, base_index + 4, base_index + 1],
            [base_index + 5, base_index + 2, base_index + 1],
            [base_index + 5, base_index + 3, base_index + 2],
            [base_index + 5, base_index + 4, base_index + 3],
            [base_index + 5, base_index + 1, base_index + 4],
        ]
    )


def pin_marker_geometry(pins, marker_scale):
    import numpy as np

    vertices = []
    faces = []
    colors = []
    for pin in pins:
        add_pin_marker(
            vertices,
            faces,
            colors,
            (pin["x"], pin["y"], pin["z"]),
            (0.0, 0.0, 1.0),
            marker_scale,
            pin["color"],
        )

    return (
        np.asarray(vertices, dtype=np.float64)
        if vertices
        else np.empty((0, 3), dtype=np.float64),
        faces,
        np.asarray(colors, dtype=np.uint8)
        if colors
        else np.empty((0, 3), dtype=np.uint8),
    )


def write_combined_pin_mesh(
    path,
    base_vertices,
    base_faces,
    base_colors,
    pins,
    marker_scale,
    node_indices=None,
    node_ids=None,
    pin_node_indices=None,
):
    import numpy as np

    pin_vertices, pin_faces, pin_colors = pin_marker_geometry(pins, marker_scale)
    lookup_by_node_id = {
        node_id: index for index, node_id in (node_ids or {}).items()
    }
    marker_pin_indices = np.asarray(
        [lookup_by_node_id[pin["id"]] for pin in pins for _ in range(6)],
        dtype=np.int32,
    )
    base_pin_indices = (
        np.asarray(pin_node_indices, dtype=np.int32)
        if pin_node_indices is not None
        else np.full(len(base_vertices), -1, dtype=np.int32)
    )
    if len(pin_vertices):
        face_offset = len(base_vertices)
        combined_vertices = np.vstack([base_vertices, pin_vertices])
        combined_colors = np.vstack([base_colors, pin_colors])
        combined_faces = list(base_faces) + [
            [int(index) + face_offset for index in face]
            for face in pin_faces
        ]
        combined_node_indices = (
            np.concatenate(
                [
                    np.asarray(node_indices, dtype=np.int32),
                    np.full(len(pin_vertices), -1, dtype=np.int32),
                ]
            )
            if node_indices is not None
            else None
        )
        combined_pin_node_indices = np.concatenate(
            [base_pin_indices, marker_pin_indices]
        )
    else:
        combined_vertices = base_vertices
        combined_colors = base_colors
        combined_faces = base_faces
        combined_node_indices = node_indices
        combined_pin_node_indices = base_pin_indices

    write_colored_ply(
        path,
        combined_vertices,
        combined_faces,
        combined_colors,
        node_indices=combined_node_indices,
        node_ids=node_ids,
        pin_node_indices=combined_pin_node_indices,
    )
    return combined_vertices, combined_faces, combined_colors


def fit_vertical_level_transform(vertices):
    import numpy as np

    xy_center = vertices[:, :2].mean(axis=0)
    centered_x = vertices[:, 0] - xy_center[0]
    centered_y = vertices[:, 1] - xy_center[1]
    design = np.column_stack(
        [centered_x, centered_y, np.ones(len(vertices), dtype=np.float64)]
    )
    slope_x, slope_y, intercept = np.linalg.lstsq(
        design,
        vertices[:, 2],
        rcond=None,
    )[0]
    normal = _normalized((-slope_x, -slope_y, 1.0), (0.0, 0.0, 1.0))
    angle_degrees = float(
        np.degrees(
            np.arccos(
                np.clip(float(np.dot(normal, np.array([0.0, 0.0, 1.0]))), -1.0, 1.0)
            )
        )
    )
    return xy_center, float(slope_x), float(slope_y), float(intercept), normal, angle_degrees


def apply_vertical_leveling(vertices, xy_center, slope_x, slope_y):
    leveled = vertices.copy()
    leveled[:, 2] = (
        vertices[:, 2]
        - slope_x * (vertices[:, 0] - xy_center[0])
        - slope_y * (vertices[:, 1] - xy_center[1])
    )
    return leveled


def write_level_transform(
    path,
    xy_center,
    slope_x,
    slope_y,
    intercept,
    normal,
    angle_degrees,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("level_transform_version 1\n")
        file.write(
            "description subtract best-fit plane slope from z while preserving x/y\n"
        )
        file.write(f"xy_center {' '.join(f'{value:.17g}' for value in xy_center)}\n")
        file.write(f"plane_z_at_center {intercept:.17g}\n")
        file.write(f"slope_x {slope_x:.17g}\n")
        file.write(f"slope_y {slope_y:.17g}\n")
        file.write(f"source_plane_normal {' '.join(f'{value:.17g}' for value in normal)}\n")
        file.write("target_plane_normal 0 0 1\n")
        file.write(f"angle_degrees {angle_degrees:.17g}\n")
        file.write(
            "formula z_leveled = z - slope_x * (x - xy_center_x) "
            "- slope_y * (y - xy_center_y)\n"
        )


def level_base_mesh(vertices):
    xy_center, slope_x, slope_y, intercept, normal, angle_degrees = (
        fit_vertical_level_transform(vertices)
    )
    leveled_vertices = apply_vertical_leveling(vertices, xy_center, slope_x, slope_y)
    return (
        leveled_vertices,
        xy_center,
        slope_x,
        slope_y,
        intercept,
        normal,
        angle_degrees,
    )


def project_object_pins(reconstruction, graph_db):
    """Project graph.db object nodes onto a georeferenced reconstructed mesh."""
    if not isinstance(reconstruction, ReconstructionResult):
        raise TypeError("reconstruction must be a ReconstructionResult")

    graph_db = Path(graph_db)
    output_dir = reconstruction.output_dir / "object_pins"
    pins_csv_path = output_dir / "object_pins.csv"
    output_mesh_path = output_dir / "object_pins_on_mesh.ply"
    leveled_base_mesh_path = output_dir / "leveled_reconstruction.ply"
    leveled_output_mesh_path = output_dir / "object_pins_on_mesh_leveled.ply"
    level_transform_path = output_dir / "object_pins_level_transform.txt"
    geometry_path = find_geometry(reconstruction.output_dir, reconstruction.mesh_path)
    base_mesh_path = preferred_base_mesh(reconstruction.output_dir, geometry_path)

    nodes = load_graph_nodes(graph_db)
    (
        vertices,
        faces,
        colors,
        node_indices,
        pin_node_indices,
        node_ids,
    ) = read_pinned_ply_mesh(base_mesh_path)
    marker_scale = default_marker_scale(vertices)
    # Reconstructed surfaces in this pipeline face negative Z, so pins belong
    # on that visible side of the mesh.
    pin_up_offset = -default_pin_up_offset(marker_scale)
    use_vertex_associations = node_indices is not None and bool(node_ids)
    if use_vertex_associations:
        original_pins, unmatched_nodes = place_nodes_from_vertex_associations(
            vertices,
            nodes,
            node_indices,
            node_ids,
            pin_up_offset,
        )
    else:
        original_pins = place_nodes_with_nearest_mesh_height(
            vertices,
            nodes,
            pin_up_offset,
        )
        unmatched_nodes = ()
    pin_output_node_ids = node_ids or {
        index: pin["id"]
        for index, pin in enumerate(sorted(original_pins, key=lambda pin: pin["id"]))
    }
    (
        leveled_vertices,
        xy_center,
        slope_x,
        slope_y,
        intercept,
        normal,
        angle_degrees,
    ) = level_base_mesh(vertices)
    if use_vertex_associations:
        leveled_pins, _ = place_nodes_from_vertex_associations(
            leveled_vertices,
            nodes,
            node_indices,
            node_ids,
            pin_up_offset,
        )
    else:
        leveled_pins = place_nodes_with_nearest_mesh_height(
            leveled_vertices,
            nodes,
            pin_up_offset,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_pins_csv(pins_csv_path, leveled_pins)
    write_combined_pin_mesh(
        output_mesh_path,
        vertices,
        faces,
        colors,
        original_pins,
        marker_scale,
        node_indices=node_indices,
        node_ids=node_ids,
        pin_node_indices=pin_node_indices,
    )
    write_colored_ply(
        leveled_base_mesh_path,
        leveled_vertices,
        faces,
        colors,
        node_indices=node_indices,
        node_ids=pin_output_node_ids,
        pin_node_indices=pin_node_indices,
    )
    write_level_transform(
        level_transform_path,
        xy_center,
        slope_x,
        slope_y,
        intercept,
        normal,
        angle_degrees,
    )
    write_combined_pin_mesh(
        leveled_output_mesh_path,
        leveled_vertices,
        faces,
        colors,
        leveled_pins,
        marker_scale,
        node_indices=node_indices,
        node_ids=pin_output_node_ids,
        pin_node_indices=pin_node_indices,
    )

    return ObjectPinProjectionResult(
        reconstruction=reconstruction,
        graph_db=graph_db,
        output_dir=output_dir,
        pins_csv_path=pins_csv_path,
        output_mesh_path=output_mesh_path,
        leveled_base_mesh_path=leveled_base_mesh_path,
        leveled_output_mesh_path=leveled_output_mesh_path,
        level_transform_path=level_transform_path,
        base_mesh_path=base_mesh_path,
        matched_nodes=tuple(pin["id"] for pin in leveled_pins),
        unmatched_nodes=unmatched_nodes,
        pin_count=len(leveled_pins),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Project graph.db object nodes onto a georeferenced COLMAP mesh."
    )
    parser.add_argument("colmap_output", help="Path to a completed COLMAP output folder")
    parser.add_argument("graph_db", help="Path to graph.db from priority_map")
    args = parser.parse_args()

    reconstruction = ReconstructionResult.from_output_dir(args.colmap_output)
    try:
        result = project_object_pins(reconstruction, args.graph_db)
    except (FileNotFoundError, RuntimeError, ValueError, sqlite3.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"Pins: {result.pin_count}")
    print(f"Matched nodes: {len(result.matched_nodes)}")
    print(f"Unmatched nodes: {len(result.unmatched_nodes)}")
    print(f"Base mesh: {result.base_mesh_path}")
    print(f"Pin CSV: {result.pins_csv_path}")
    print(f"Object pin mesh: {result.output_mesh_path}")
    print(f"Leveled reconstruction mesh: {result.leveled_base_mesh_path}")
    print(f"Leveled object pin mesh: {result.leveled_output_mesh_path}")
    print(f"Level transform: {result.level_transform_path}")


if __name__ == "__main__":
    main()
