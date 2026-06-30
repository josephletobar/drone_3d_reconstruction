#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import struct
from dataclasses import dataclass
from pathlib import Path

from .orchestrate import ReconstructionResult


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_COLMAP = PROJECT_ROOT / "tools" / "colmap" / "COLMAP.bat"
DEFAULT_HEATMAP_ALPHA = 0.6

DEFAULT_HEATMAP_GLOBAL_BLUR_GRID_DIVISIONS = 6
DEFAULT_HEATMAP_GLOBAL_BLUR_FRACTION = 0.025
DEFAULT_HEATMAP_GLOBAL_BLUR_STRENGTH = 1.0
DEFAULT_HEATMAP_SMOOTH_ITERATIONS = 1
DEFAULT_HEATMAP_SMOOTH_STRENGTH = 1

PLY_SCALAR_FORMATS = {
    "char": "b",
    "uchar": "B",
    "int8": "b",
    "uint8": "B",
    "short": "h",
    "ushort": "H",
    "int16": "h",
    "uint16": "H",
    "int": "i",
    "uint": "I",
    "int32": "i",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


@dataclass(frozen=True)
class HeatmapProjectionResult:
    """Paths and matching metadata produced by heatmap projection."""

    reconstruction: ReconstructionResult
    heatmap_dir: Path
    method: str
    output_mesh_path: Path
    texture_path: Path | None
    matched_images: tuple[str, ...]
    missing_heatmaps: tuple[str, ...]
    extra_heatmaps: tuple[str, ...]
    assigned_vertices: int | None
    total_vertices: int | None


def colmap_command():
    """Return the preferred COLMAP command."""
    custom_colmap = os.environ.get("COLMAP_EXE")
    if custom_colmap:
        return custom_colmap
    if LOCAL_COLMAP.exists():
        return str(LOCAL_COLMAP)
    return "colmap"


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]


@dataclass(frozen=True)
class ImagePose:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str


def parse_cameras_txt(path):
    """Parse COLMAP cameras.txt into Camera objects keyed by camera id."""
    cameras = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid cameras.txt line {line_number}: {line}")

            camera_id = int(parts[0])
            cameras[camera_id] = Camera(
                camera_id=camera_id,
                model=parts[1],
                width=int(parts[2]),
                height=int(parts[3]),
                params=[float(value) for value in parts[4:]],
            )

    return cameras


def parse_images_txt(path):
    """Parse COLMAP images.txt metadata lines into ImagePose objects."""
    images = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0

    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue

        parts = line.split(maxsplit=9)
        if len(parts) < 10:
            raise ValueError(f"Invalid images.txt line {index + 1}: {line}")

        image_id = int(parts[0])
        images[parts[9]] = ImagePose(
            image_id=image_id,
            qvec=tuple(float(value) for value in parts[1:5]),
            tvec=tuple(float(value) for value in parts[5:8]),
            camera_id=int(parts[8]),
            name=parts[9],
        )

        index += 2

    return images


def list_heatmaps(heatmap_folder):
    """Return heatmap image files keyed by exact filename."""
    heatmaps = {}

    for path in sorted(heatmap_folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            heatmaps[path.name] = path

    return heatmaps


def load_image_name_map(colmap_output):
    """Return COLMAP staged names mapped to original source names and paths."""
    manifest_path = colmap_output / "image_name_map.csv"
    if not manifest_path.exists():
        return {}, {}, None

    name_map = {}
    path_map = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"original_name", "staged_name"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Invalid image name map: {manifest_path} must include "
                "original_name and staged_name columns"
            )

        for row in reader:
            staged_name = row["staged_name"].strip()
            original_name = row["original_name"].strip()
            if staged_name and original_name:
                name_map[staged_name] = original_name
            original_path = row.get("original_path", "").strip()
            if staged_name and original_path:
                path_map[staged_name] = Path(original_path)

    return name_map, path_map, manifest_path


def validate_inputs(colmap_output, heatmap_folder):
    if not colmap_output.exists() or not colmap_output.is_dir():
        raise FileNotFoundError(f"COLMAP output folder not found: {colmap_output}")
    if not heatmap_folder.exists() or not heatmap_folder.is_dir():
        raise FileNotFoundError(f"Heatmap folder not found: {heatmap_folder}")

    cameras_txt = colmap_output / "sparse" / "cameras.txt"
    images_txt = colmap_output / "sparse" / "images.txt"
    if not cameras_txt.exists():
        raise FileNotFoundError(f"Missing COLMAP cameras file: {cameras_txt}")
    if not images_txt.exists():
        raise FileNotFoundError(f"Missing COLMAP images file: {images_txt}")

    return cameras_txt, images_txt


def find_geometry(colmap_output, mesh_path=None):
    """Return the mesh path to texture, defaulting to colmap_output/dense.ply."""
    geometry_path = Path(mesh_path) if mesh_path is not None else colmap_output / "dense.ply"
    if not geometry_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {geometry_path}")
    return geometry_path


def print_name_list(title, names, limit=10):
    if not names:
        return

    print(f"{title}:")
    for name in names[:limit]:
        print(f"  - {name}")
    if len(names) > limit:
        print(f"  ... and {len(names) - limit} more")


def require_heatmap_matches(sync_data):
    """Ensure there is at least one matching heatmap to project."""
    if sync_data["heatmap_matches"]:
        return

    raise ValueError("No COLMAP images have matching heatmaps.")


def warn_partial_heatmap_matches(sync_data):
    """Warn about imperfect heatmap sync without blocking projection."""
    missing_heatmaps = sync_data["missing_heatmaps"]
    if not missing_heatmaps:
        return

    print(
        "Warning: Some COLMAP images do not have matching heatmaps; "
        f"continuing with {len(sync_data['heatmap_matches'])} matched image(s).",
        file=sys.stderr,
    )


def load_sync_data(colmap_output, heatmap_folder, mesh_path=None):
    cameras_txt, images_txt = validate_inputs(colmap_output, heatmap_folder)

    cameras = parse_cameras_txt(cameras_txt)
    images = parse_images_txt(images_txt)
    heatmaps = list_heatmaps(heatmap_folder)
    image_name_map, original_path_map, image_name_map_path = load_image_name_map(colmap_output)
    geometry_path = find_geometry(colmap_output, mesh_path)

    expected_by_colmap_name = {
        image_name: image_name_map.get(image_name, image_name)
        for image_name in images
    }
    expected_heatmap_names = set(expected_by_colmap_name.values())
    heatmap_names = set(heatmaps)
    missing_heatmaps = sorted(expected_heatmap_names - heatmap_names)
    extra_heatmaps = sorted(heatmap_names - expected_heatmap_names)
    matched_names = sorted(expected_heatmap_names & heatmap_names)
    heatmap_matches = {
        image_name: heatmaps[heatmap_name]
        for image_name, heatmap_name in expected_by_colmap_name.items()
        if heatmap_name in heatmaps
    }

    return {
        "cameras": cameras,
        "images": images,
        "heatmaps": heatmaps,
        "image_name_map": image_name_map,
        "original_path_map": original_path_map,
        "image_name_map_path": image_name_map_path,
        "geometry_path": geometry_path,
        "matched_names": matched_names,
        "heatmap_matches": heatmap_matches,
        "missing_heatmaps": missing_heatmaps,
        "extra_heatmaps": extra_heatmaps,
    }


def blend_images(base_path, heatmap_path, output_path, heatmap_alpha):
    """Blend a heatmap over a base image and write the result."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required for blended heatmap textures. "
            "Install dependencies with `pip install -r requirements.txt`, "
            "or use --heatmap-only."
        ) from error

    with Image.open(base_path) as base_image, Image.open(heatmap_path) as heatmap_image:
        base_image = base_image.convert("RGB")
        heatmap_image = heatmap_image.convert("RGB")
        if heatmap_image.size != base_image.size:
            heatmap_image = heatmap_image.resize(base_image.size, Image.Resampling.BILINEAR)
        blended = Image.blend(base_image, heatmap_image, heatmap_alpha)
        blended.save(output_path)


def filter_sparse_images_txt(sparse_dir, kept_image_names):
    """Keep only matched images in a copied COLMAP text sparse model."""
    images_txt = sparse_dir / "images.txt"
    kept_image_names = set(kept_image_names)
    lines = images_txt.read_text(encoding="utf-8").splitlines()
    filtered_lines = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            filtered_lines.append(line)
            index += 1
            continue

        parts = stripped.split(maxsplit=9)
        if len(parts) < 10:
            raise ValueError(f"Invalid images.txt line {index + 1}: {line}")

        points_line = lines[index + 1] if index + 1 < len(lines) else ""
        if parts[9] in kept_image_names:
            filtered_lines.append(line)
            filtered_lines.append(points_line)
        index += 2

    images_txt.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")


def qvec_to_rotmat(qvec):
    """Convert COLMAP qvec to a world-to-camera rotation matrix."""
    import numpy as np

    qw, qx, qy, qz = qvec
    return np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qz * qw,
                2 * qz * qx + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qz * qx - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def project_points(points, camera, image_pose):
    """Project 3D world points into one COLMAP camera image."""
    import numpy as np

    rotation = qvec_to_rotmat(image_pose.qvec)
    translation = np.array(image_pose.tvec, dtype=np.float64)
    camera_points = points @ rotation.T + translation
    z = camera_points[:, 2]
    in_front = z > 1e-8

    x = np.empty_like(z)
    y = np.empty_like(z)
    x[in_front] = camera_points[in_front, 0] / z[in_front]
    y[in_front] = camera_points[in_front, 1] / z[in_front]

    params = camera.params
    model = camera.model.upper()
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        u = f * x + cx
        v = f * y + cy
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        u = fx * x + cx
        v = fy * y + cy
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params[:4]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2
        u = f * x * radial + cx
        v = f * y * radial + cy
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params[:5]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        u = f * x * radial + cx
        v = f * y * radial + cy
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        u = fx * x_distorted + cx
        v = fy * y_distorted + cy
    else:
        raise ValueError(f"Unsupported camera model for vertex projection: {camera.model}")

    valid = (
        in_front
        & (u >= 0.0)
        & (v >= 0.0)
        & (u < camera.width)
        & (v < camera.height)
    )
    return u, v, valid


def parse_ply_header(file):
    header_lines = []
    while True:
        line = file.readline()
        if not line:
            raise ValueError("Invalid PLY: missing end_header")
        decoded = line.decode("ascii").rstrip("\n")
        header_lines.append(decoded)
        if decoded == "end_header":
            break

    if not header_lines or header_lines[0] != "ply":
        raise ValueError("Invalid PLY: missing ply header")

    fmt = None
    elements = []
    current = None
    for line in header_lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current)
        elif parts[0] == "property" and current is not None:
            if parts[1] == "list":
                current["properties"].append(("list", parts[2], parts[3], parts[4]))
            else:
                current["properties"].append(("scalar", parts[1], parts[2]))

    if fmt not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"Unsupported PLY format: {fmt}")
    return fmt, elements


def read_ascii_scalar(value, value_type):
    if PLY_SCALAR_FORMATS[value_type].islower() and value_type not in {"float", "double", "float32", "float64"}:
        return int(value)
    if value_type in {"float", "double", "float32", "float64"}:
        return float(value)
    return int(value)


def read_binary_scalar(file, value_type):
    fmt = PLY_SCALAR_FORMATS[value_type]
    size = struct.calcsize("<" + fmt)
    data = file.read(size)
    if len(data) != size:
        raise ValueError("Invalid PLY: unexpected end of binary data")
    return struct.unpack("<" + fmt, data)[0]


def read_ply_mesh(path):
    """Read vertex positions and triangular/polygon faces from an ASCII or little-endian PLY."""
    import numpy as np

    with path.open("rb") as file:
        fmt, elements = parse_ply_header(file)
        vertices = []
        faces = []

        for element in elements:
            name = element["name"]
            properties = element["properties"]
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
                elif name == "face":
                    indices = row.get("vertex_indices", row.get("vertex_index"))
                    if indices:
                        faces.append(indices)

    return np.asarray(vertices, dtype=np.float64), faces


def write_colored_ply(path, vertices, faces, colors):
    """Write an ASCII PLY with per-vertex RGB colors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(vertices)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write(f"element face {len(faces)}\n")
        file.write("property list uchar int vertex_indices\n")
        file.write("end_header\n")
        for vertex, color in zip(vertices, colors):
            file.write(
                f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            indices = " ".join(str(int(index)) for index in face)
            file.write(f"{len(face)} {indices}\n")


def smooth_colors_on_mesh(colors, faces, assigned, iterations, strength):
    """Laplacian-smooth vertex colors over mesh connectivity."""
    if iterations <= 0 or strength <= 0.0:
        return colors

    import numpy as np

    edges = []
    for face in faces:
        face_indices = [int(index) for index in face if assigned[int(index)]]
        if len(face_indices) < 2:
            continue
        for index, source in enumerate(face_indices):
            for target in face_indices[index + 1 :]:
                edges.append((source, target))
                edges.append((target, source))

    if not edges:
        return colors

    edge_array = np.asarray(edges, dtype=np.int64)
    edge_sources = edge_array[:, 0]
    edge_targets = edge_array[:, 1]
    smoothable = np.zeros(len(colors), dtype=bool)
    smoothable[edge_sources] = True

    smoothed = colors.astype(np.float32, copy=True)
    for _ in range(iterations):
        sums = np.zeros_like(smoothed, dtype=np.float32)
        counts = np.zeros(len(smoothed), dtype=np.float32)

        np.add.at(sums, edge_sources, smoothed[edge_targets])
        np.add.at(counts, edge_sources, 1)

        valid = smoothable & (counts > 0)
        if not valid.any():
            break
        neighbor_average = sums[valid] / counts[valid, None]
        smoothed[valid] = (
            smoothed[valid] * (1.0 - strength) + neighbor_average * strength
        )

    return smoothed


def blur_colors_spatial(vertices, colors, assigned, radius_fraction, strength, voxel_divisions):
    """Blur colors in the horizontal XY plane using a sparse grid field."""
    if radius_fraction <= 0.0 or strength <= 0.0:
        return colors

    import math
    import numpy as np

    assigned_indices = np.flatnonzero(assigned)
    if len(assigned_indices) == 0:
        return colors

    xy_vertices = vertices[:, :2]
    bbox_min = xy_vertices.min(axis=0)
    bbox_max = xy_vertices.max(axis=0)
    scene_diag = float(np.linalg.norm(bbox_max - bbox_min))
    if scene_diag <= 0.0:
        return colors

    blur_radius = scene_diag * radius_fraction
    cell_size = blur_radius / max(float(voxel_divisions), 1.0)
    if cell_size <= 0.0:
        return colors

    assigned_vertices = xy_vertices[assigned_indices]
    assigned_colors = colors[assigned_indices].astype(np.float32, copy=False)
    voxel_coords = np.floor((assigned_vertices - bbox_min) / cell_size).astype(np.int64)
    unique_coords, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
    voxel_count = len(unique_coords)

    color_sums = np.zeros((voxel_count, 3), dtype=np.float32)
    position_sums = np.zeros((voxel_count, 2), dtype=np.float64)
    counts = np.zeros(voxel_count, dtype=np.float32)
    np.add.at(color_sums, inverse, assigned_colors)
    np.add.at(position_sums, inverse, assigned_vertices)
    np.add.at(counts, inverse, 1)

    voxel_colors = color_sums / counts[:, None]
    voxel_positions = position_sums / counts[:, None]
    coord_to_index = {tuple(coord): index for index, coord in enumerate(unique_coords)}

    offset_radius = int(math.ceil(blur_radius / cell_size))
    offsets = [
        np.array((dx, dy), dtype=np.int64)
        for dx in range(-offset_radius, offset_radius + 1)
        for dy in range(-offset_radius, offset_radius + 1)
    ]
    sigma = blur_radius / 2.0
    blurred_voxel_colors = np.zeros_like(voxel_colors)

    for voxel_index, coord in enumerate(unique_coords):
        weighted_sum = np.zeros(3, dtype=np.float64)
        weight_sum = 0.0
        center = voxel_positions[voxel_index]

        for offset in offsets:
            neighbor_index = coord_to_index.get(tuple(coord + offset))
            if neighbor_index is None:
                continue
            distance = float(np.linalg.norm(center - voxel_positions[neighbor_index]))
            if distance > blur_radius:
                continue
            weight = math.exp(-0.5 * (distance / sigma) ** 2)
            weighted_sum += voxel_colors[neighbor_index] * weight
            weight_sum += weight

        if weight_sum > 0.0:
            blurred_voxel_colors[voxel_index] = weighted_sum / weight_sum
        else:
            blurred_voxel_colors[voxel_index] = voxel_colors[voxel_index]

    blurred = colors.astype(np.float32, copy=True)
    blurred_assigned = blurred_voxel_colors[inverse]
    blurred[assigned_indices] = (
        blurred[assigned_indices] * (1.0 - strength) + blurred_assigned * strength
    )
    return blurred


def apply_heatmaps_as_vertex_colors(
    colmap_output,
    heatmap_folder,
    output_path=None,
    mesh_path=None,
    heatmap_alpha=DEFAULT_HEATMAP_ALPHA,
    heatmap_only=False,
    heatmap_global_blur_fraction=DEFAULT_HEATMAP_GLOBAL_BLUR_FRACTION,
    heatmap_global_blur_strength=DEFAULT_HEATMAP_GLOBAL_BLUR_STRENGTH,
    heatmap_global_blur_grid_divisions=DEFAULT_HEATMAP_GLOBAL_BLUR_GRID_DIVISIONS,
    heatmap_smooth_iterations=DEFAULT_HEATMAP_SMOOTH_ITERATIONS,
    heatmap_smooth_strength=DEFAULT_HEATMAP_SMOOTH_STRENGTH,
    warn_on_partial=True,
):
    """Color mesh vertices by projecting them into the latest valid heatmap image."""
    import numpy as np
    from PIL import Image

    sync_data = load_sync_data(colmap_output, heatmap_folder, mesh_path)
    require_heatmap_matches(sync_data)
    if warn_on_partial:
        warn_partial_heatmap_matches(sync_data)

    vertices, faces = read_ply_mesh(sync_data["geometry_path"])
    base_colors = np.full((len(vertices), 3), 180, dtype=np.float32)
    heatmap_colors = np.full((len(vertices), 3), 180, dtype=np.float32)
    assigned = np.zeros(len(vertices), dtype=bool)

    image_poses = list(sync_data["images"].values())
    for image_pose in reversed(image_poses):
        unassigned_indices = np.flatnonzero(~assigned)
        if len(unassigned_indices) == 0:
            break

        camera = sync_data["cameras"][image_pose.camera_id]
        heatmap_path = sync_data["heatmap_matches"].get(image_pose.name)
        if heatmap_path is None:
            continue

        u, v, valid = project_points(vertices[unassigned_indices], camera, image_pose)
        if not valid.any():
            continue

        with Image.open(heatmap_path) as heatmap_image:
            heatmap_image = heatmap_image.convert("RGB")
            if heatmap_image.size != (camera.width, camera.height):
                heatmap_image = heatmap_image.resize(
                    (camera.width, camera.height), Image.Resampling.BILINEAR
                )
            heatmap_pixels = np.asarray(heatmap_image, dtype=np.float32)

        if not heatmap_only:
            original_path = sync_data["original_path_map"].get(image_pose.name)
            if not original_path or not original_path.exists():
                raise FileNotFoundError(
                    f"Missing original image for blending {image_pose.name}. "
                    "Use --heatmap-only to color with heatmaps directly."
                )
            with Image.open(original_path) as base_image:
                base_image = base_image.convert("RGB")
                if base_image.size != (camera.width, camera.height):
                    base_image = base_image.resize(
                        (camera.width, camera.height), Image.Resampling.BILINEAR
                    )
                base_pixels = np.asarray(base_image, dtype=np.float32)

        target_indices = unassigned_indices[valid]
        pixel_x = np.clip(np.rint(u[valid]).astype(int), 0, camera.width - 1)
        pixel_y = np.clip(np.rint(v[valid]).astype(int), 0, camera.height - 1)
        heatmap_colors[target_indices] = heatmap_pixels[pixel_y, pixel_x]
        if heatmap_only:
            base_colors[target_indices] = heatmap_pixels[pixel_y, pixel_x]
        else:
            base_colors[target_indices] = base_pixels[pixel_y, pixel_x]
        assigned[target_indices] = True

    heatmap_colors = blur_colors_spatial(
        vertices,
        heatmap_colors,
        assigned,
        heatmap_global_blur_fraction,
        heatmap_global_blur_strength,
        heatmap_global_blur_grid_divisions,
    )
    heatmap_colors = smooth_colors_on_mesh(
        heatmap_colors,
        faces,
        assigned,
        heatmap_smooth_iterations,
        heatmap_smooth_strength,
    )
    if heatmap_only:
        colors = heatmap_colors
    else:
        colors = base_colors * (1.0 - heatmap_alpha) + heatmap_colors * heatmap_alpha

    output_path = (
        Path(output_path)
        if output_path is not None
        else colmap_output / "heatmapped_vertex_colors.ply"
    )
    write_colored_ply(output_path, vertices, faces, np.clip(colors, 0, 255).astype(np.uint8))
    return sync_data, output_path, int(assigned.sum()), len(vertices)


def build_heatmap_workspace(
    sync_data,
    colmap_output,
    workspace_path,
    heatmap_alpha=DEFAULT_HEATMAP_ALPHA,
    heatmap_only=False,
):
    """Create a COLMAP texturing workspace with heatmaps staged as images."""
    sparse_src = colmap_output / "sparse"
    sparse_dst = workspace_path / "sparse"
    images_dst = workspace_path / "images"

    workspace_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(sparse_src, sparse_dst, dirs_exist_ok=True)
    filter_sparse_images_txt(sparse_dst, sync_data["heatmap_matches"])
    images_dst.mkdir(parents=True, exist_ok=True)

    for image_name, heatmap_path in sync_data["heatmap_matches"].items():
        staged_path = images_dst / image_name
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        original_path = sync_data["original_path_map"].get(image_name)
        if heatmap_only:
            shutil.copy2(heatmap_path, staged_path)
        elif original_path and original_path.exists():
            blend_images(original_path, heatmap_path, staged_path, heatmap_alpha)
        else:
            raise FileNotFoundError(
                f"Missing original image for blending {image_name}. "
                "Use --heatmap-only to texture with heatmaps directly."
            )


def run_mesh_texturer(
    workspace_path,
    mesh_path,
    output_dir,
):
    """Run COLMAP mesh_texturer on the staged heatmap workspace."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap_command(),
        "mesh_texturer",
        "--workspace_path",
        str(workspace_path),
        "--input_path",
        str(mesh_path),
        "--output_path",
        str(output_dir),
        "--MeshTextureMapping.apply_color_correction",
        "0",
    ]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "COLMAP command not found. Install COLMAP or place it at "
            f"{LOCAL_COLMAP}."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"COLMAP mesh_texturer failed: {error}") from error


def apply_heatmaps_to_mesh(
    colmap_output,
    heatmap_folder,
    output_dir=None,
    mesh_path=None,
    keep_workspace=False,
    heatmap_alpha=DEFAULT_HEATMAP_ALPHA,
    heatmap_only=False,
    warn_on_partial=True,
):
    """Validate heatmaps and texture the mesh with COLMAP mesh_texturer."""
    sync_data = load_sync_data(colmap_output, heatmap_folder, mesh_path)
    require_heatmap_matches(sync_data)
    if warn_on_partial:
        warn_partial_heatmap_matches(sync_data)

    output_dir = Path(output_dir) if output_dir is not None else colmap_output / "heatmapped_mesh"
    temp_root = Path(tempfile.mkdtemp(prefix="heatmap_texturing_"))
    workspace_path = temp_root / "workspace"

    try:
        build_heatmap_workspace(
            sync_data,
            colmap_output,
            workspace_path,
            heatmap_alpha=heatmap_alpha,
            heatmap_only=heatmap_only,
        )
        run_mesh_texturer(
            workspace_path,
            sync_data["geometry_path"],
            output_dir,
        )
    finally:
        if keep_workspace:
            print(f"Kept temporary workspace: {workspace_path}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    mesh_output = output_dir / "mesh.ply"
    texture_output = output_dir / "texture.png"
    if not mesh_output.exists() or not texture_output.exists():
        raise RuntimeError(
            "COLMAP mesh_texturer finished, but expected outputs were not found: "
            f"{mesh_output}, {texture_output}"
        )

    return sync_data, mesh_output, texture_output


def project_heatmaps(
    reconstruction,
    heatmap_dir,
    method="vertex-colors",
    output_path=None,
    output_dir=None,
    mesh_path=None,
    keep_workspace=False,
    heatmap_alpha=DEFAULT_HEATMAP_ALPHA,
    heatmap_only=False,
    heatmap_global_blur_fraction=DEFAULT_HEATMAP_GLOBAL_BLUR_FRACTION,
    heatmap_global_blur_strength=DEFAULT_HEATMAP_GLOBAL_BLUR_STRENGTH,
    heatmap_global_blur_grid_divisions=DEFAULT_HEATMAP_GLOBAL_BLUR_GRID_DIVISIONS,
    heatmap_smooth_iterations=DEFAULT_HEATMAP_SMOOTH_ITERATIONS,
    heatmap_smooth_strength=DEFAULT_HEATMAP_SMOOTH_STRENGTH,
    warn_on_partial=True,
):
    """Project heatmaps onto a reconstructed mesh and return structured outputs."""
    if method not in {"vertex-colors", "mesh-texturer"}:
        raise ValueError("method must be 'vertex-colors' or 'mesh-texturer'")

    if not isinstance(reconstruction, ReconstructionResult):
        raise TypeError("reconstruction must be a ReconstructionResult")

    heatmap_dir = Path(heatmap_dir)
    geometry_path = Path(mesh_path) if mesh_path is not None else reconstruction.mesh_path

    if method == "vertex-colors":
        sync_data, mesh_output, assigned_count, vertex_count = apply_heatmaps_as_vertex_colors(
            reconstruction.output_dir,
            heatmap_dir,
            output_path=output_path,
            mesh_path=geometry_path,
            heatmap_alpha=heatmap_alpha,
            heatmap_only=heatmap_only,
            heatmap_global_blur_fraction=heatmap_global_blur_fraction,
            heatmap_global_blur_strength=heatmap_global_blur_strength,
            heatmap_global_blur_grid_divisions=heatmap_global_blur_grid_divisions,
            heatmap_smooth_iterations=heatmap_smooth_iterations,
            heatmap_smooth_strength=heatmap_smooth_strength,
            warn_on_partial=warn_on_partial,
        )
        return HeatmapProjectionResult(
            reconstruction=reconstruction,
            heatmap_dir=heatmap_dir,
            method=method,
            output_mesh_path=mesh_output,
            texture_path=None,
            matched_images=tuple(sync_data["matched_names"]),
            missing_heatmaps=tuple(sync_data["missing_heatmaps"]),
            extra_heatmaps=tuple(sync_data["extra_heatmaps"]),
            assigned_vertices=assigned_count,
            total_vertices=vertex_count,
        )

    sync_data, mesh_output, texture_output = apply_heatmaps_to_mesh(
        reconstruction.output_dir,
        heatmap_dir,
        output_dir=output_dir,
        mesh_path=geometry_path,
        keep_workspace=keep_workspace,
        heatmap_alpha=heatmap_alpha,
        heatmap_only=heatmap_only,
        warn_on_partial=warn_on_partial,
    )
    return HeatmapProjectionResult(
        reconstruction=reconstruction,
        heatmap_dir=heatmap_dir,
        method=method,
        output_mesh_path=mesh_output,
        texture_path=texture_output,
        matched_images=tuple(sync_data["matched_names"]),
        missing_heatmaps=tuple(sync_data["missing_heatmaps"]),
        extra_heatmaps=tuple(sync_data["extra_heatmaps"]),
        assigned_vertices=None,
        total_vertices=None,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Project heatmap images onto a COLMAP mesh."
    )
    parser.add_argument("colmap_output", help="Path to a completed COLMAP output folder")
    parser.add_argument("heatmap_folder", help="Path to heatmap images matching COLMAP names")
    parser.add_argument(
        "--output-dir",
        help="Output directory for heatmapped mesh files. Defaults to colmap_output/heatmapped_mesh",
    )
    parser.add_argument(
        "--output-ply",
        help=(
            "Output PLY for --method vertex-colors. Defaults to "
            "colmap_output/heatmapped_vertex_colors.ply"
        ),
    )
    parser.add_argument(
        "--mesh",
        help="Mesh file to texture. Defaults to colmap_output/dense.ply",
    )
    parser.add_argument(
        "--method",
        choices=("vertex-colors", "mesh-texturer"),
        default="vertex-colors",
        help=(
            "Projection method. 'vertex-colors' is pure Python and picks the latest "
            "valid heatmap per vertex. 'mesh-texturer' uses COLMAP UV texture baking. "
            "Defaults to vertex-colors."
        ),
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep the temporary COLMAP texturing workspace for debugging",
    )
    parser.add_argument(
        "--heatmap-alpha",
        type=float,
        default=DEFAULT_HEATMAP_ALPHA,
        help="Heatmap blend amount from 0.0 to 1.0. Defaults to 0.4",
    )
    parser.add_argument(
        "--heatmap-only",
        action="store_true",
        help="Use heatmap images directly instead of blending over original images",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate heatmap inputs; do not run COLMAP mesh_texturer",
    )
    args = parser.parse_args()
    if not 0.0 <= args.heatmap_alpha <= 1.0:
        print("Error: --heatmap-alpha must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    colmap_output = Path(args.colmap_output)
    heatmap_folder = Path(args.heatmap_folder)
    reconstruction = ReconstructionResult.from_output_dir(
        colmap_output,
        mesh_path=args.mesh,
    )

    try:
        sync_data = load_sync_data(colmap_output, heatmap_folder, args.mesh)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    missing_heatmaps = sync_data["missing_heatmaps"]
    extra_heatmaps = sync_data["extra_heatmaps"]

    print("Heatmap sync data loaded")
    print(f"COLMAP output: {colmap_output}")
    print(f"Heatmap folder: {heatmap_folder}")
    print(f"Geometry input: {sync_data['geometry_path']}")
    if sync_data["image_name_map_path"]:
        print(f"Image name map: {sync_data['image_name_map_path']}")
        print(f"Mapped image names: {len(sync_data['image_name_map'])}")
    else:
        print("Image name map: not found, matching heatmaps directly to COLMAP names")
    print(f"Cameras: {len(sync_data['cameras'])}")
    print(f"COLMAP images: {len(sync_data['images'])}")
    print(f"Heatmap images: {len(sync_data['heatmaps'])}")
    print(f"Matched images: {len(sync_data['matched_names'])}")
    print(f"Missing heatmaps: {len(missing_heatmaps)}")
    print(f"Extra heatmaps: {len(extra_heatmaps)}")

    print_name_list("Missing heatmap files", missing_heatmaps)
    print_name_list("Extra heatmap files", extra_heatmaps)

    try:
        require_heatmap_matches(sync_data)
    except ValueError as error:
        if not sync_data["image_name_map_path"]:
            print(
                "Hint: If COLMAP renamed your source images, rerun colmap-orchestrate "
                "to create image_name_map.csv in the output folder.",
                file=sys.stderr,
            )
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    if missing_heatmaps:
        if not sync_data["image_name_map_path"]:
            print(
                "Hint: If COLMAP renamed your source images, rerun colmap-orchestrate "
                "to create image_name_map.csv in the output folder.",
                file=sys.stderr,
            )
        warn_partial_heatmap_matches(sync_data)

    if args.validate_only:
        return

    if args.method == "vertex-colors":
        output_ply = Path(args.output_ply) if args.output_ply else None
        print("Projecting latest valid heatmaps onto mesh vertices...")
        print("Method: pure Python vertex colors, no COLMAP binary changes needed")
        if DEFAULT_HEATMAP_GLOBAL_BLUR_FRACTION:
            print(
                "Global heatmap blur: "
                f"{DEFAULT_HEATMAP_GLOBAL_BLUR_FRACTION:.3f} scene fraction, "
                f"strength {DEFAULT_HEATMAP_GLOBAL_BLUR_STRENGTH}, "
                f"grid divisions {DEFAULT_HEATMAP_GLOBAL_BLUR_GRID_DIVISIONS}"
            )
        if DEFAULT_HEATMAP_SMOOTH_ITERATIONS:
            print(
                "Heatmap smoothing: "
                f"{DEFAULT_HEATMAP_SMOOTH_ITERATIONS} iteration(s), "
                f"strength {DEFAULT_HEATMAP_SMOOTH_STRENGTH}"
            )
        try:
            result = project_heatmaps(
                reconstruction,
                heatmap_folder,
                method="vertex-colors",
                output_path=output_ply,
                mesh_path=args.mesh,
                heatmap_alpha=args.heatmap_alpha,
                heatmap_only=args.heatmap_only,
                warn_on_partial=False,
            )
        except (FileNotFoundError, RuntimeError, ValueError, ImportError) as error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)

        print("Heatmap vertex coloring complete")
        print(f"Mesh output: {result.output_mesh_path}")
        print(f"Colored vertices: {result.assigned_vertices} / {result.total_vertices}")
        if result.assigned_vertices < result.total_vertices:
            print("Note: unprojected vertices were left neutral gray")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    print("Building heatmap texture workspace and running COLMAP mesh_texturer...")
    if args.heatmap_only:
        print("Texture source: heatmaps only")
    else:
        print(f"Texture source: original images blended with heatmaps at alpha {args.heatmap_alpha}")
    print("View selection: COLMAP best projected-area image per face")
    try:
        result = project_heatmaps(
            reconstruction,
            heatmap_folder,
            method="mesh-texturer",
            output_dir=output_dir,
            mesh_path=args.mesh,
            keep_workspace=args.keep_workspace,
            heatmap_alpha=args.heatmap_alpha,
            heatmap_only=args.heatmap_only,
            warn_on_partial=False,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    print("Heatmap texturing complete")
    print(f"Mesh output: {result.output_mesh_path}")
    print(f"Texture output: {result.texture_path}")


if __name__ == "__main__":
    main()
