#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import shutil
import tempfile
from PIL import Image

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_COLMAP = PROJECT_ROOT / "tools" / "colmap" / "COLMAP.bat"
DEFAULT_MAX_IMAGE_SIZE = 640
PATCH_MATCH_STEREO_NUM_ITERATIONS = 2
DEFAULT_GEOREF_ALIGNMENT_MAX_ERROR = 50.0


@dataclass(frozen=True)
class ReconstructionResult:
    """Paths and metadata produced by a COLMAP reconstruction run."""

    input_path: Path | None
    output_dir: Path
    mesh_path: Path
    sparse_dir: Path
    dense_dir: Path
    image_name_map_path: Path | None
    artifact_paths: tuple[Path, ...]
    frame_count: int
    dense: bool
    georeferenced: bool = False
    georef_csv_path: Path | None = None
    georef_reference_path: Path | None = None
    georef_transform_path: Path | None = None

    @classmethod
    def from_output_dir(
        cls,
        output_dir,
        input_path=None,
        mesh_path=None,
        dense=True,
        artifact_paths=(),
        frame_count=0,
        georeferenced=False,
        georef_csv_path=None,
    ):
        """Build a result object for an existing reconstruction output folder."""
        output_dir = Path(output_dir)
        image_name_map_path = output_dir / "image_name_map.csv"
        georef_reference_path = output_dir / "georef_reference.txt"
        georef_transform_path = output_dir / "georef_transform.txt"
        return cls(
            input_path=Path(input_path) if input_path is not None else None,
            output_dir=output_dir,
            mesh_path=Path(mesh_path) if mesh_path is not None else output_dir / "dense.ply",
            sparse_dir=output_dir / "sparse",
            dense_dir=output_dir / "dense",
            image_name_map_path=(
                image_name_map_path if image_name_map_path.exists() else None
            ),
            artifact_paths=tuple(Path(path) for path in artifact_paths),
            frame_count=frame_count,
            dense=dense,
            georeferenced=georeferenced or georef_reference_path.exists(),
            georef_csv_path=Path(georef_csv_path) if georef_csv_path is not None else None,
            georef_reference_path=(
                georef_reference_path if georef_reference_path.exists() else None
            ),
            georef_transform_path=(
                georef_transform_path if georef_transform_path.exists() else None
            ),
        )


def colmap_command():
    """Return the preferred COLMAP command."""
    custom_colmap = os.environ.get("COLMAP_EXE")
    if custom_colmap:
        return custom_colmap
    if LOCAL_COLMAP.exists():
        return str(LOCAL_COLMAP)
    return "colmap"


def format_duration(seconds):
    """Format elapsed seconds as h:mm:ss or m:ss."""
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def get_media_files(folder, extensions):
    """Get media files from folder matching the provided extensions."""
    return sorted(
        path
        for path in Path(folder).iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in extensions
    )


def get_image_files(folder):
    """Get all image files from folder."""
    return get_media_files(folder, IMAGE_EXTENSIONS)


def get_video_files(folder):
    """Get all video files from folder."""
    return get_media_files(folder, VIDEO_EXTENSIONS)


def stage_images(
    image_files,
    output_dir,
    skip_frames=0,
    max_image_size=DEFAULT_MAX_IMAGE_SIZE,
    manifest_path=None,
):
    """Copy input images into the COLMAP frames directory, downscaling if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)

    step = skip_frames if skip_frames > 0 else 1
    selected_images = image_files[::step]
    manifest_rows = []

    for index, image_file in enumerate(tqdm(selected_images, desc="Staging images"), start=1):
        output_name = f"image_{index:06d}{image_file.suffix.lower()}"
        output_path = output_dir / output_name

        img = Image.open(image_file)
        longest_edge = max(img.width, img.height)
        if longest_edge > max_image_size:
            ratio = max_image_size / longest_edge
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img.save(output_path, quality=95)

        manifest_rows.append({
            "original_name": image_file.name,
            "original_path": str(image_file),
            "staged_name": output_name,
            "staged_path": str(output_path),
        })

    if manifest_path is not None:
        write_image_name_map(manifest_path, manifest_rows)

    return len(selected_images)


def ffmpeg_max_edge_scale_filter(max_image_size):
    """Return an ffmpeg scale filter that preserves aspect ratio by longest edge."""
    return (
        f"scale='if(gte(iw,ih),min({max_image_size},iw),-2)':"
        f"'if(gte(iw,ih),-2,min({max_image_size},ih))'"
    )


def write_image_name_map(manifest_path, rows):
    """Write the source-to-COLMAP staged image filename map."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["original_name", "original_path", "staged_name", "staged_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def read_image_name_map(manifest_path):
    """Read the source-to-COLMAP staged image filename map."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Image name map not found: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"original_name", "staged_name"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Invalid image name map: {manifest_path} must include "
                "original_name and staged_name columns"
            )
        return list(reader)


def load_georef_positions(georef_csv):
    """Read image camera centers from a query.csv-style georeference file."""
    georef_csv = Path(georef_csv)
    if not georef_csv.exists():
        raise FileNotFoundError(f"Georeference CSV not found: {georef_csv}")

    positions = {}
    lower_positions = {}
    with georef_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"name", "easting", "northing", "altitude"}
        if not required_columns.issubset(reader.fieldnames or []):
            missing = ", ".join(sorted(required_columns - set(reader.fieldnames or [])))
            raise ValueError(
                f"Invalid georeference CSV: {georef_csv} is missing required "
                f"column(s): {missing}"
            )

        for row_number, row in enumerate(reader, start=2):
            image_name = row["name"].strip()
            if not image_name:
                raise ValueError(f"Missing image name in georeference CSV row {row_number}")
            if image_name in positions:
                raise ValueError(
                    f"Duplicate image name in georeference CSV row {row_number}: "
                    f"{image_name}"
                )

            try:
                position = (
                    float(row["easting"]),
                    float(row["northing"]),
                    float(row["altitude"]),
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid georeference coordinate in row {row_number}: {row}"
                ) from error

            positions[image_name] = position
            lower_positions.setdefault(image_name.lower(), position)

    return positions, lower_positions


def build_georef_reference_file(
    georef_csv,
    image_name_map_path,
    reference_path,
    min_matches=3,
):
    """
    Write a COLMAP model_aligner reference file from query.csv and image_name_map.csv.

    The output uses staged COLMAP image names with easting/northing/altitude values:
    image_000001.png 498224.607 4433452.523 530.817
    """
    reference_path = Path(reference_path)
    if reference_path.exists():
        reference_path.unlink()

    positions, lower_positions = load_georef_positions(georef_csv)
    image_name_rows = read_image_name_map(image_name_map_path)

    reference_rows = []
    missing_original_names = []
    for row in image_name_rows:
        original_name = row["original_name"].strip()
        staged_name = row["staged_name"].strip()
        if not original_name or not staged_name:
            continue

        position = positions.get(original_name)
        if position is None:
            position = lower_positions.get(original_name.lower())
        if position is None:
            missing_original_names.append(original_name)
            continue

        reference_rows.append((staged_name, *position))

    if len(reference_rows) < min_matches:
        raise ValueError(
            "Georeferencing requires at least "
            f"{min_matches} images matched between the input folder and CSV, "
            f"but only found {len(reference_rows)}."
        )

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    with reference_path.open("w", encoding="utf-8", newline="\n") as file:
        for staged_name, easting, northing, altitude in reference_rows:
            file.write(
                f"{staged_name} {easting:.17g} {northing:.17g} {altitude:.17g}\n"
            )

    return reference_path, len(reference_rows), missing_original_names


def extract_frames(
    video_file,
    output_dir,
    skip_frames,
    max_image_size=DEFAULT_MAX_IMAGE_SIZE,
):
    """Extract frames from one video using ffmpeg, downscaling if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            "ffmpeg",
            "-i", str(video_file),
            "-q:v", "2",
        ]
        # Build filter chain: fps filter + scale filter
        filters = []
        if skip_frames > 0:
            filters.append(f"fps=1/{skip_frames}")
        filters.append(ffmpeg_max_edge_scale_filter(max_image_size))

        cmd.extend(["-vf", ",".join(filters)])
        cmd.append(str(output_dir / f"{video_file.stem}_%06d.jpg"))

        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract frames from {video_file}: {e.stderr.decode()}")

    return len(list(output_dir.glob(f"{video_file.stem}_*.jpg")))

def run_colmap_with_progress(cmd, step_name):
    """Run COLMAP command and parse progress from stderr."""
    print(f"{step_name}...")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        pbar = None
        last_current = 0
        import re

        for line in proc.stderr:
            # Parse progress patterns like "Processed file [77/817]"
            if "Processed file [" in line:
                match = re.search(r'\[(\d+)/(\d+)\]', line)
                if match:
                    current, total = int(match.group(1)), int(match.group(2))
                    if pbar is None:
                        pbar = tqdm(total=total, desc=step_name, unit="img")
                    # Update by increment
                    if current > last_current:
                        pbar.update(current - last_current)
                        last_current = current

        if pbar:
            pbar.close()

        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{step_name} failed: {e}")


def run_colmap_sparse(frames_dir, workspace_dir):
    """Run COLMAP sparse reconstruction (manual pipeline, no dense)."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    db_path = workspace_dir / "database.db"

    # Feature extraction
    print("Extracting features...")
    cmd = [
        colmap_command(), "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.gpu_index", "0",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Feature extraction failed: {e}")

    # Feature matching (exhaustive)
    print("Matching features...")
    cmd = [
        colmap_command(), "exhaustive_matcher",
        "--database_path", str(db_path),
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.gpu_index", "0",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Feature matching failed: {e}")

    # Sparse reconstruction (global mapper for speed)
    print("Running sparse reconstruction...")
    sparse_dir = workspace_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap_command(), "global_mapper",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse_dir),
        "--GlobalMapper.gp_use_gpu", "1",
        "--GlobalMapper.gp_gpu_index", "0",
        "--GlobalMapper.ba_ceres_use_gpu", "1",
        "--GlobalMapper.ba_ceres_gpu_index", "0",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Sparse reconstruction failed: {e}")

def run_colmap_dense(
    workspace_dir,
    frames_dir,
    max_image_size=DEFAULT_MAX_IMAGE_SIZE,
):
    """Run COLMAP dense reconstruction pipeline."""
    dense_dir = workspace_dir / "dense"
    sparse_model = find_sparse_model_dir(workspace_dir)
    if sparse_model is None:
        raise RuntimeError("Dense reconstruction failed: no sparse model found")

    # Step 1: Image undistortion
    cmd = [
        colmap_command(), "image_undistorter",
        "--image_path", str(frames_dir),
        "--input_path", str(sparse_model),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size),
    ]
    print("Running image undistortion...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Image undistortion failed: {e}")

    # Step 2: Stereo matching
    cmd = [
        colmap_command(), "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.gpu_index", "0",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.num_iterations", str(PATCH_MATCH_STEREO_NUM_ITERATIONS),
    ]
    print("Running stereo matching...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Stereo matching failed: {e}")

    # Step 3: Stereo fusion
    output_ply = dense_dir / "fused.ply"
    cmd = [
        colmap_command(), "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(output_ply),
    ]
    print("Running stereo fusion...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Stereo fusion failed: {e}")

    return output_ply


def run_colmap_model_aligner(
    workspace_dir,
    reference_path,
    transform_path,
    alignment_max_error=DEFAULT_GEOREF_ALIGNMENT_MAX_ERROR,
):
    """Align the sparse model into the custom easting/northing/altitude frame."""
    if alignment_max_error <= 0:
        raise ValueError("Georeference alignment max error must be greater than 0")

    sparse_model_dir = find_sparse_model_dir(workspace_dir)
    if sparse_model_dir is None:
        raise RuntimeError("Georeferencing failed: no sparse model found")

    aligned_model_dir = workspace_dir / "sparse_aligned"
    if aligned_model_dir.exists():
        shutil.rmtree(aligned_model_dir)
    aligned_model_dir.mkdir(parents=True, exist_ok=True)

    transform_path = Path(transform_path)
    transform_path.parent.mkdir(parents=True, exist_ok=True)
    if transform_path.exists():
        transform_path.unlink()

    print("Aligning sparse reconstruction to georeference CSV...")
    cmd = [
        colmap_command(),
        "model_aligner",
        "--input_path",
        str(sparse_model_dir),
        "--output_path",
        str(aligned_model_dir),
        "--ref_images_path",
        str(reference_path),
        "--ref_is_gps",
        "0",
        "--alignment_type",
        "custom",
        "--min_common_images",
        "3",
        "--alignment_max_error",
        str(alignment_max_error),
        "--transform_path",
        str(transform_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Georeferencing failed: {e}")

    if not has_sparse_model_files(aligned_model_dir):
        raise RuntimeError(
            f"Georeferencing failed: aligned model files not found in {aligned_model_dir}"
        )
    if not transform_path.exists():
        raise RuntimeError(
            f"Georeferencing failed: transform file was not written: {transform_path}"
        )

    if sparse_model_dir.exists():
        shutil.rmtree(sparse_model_dir)
    shutil.copytree(aligned_model_dir, sparse_model_dir)
    return sparse_model_dir


def run_colmap_poisson_mesher(point_cloud_path, output_mesh_path, depth=10):
    """Convert point cloud to mesh using COLMAP's Poisson mesher."""
    print("Running Poisson meshing...")
    cmd = [
        colmap_command(), "poisson_mesher",
        "--input_path", str(point_cloud_path),
        "--output_path", str(output_mesh_path),
        "--PoissonMeshing.depth", str(depth),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Poisson meshing failed: {e}")


def has_sparse_model_files(model_dir):
    """Return True when a COLMAP sparse model directory has camera and image files."""
    return (
        (model_dir / "cameras.bin").exists()
        or (model_dir / "cameras.txt").exists()
    ) and (
        (model_dir / "images.bin").exists()
        or (model_dir / "images.txt").exists()
    )


def find_sparse_model_dir(workspace_dir):
    """Find the first sparse model directory COLMAP produced."""
    sparse_root = workspace_dir / "sparse"
    candidates = [sparse_root / "0", sparse_root]

    if sparse_root.exists():
        candidates.extend(
            path
            for path in sorted(sparse_root.iterdir())
            if path.is_dir() and path not in candidates
        )

    for candidate in candidates:
        if candidate.exists() and has_sparse_model_files(candidate):
            return candidate

    return None


def save_colmap_artifacts(workspace_dir, output_folder, verbose=True):
    """Save sparse metadata and dense fusion artifacts before cleanup."""
    sparse_model_dir = find_sparse_model_dir(workspace_dir)
    if sparse_model_dir is None:
        raise RuntimeError("Could not find sparse model files to save")

    saved_paths = []
    output_sparse_dir = output_folder / "sparse"
    output_sparse_dir.mkdir(parents=True, exist_ok=True)

    for artifact in sparse_model_dir.iterdir():
        if artifact.is_file() and artifact.suffix.lower() in {".bin", ".txt"}:
            destination = output_sparse_dir / artifact.name
            shutil.copy2(artifact, destination)
            saved_paths.append(destination)

    # Also write text versions when COLMAP produced binary sparse files.
    if not (output_sparse_dir / "cameras.txt").exists() or not (output_sparse_dir / "images.txt").exists():
        cmd = [
            colmap_command(), "model_converter",
            "--input_path", str(sparse_model_dir),
            "--output_path", str(output_sparse_dir),
            "--output_type", "TXT",
        ]
        try:
            subprocess.run(cmd, check=True)
            saved_paths.extend(
                path
                for path in output_sparse_dir.glob("*.txt")
                if path not in saved_paths
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not export sparse text files: {e}")

    dense_dir = workspace_dir / "dense"
    output_dense_dir = output_folder / "dense"
    for artifact_name in ("fused.ply", "fused.ply.vis"):
        artifact = dense_dir / artifact_name
        if artifact.exists():
            output_dense_dir.mkdir(parents=True, exist_ok=True)
            destination = output_dense_dir / artifact.name
            shutil.copy2(artifact, destination)
            saved_paths.append(destination)

    if verbose:
        print(f"Saved COLMAP artifacts to {output_folder}")

    return saved_paths


def reconstruct(
    input_path,
    output_dir,
    skip_frames=0,
    max_image_size=DEFAULT_MAX_IMAGE_SIZE,
    workspace_dir=None,
    frames_dir=None,
    verbose=True,
    dense=True,
    georef_csv=None,
    georef_alignment_max_error=DEFAULT_GEOREF_ALIGNMENT_MAX_ERROR,
):
    """
    Run full COLMAP pipeline from one video, a video folder, or an image-frame folder.

    Args:
        input_path: Path to a video file, folder of videos, or folder containing image frames
        output_dir: Path where to save reconstruction artifacts
        skip_frames: For video input only, sample at 1/N fps. 0 extracts all frames
        max_image_size: Resize prepared images so the longest edge is at most this many pixels
        workspace_dir: Override temp workspace directory
        frames_dir: Override temp frames directory
        verbose: Print progress messages
        dense: If False, sparse only (default True for dense reconstruction + meshing)
        georef_csv: Optional query.csv-style file with name/easting/northing/altitude
        georef_alignment_max_error: Max model_aligner error in CSV coordinate units

    Returns:
        ReconstructionResult with output paths and run metadata

    Raises:
        RuntimeError: If any step fails
        ValueError: If inputs are invalid
    """
    pipeline_start = time.perf_counter()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    georef_csv_path = Path(georef_csv) if georef_csv is not None else None
    georef_alignment_max_error = float(georef_alignment_max_error)

    temp_root = None
    if workspace_dir is None or frames_dir is None:
        temp_root = Path(tempfile.mkdtemp(prefix="colmap_orchestrate_"))

    if workspace_dir is None:
        workspace_dir = temp_root / "colmap_work"
    else:
        workspace_dir = Path(workspace_dir)

    if frames_dir is None:
        frames_dir = temp_root / "colmap_frames"
    else:
        frames_dir = Path(frames_dir)

    # Validate inputs
    if not input_path.exists():
        raise ValueError(f"Input path '{input_path}' does not exist")
    if georef_csv_path is not None and not georef_csv_path.exists():
        raise ValueError(f"Georeference CSV '{georef_csv_path}' does not exist")
    if georef_csv_path is not None and georef_alignment_max_error <= 0:
        raise ValueError("Georeference alignment max error must be greater than 0")

    output_dir.mkdir(parents=True, exist_ok=True)

    is_video_input = input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS
    is_folder_input = input_path.is_dir()
    videos = get_video_files(input_path) if is_folder_input else []
    images = get_image_files(input_path) if is_folder_input and not videos else []
    is_video_folder_input = bool(videos)
    is_image_folder_input = bool(images)

    if not is_video_input and not is_video_folder_input and not is_image_folder_input:
        video_extensions = ", ".join(sorted(VIDEO_EXTENSIONS))
        image_extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(
            f"Input must be one video file, one folder of videos, or one folder of image frames. "
            f"Supported video extensions: {video_extensions}. "
            f"Supported image extensions: {image_extensions}"
        )

    if georef_csv_path is not None and not is_image_folder_input:
        raise ValueError(
            "Georeferencing with --georef-csv is only supported for image-frame "
            "folder inputs because the CSV name column must match source image files."
        )

    if verbose:
        if is_video_input:
            print(f"Found video input: {input_path}")
        elif is_video_folder_input:
            print(f"Found {len(videos)} video file(s)")
        else:
            print(f"Found {len(images)} image frame(s)")

    try:
        num_frames = 0
        image_name_map_path = output_dir / "image_name_map.csv"
        georeferenced = False
        georef_reference_path = None
        georef_transform_path = None

        if is_image_folder_input:
            if verbose:
                if skip_frames > 0:
                    print(f"Staging input images (every {skip_frames}th image)")
                else:
                    print("Staging input images")
            num_frames += stage_images(
                images,
                frames_dir,
                skip_frames,
                max_image_size=max_image_size,
                manifest_path=image_name_map_path,
            )

        if is_video_input or is_video_folder_input:
            video_inputs = [input_path] if is_video_input else videos
            if verbose:
                if skip_frames > 0:
                    print(f"Extracting video frames at 1/{skip_frames} fps")
                else:
                    print("Extracting all frames from video input")
            for video_file in video_inputs:
                if verbose and is_video_folder_input:
                    print(f"Extracting frames from {video_file.name}")
                num_frames += extract_frames(
                    video_file,
                    frames_dir,
                    skip_frames,
                    max_image_size=max_image_size,
                )

        if verbose:
            print(f"Prepared {num_frames} image(s) for reconstruction")

        if georef_csv_path is not None:
            georef_reference_path = output_dir / "georef_reference.txt"
            georef_transform_path = output_dir / "georef_transform.txt"
            georef_reference_path, matched_count, missing_names = build_georef_reference_file(
                georef_csv_path,
                image_name_map_path,
                georef_reference_path,
            )
            if verbose:
                print(
                    f"Prepared georeference positions for {matched_count} staged image(s)"
                )
                if missing_names:
                    print(
                        "Warning: "
                        f"{len(missing_names)} staged image(s) had no georeference row"
                    )

        # Run COLMAP pipeline
        print("\n--- Starting sparse reconstruction ---")
        run_colmap_sparse(frames_dir, workspace_dir)
        print("Sparse reconstruction complete\n")

        if georef_reference_path is not None:
            print("--- Starting sparse georeferencing ---")
            run_colmap_model_aligner(
                workspace_dir,
                georef_reference_path,
                georef_transform_path,
                alignment_max_error=georef_alignment_max_error,
            )
            georeferenced = True
            print("Sparse georeferencing complete\n")

        # Optional dense reconstruction
        if dense:
            print("--- Starting dense reconstruction ---")
            run_colmap_dense(
                workspace_dir,
                frames_dir,
                max_image_size=max_image_size,
            )
            print("Dense reconstruction complete\n")

        # Find point cloud (sparse or dense)
        if dense:
            # Dense output
            point_cloud_ply = workspace_dir / "dense" / "fused.ply"
            if not point_cloud_ply.exists():
                raise RuntimeError("Dense reconstruction failed: fused.ply not found")

            # Run Poisson meshing on dense cloud
            final_output = output_dir / "dense.ply"
            run_colmap_poisson_mesher(point_cloud_ply, final_output, depth=10)

            if verbose:
                print(f"Meshing complete\nSuccess! Mesh saved to {final_output}")
        else:
            # Sparse output - convert to PLY
            sparse_model_dir = find_sparse_model_dir(workspace_dir)

            if sparse_model_dir is None:
                raise RuntimeError("Sparse reconstruction failed: no model files found")

            final_output = output_dir / "dense.ply"
            cmd = [
                colmap_command(), "model_converter",
                "--input_path", str(sparse_model_dir),
                "--output_path", str(final_output),
                "--output_type", "PLY",
            ]
            print("Converting sparse model to PLY...")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Model conversion failed: {e}")

            if verbose:
                print(f"Success! Point cloud saved to {final_output}")

        artifact_paths = list(save_colmap_artifacts(workspace_dir, output_dir, verbose))
        for georef_artifact in (georef_reference_path, georef_transform_path):
            if georef_artifact is not None and Path(georef_artifact).exists():
                artifact_paths.append(Path(georef_artifact))

        return ReconstructionResult(
            input_path=input_path,
            output_dir=output_dir,
            mesh_path=final_output,
            sparse_dir=output_dir / "sparse",
            dense_dir=output_dir / "dense",
            image_name_map_path=(
                image_name_map_path if image_name_map_path.exists() else None
            ),
            artifact_paths=tuple(artifact_paths),
            frame_count=num_frames,
            dense=dense,
            georeferenced=georeferenced,
            georef_csv_path=georef_csv_path,
            georef_reference_path=(
                georef_reference_path
                if georef_reference_path is not None and Path(georef_reference_path).exists()
                else None
            ),
            georef_transform_path=(
                georef_transform_path
                if georef_transform_path is not None and Path(georef_transform_path).exists()
                else None
            ),
        )

    except Exception as e:
        raise RuntimeError(f"Pipeline failed: {e}") from e

    finally:
        elapsed = format_duration(time.perf_counter() - pipeline_start)
        if verbose:
            print(f"Pipeline elapsed time: {elapsed}")

        # Cleanup
        if verbose:
            print("Cleaning up temporary files...")
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root)


def orchestrate(
    input_path,
    output_folder,
    skip_frames=0,
    max_image_size=DEFAULT_MAX_IMAGE_SIZE,
    workspace_dir=None,
    frames_dir=None,
    verbose=True,
    dense=True,
    georef_csv=None,
    georef_alignment_max_error=DEFAULT_GEOREF_ALIGNMENT_MAX_ERROR,
):
    """
    Compatibility wrapper for the older path-returning API.

    Prefer reconstruct(...) for new Python code.
    """
    result = reconstruct(
        input_path,
        output_folder,
        skip_frames=skip_frames,
        max_image_size=max_image_size,
        workspace_dir=workspace_dir,
        frames_dir=frames_dir,
        verbose=verbose,
        dense=dense,
        georef_csv=georef_csv,
        georef_alignment_max_error=georef_alignment_max_error,
    )
    return result.mesh_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="COLMAP video/video-folder/file-frames-to-dense-cloud orchestrator"
    )
    parser.add_argument(
        "input_path",
        help="Path to one video file, one folder of videos, or one folder of image frames",
    )
    parser.add_argument("output_folder", help="Path where to save dense.ply")
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="For video input only, sample at 1/N fps. 0 extracts all frames",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=DEFAULT_MAX_IMAGE_SIZE,
        help=(
            "Resize prepared images so the longest edge is at most this many "
            f"pixels. Defaults to {DEFAULT_MAX_IMAGE_SIZE}"
        ),
    )
    parser.add_argument(
        "--georef-csv",
        "--gps-csv",
        dest="georef_csv",
        help=(
            "Optional query.csv-style file with name,easting,northing,altitude "
            "columns. Supported for image-frame folder inputs only."
        ),
    )
    parser.add_argument(
        "--georef-alignment-max-error",
        type=float,
        default=DEFAULT_GEOREF_ALIGNMENT_MAX_ERROR,
        help=(
            "Maximum COLMAP model_aligner error in CSV coordinate units. "
            f"Defaults to {DEFAULT_GEOREF_ALIGNMENT_MAX_ERROR:g}."
        ),
    )
    args = parser.parse_args()

    try:
        result = reconstruct(
            args.input_path,
            args.output_folder,
            skip_frames=args.skip_frames,
            max_image_size=args.max_image_size,
            georef_csv=args.georef_csv,
            georef_alignment_max_error=args.georef_alignment_max_error,
        )
        print(f"Mesh output: {result.mesh_path}")
        if result.georeferenced:
            print(f"Georeference reference: {result.georef_reference_path}")
            print(f"Georeference transform: {result.georef_transform_path}")
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
