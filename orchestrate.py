#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import shutil
import tempfile

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
LOCAL_COLMAP = Path(__file__).resolve().parent / "tools" / "colmap" / "COLMAP.bat"


def colmap_command():
    """Return the preferred COLMAP command."""
    if LOCAL_COLMAP.exists():
        return str(LOCAL_COLMAP)
    return "colmap"


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


def stage_images(image_files, output_dir):
    """Copy input images into the COLMAP frames directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, image_file in enumerate(tqdm(image_files, desc="Staging images"), start=1):
        output_name = f"image_{index:06d}{image_file.suffix.lower()}"
        shutil.copy2(image_file, output_dir / output_name)

    return len(image_files)

def extract_frames(video_file, output_dir, skip_frames):
    """Extract frames from one video using ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            "ffmpeg",
            "-i", str(video_file),
            "-q:v", "2",
        ]
        # Only add fps filter if skip_frames is set
        if skip_frames > 0:
            cmd.extend(["-vf", f"fps=1/{skip_frames}"])

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
    """Run COLMAP sparse reconstruction."""
    workspace_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        colmap_command(), "automatic_reconstructor",
        "--workspace_path", str(workspace_dir),
        "--image_path", str(frames_dir),
        "--data_type", "VIDEO",
    ]

    run_colmap_with_progress(cmd, "Sparse reconstruction")

def run_colmap_dense(workspace_dir, frames_dir):
    """Run COLMAP dense reconstruction pipeline."""
    dense_dir = workspace_dir / "dense"
    sparse_model = workspace_dir / "sparse" / "0"

    # Step 1: Image undistortion
    cmd = [
        colmap_command(), "image_undistorter",
        "--image_path", str(frames_dir),
        "--input_path", str(sparse_model),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
        "--max_image_size", "2000",
    ]
    run_colmap_with_progress(cmd, "Undistorting images")

    # Step 2: Stereo matching
    cmd = [
        colmap_command(), "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
    ]
    run_colmap_with_progress(cmd, "Computing depth maps")

    # Step 3: Stereo fusion
    output_ply = dense_dir / "fused.ply"
    cmd = [
        colmap_command(), "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(output_ply),
    ]
    run_colmap_with_progress(cmd, "Fusing depth maps")

    return output_ply

def orchestrate(input_path, output_folder, skip_frames=0, workspace_dir=None, frames_dir=None, verbose=True):
    """
    Run full COLMAP pipeline from one video or one image-frame folder.

    Args:
        input_path: Path to a video file or folder containing image frames
        output_folder: Path where to save dense.ply
        skip_frames: For video input only, sample at 1/N fps. 0 extracts all frames
        workspace_dir: Override temp workspace directory
        frames_dir: Override temp frames directory
        verbose: Print progress messages

    Returns:
        Path to output dense.ply file

    Raises:
        RuntimeError: If any step fails
        ValueError: If inputs are invalid
    """
    input_path = Path(input_path)
    output_folder = Path(output_folder)

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

    output_folder.mkdir(parents=True, exist_ok=True)

    is_video_input = input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS
    is_image_folder_input = input_path.is_dir()

    if not is_video_input and not is_image_folder_input:
        video_extensions = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(
            f"Input must be one video file or one folder of image frames. "
            f"Supported video extensions: {video_extensions}"
        )

    images = get_image_files(input_path) if is_image_folder_input else []
    if is_image_folder_input and not images:
        image_extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(
            f"No image files found in '{input_path}'. "
            f"Supported image extensions: {image_extensions}"
        )

    if verbose:
        if is_video_input:
            print(f"Found video input: {input_path}")
        else:
            print(f"Found {len(images)} image frame(s)")

    try:
        num_frames = 0

        if is_image_folder_input:
            if verbose:
                print("Staging input images")
            num_frames += stage_images(images, frames_dir)

        if is_video_input:
            if verbose:
                if skip_frames > 0:
                    print(f"Extracting video frames at 1/{skip_frames} fps")
                else:
                    print("Extracting all frames from video")
            num_frames += extract_frames(input_path, frames_dir, skip_frames)

        if verbose:
            print(f"Prepared {num_frames} image(s) for reconstruction")

        # Run COLMAP pipeline
        run_colmap_sparse(frames_dir, workspace_dir)
        output_ply = run_colmap_dense(workspace_dir, frames_dir)

        # Copy output
        final_output = output_folder / "dense.ply"
        if verbose:
            print(f"Copying output to {final_output}...")
        shutil.copy(output_ply, final_output)

        if verbose:
            print(f"Success! Dense point cloud saved to {final_output}")

        return final_output

    except Exception as e:
        raise RuntimeError(f"Pipeline failed: {e}") from e

    finally:
        # Cleanup
        if verbose:
            print("Cleaning up temporary files...")
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="COLMAP video/file-frames-to-dense-cloud orchestrator"
    )
    parser.add_argument("input_path", help="Path to one video file or one folder of image frames")
    parser.add_argument("output_folder", help="Path where to save dense.ply")
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="For video input only, sample at 1/N fps. 0 extracts all frames",
    )
    args = parser.parse_args()

    try:
        orchestrate(args.input_path, args.output_folder, args.skip_frames)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
