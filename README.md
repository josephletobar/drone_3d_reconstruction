# COLMAP CUDA Reconstruction Orchestrator

Automated pipeline for converting one prerecorded video or one folder of image
frames into a 3D mesh using GPU-accelerated COLMAP reconstruction.

## Setup

Create the Python/ffmpeg conda environment:

```powershell
conda env create -f environment.yml
conda activate colmap-reconstruction
```

Install the official CUDA COLMAP Windows bundle:

1. Download `colmap-x64-windows-cuda.zip` from the COLMAP GitHub releases page.
2. Extract it to `tools/colmap`.
3. Verify the install:

```powershell
.\tools\colmap\COLMAP.bat -h
```

The first line should include `with CUDA`.

## Usage

Run from this project folder:

```powershell
conda activate colmap-reconstruction
# Option 1: video input
python orchestrate.py C:\path\to\video.mp4 C:\path\to\output
# OR option 2: image-frame folder input
python orchestrate.py C:\path\to\image_frames C:\path\to\output
```

The input must be either:

- A single video file
- A folder containing image frames

Supported video extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.flv`, `.wmv`.

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`.

## Python Module

```python
from orchestrate import orchestrate

mesh_path = orchestrate(
    input_path="/path/to/video.mp4",  # Or "/path/to/image_frames"
    output_folder="/path/to/output",
    skip_frames=0,
)
```

`requirements.txt` lists only Python package dependencies for apps that import
this module. COLMAP CUDA and ffmpeg are still runtime requirements.

## Options

| Flag | Purpose | Default |
|------|---------|---------|
| `--skip-frames N` | For video input, sample every Nth frame | `0` = all frames |

For image-frame folder input, `--skip-frames N` samples every Nth image.

> **Note:** Recommend starting with `--skip-frames` at 0 for best results.
> Increase from there based on results to speed up processing or if your video
> has redundant high framerate footage.

## Output

- `dense.ply` - Triangulated 3D mesh generated from dense reconstruction

Pipeline: video/images -> sparse SfM -> dense stereo -> Poisson mesh.

Intermediate files are automatically cleaned up.

Open the `.ply` file in CloudCompare, MeshLab, or another PLY viewer.

**Example:**
<img width="2570" height="1618" alt="image" src="https://github.com/user-attachments/assets/9e4aa971-a5f4-4950-8b0e-e6775a34a076" />

