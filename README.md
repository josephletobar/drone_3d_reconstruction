# COLMAP CUDA Reconstruction Orchestrator

Automated pipeline for converting one prerecorded video or one folder of image
frames into a 3D mesh using GPU-accelerated COLMAP reconstruction.

## Setup

Create the Python/ffmpeg conda environment:

```powershell
conda env create -f environment.yml
conda activate colmap-reconstruction
python -m pip install -e .
```

If the environment already exists and `python -m pip` is missing, install pip
once:

```powershell
conda install -n colmap-reconstruction pip
conda activate colmap-reconstruction
python -m pip install -e .
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
colmap-orchestrate C:\path\to\video.mp4 C:\path\to\output
# OR option 2: image-frame folder input
colmap-orchestrate C:\path\to\image_frames C:\path\to\output
```

The input must be either:

- A single video file
- A folder containing image frames

Supported video extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.flv`, `.wmv`.

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`.

## Python Module

```python
from colmap_reconstruction import project_heatmaps, reconstruct

reconstruction = reconstruct(
    input_path="/path/to/video.mp4",  # Or "/path/to/image_frames"
    output_dir="/path/to/output",
    skip_frames=0,
    max_image_size=640,
)

heatmapped = project_heatmaps(
    reconstruction=reconstruction,
    heatmap_dir="/path/to/heatmaps",
)

print(reconstruction.mesh_path)
print(heatmapped.output_mesh_path)
```

Python package dependencies are declared in `pyproject.toml`. COLMAP CUDA and
ffmpeg are still runtime requirements.

## Options

| Flag | Purpose | Default |
|------|---------|---------|
| `--skip-frames N` | For video input, sample every Nth frame | `0` = all frames |
| `--max-image-size N` | Resize prepared images so the longest edge is at most N pixels | `640` |
| `--georef-csv PATH` | Georeference image-folder reconstructions from `name,easting,northing,altitude` CSV | disabled |
| `--georef-alignment-max-error N` | Maximum COLMAP alignment error in CSV coordinate units | `50` |

For image-frame folder input, `--skip-frames N` samples every Nth image.
`--max-image-size N` preserves aspect ratio for landscape, portrait, and square
inputs by scaling the longest edge.

> **Note:** Recommend starting with `--skip-frames` at 0 for best results.
> Increase from there based on results to speed up processing or if your video
> has redundant high framerate footage.
> Likewise, start with the default `--max-image-size 640`; increase to 768 or
> 1024 only if the reconstruction is too sparse or loses detail.

## Georeferenced Reconstruction

For image-frame folders with a `query.csv`-style metadata file, pass
`--georef-csv` to align the sparse reconstruction before dense reconstruction:

```powershell
colmap-orchestrate C:\path\to\image_frames C:\path\to\output --georef-csv C:\path\to\query.csv
```

The CSV must include `name`, `easting`, `northing`, and `altitude` columns.
Image names are matched through `image_name_map.csv`, so the final dense cloud,
mesh, and saved sparse model are written in the CSV coordinate frame.

## Output

- `dense.ply` - Triangulated 3D mesh generated from dense reconstruction

Pipeline: video/images -> sparse SfM -> dense stereo -> Poisson mesh.

Intermediate files are automatically cleaned up.

Open the `.ply` file in CloudCompare, MeshLab, or another PLY viewer.

## Interactive Heatmap Painting

Open any PLY in the desktop painter:

```powershell
colmap-paint-heatmap C:\path\to\mesh.ply
```

The painter always treats the colors already in the PLY as the base, whether
they came from reconstruction, projected heatmaps, or earlier painting. Press
Use the on-screen heat slider to choose a color from the OpenCV JET scale, then
press `P` to paint. `X` erases toward the colors present when the file was
opened, and `N` navigates the camera. Left-drag applies the selected brush. Use
`-` and `=` for brush size, `U` to undo the last stroke, `C` to clear all
current-session painting, and `S` to save a new PLY copy.
The brush has a feathered radial edge, and repeated strokes are capped at 55%
overlay opacity so the mesh imagery remains visible. Override that ceiling with
`--max-overlay-opacity` when launching the painter.
The input file is never overwritten automatically.

From Python:

```python
from colmap_reconstruction import paint_heatmap

paint_heatmap("/path/to/mesh.ply")
```

**Example:**
<img width="1285" height="809" alt="image" src="https://github.com/user-attachments/assets/9e4aa971-a5f4-4950-8b0e-e6775a34a076" />

## Heatmap Projection

After installing editable, project heatmaps onto a completed COLMAP output:

```powershell
colmap-heatmaps C:\path\to\colmap_output C:\path\to\heatmaps
```

From Python, pass the reconstruction result into heatmap projection:

```python
from colmap_reconstruction import project_heatmaps

heatmap_result = project_heatmaps(reconstruction, "/path/to/heatmaps")
print(heatmap_result.output_mesh_path)
```

## Object Pins

After a georeferenced reconstruction, project object nodes from a saved
`graph.db` onto the mesh:

```powershell
colmap-object-pins C:\path\to\colmap_output C:\path\to\graph.db
```

This reads the base `nodes` table and writes outputs under
`colmap_output\object_pins`:

- `object_pins.csv` - graph node `x/y` positions with nearest-mesh height plus a small vertical offset
- `object_pins_on_mesh.ply` - the heatmapped mesh, when available, with colored object pins appended into the same PLY
- `leveled_reconstruction.ply` - the reconstruction leveled by correcting `z` while preserving geospatial `x/y`
- `object_pins_on_mesh_leveled.ply` - the leveled reconstruction with object pins placed using the same `geo_pos_x/y` logic
- `object_pins_level_transform.txt` - the vertical leveling transform applied to the reconstruction

From Python:

```python
from colmap_reconstruction import project_object_pins

pin_result = project_object_pins(reconstruction, "/path/to/graph.db")
print(pin_result.output_mesh_path)
```
