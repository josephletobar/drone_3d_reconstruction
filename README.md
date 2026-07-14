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

When Priority Map writes a matching ownership file beside a heatmap image,
vertex-color projection also embeds the exact graph node ID into the output
PLY:

```text
000427.png
000427.nodes.npz   # contains a 2D string array named node_ids
```

The PNG supplies the projected RGB value and `node_ids` is sampled at the same
pixel. Missing ownership files do not block projection; affected vertices are
left unassociated. Malformed ownership files fail with a validation error.

The self-contained PLY stores an integer `node_index` on each vertex (`-1`
means no node) and a `node_label` lookup element containing the exact UTF-8
IDs, such as `buildings_0`. Generic PLY viewers may ignore these custom fields
while continuing to display the mesh normally. The interactive painter and
object-pin/leveling outputs preserve the association data.

## Object Pins

Add pins for object nodes from a saved `graph.db` to a reconstructed mesh that
contains embedded vertex-to-node associations:

```powershell
colmap-object-pins C:\path\to\colmap_output C:\path\to\graph.db
```

Add `--view` to open the generated pinned mesh immediately in the custom
PyVista viewer. Human-readable database names appear in translucent labels
above the pins, and the viewer starts in navigation mode; painting is optional.

```powershell
colmap-object-pins C:\path\to\colmap_output C:\path\to\graph.db --view
```

This reads the base `nodes` table and writes outputs under
`colmap_output\object_pins`:

Vertices are grouped by exact node ID and one pin is placed near each group's
3D median. The associations, rather than heatmap coloring or georeferencing,
connect regions of the mesh to database nodes. Generated marker vertices also
receive a `pin_node_index` property so the interactive painter can recolor the
correct pin when its graph node changes.

- `object_pins.csv` - object pin IDs, colors, and 3D positions derived from their associated mesh regions
- `object_pins_on_mesh.ply` - the associated mesh with colored object pins appended into the same PLY
- `leveled_reconstruction.ply` - the reconstruction leveled by correcting `z` while preserving its `x/y` coordinates
- `object_pins_on_mesh_leveled.ply` - the leveled reconstruction with pins placed from the same vertex-to-node associations
- `object_pins_level_transform.txt` - the vertical leveling transform applied to the reconstruction

From Python:

```python
from colmap_reconstruction import project_object_pins

pin_result = project_object_pins(reconstruction, "/path/to/graph.db")
print(pin_result.output_mesh_path)
```
## Interactive Heatmap Painting

For visual-only painting, open any PLY in the desktop painter:

```powershell
colmap-paint-heatmap C:\path\to\mesh.ply
```

To also recolor object pins and update Priority Map, pass a pinned PLY containing
embedded node associations together with its database:

```powershell
colmap-paint-heatmap C:\path\to\object_pins_on_mesh.ply C:\path\to\graph.db
```

The painter always treats the colors already in the PLY as the base, whether
they came from reconstruction, projected heatmaps, or earlier painting. Use
the on-screen heat slider to choose a color from the OpenCV JET scale, then
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
paint_heatmap("/path/to/object_pins_on_mesh.ply", graph_db="/path/to/graph.db")
```

With a database supplied, the painter groups touched surface vertices by their
embedded node ID at the end of each stroke. For every node with a pin, it takes
the median visible RGB, recolors the pin, writes that RGB to the base `nodes`
table, and converts the nearest OpenCV JET color to a `0`-`100` priority score.
`U` restores mesh, pin, and database state for the previous stroke; `C` restores
the database state from when the session opened. Database writes are immediate,
while saving the painted PLY remains explicit. Without a database, painting is
visual only.
