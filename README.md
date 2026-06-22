# COLMAP Video/Frames-to-Mesh Orchestrator

Automated pipeline for converting one prerecorded video or one folder of image
frames into a 3D mesh using COLMAP sparse/dense reconstruction + Poisson meshing.

## Usage

### Docker
```bash
docker build -t colmap-orchestrator .
docker run --volume /path/to/video.mp4:/input/video.mp4 --volume /path/to/output:/output colmap-orchestrator /input/video.mp4 /output
docker run --volume /path/to/image_frames:/input/frames --volume /path/to/output:/output colmap-orchestrator /input/frames /output
```

### Standalone (if dependencies installed)
```bash
python3 orchestrate.py /path/to/video.mp4 /path/to/output
python3 orchestrate.py /path/to/image_frames /path/to/output
```

### Conda Environment
```powershell
conda env create -f environment.yml
conda activate colmap-reconstruction
python orchestrate.py C:\path\to\video.mp4 C:\path\to\output
python orchestrate.py C:\path\to\image_frames C:\path\to\output
```

The conda environment includes Python, `tqdm`, and `ffmpeg`.

On Windows, use the official COLMAP release ZIP instead of the conda COLMAP
package. For GPU acceleration, download `colmap-x64-windows-cuda.zip` from the
COLMAP GitHub releases page and extract it to `tools/colmap`, so this file
exists:

```text
tools/colmap/COLMAP.bat
```

The script uses `tools/colmap/COLMAP.bat` automatically when it is present. This
batch file sets the required COLMAP library paths before launching the CLI.
Verify the local COLMAP install with:

```powershell
.\tools\colmap\COLMAP.bat -h
```

The first line should include `with CUDA` for GPU support.

The script enables GPU acceleration for feature extraction, feature matching,
global mapping/bundle adjustment, and PatchMatch stereo.

`requirements.txt` only lists Python package dependencies. It is useful when
another Python app imports this module and already provides `ffmpeg` and
`colmap` separately.

### As Python Module
```python
from orchestrate import orchestrate

mesh_path = orchestrate(
    input_path="/path/to/video.mp4",  # Or "/path/to/image_frames"
    output_folder="/path/to/output",
    skip_frames=0  # Video input only: 0 = all frames, N = sample at 1/N fps
)
```

## Input

The input must be exactly one of:

- A single video file
- A folder containing image frames

Supported video extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.flv`, `.wmv`.

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`.

## Options

- `--skip-frames N` - For video input only, sample at 1/N fps (default: 0 = extract all frames)

## Output

- `dense.ply` - Triangulated 3D mesh (generated from dense point cloud via Poisson reconstruction)

**Pipeline:** Video/images → Sparse SfM → Dense stereo matching → Poisson mesh

Intermediate files are automatically cleaned up.

## View Results

Open the `.ply` file in CloudCompare, MeshLab, or any PLY viewer.
