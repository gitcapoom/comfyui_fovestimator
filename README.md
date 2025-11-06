# ComfyUI FOV Estimator

A ComfyUI custom node that estimates the **Field of View (FOV)** and **tilt angle (horizon angle)** of images using computer vision techniques.

## Features

- **FOV Estimation**: Estimates the camera's field of view in degrees by detecting vanishing points
- **Tilt Detection**: Detects the horizon angle and tilt of the camera in degrees
- **Visual Debugging**: Optionally overlays detected lines, vanishing points, and measurements on the output image
- **Configurable Parameters**: Adjust edge detection and line detection thresholds for different image types

## Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/gitcapoom/comfyui_fovestimator.git
   ```

3. Install dependencies:
   ```bash
   cd comfyui_fovestimator
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

## Usage

The node will appear in ComfyUI under the **image/analysis** category as **"FOV & Tilt Estimator"**.

### Inputs

- **image** (required): The input image to analyze
- **edge_threshold_low** (optional, default: 50): Lower threshold for Canny edge detection (0-255)
- **edge_threshold_high** (optional, default: 150): Upper threshold for Canny edge detection (0-255)
- **line_threshold** (optional, default: 100): Threshold for Hough line detection (10-500)
- **visualize** (optional, default: True): Whether to draw detected features on the output image

### Outputs

- **annotated_image**: The input image with visualization overlays (if enabled)
- **fov_degrees**: Estimated field of view in degrees (float)
- **tilt_degrees**: Estimated tilt/horizon angle in degrees (float)
- **info**: Text string with formatted results

### Example Workflow

1. Load an image
2. Connect it to the FOV & Tilt Estimator node
3. View the results:
   - The annotated image shows detected horizon lines (blue) and vanishing points (green)
   - FOV and tilt values are displayed as text overlay and available as outputs

## How It Works

### FOV Estimation
Uses the pinhole camera model and vanishing point analysis:

1. Detects edge lines in the image using Canny edge detection and Hough transform
2. Filters for non-horizontal lines (converging lines from buildings, roads, etc.)
3. Finds intersections between line pairs to identify potential vanishing points
4. Uses RANSAC-like clustering to find the dominant vanishing point (most supported by line intersections)
5. Calculates focal length from the distance between the image center and vanishing point
6. Converts focal length to horizontal FOV using: `hfov = 2 × arctan(width / (2 × focal_length))`
7. Falls back to aspect ratio-based estimation if no good vanishing point is found

**Key Principle:** The distance from the image center (principal point) to a vanishing point created by receding parallel lines approximates the camera's focal length in pixels. This relationship allows accurate FOV estimation from image geometry.

### Tilt Detection
1. Converts image to grayscale
2. Applies Gaussian blur to reduce noise
3. Uses Canny edge detection to find edges
4. Applies Hough line transform to detect horizon lines
5. Filters for near-horizontal lines
6. Calculates where the horizon intersects the center of the frame
7. Converts horizontal FOV to vertical FOV using the image aspect ratio
8. Measures tilt based on horizon position relative to frame center using the estimated vertical FOV:
   - **Tilt = 0°** when horizon is at the center of the frame
   - **Tilt < 0°** when horizon is above center (camera looking up)
   - **Tilt > 0°** when horizon is below center (camera looking down)

## Tips

- **For architecture/buildings**: Default settings work well
- **For outdoor landscapes**: Reduce line_threshold to 50-70
- **For noisy images**: Increase edge_threshold_low to 70-100
- **For subtle edges**: Decrease edge_threshold_high to 100-120

## Limitations

- **FOV estimation** works best on images with clear converging lines (buildings, roads, railroad tracks)
- **Tilt estimation** works best on images with visible horizons
- May be less accurate on images with:
  - Extreme wide-angle or fisheye distortion (non-linear distortion)
  - Very cluttered scenes with few clear lines
  - No clear horizon or parallel lines
  - Mostly organic/natural scenes without geometric features

## Technical Details

- **Computer Vision**: Uses OpenCV for all image processing operations
- **Edge Detection**: Canny edge detection with configurable thresholds
- **Line Detection**: Hough line transform for finding straight lines
- **FOV Estimation**:
  - RANSAC-like clustering for robust vanishing point detection
  - Pinhole camera model for focal length estimation
  - Handles images without clear vanishing points via aspect ratio fallback
- **Tilt Estimation**:
  - Horizon line detection and position analysis
  - Converts horizontal to vertical FOV using aspect ratio
  - Median-based robust estimation resistant to outliers

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
