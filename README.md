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

### Tilt Detection
1. Converts image to grayscale
2. Applies Gaussian blur to reduce noise
3. Uses Canny edge detection to find edges
4. Applies Hough line transform to detect horizon lines
5. Filters for near-horizontal lines
6. Calculates where the horizon intersects the center of the frame
7. Measures tilt based on horizon position relative to frame center:
   - **Tilt = 0°** when horizon is at the center of the frame
   - **Tilt < 0°** when horizon is above center (camera looking up)
   - **Tilt > 0°** when horizon is below center (camera looking down)

### FOV Estimation
1. Detects edge lines in the image
2. Finds intersections between lines to locate vanishing points
3. Calculates the angular separation between vanishing points
4. Estimates FOV based on vanishing point geometry
5. Falls back to aspect ratio-based estimation if vanishing points aren't found

## Tips

- **For architecture/buildings**: Default settings work well
- **For outdoor landscapes**: Reduce line_threshold to 50-70
- **For noisy images**: Increase edge_threshold_low to 70-100
- **For subtle edges**: Decrease edge_threshold_high to 100-120

## Limitations

- Works best on images with clear linear features (buildings, roads, horizons)
- May be less accurate on images with:
  - Extreme wide-angle or fisheye distortion
  - Very cluttered scenes
  - No clear horizon or parallel lines

## Technical Details

- Uses OpenCV for computer vision operations
- Implements Canny edge detection and Hough transforms
- Performs vanishing point analysis for FOV estimation
- Median-based robust estimation for tilt angle

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
