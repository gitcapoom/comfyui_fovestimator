"""
ComfyUI Node for estimating FOV and tilt (horizon angle) of images
"""

import numpy as np
import torch
import cv2
from typing import Tuple, Dict, Any


class FOVEstimatorNode:
    """
    A ComfyUI node that estimates the Field of View (FOV) and tilt angle (horizon angle) of an image.

    Uses computer vision techniques including:
    - Canny edge detection
    - Hough line transform for line detection
    - Vanishing point analysis for FOV estimation
    - Horizon line detection for tilt estimation
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "edge_threshold_low": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
                "edge_threshold_high": ("INT", {
                    "default": 150,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
                "line_threshold": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "display": "number"
                }),
                "visualize": ("BOOLEAN", {
                    "default": True,
                    "display": "checkbox"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("annotated_image", "fov_degrees", "tilt_degrees", "info")
    FUNCTION = "estimate_fov_tilt"
    CATEGORY = "image/analysis"

    def estimate_fov_tilt(
        self,
        image: torch.Tensor,
        edge_threshold_low: int = 50,
        edge_threshold_high: int = 150,
        line_threshold: int = 100,
        visualize: bool = True
    ) -> Tuple[torch.Tensor, float, float, str]:
        """
        Estimate FOV and tilt angle from an image.

        Args:
            image: Input image tensor in ComfyUI format [B, H, W, C]
            edge_threshold_low: Lower threshold for Canny edge detection
            edge_threshold_high: Upper threshold for Canny edge detection
            line_threshold: Threshold for Hough line detection
            visualize: Whether to draw detected lines on the output image

        Returns:
            Tuple of (annotated_image, fov_degrees, tilt_degrees, info_string)
        """
        # Convert from ComfyUI tensor format [B, H, W, C] to numpy
        batch_size = image.shape[0]
        results_images = []
        results_fov = []
        results_tilt = []
        results_info = []

        for i in range(batch_size):
            img_np = image[i].cpu().numpy()

            # Convert from [0, 1] float to [0, 255] uint8
            img_np = (img_np * 255).astype(np.uint8)

            # Estimate FOV first, then use it for tilt calculation
            fov_angle, vanishing_points = self._estimate_fov(
                img_np, edge_threshold_low, edge_threshold_high, line_threshold
            )

            # Estimate tilt using the calculated FOV
            tilt_angle, horizon_lines = self._estimate_tilt(
                img_np, edge_threshold_low, edge_threshold_high, line_threshold, fov_angle
            )

            # Create visualization if requested
            if visualize:
                vis_img = self._visualize_results(
                    img_np.copy(), horizon_lines, vanishing_points, tilt_angle, fov_angle
                )
            else:
                vis_img = img_np

            # Convert back to ComfyUI tensor format
            vis_tensor = torch.from_numpy(vis_img.astype(np.float32) / 255.0)
            results_images.append(vis_tensor)
            results_fov.append(fov_angle)
            results_tilt.append(tilt_angle)

            # Create info string
            info = f"FOV: {fov_angle:.2f}°, Tilt: {tilt_angle:.2f}°"
            results_info.append(info)

        # Stack batch results
        output_image = torch.stack(results_images, dim=0)
        avg_fov = float(np.mean(results_fov))
        avg_tilt = float(np.mean(results_tilt))
        combined_info = "\n".join(results_info)

        return (output_image, avg_fov, avg_tilt, combined_info)

    def _estimate_tilt(
        self,
        img: np.ndarray,
        threshold1: int,
        threshold2: int,
        line_threshold: int,
        fov_horizontal: float
    ) -> Tuple[float, list]:
        """
        Estimate the tilt angle (horizon angle) of the image based on horizon position.

        Tilt is calculated relative to frame center:
        - Tilt = 0° when horizon is at frame center
        - Tilt < 0° when horizon is above center (camera looking up)
        - Tilt > 0° when horizon is below center (camera looking down)

        Args:
            fov_horizontal: Estimated horizontal field of view in degrees

        Returns:
            Tuple of (tilt_angle_degrees, detected_horizon_lines)
        """
        height, width = img.shape[:2]
        frame_center_y = height / 2.0

        # Calculate vertical FOV from horizontal FOV using aspect ratio
        aspect_ratio = width / height
        # tan(vfov/2) = tan(hfov/2) / aspect_ratio
        fov_vertical = 2 * np.degrees(np.arctan(np.tan(np.radians(fov_horizontal / 2)) / aspect_ratio))

        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, threshold1, threshold2)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, line_threshold)

        horizon_lines = []
        horizon_y_positions = []

        if lines is not None:
            for line in lines:
                rho, theta = line[0]

                # Convert theta to degrees
                angle_deg = np.degrees(theta)

                # Filter for near-horizontal lines (within 15 degrees of horizontal)
                # Theta is measured from vertical, so horizontal is around 90 degrees
                if 75 <= angle_deg <= 105:
                    # Calculate where this line crosses the center x of the frame
                    # Line equation: x*cos(theta) + y*sin(theta) = rho
                    # At x = width/2, solve for y: y = (rho - x*cos(theta)) / sin(theta)

                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)

                    if abs(sin_theta) > 0.1:  # Avoid near-vertical lines
                        x_center = width / 2.0
                        y_at_center = (rho - x_center * cos_theta) / sin_theta

                        # Only consider lines that cross within or near the frame
                        if -height * 0.5 <= y_at_center <= height * 1.5:
                            horizon_lines.append((rho, theta))
                            horizon_y_positions.append(y_at_center)

        # Calculate tilt based on horizon position relative to frame center
        if horizon_y_positions:
            # Use median horizon position to be robust to outliers
            median_horizon_y = float(np.median(horizon_y_positions))

            # Calculate offset from center (positive = below center)
            y_offset = median_horizon_y - frame_center_y

            # Calculate tilt angle using the estimated vertical FOV
            # Each pixel represents: vertical_fov / height degrees
            degrees_per_pixel = fov_vertical / height

            # Calculate tilt: positive when horizon is below center (camera tilted up)
            tilt_angle = y_offset * degrees_per_pixel
        else:
            tilt_angle = 0.0  # No horizon detected, assume level

        return tilt_angle, horizon_lines

    def _estimate_fov(
        self,
        img: np.ndarray,
        threshold1: int,
        threshold2: int,
        line_threshold: int
    ) -> Tuple[float, list]:
        """
        Estimate the Field of View using vanishing point and focal length estimation.

        Uses the relationship: focal_length = width / (2 * tan(hfov/2))
        The distance from image center to a vanishing point approximates the focal length.

        Returns:
            Tuple of (fov_angle_degrees, vanishing_points)
        """
        height, width = img.shape[:2]

        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, threshold1, threshold2)

        # Detect lines using probabilistic Hough transform for better line segments
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=line_threshold // 2,
            minLineLength=min(width, height) // 10,
            maxLineGap=min(width, height) // 20
        )

        vanishing_points = []
        fov = None

        if lines is not None and len(lines) > 3:
            # Find the dominant vanishing point
            vanishing_point, supporting_lines = self._find_dominant_vanishing_point(lines, (height, width))

            if vanishing_point is not None:
                vanishing_points.append(vanishing_point)

                # Calculate FOV from vanishing point position
                # The distance from principal point (image center) to vanishing point
                # approximates the focal length in pixels
                principal_point = np.array([width / 2.0, height / 2.0])
                vp_array = np.array(vanishing_point)

                # For horizontal FOV, use the horizontal distance
                focal_length_estimate = abs(vp_array[0] - principal_point[0])

                # If vanishing point is too close to center, use the Euclidean distance
                if focal_length_estimate < width * 0.1:
                    focal_length_estimate = np.linalg.norm(vp_array - principal_point)

                # Ensure reasonable focal length
                focal_length_estimate = max(focal_length_estimate, width * 0.2)

                # Calculate horizontal FOV from focal length
                # hfov = 2 * arctan(width / (2 * focal_length))
                fov = 2 * np.degrees(np.arctan(width / (2 * focal_length_estimate)))

                # Clamp to reasonable range
                fov = np.clip(fov, 15, 150)

        # If no good vanishing point found, use a more conservative default
        if fov is None:
            # Default based on typical camera FOV (smartphone ~70°, DSLR ~50°, wide ~90°)
            # Use aspect ratio as a hint
            aspect_ratio = width / height
            if aspect_ratio > 2.0:  # Ultra-wide panorama
                fov = 100.0
            elif aspect_ratio > 1.7:  # Wide aspect (16:9, etc.)
                fov = 70.0
            elif aspect_ratio > 1.2:  # Standard aspect (4:3, 3:2)
                fov = 55.0
            else:  # Square or portrait
                fov = 50.0

        return float(fov), vanishing_points

    def _find_dominant_vanishing_point(
        self,
        lines: np.ndarray,
        img_shape: Tuple[int, int]
    ) -> Tuple[Tuple[float, float], int]:
        """
        Find the most dominant vanishing point using RANSAC-like clustering.

        Returns:
            Tuple of (vanishing_point, num_supporting_lines) or (None, 0) if none found
        """
        height, width = img_shape
        principal_point = np.array([width / 2.0, height / 2.0])

        # Convert lines to point pairs and filter out near-horizontal lines
        # (we want lines that converge, like roads, buildings)
        line_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate line angle
            dx = x2 - x1
            dy = y2 - y1
            angle = abs(np.degrees(np.arctan2(dy, dx)))

            # Filter: keep lines that are NOT horizontal (skip 80-100 degrees from vertical)
            # We want converging lines (vertical-ish or diagonal)
            if not (10 <= angle <= 170):  # Skip near-horizontal
                continue

            line_list.append(((float(x1), float(y1)), (float(x2), float(y2))))

        if len(line_list) < 3:
            return None, 0

        # Find all pairwise intersections
        intersections = []
        intersection_line_pairs = []

        for i in range(len(line_list)):
            for j in range(i + 1, min(i + 50, len(line_list))):  # Limit combinations for performance
                pt = self._line_intersection(line_list[i], line_list[j])
                if pt is not None:
                    # Only consider intersections within reasonable bounds
                    if (-width <= pt[0] <= 2*width) and (-height <= pt[1] <= 2*height):
                        intersections.append(pt)
                        intersection_line_pairs.append((i, j))

        if len(intersections) < 3:
            return None, 0

        intersections = np.array(intersections)

        # Cluster intersections to find dominant vanishing point
        # Use a voting scheme: find the point with most nearby intersections
        best_vp = None
        best_support = 0

        # Sample potential vanishing points
        sample_indices = np.linspace(0, len(intersections)-1, min(20, len(intersections)), dtype=int)

        for idx in sample_indices:
            candidate = intersections[idx]

            # Count how many intersections are near this candidate
            distances = np.linalg.norm(intersections - candidate, axis=1)

            # Dynamic threshold based on image size
            threshold = max(width, height) * 0.15
            support = np.sum(distances < threshold)

            if support > best_support:
                best_support = support
                # Use median of supporting points for robustness
                supporting_points = intersections[distances < threshold]
                best_vp = tuple(np.median(supporting_points, axis=0))

        # Require at least 3 supporting intersections
        if best_support >= 3:
            return best_vp, best_support

        return None, 0

    def _line_intersection(
        self,
        line1: Tuple[Tuple[int, int], Tuple[int, int]],
        line2: Tuple[Tuple[int, int], Tuple[int, int]]
    ) -> Tuple[float, float]:
        """
        Find intersection point of two lines.
        """
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2

        # Convert to float to avoid integer overflow
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        x3, y3, x4, y4 = float(x3), float(y3), float(x4), float(y4)

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-6:  # Lines are parallel
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        return (px, py)

    def _visualize_results(
        self,
        img: np.ndarray,
        horizon_lines: list,
        vanishing_points: list,
        tilt_angle: float,
        fov_angle: float
    ) -> np.ndarray:
        """
        Draw detected lines and vanishing points on the image.
        """
        vis_img = img.copy()
        height, width = img.shape[:2]

        # Draw horizon lines in blue
        for rho, theta in horizon_lines[:5]:  # Draw top 5 lines
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            cv2.line(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw vanishing points in green
        for vp in vanishing_points:
            x, y = int(vp[0]), int(vp[1])
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(vis_img, (x, y), 10, (0, 255, 0), -1)
                cv2.circle(vis_img, (x, y), 15, (0, 255, 0), 2)

        # Add text overlay with results
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_fov = f"FOV: {fov_angle:.1f} deg"
        text_tilt = f"Tilt: {tilt_angle:.1f} deg"

        # Draw text with background for visibility
        cv2.rectangle(vis_img, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.putText(vis_img, text_fov, (20, 40), font, 0.8, (255, 255, 255), 2)
        cv2.putText(vis_img, text_tilt, (20, 70), font, 0.8, (255, 255, 255), 2)

        return vis_img


class DepthFOVEstimatorNode:
    """
    A ComfyUI node that estimates FOV and tilt from depth maps.

    Uses depth information for more robust geometric analysis:
    - Depth discontinuities for edge detection
    - 3D structure analysis for vanishing points
    - Depth-aware horizon detection
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "depth_map": ("IMAGE",),  # Accepts depth map from depth estimation nodes
                "image": ("IMAGE",),       # Original image for visualization
            },
            "optional": {
                "depth_edge_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "line_threshold": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 300,
                    "step": 5,
                    "display": "number"
                }),
                "visualize": ("BOOLEAN", {
                    "default": True,
                    "display": "checkbox"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("annotated_image", "fov_degrees", "tilt_degrees", "info")
    FUNCTION = "estimate_fov_tilt_from_depth"
    CATEGORY = "image/analysis"

    def estimate_fov_tilt_from_depth(
        self,
        depth_map: torch.Tensor,
        image: torch.Tensor,
        depth_edge_threshold: float = 0.1,
        line_threshold: int = 50,
        visualize: bool = True
    ) -> Tuple[torch.Tensor, float, float, str]:
        """
        Estimate FOV and tilt from depth map.

        Args:
            depth_map: Depth map tensor [B, H, W, C]
            image: Original RGB image for visualization [B, H, W, C]
            depth_edge_threshold: Threshold for depth discontinuities
            line_threshold: Threshold for Hough line detection
            visualize: Whether to draw detected features

        Returns:
            Tuple of (annotated_image, fov_degrees, tilt_degrees, info_string)
        """
        batch_size = depth_map.shape[0]
        results_images = []
        results_fov = []
        results_tilt = []
        results_info = []

        for i in range(batch_size):
            # Process depth map
            depth_np = depth_map[i].cpu().numpy()
            img_np = image[i].cpu().numpy()

            # Convert image from [0, 1] float to [0, 255] uint8
            img_np = (img_np * 255).astype(np.uint8)

            # Handle depth map format (could be 1 or 3 channels)
            if depth_np.shape[-1] == 3:
                # If 3 channels, take first channel (should all be same for grayscale depth)
                depth_np = depth_np[:, :, 0]
            elif depth_np.shape[-1] == 1:
                depth_np = depth_np[:, :, 0]

            # Normalize depth to [0, 1] if not already
            if depth_np.max() > 1.0:
                depth_np = depth_np / 255.0

            # Estimate FOV from depth map
            fov_angle, vanishing_points = self._estimate_fov_from_depth(
                depth_np, depth_edge_threshold, line_threshold
            )

            # Estimate tilt from depth map
            tilt_angle, horizon_lines = self._estimate_tilt_from_depth(
                depth_np, depth_edge_threshold, fov_angle
            )

            # Create visualization if requested
            if visualize:
                vis_img = self._visualize_depth_results(
                    img_np.copy(), depth_np, horizon_lines, vanishing_points, tilt_angle, fov_angle
                )
            else:
                vis_img = img_np

            # Convert back to ComfyUI tensor format
            vis_tensor = torch.from_numpy(vis_img.astype(np.float32) / 255.0)
            results_images.append(vis_tensor)
            results_fov.append(fov_angle)
            results_tilt.append(tilt_angle)

            # Create info string
            info = f"FOV: {fov_angle:.2f}°, Tilt: {tilt_angle:.2f}° (depth-based)"
            results_info.append(info)

        # Stack batch results
        output_image = torch.stack(results_images, dim=0)
        avg_fov = float(np.mean(results_fov))
        avg_tilt = float(np.mean(results_tilt))
        combined_info = "\n".join(results_info)

        return (output_image, avg_fov, avg_tilt, combined_info)

    def _estimate_fov_from_depth(
        self,
        depth: np.ndarray,
        edge_threshold: float,
        line_threshold: int
    ) -> Tuple[float, list]:
        """
        Estimate FOV using depth discontinuities and vanishing points.

        Depth discontinuities represent real 3D edges in the scene,
        providing cleaner geometric information than RGB edges.
        """
        height, width = depth.shape

        # Find depth discontinuities (edges in 3D structure)
        depth_edges = self._detect_depth_discontinuities(depth, edge_threshold)

        # Detect lines from depth edges
        lines = cv2.HoughLinesP(
            depth_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=line_threshold,
            minLineLength=min(width, height) // 8,  # Longer minimum lines
            maxLineGap=min(width, height) // 25      # Smaller gaps
        )

        vanishing_points = []
        fov = None

        if lines is not None and len(lines) > 5:  # Require more lines for reliability
            # Find dominant vanishing point from depth-based lines
            vanishing_point, supporting_lines = self._find_dominant_vanishing_point_from_lines(
                lines, (height, width)
            )

            if vanishing_point is not None:
                vanishing_points.append(vanishing_point)

                # Calculate FOV from vanishing point
                principal_point = np.array([width / 2.0, height / 2.0])
                vp_array = np.array(vanishing_point)

                # Calculate distance from center
                vp_distance = np.linalg.norm(vp_array - principal_point)

                # For horizontal FOV, primarily use horizontal distance
                # but validate it makes sense
                horizontal_distance = abs(vp_array[0] - principal_point[0])

                # Use horizontal distance if VP is reasonably off-center horizontally
                # Otherwise use Euclidean distance
                if horizontal_distance > width * 0.15:
                    focal_length_estimate = horizontal_distance
                else:
                    focal_length_estimate = vp_distance

                # Validate: reject if vanishing point is too close to center
                # (indicates wide FOV or bad detection)
                # For typical images, minimum focal length should be at least 0.5 * width
                # This corresponds to ~90° FOV maximum
                min_focal_length = width * 0.5

                if focal_length_estimate < min_focal_length:
                    # VP too close - likely a bad detection, use conservative fallback
                    fov = None
                else:
                    # Calculate horizontal FOV
                    fov = 2 * np.degrees(np.arctan(width / (2 * focal_length_estimate)))
                    # Clamp to reasonable range
                    fov = np.clip(fov, 15, 120)

        # Fallback if no good vanishing point found
        if fov is None:
            # Use conservative default based on aspect ratio
            # Most cameras have FOV between 40-70 degrees
            aspect_ratio = width / height
            if aspect_ratio > 2.0:
                fov = 90.0  # Ultra-wide panorama
            elif aspect_ratio > 1.7:
                fov = 65.0  # Wide aspect (16:9)
            elif aspect_ratio > 1.2:
                fov = 50.0  # Standard aspect (4:3, 3:2)
            else:
                fov = 45.0  # Square or portrait

        return float(fov), vanishing_points

    def _estimate_tilt_from_depth(
        self,
        depth: np.ndarray,
        edge_threshold: float,
        fov_horizontal: float
    ) -> Tuple[float, list]:
        """
        Estimate tilt using depth-based horizon detection.

        The horizon in depth maps appears as a transition from near to far depths,
        often more clearly visible than in RGB images.
        """
        height, width = depth.shape
        frame_center_y = height / 2.0

        # Calculate vertical FOV
        aspect_ratio = width / height
        fov_vertical = 2 * np.degrees(np.arctan(np.tan(np.radians(fov_horizontal / 2)) / aspect_ratio))

        # Find depth discontinuities
        depth_edges = self._detect_depth_discontinuities(depth, edge_threshold)

        # Detect horizontal lines (potential horizons)
        lines = cv2.HoughLines(depth_edges, 1, np.pi / 180, int(width * 0.3))

        horizon_lines = []
        horizon_y_positions = []

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                angle_deg = np.degrees(theta)

                # Filter for near-horizontal lines
                if 75 <= angle_deg <= 105:
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)

                    if abs(sin_theta) > 0.1:
                        x_center = width / 2.0
                        y_at_center = (rho - x_center * cos_theta) / sin_theta

                        if -height * 0.5 <= y_at_center <= height * 1.5:
                            horizon_lines.append((rho, theta))
                            horizon_y_positions.append(y_at_center)

        # Calculate tilt from horizon position
        if horizon_y_positions:
            median_horizon_y = float(np.median(horizon_y_positions))
            y_offset = median_horizon_y - frame_center_y
            degrees_per_pixel = fov_vertical / height
            tilt_angle = y_offset * degrees_per_pixel
        else:
            # Alternative: analyze depth gradient
            # Lower half of image should generally have nearer depths (ground)
            # Upper half should have farther depths (sky/background)
            tilt_angle = self._estimate_tilt_from_depth_gradient(depth, fov_vertical)

        return tilt_angle, horizon_lines

    def _detect_depth_discontinuities(
        self,
        depth: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Detect depth discontinuities (3D edges) in the depth map.

        Depth discontinuities represent real boundaries between objects,
        independent of texture or lighting.
        """
        # Normalize depth to [0, 1]
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Convert to uint8 for edge detection
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        # Apply bilateral filter to preserve edges while smoothing
        depth_filtered = cv2.bilateralFilter(depth_uint8, 5, 50, 50)

        # Calculate depth gradients
        grad_x = cv2.Sobel(depth_filtered, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_filtered, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize gradient
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)

        # Threshold to get binary edge map
        threshold_uint8 = int(threshold * 255)
        _, edges = cv2.threshold(gradient_magnitude, threshold_uint8, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return edges

    def _estimate_tilt_from_depth_gradient(
        self,
        depth: np.ndarray,
        fov_vertical: float
    ) -> float:
        """
        Estimate tilt from overall depth gradient when no clear horizon is found.
        """
        height, width = depth.shape

        # Divide image into top and bottom halves
        top_half = depth[:height//2, :]
        bottom_half = depth[height//2:, :]

        # Calculate median depths
        top_median = np.median(top_half)
        bottom_median = np.median(bottom_half)

        # If bottom is significantly nearer than top, camera is level or tilted down
        # If top is nearer, camera is tilted up significantly
        depth_ratio = (top_median - bottom_median) / (top_median + bottom_median + 1e-8)

        # Estimate tilt angle (heuristic)
        # Positive depth_ratio means top is farther (normal/level view)
        # Negative means top is nearer (looking up at something)
        tilt_angle = -depth_ratio * 30.0  # Scale factor is heuristic

        return np.clip(tilt_angle, -45.0, 45.0)

    def _find_dominant_vanishing_point_from_lines(
        self,
        lines: np.ndarray,
        img_shape: Tuple[int, int]
    ) -> Tuple[Tuple[float, float], int]:
        """
        Find dominant vanishing point from depth-based lines.
        Optimized for depth discontinuities - focuses on near-vertical converging lines.
        """
        height, width = img_shape
        principal_point = np.array([width / 2.0, height / 2.0])

        # Filter for near-vertical lines that are likely to converge
        line_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1

            # Calculate line length
            line_length = np.sqrt(dx**2 + dy**2)

            # Skip very short lines
            if line_length < min(width, height) * 0.05:
                continue

            # Calculate angle from horizontal (0-180 degrees)
            angle = abs(np.degrees(np.arctan2(dy, dx)))

            # Focus on near-vertical lines (converging lines like building edges)
            # Accept lines between 20-70 degrees and 110-160 degrees
            # This filters out horizontal and diagonal lines
            is_near_vertical = (20 <= angle <= 70) or (110 <= angle <= 160)

            if is_near_vertical:
                line_list.append(((float(x1), float(y1)), (float(x2), float(y2))))

        if len(line_list) < 5:  # Need at least 5 good lines
            return None, 0

        # Find intersections between line pairs
        intersections = []
        for i in range(len(line_list)):
            for j in range(i + 1, min(i + 40, len(line_list))):
                pt = self._line_intersection(line_list[i], line_list[j])
                if pt is not None:
                    # Allow intersections within extended bounds
                    if (-width * 0.5 <= pt[0] <= width * 2.5) and (-height * 0.5 <= pt[1] <= height * 2.5):
                        intersections.append(pt)

        if len(intersections) < 5:  # Need enough intersections
            return None, 0

        intersections = np.array(intersections)

        # Filter intersections: prefer those farther from center
        # (narrow FOV means VP should be far from center)
        distances_from_center = np.linalg.norm(intersections - principal_point, axis=1)

        # Only consider intersections reasonably far from center
        # This helps avoid false positives that would indicate very wide FOV
        min_distance = width * 0.4  # At least 40% of width from center
        far_intersections = intersections[distances_from_center > min_distance]

        if len(far_intersections) < 3:
            # If we don't have enough far intersections, the scene may not have
            # clear converging lines - return None to use fallback
            return None, 0

        # Use the far intersections for clustering
        intersections = far_intersections

        # Find best vanishing point through clustering
        best_vp = None
        best_support = 0

        sample_indices = np.linspace(0, len(intersections)-1, min(15, len(intersections)), dtype=int)

        for idx in sample_indices:
            candidate = intersections[idx]
            distances = np.linalg.norm(intersections - candidate, axis=1)

            # Tighter clustering threshold for more precision
            threshold = max(width, height) * 0.10
            support = np.sum(distances < threshold)

            if support > best_support:
                best_support = support
                supporting_points = intersections[distances < threshold]
                best_vp = tuple(np.median(supporting_points, axis=0))

        # Require stronger support (more intersections agreeing)
        if best_support >= 5:
            return best_vp, best_support

        return None, 0

    def _line_intersection(
        self,
        line1: Tuple[Tuple[float, float], Tuple[float, float]],
        line2: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Find intersection point of two lines."""
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-6:
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        return (px, py)

    def _visualize_depth_results(
        self,
        img: np.ndarray,
        depth: np.ndarray,
        horizon_lines: list,
        vanishing_points: list,
        tilt_angle: float,
        fov_angle: float
    ) -> np.ndarray:
        """
        Visualize depth-based analysis results on the original image.
        """
        vis_img = img.copy()
        height, width = img.shape[:2]

        # Draw horizon lines in blue
        for rho, theta in horizon_lines[:5]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            cv2.line(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw vanishing points in green
        for vp in vanishing_points:
            x, y = int(vp[0]), int(vp[1])
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(vis_img, (x, y), 10, (0, 255, 0), -1)
                cv2.circle(vis_img, (x, y), 15, (0, 255, 0), 2)

        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_fov = f"FOV: {fov_angle:.1f} deg (depth)"
        text_tilt = f"Tilt: {tilt_angle:.1f} deg"

        cv2.rectangle(vis_img, (10, 10), (340, 80), (0, 0, 0), -1)
        cv2.putText(vis_img, text_fov, (20, 40), font, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_img, text_tilt, (20, 70), font, 0.7, (255, 255, 255), 2)

        return vis_img


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "FOVEstimator": FOVEstimatorNode,
    "DepthFOVEstimator": DepthFOVEstimatorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FOVEstimator": "FOV & Tilt Estimator (RGB)",
    "DepthFOVEstimator": "FOV & Tilt Estimator (Depth)"
}
