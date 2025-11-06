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


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "FOVEstimator": FOVEstimatorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FOVEstimator": "FOV & Tilt Estimator"
}
