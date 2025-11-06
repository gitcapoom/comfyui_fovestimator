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

            # Estimate tilt and FOV
            tilt_angle, horizon_lines = self._estimate_tilt(
                img_np, edge_threshold_low, edge_threshold_high, line_threshold
            )

            fov_angle, vanishing_points = self._estimate_fov(
                img_np, edge_threshold_low, edge_threshold_high, line_threshold
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
        line_threshold: int
    ) -> Tuple[float, list]:
        """
        Estimate the tilt angle (horizon angle) of the image based on horizon position.

        Tilt is calculated relative to frame center:
        - Tilt = 0° when horizon is at frame center
        - Tilt < 0° when horizon is above center (camera looking up)
        - Tilt > 0° when horizon is below center (camera looking down)

        Returns:
            Tuple of (tilt_angle_degrees, detected_horizon_lines)
        """
        height, width = img.shape[:2]
        frame_center_y = height / 2.0

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

            # Estimate tilt angle
            # Assume a typical FOV of 60 degrees vertically
            # This means each pixel represents: 60 / height degrees
            estimated_vertical_fov = 60.0  # degrees
            degrees_per_pixel = estimated_vertical_fov / height

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
        Estimate the Field of View by detecting vanishing points.

        Returns:
            Tuple of (fov_angle_degrees, vanishing_points)
        """
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
            minLineLength=img.shape[1] // 10,
            maxLineGap=img.shape[1] // 20
        )

        vanishing_points = []

        if lines is not None and len(lines) > 1:
            # Find vanishing points by finding line intersections
            vanishing_points = self._find_vanishing_points(lines, img.shape)

        # Estimate FOV based on vanishing points
        if len(vanishing_points) >= 2:
            # Calculate angle between vanishing points
            fov = self._calculate_fov_from_vanishing_points(vanishing_points, img.shape)
        else:
            # Default FOV estimate based on image aspect ratio
            # Typical camera FOV ranges from 50-90 degrees
            aspect_ratio = img.shape[1] / img.shape[0]
            # Estimate: wider aspect ratio suggests wider FOV
            fov = 50 + (aspect_ratio - 1.0) * 30
            fov = np.clip(fov, 40, 120)

        return float(fov), vanishing_points

    def _find_vanishing_points(
        self,
        lines: np.ndarray,
        img_shape: Tuple[int, ...]
    ) -> list:
        """
        Find vanishing points from line intersections.
        """
        vanishing_points = []
        height, width = img_shape[:2]

        # Convert lines to point pairs
        line_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_list.append(((x1, y1), (x2, y2)))

        # Find intersections between non-parallel lines
        intersections = []
        for i in range(len(line_list)):
            for j in range(i + 1, len(line_list)):
                pt = self._line_intersection(line_list[i], line_list[j])
                if pt is not None:
                    intersections.append(pt)

        if not intersections:
            return []

        # Cluster intersections to find vanishing points
        # Use simple spatial clustering
        intersections = np.array(intersections)

        # Filter out intersections too far from image center
        center = np.array([width / 2, height / 2])
        max_dist = max(width, height) * 2  # Allow points outside image but not too far
        distances = np.linalg.norm(intersections - center, axis=1)
        valid_intersections = intersections[distances < max_dist]

        if len(valid_intersections) > 0:
            # Simple clustering: find dense regions
            # For now, just return the median point if we have enough intersections
            if len(valid_intersections) >= 3:
                median_point = np.median(valid_intersections, axis=0)
                vanishing_points.append(tuple(median_point))

        return vanishing_points

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

    def _calculate_fov_from_vanishing_points(
        self,
        vanishing_points: list,
        img_shape: Tuple[int, ...]
    ) -> float:
        """
        Calculate FOV from vanishing points.
        """
        if len(vanishing_points) < 2:
            return 60.0  # Default

        height, width = img_shape[:2]
        center = np.array([width / 2, height / 2])

        # Calculate angles from center to vanishing points
        angles = []
        for vp in vanishing_points[:2]:  # Use first two vanishing points
            vp_array = np.array(vp)
            # Calculate angle from center to vanishing point
            diff = vp_array - center
            angle = np.degrees(np.arctan2(diff[1], diff[0]))
            angles.append(angle)

        # FOV is related to the angular separation of vanishing points
        angular_separation = abs(angles[1] - angles[0])
        if angular_separation > 180:
            angular_separation = 360 - angular_separation

        # The angular separation of vanishing points relates to FOV
        # For perpendicular lines, this gives us a good FOV estimate
        fov = angular_separation

        # Clamp to reasonable range
        fov = np.clip(fov, 30, 150)

        return float(fov)

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
