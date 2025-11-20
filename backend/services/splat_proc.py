# -*- coding: utf-8 -*-
"""
Gaussian Splat Processing for Aether-GS
3D Lifting, .splat/.ply Verarbeitung und Optimierung fuer Spark Renderer
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from pathlib import Path
from PIL import Image
import logging
import struct

logger = logging.getLogger(__name__)


class GaussianSplat:
    """
    Repr�sentiert ein einzelnes Gaussian Splat

    Ein Splat hat:
    - Position (x, y, z)
    - Rotation (quaternion)
    - Scale (sx, sy, sz)
    - Color (r, g, b)
    - Opacity (alpha)
    """

    def __init__(
        self,
        position: np.ndarray,  # [x, y, z]
        rotation: np.ndarray,  # [qx, qy, qz, qw]
        scale: np.ndarray,     # [sx, sy, sz]
        color: np.ndarray,     # [r, g, b] (0-255)
        opacity: float,        # 0.0-1.0
    ):
        self.position = position
        self.rotation = rotation
        self.scale = scale
        self.color = color
        self.opacity = opacity

    @classmethod
    def from_point_cloud(
        cls,
        xyz: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        default_scale: float = 0.1,
    ) -> "GaussianSplat":
        """Erstellt Splat aus Point Cloud Point"""
        rotation = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        scale = np.array([default_scale] * 3)
        color = rgb if rgb is not None else np.array([128, 128, 128])
        opacity = 1.0

        return cls(xyz, rotation, scale, color, opacity)


class SplatFile:
    """Handler f�r .splat/.ply Gaussian Splatting Dateien"""

    def __init__(self):
        self.splats: List[GaussianSplat] = []
        self.metadata: Dict[str, Any] = {}

    def load_ply(self, filepath: Path) -> None:
        """
        L�dt Gaussian Splats aus .ply Datei

        PLY Format kann von verschiedenen Tools kommen:
        - Luma AI
        - Polycam
        - 3D Gaussian Splatting Research Code
        """
        try:
            from plyfile import PlyData

            logger.info(f"Loading PLY file: {filepath}")
            plydata = PlyData.read(str(filepath))

            vertex = plydata["vertex"]
            num_points = len(vertex)

            logger.info(f"Found {num_points} splats")

            # Extract data
            positions = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T

            # Colors (falls vorhanden)
            if "red" in vertex:
                colors = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T
            else:
                colors = np.full((num_points, 3), 128)

            # Scales (falls vorhanden)
            if "scale_0" in vertex:
                scales = np.vstack(
                    [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]]
                ).T
            else:
                scales = np.full((num_points, 3), 0.1)

            # Rotation (als Quaternion)
            if "rot_0" in vertex:
                rotations = np.vstack(
                    [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]]
                ).T
            else:
                rotations = np.tile([0.0, 0.0, 0.0, 1.0], (num_points, 1))

            # Opacity
            if "opacity" in vertex:
                opacities = vertex["opacity"]
            else:
                opacities = np.ones(num_points)

            # Create Splat objects
            self.splats = []
            for i in range(num_points):
                splat = GaussianSplat(
                    position=positions[i],
                    rotation=rotations[i],
                    scale=scales[i],
                    color=colors[i],
                    opacity=opacities[i],
                )
                self.splats.append(splat)

            logger.info(f"Successfully loaded {len(self.splats)} splats")

        except ImportError:
            logger.error("plyfile library not found. Install with: pip install plyfile")
            raise
        except Exception as e:
            logger.error(f"Failed to load PLY: {e}")
            raise

    def save_ply(self, filepath: Path) -> None:
        """Speichert Splats als .ply Datei"""
        try:
            from plyfile import PlyData, PlyElement

            logger.info(f"Saving {len(self.splats)} splats to: {filepath}")

            # Prepare vertex data
            vertices = []
            for splat in self.splats:
                vertex = (
                    splat.position[0],  # x
                    splat.position[1],  # y
                    splat.position[2],  # z
                    int(splat.color[0]),  # r
                    int(splat.color[1]),  # g
                    int(splat.color[2]),  # b
                    splat.scale[0],     # scale_0
                    splat.scale[1],     # scale_1
                    splat.scale[2],     # scale_2
                    splat.rotation[0],  # rot_0
                    splat.rotation[1],  # rot_1
                    splat.rotation[2],  # rot_2
                    splat.rotation[3],  # rot_3
                    splat.opacity,      # opacity
                )
                vertices.append(vertex)

            # Define PLY element
            vertex_array = np.array(
                vertices,
                dtype=[
                    ("x", "f4"),
                    ("y", "f4"),
                    ("z", "f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                    ("scale_0", "f4"),
                    ("scale_1", "f4"),
                    ("scale_2", "f4"),
                    ("rot_0", "f4"),
                    ("rot_1", "f4"),
                    ("rot_2", "f4"),
                    ("rot_3", "f4"),
                    ("opacity", "f4"),
                ],
            )

            el = PlyElement.describe(vertex_array, "vertex")
            plydata = PlyData([el])
            plydata.write(str(filepath))

            logger.info(f"PLY saved successfully")

        except Exception as e:
            logger.error(f"Failed to save PLY: {e}")
            raise

    def optimize_for_web(
        self,
        max_splats: Optional[int] = None,
        quality_threshold: float = 0.1,
    ) -> None:
        """
        Optimiert Splat-Cloud f�r Web-Rendering

        Args:
            max_splats: Maximum Anzahl Splats (entfernt unwichtigste)
            quality_threshold: Mindest-Opacity (entfernt transparente Splats)
        """
        logger.info("Optimizing splat cloud for web rendering...")

        original_count = len(self.splats)

        # Filter: Entferne sehr transparente Splats
        self.splats = [s for s in self.splats if s.opacity >= quality_threshold]

        # Filter: Limitiere auf max_splats (nach Opacity sortiert)
        if max_splats and len(self.splats) > max_splats:
            self.splats.sort(key=lambda s: s.opacity, reverse=True)
            self.splats = self.splats[:max_splats]

        logger.info(f"Optimization: {original_count} � {len(self.splats)} splats")

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Berechnet Bounding Box (min, max)"""
        if not self.splats:
            return np.zeros(3), np.zeros(3)

        positions = np.array([s.position for s in self.splats])
        return positions.min(axis=0), positions.max(axis=0)


class ImageToSplatConverter:
    """
    Konvertiert 2D-Bilder + Depth Maps zu 3D Gaussian Splats
    (Vereinfachte Version - echte 3DGS braucht Multi-View Training)
    """

    @staticmethod
    def depth_image_to_pointcloud(
        image: Image.Image,
        depth_map: np.ndarray,
        fov_deg: float = 60.0,
        max_depth: float = 100.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Liftet 2D Bild + Depth zu 3D Point Cloud

        Args:
            image: RGB Bild
            depth_map: Depth Map (normalisiert 0-1)
            fov_deg: Field of View in Grad
            max_depth: Maximale Tiefe in Metern

        Returns:
            (xyz_points, rgb_colors)
        """
        w, h = image.size
        rgb = np.array(image)

        # Ensure depth_map matches image size
        if depth_map.shape != (h, w):
            from scipy.ndimage import zoom
            scale_y = h / depth_map.shape[0]
            scale_x = w / depth_map.shape[1]
            depth_map = zoom(depth_map, (scale_y, scale_x), order=1)

        # Camera intrinsics (simplified)
        fov_rad = np.radians(fov_deg)
        focal_length = w / (2 * np.tan(fov_rad / 2))

        # Create pixel grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Unproject to 3D
        z = depth_map * max_depth
        x = (u - w / 2) * z / focal_length
        y = (v - h / 2) * z / focal_length

        # Stack to point cloud
        xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        rgb_flat = rgb.reshape(-1, 3)

        # Filter invalid points
        valid_mask = z.flatten() > 0.01  # Remove too-close points
        xyz = xyz[valid_mask]
        rgb_flat = rgb_flat[valid_mask]

        logger.info(f"Generated point cloud: {xyz.shape[0]} points")
        return xyz, rgb_flat

    @staticmethod
    def pointcloud_to_splats(
        xyz: np.ndarray,
        rgb: np.ndarray,
        splat_scale: float = 0.5,
    ) -> List[GaussianSplat]:
        """Konvertiert Point Cloud zu Gaussian Splats"""
        splats = []

        for i in range(len(xyz)):
            splat = GaussianSplat.from_point_cloud(
                xyz=xyz[i],
                rgb=rgb[i],
                default_scale=splat_scale,
            )
            splats.append(splat)

        logger.info(f"Created {len(splats)} splats")
        return splats

    @classmethod
    def convert_image_to_splat(
        cls,
        image: Image.Image,
        depth_map: np.ndarray,
        output_path: Path,
        fov_deg: float = 60.0,
        max_depth: float = 100.0,
        splat_scale: float = 0.5,
        optimize: bool = True,
        max_splats: int = 500000,
    ) -> SplatFile:
        """
        Komplette Pipeline: Image + Depth � .ply Splat File

        Args:
            image: Input RGB image
            depth_map: Depth map
            output_path: Ausgabe .ply Datei
            fov_deg: Field of View
            max_depth: Max depth in meters
            splat_scale: Gr��e der Splats
            optimize: Optimierung f�r Web
            max_splats: Max Anzahl Splats

        Returns:
            SplatFile object
        """
        logger.info("Converting image to Gaussian Splat...")

        # Step 1: Depth Image � Point Cloud
        xyz, rgb = cls.depth_image_to_pointcloud(
            image, depth_map, fov_deg=fov_deg, max_depth=max_depth
        )

        # Step 2: Point Cloud � Splats
        splats = cls.pointcloud_to_splats(xyz, rgb, splat_scale=splat_scale)

        # Step 3: Create SplatFile
        splat_file = SplatFile()
        splat_file.splats = splats

        # Step 4: Optimize
        if optimize:
            splat_file.optimize_for_web(max_splats=max_splats)

        # Step 5: Save
        splat_file.save_ply(output_path)

        logger.info(f"Splat file saved to: {output_path}")
        return splat_file


class SplatRenderer:
    """
    Hilfsklasse f�r Server-Side Splat-Rendering (Preview/Thumbnails)
    Echtes Rendering passiert im Frontend via Spark Renderer
    """

    @staticmethod
    def generate_thumbnail(
        splat_file: SplatFile,
        size: Tuple[int, int] = (512, 512),
    ) -> Image.Image:
        """
        Erzeugt einfaches 2D-Thumbnail aus Splat Cloud
        (Orthographic Projection von oben)
        """
        width, height = size

        # Get bounds
        min_bound, max_bound = splat_file.get_bounds()

        # Create canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Project splats
        for splat in splat_file.splats:
            # Normalize position to image space
            x_norm = (splat.position[0] - min_bound[0]) / (max_bound[0] - min_bound[0] + 1e-8)
            y_norm = (splat.position[1] - min_bound[1]) / (max_bound[1] - min_bound[1] + 1e-8)

            px = int(x_norm * (width - 1))
            py = int(y_norm * (height - 1))

            if 0 <= px < width and 0 <= py < height:
                # Blend color
                alpha = splat.opacity
                canvas[py, px] = (
                    canvas[py, px] * (1 - alpha) + splat.color * alpha
                ).astype(np.uint8)

        return Image.fromarray(canvas)
