# -*- coding: utf-8 -*-
"""
Map & Scenario Generator for Aether-GS
Generiert synthetische Satellitenbilder und Heightmaps via AI
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TerrainType:
    """Vordefinierte Terrain-Typen mit optimierten Prompts"""

    URBAN = {
        "prompt": "satellite view of urban area, city blocks, streets, buildings, high resolution, realistic, top-down view, aerial photography",
        "negative_prompt": "blurry, low quality, cartoon, painting, people, cars",
    }

    FOREST = {
        "prompt": "satellite view of dense forest, trees, vegetation, realistic terrain, top-down view, aerial photography",
        "negative_prompt": "blurry, low quality, cartoon, painting, desert, ocean",
    }

    DESERT = {
        "prompt": "satellite view of desert terrain, sand dunes, rocky formations, arid landscape, top-down view, aerial photography",
        "negative_prompt": "blurry, low quality, cartoon, painting, vegetation, water",
    }

    MOUNTAINS = {
        "prompt": "satellite view of mountainous terrain, peaks, valleys, rocky landscape, high altitude, top-down view, aerial photography",
        "negative_prompt": "blurry, low quality, cartoon, painting, flat terrain, ocean",
    }

    MIXED = {
        "prompt": "satellite view of mixed terrain, varied landscape, realistic aerial photography, top-down view",
        "negative_prompt": "blurry, low quality, cartoon, painting",
    }


class ScenarioGenerator:
    """
    Generiert komplette Szenarien (Sat-Image + Heightmap) via AI

    Pipeline:
    1. Text � Satellite Image (SDXL)
    2. Image � Depth Map (DepthAnything/MiDaS)
    3. Depth � Heightmap (Normalisierung)
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_cache_dir: Optional[str] = None,
    ):
        self.device = device
        self.model_cache_dir = model_cache_dir or "./models"

        # Lazy Loading - Models werden erst geladen wenn gebraucht
        self._diffusion_pipeline = None
        self._depth_model = None

        logger.info(f"ScenarioGenerator initialized on device: {self.device}")

    @property
    def diffusion_pipeline(self):
        """Lazy-loading SDXL Pipeline"""
        if self._diffusion_pipeline is None:
            try:
                from diffusers import StableDiffusionXLPipeline

                logger.info("Loading SDXL model...")
                self._diffusion_pipeline = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True,
                    cache_dir=self.model_cache_dir,
                )
                self._diffusion_pipeline.to(self.device)

                # Memory-Optimierung
                if self.device == "cuda":
                    self._diffusion_pipeline.enable_model_cpu_offload()
                    self._diffusion_pipeline.enable_vae_slicing()

                logger.info("SDXL loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load SDXL: {e}")
                raise

        return self._diffusion_pipeline

    @property
    def depth_model(self):
        """Lazy-loading Depth Estimation Model"""
        if self._depth_model is None:
            try:
                from transformers import pipeline

                logger.info("Loading Depth Estimation model...")
                self._depth_model = pipeline(
                    task="depth-estimation",
                    model="LiheYoung/depth-anything-small-hf",  # Leichtes Model
                    device=0 if self.device == "cuda" else -1,
                )
                logger.info("Depth model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load depth model: {e}")
                raise

        return self._depth_model

    def generate_satellite_image(
        self,
        terrain_type: str = "mixed",
        custom_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generiert synthetisches Satellitenbild via SDXL

        Args:
            terrain_type: Einer von ["urban", "forest", "desert", "mountains", "mixed"]
            custom_prompt: Optional eigener Prompt (�berschreibt terrain_type)
            width: Bildbreite (Standard: 1024)
            height: Bildh�he (Standard: 1024)
            num_inference_steps: Anzahl Diffusion Steps (h�her = besser aber langsamer)
            guidance_scale: CFG Scale (h�her = mehr Prompt-Adherence)
            seed: Random Seed f�r Reproduzierbarkeit

        Returns:
            PIL Image
        """
        # Lade Prompt f�r Terrain-Type
        terrain_config = getattr(TerrainType, terrain_type.upper(), TerrainType.MIXED)
        prompt = custom_prompt or terrain_config["prompt"]
        negative_prompt = terrain_config.get("negative_prompt", "")

        logger.info(f"Generating {terrain_type} satellite image ({width}x{height})")

        # Generator f�r Seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate Image
        result = self.diffusion_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        image = result.images[0]
        logger.info("Satellite image generated successfully")

        return image

    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """
        Estimiert Depth Map aus RGB-Bild

        Args:
            image: Input PIL Image

        Returns:
            Depth map als NumPy Array (normalisiert 0-1)
        """
        logger.info("Estimating depth map...")

        # Depth Estimation
        result = self.depth_model(image)
        depth_pil = result["depth"]

        # Konvertiere zu NumPy und normalisiere
        depth_array = np.array(depth_pil).astype(np.float32)
        depth_normalized = (depth_array - depth_array.min()) / (
            depth_array.max() - depth_array.min() + 1e-8
        )

        logger.info(f"Depth map estimated: {depth_normalized.shape}")
        return depth_normalized

    def depth_to_heightmap(
        self,
        depth_map: np.ndarray,
        height_range: Tuple[float, float] = (0.0, 100.0),
        invert: bool = True,
    ) -> np.ndarray:
        """
        Konvertiert Depth Map zu Heightmap

        Args:
            depth_map: Normalisierter Depth Array (0-1)
            height_range: (min_height, max_height) in Metern
            invert: Invertiere Depth (Standard True, da Depth ` Height)

        Returns:
            Heightmap in Metern
        """
        if invert:
            # Depth-Map invertieren: N�her = H�her
            depth_map = 1.0 - depth_map

        # Skaliere auf Height-Range
        min_h, max_h = height_range
        heightmap = depth_map * (max_h - min_h) + min_h

        logger.info(f"Heightmap created: range [{heightmap.min():.1f}, {heightmap.max():.1f}]m")
        return heightmap

    def generate_scenario(
        self,
        terrain_type: str = "mixed",
        custom_prompt: Optional[str] = None,
        output_dir: Optional[Path] = None,
        image_size: Tuple[int, int] = (1024, 1024),
        height_range: Tuple[float, float] = (0.0, 100.0),
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Komplette Scenario-Generierung Pipeline

        Args:
            terrain_type: Terrain-Typ
            custom_prompt: Eigener Prompt
            output_dir: Ordner f�r Output-Dateien (optional)
            image_size: (width, height) f�r Satellitenbild
            height_range: (min, max) Height in Metern
            seed: Random Seed

        Returns:
            Dict mit {
                "image": PIL Image,
                "depth_map": np.ndarray,
                "heightmap": np.ndarray,
                "metadata": {...}
            }
        """
        logger.info(f"Starting full scenario generation: {terrain_type}")

        # Step 1: Generate Satellite Image
        sat_image = self.generate_satellite_image(
            terrain_type=terrain_type,
            custom_prompt=custom_prompt,
            width=image_size[0],
            height=image_size[1],
            seed=seed,
        )

        # Step 2: Estimate Depth
        depth_map = self.estimate_depth(sat_image)

        # Step 3: Create Heightmap
        heightmap = self.depth_to_heightmap(depth_map, height_range=height_range)

        # Save Files (optional)
        metadata = {
            "terrain_type": terrain_type,
            "image_size": image_size,
            "height_range": height_range,
            "seed": seed,
        }

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save Image
            img_path = output_dir / "satellite.png"
            sat_image.save(img_path)

            # Save Depth Map (als Bild)
            depth_img = Image.fromarray((depth_map * 255).astype(np.uint8))
            depth_img.save(output_dir / "depth.png")

            # Save Heightmap (als NumPy)
            np.save(output_dir / "heightmap.npy", heightmap)

            logger.info(f"Scenario saved to {output_dir}")
            metadata["output_dir"] = str(output_dir)

        return {
            "image": sat_image,
            "depth_map": depth_map,
            "heightmap": heightmap,
            "metadata": metadata,
        }


class QuickMapGenerator:
    """
    Schnelle Map-Generierung ohne AI (f�r Testing/Development)
    Generiert prozedurale Heightmaps mit Perlin Noise
    """

    @staticmethod
    def generate_perlin_heightmap(
        width: int = 1024,
        height: int = 1024,
        scale: float = 100.0,
        octaves: int = 6,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        height_range: Tuple[float, float] = (0.0, 100.0),
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generiert Heightmap mit Perlin/Simplex Noise (numpy-basiert)

        Fallback ohne externe Noise-Library (vereinfachtes Multi-Octave Noise)
        """
        if seed is not None:
            np.random.seed(seed)

        # Einfaches Multi-Scale Random Noise (Ersatz f�r echtes Perlin)
        heightmap = np.zeros((height, width), dtype=np.float32)

        for octave in range(octaves):
            frequency = lacunarity ** octave
            amplitude = persistence ** octave

            # Random Noise f�r diese Octave
            noise = np.random.rand(
                int(height / frequency) + 1,
                int(width / frequency) + 1
            )

            # Resize auf volle Gr��e (bilinear interpolation)
            from scipy.ndimage import zoom
            noise_scaled = zoom(noise, frequency, order=1)[:height, :width]

            heightmap += noise_scaled * amplitude

        # Normalisieren und Skalieren
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
        min_h, max_h = height_range
        heightmap = heightmap * (max_h - min_h) + min_h

        logger.info(f"Perlin heightmap generated: {width}x{height}")
        return heightmap
