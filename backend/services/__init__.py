"""
Backend Services für Aether-GS Tactical Wargame System

Verfügbare Module:
- gamestate: Game State Management, Unit-Verwaltung, Line of Sight
- generator: Map & Scenario Generation via AI (SDXL, Depth Estimation)
- splat_proc: Gaussian Splat Processing & 3D Lifting
"""

from .gamestate import (
    GameState,
    StateManager,
    Unit,
    UnitType,
    Team,
    UnitStatus,
    Position,
    Heightmap,
    state_manager,
)

from .generator import (
    ScenarioGenerator,
    QuickMapGenerator,
    TerrainType,
)

from .splat_proc import (
    GaussianSplat,
    SplatFile,
    ImageToSplatConverter,
    SplatRenderer,
)

__all__ = [
    # Game State
    "GameState",
    "StateManager",
    "state_manager",
    "Unit",
    "UnitType",
    "Team",
    "UnitStatus",
    "Position",
    "Heightmap",
    # Generator
    "ScenarioGenerator",
    "QuickMapGenerator",
    "TerrainType",
    # Splat Processing
    "GaussianSplat",
    "SplatFile",
    "ImageToSplatConverter",
    "SplatRenderer",
]
