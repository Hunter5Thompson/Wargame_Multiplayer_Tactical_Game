# -*- coding: utf-8 -*-
"""
Game State Management for Aether-GS Tactical Wargame
Verwaltet Units, Line of Sight (LOS), Bewegung und Multiplayer-Synchronisation
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import json


class UnitType(str, Enum):
    """NATO APP-6 konforme Unit-Typen"""
    INFANTRY = "infantry"
    ARMOR = "armor"
    ARTILLERY = "artillery"
    RECON = "recon"
    COMMAND = "command"
    LOGISTICS = "logistics"


class Team(str, Enum):
    """Team-Zugeh�rigkeit"""
    BLUE = "blue"  # Friendly Forces
    RED = "red"    # Opposition Forces
    NEUTRAL = "neutral"


class UnitStatus(str, Enum):
    """Unit-Status"""
    ACTIVE = "active"
    SUPPRESSED = "suppressed"
    DESTROYED = "destroyed"
    WITHDRAWN = "withdrawn"


@dataclass
class Position:
    """3D Position auf dem Schlachtfeld"""
    x: float
    y: float
    z: float  # H�he von Heightmap

    def distance_to(self, other: "Position") -> float:
        """Berechnet 3D Distanz zu anderer Position"""
        return np.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def distance_2d(self, other: "Position") -> float:
        """Berechnet 2D Distanz (nur x, y)"""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class Unit:
    """Taktische Einheit"""
    id: str
    name: str
    unit_type: UnitType
    team: Team
    position: Position
    status: UnitStatus = UnitStatus.ACTIVE
    strength: int = 100  # 0-100%
    speed: float = 1.0  # Bewegungsgeschwindigkeit (Einheiten/Sekunde)
    visibility_range: float = 500.0  # Sichtweite in Metern
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self):
        """Serialisierung f�r WebSocket-�bertragung"""
        return {
            "id": self.id,
            "name": self.name,
            "unit_type": self.unit_type.value,
            "team": self.team.value,
            "position": self.position.to_dict(),
            "status": self.status.value,
            "strength": self.strength,
            "speed": self.speed,
            "visibility_range": self.visibility_range,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }


class Heightmap:
    """Heightmap f�r Physics Layer (Bewegungsvalidierung & LOS)"""

    def __init__(self, width: int, height: int, resolution: float = 1.0):
        """
        Args:
            width: Breite der Heightmap in Pixeln
            height: H�he der Heightmap in Pixeln
            resolution: Meter pro Pixel (Standard: 1m/px)
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.data: np.ndarray = np.zeros((height, width), dtype=np.float32)

    def load_from_array(self, data: np.ndarray):
        """L�dt Heightmap-Daten aus NumPy Array"""
        if data.shape != (self.height, self.width):
            raise ValueError(f"Array shape {data.shape} doesn't match heightmap dimensions")
        self.data = data.astype(np.float32)

    def get_height(self, x: float, y: float) -> Optional[float]:
        """Gibt H�he an Position (x, y) zur�ck (mit bilinearer Interpolation)"""
        # Konvertiere World-Space zu Pixel-Coordinates
        px = int(x / self.resolution)
        py = int(y / self.resolution)

        if 0 <= px < self.width and 0 <= py < self.height:
            return float(self.data[py, px])
        return None

    def is_valid_position(self, pos: Position) -> bool:
        """Pr�ft ob Position auf g�ltiger Heightmap liegt"""
        h = self.get_height(pos.x, pos.y)
        return h is not None and abs(pos.z - h) < 5.0  # 5m Toleranz


class GameState:
    """Zentraler Spielzustand f�r eine Session"""

    def __init__(self, session_id: str, map_size: Tuple[int, int] = (1024, 1024)):
        self.session_id = session_id
        self.units: Dict[str, Unit] = {}
        self.heightmap = Heightmap(map_size[0], map_size[1])
        self.created_at = datetime.utcnow()
        self.turn_number = 0

    def add_unit(self, unit: Unit) -> bool:
        """F�gt Unit hinzu"""
        if unit.id in self.units:
            return False

        # Validiere Position auf Heightmap
        if not self.heightmap.is_valid_position(unit.position):
            print(f"Warning: Unit {unit.id} position invalid, adjusting height")
            height = self.heightmap.get_height(unit.position.x, unit.position.y)
            if height is not None:
                unit.position.z = height

        self.units[unit.id] = unit
        return True

    def remove_unit(self, unit_id: str) -> bool:
        """Entfernt Unit"""
        if unit_id in self.units:
            del self.units[unit_id]
            return True
        return False

    def move_unit(self, unit_id: str, target_pos: Position) -> bool:
        """
        Bewegt Unit zu neuer Position (mit Validierung)
        Returns: True wenn Bewegung erfolgreich
        """
        if unit_id not in self.units:
            return False

        unit = self.units[unit_id]

        # Validiere Zielposition
        height = self.heightmap.get_height(target_pos.x, target_pos.y)
        if height is None:
            return False  # Au�erhalb der Map

        # Snap auf Heightmap
        target_pos.z = height

        # Update Position
        unit.position = target_pos
        unit.last_updated = datetime.utcnow()
        return True

    def calculate_los(self, from_unit_id: str, to_unit_id: str,
                      sample_points: int = 20) -> bool:
        """
        Berechnet Line of Sight zwischen zwei Units
        Verwendet Raycast �ber Heightmap

        Returns: True wenn LOS frei ist
        """
        if from_unit_id not in self.units or to_unit_id not in self.units:
            return False

        from_unit = self.units[from_unit_id]
        to_unit = self.units[to_unit_id]

        # Berechne Ray zwischen Units
        start = from_unit.position
        end = to_unit.position

        for i in range(1, sample_points):
            t = i / sample_points
            x = start.x + t * (end.x - start.x)
            y = start.y + t * (end.y - start.y)
            z = start.z + t * (end.z - start.z)

            # Check ob Terrain im Weg ist
            terrain_height = self.heightmap.get_height(x, y)
            if terrain_height is not None and terrain_height > z:
                return False  # Blocked by terrain

        return True

    def get_units_in_range(self, position: Position, range_m: float,
                          team: Optional[Team] = None) -> List[Unit]:
        """Findet alle Units in Reichweite einer Position"""
        units_in_range = []

        for unit in self.units.values():
            distance = position.distance_2d(unit.position)
            if distance <= range_m:
                if team is None or unit.team == team:
                    units_in_range.append(unit)

        return units_in_range

    def get_visible_units(self, observer_unit_id: str) -> List[Unit]:
        """
        Gibt alle Units zur�ck die von observer_unit sichtbar sind
        (Ber�cksichtigt Reichweite UND Line of Sight)
        """
        if observer_unit_id not in self.units:
            return []

        observer = self.units[observer_unit_id]
        visible = []

        for unit_id, unit in self.units.items():
            if unit_id == observer_unit_id:
                continue  # Don't see yourself

            # Check Reichweite
            distance = observer.position.distance_2d(unit.position)
            if distance > observer.visibility_range:
                continue

            # Check LOS
            if self.calculate_los(observer_unit_id, unit_id):
                visible.append(unit)

        return visible

    def to_dict(self, include_heightmap: bool = False) -> dict:
        """Serialisiert kompletten Game State"""
        data = {
            "session_id": self.session_id,
            "units": {uid: unit.to_dict() for uid, unit in self.units.items()},
            "turn_number": self.turn_number,
            "created_at": self.created_at.isoformat(),
        }

        if include_heightmap:
            data["heightmap"] = {
                "width": self.heightmap.width,
                "height": self.heightmap.height,
                "resolution": self.heightmap.resolution,
                # Heightmap-Daten w�rden hier als Base64 oder separater Download kommen
            }

        return data


# Global State Manager (f�r schnelle Entwicklung - sp�ter Redis)
class StateManager:
    """Verwaltet mehrere Game Sessions"""

    def __init__(self):
        self.sessions: Dict[str, GameState] = {}

    def create_session(self, session_id: str, map_size: Tuple[int, int] = (1024, 1024)) -> GameState:
        """Erstellt neue Game Session"""
        if session_id in self.sessions:
            return self.sessions[session_id]

        state = GameState(session_id, map_size)
        self.sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> Optional[GameState]:
        """Holt existierende Session"""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """L�scht Session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


# Singleton Instance f�r globalen Zugriff
state_manager = StateManager()
