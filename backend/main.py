from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Dict
import json
import logging
import numpy as np
from PIL import Image
from pathlib import Path
import os

from backend.services.gamestate import (
    StateManager, GameState, Unit, UnitType, Team, UnitStatus, Position
)
from backend.services.generator import ScenarioGenerator, QuickMapGenerator

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aether-GS Backend", version="0.1.0")

# --- CORS Configuration (Damit Frontend auf Backend zugreifen darf) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Production hier nur die Frontend-URL erlauben
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Connection Manager für Multiplayer ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_to_player: Dict[WebSocket, str] = {}  # WebSocket -> player_id mapping

    async def connect(self, websocket: WebSocket, player_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if player_id:
            self.client_to_player[websocket] = player_id
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
            if websocket in self.client_to_player:
                del self.client_to_player[websocket]
        except ValueError:
            pass  # Already removed
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict, sender: WebSocket = None):
        """Sendet eine Nachricht an alle verbundenen Clients (außer optional dem Sender)"""
        json_msg = json.dumps(message)
        for connection in self.active_connections:
            # Don't echo back to sender for efficiency
            if connection != sender:
                try:
                    await connection.send_text(json_msg)
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")

manager = ConnectionManager()
state_manager = StateManager()

# Initialize default game session
default_session = state_manager.create_session("default", map_size=(200, 200))

# Initialize Generators
scenario_generator = ScenarioGenerator(device="cpu") # Force CPU for compatibility if no CUDA
# Ensure output directory exists
OUTPUT_DIR = Path("generated_maps")
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount Static Files for generated assets
app.mount("/assets", StaticFiles(directory=OUTPUT_DIR), name="assets")

# --- Routes ---

@app.get("/")
async def root():
    return {"status": "online", "system": "Skyfield Aether-GS Core"}

@app.post("/api/map/generate")
async def generate_map(type: str = "mixed", seed: int = None):
    """Generates a new map and updates the game state"""
    try:
        logger.info(f"Starting map generation: {type}")
        
        # 1. Generate Assets
        # For prototype speed, use QuickMapGenerator for heightmap if type is 'quick'
        if type == "quick":
            heightmap = QuickMapGenerator.generate_perlin_heightmap(width=200, height=200, seed=seed)
            # Create dummy image for consistency
            img = Image.new('RGB', (200, 200), color = 'green')
            img.save(OUTPUT_DIR / "satellite.png")
            np.save(OUTPUT_DIR / "heightmap.npy", heightmap)
        else:
            # Full AI Pipeline
            result = scenario_generator.generate_scenario(
                terrain_type=type,
                output_dir=OUTPUT_DIR,
                image_size=(512, 512), # Smaller for speed
                height_range=(0.0, 50.0),
                seed=seed
            )
            heightmap = result["heightmap"]

        # 2. Update Game State
        # Resize heightmap to match game grid if necessary, or update game grid size
        # For now, we assume 1px = 1m
        default_session.heightmap.load_from_array(heightmap)
        
        return {
            "status": "success", 
            "map_url": "/assets/satellite.png",
            "heightmap_url": "/assets/heightmap.npy",
            "splat_url": "/assets/scene.splat" # Placeholder until we have real splat generation
        }
    except Exception as e:
        logger.error(f"Map generation failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.on_event("startup")
async def startup_event():
    """Initialize demo units on startup"""
    # Add demo units to default session
    unit1 = Unit(
        id="u1",
        name="Leopard 2A7",
        unit_type=UnitType.ARMOR,
        team=Team.BLUE,
        position=Position(x=-10.0, y=0.0, z=0.0)
    )
    unit2 = Unit(
        id="u2",
        name="T-90M",
        unit_type=UnitType.ARMOR,
        team=Team.RED,
        position=Position(x=20.0, y=0.0, z=0.0)
    )
    default_session.add_unit(unit1)
    default_session.add_unit(unit2)
    logger.info("Demo units initialized")

@app.websocket("/ws/tactical")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    # Send initial game state to new client
    try:
        initial_state = {
            "type": "INIT",
            "units": default_session.to_dict()["units"]
        }
        await websocket.send_text(json.dumps(initial_state))
    except Exception as e:
        logger.error(f"Error sending initial state: {e}")

    try:
        while True:
            # Warten auf Nachrichten vom Frontend (z.B. Unit Move)
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)

                # SERVER-SIDE VALIDATION & PROCESSING
                if payload.get("type") == "MOVE":
                    unit_id = payload.get("id")
                    target_x = payload.get("x")
                    target_z = payload.get("z")

                    if not all([unit_id, target_x is not None, target_z is not None]):
                        logger.warning(f"Invalid MOVE payload: {payload}")
                        continue

                    # Validate and process move through GameState
                    target_pos = Position(x=float(target_x), y=0.0, z=float(target_z))

                    # Check if unit exists
                    if unit_id not in default_session.units:
                        logger.warning(f"Unit {unit_id} not found")
                        continue

                    # Validate position is within map bounds
                    map_half = 100  # CONFIG.mapWidth / 2
                    if abs(target_x) > map_half or abs(target_z) > map_half:
                        logger.warning(f"Position out of bounds: {target_x}, {target_z}")
                        continue

                    # Execute validated move
                    success = default_session.move_unit(unit_id, target_pos)

                    if success:
                        # Broadcast validated move to all clients
                        validated_payload = {
                            "type": "MOVE",
                            "id": unit_id,
                            "x": target_x,
                            "z": target_z
                        }
                        logger.info(f"Move validated: {unit_id} to ({target_x}, {target_z})")
                        await manager.broadcast(validated_payload, sender=websocket)
                    else:
                        logger.warning(f"Move validation failed for unit {unit_id}")

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {data}")
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        manager.disconnect(websocket)