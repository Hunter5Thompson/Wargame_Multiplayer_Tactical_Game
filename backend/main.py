from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import json
import logging

from backend.services.gamestate import (
    StateManager, GameState, Unit, UnitType, Team, UnitStatus, Position
)

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

# --- Routes ---

@app.get("/")
async def root():
    return {"status": "online", "system": "Skyfield Aether-GS Core"}

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