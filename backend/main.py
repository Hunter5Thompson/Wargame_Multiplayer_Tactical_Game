from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json

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

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass  # Already removed
        print(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict, sender: WebSocket = None):
        """Sendet eine Nachricht an alle verbundenen Clients (außer optional dem Sender)"""
        json_msg = json.dumps(message)
        for connection in self.active_connections:
            # Don't echo back to sender for efficiency
            if connection != sender:
                await connection.send_text(json_msg)

manager = ConnectionManager()

# --- Routes ---

@app.get("/")
async def root():
    return {"status": "online", "system": "Skyfield Aether-GS Core"}

@app.websocket("/ws/tactical")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Warten auf Nachrichten vom Frontend (z.B. Unit Move)
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)

                # Logik: Wenn "MOVE", dann Broadcast an alle anderen
                # Hier könnte später auch Server-Side Validation (Cheating prevention) stehen

                print(f"Received: {payload}")
                await manager.broadcast(payload, sender=websocket)
            except json.JSONDecodeError:
                print(f"Invalid JSON received: {data}")
            except Exception as e:
                print(f"Error processing message: {e}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Unexpected error: {e}")
        manager.disconnect(websocket)