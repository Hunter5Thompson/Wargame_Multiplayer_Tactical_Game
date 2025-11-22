# AETHER-GS (Project Skyfield)

![Status](https://img.shields.io/badge/Status-Prototype-orange) ![Stack](https://img.shields.io/badge/Stack-FastAPI%20%7C%20Three.js%20%7C%20Spark-blue) ![License](https://img.shields.io/badge/License-MIT-green)

**Next-Gen Tactical Wargaming Platform powered by Gaussian Splatting.**

Aether-GS (Skyfield) ist ein leichtgewichtiges Ausbildungssystem für taktische Entscheidungsfindung. Es ersetzt abstrakte 2D-Karten durch fotorealistische, aus Satellitendaten oder KI generierte 3D-Umgebungen (Gaussian Splats), ohne die Komplexität einer vollen Game-Engine wie UE5.

---

## 🎯 Zielsetzung

*   **Schnelligkeit:** Erstellung von Szenarien (Urban, Wald, Wüste) in Minuten statt Wochen.
*   **Realismus:** Nutzung von Gaussian Splatting für echte Gelände-Visualisierung.
*   **Fokus:** Kein "Spiel", sondern ein Führungsmittel. Reduzierte UI, Fokus auf Gelände und Feind.

## 🏗 Architektur

Das System nutzt eine **Duale Pipeline**:
1.  **Visual Layer:** Rendert `.splat` Dateien via **Spark Renderer** (High-Fidelity Visuals).
2.  **Physics Layer:** Eine unsichtbare Heightmap/Mesh-Ebene für Raycasting, LOS (Line of Sight) und Bewegung.

```mermaid
graph TD
    A[Scenario Agent / User] -->|Prompt| B(Backend / SDXL);
    B -->|Generate| C[Synthetic Sat Image];
    C -->|Depth Est.| D[Heightmap (Physics)];
    C -->|3D Lifting| E[Gaussian Splat (Visuals)];
    
    subgraph Browser [Frontend / Three.js]
    E --> F[Spark Renderer Layer];
    D --> G[Invisible Physics Mesh];
    H[User Input] -->|Raycast| G;
    G -->|Snap Position| I[Unit Marker];
    I -->|Overlay| F;
    end
```

## 🚀 Quick Start

### Voraussetzungen
*   Python 3.11+ & [uv](https://github.com/astral-sh/uv)
*   Node.js 20+

### Installation

1.  **Repo klonen:**
    ```bash
    git clone https://github.com/Hunter5Thompson/Wargame_Multiplayer_Tactical_Game.git
    cd aether-gs
    ```

2.  **Backend Setup (mit uv):**
    ```bash
    uv sync
    ```

3.  **Frontend Setup:**
    ```bash
    cd frontend
    npm install
    ```

### Running the System

**Backend (API & WebSocket):**
```bash
# Startet FastAPI Server auf http://localhost:8000
uv run uvicorn backend.main:app --reload
```

**Frontend (Client):**
```bash
cd frontend
npm run dev
# Startet UI auf http://localhost:5173
```

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| **PackageManager** | `uv` (Python), `npm` (JS) |
| **Backend** | Python FastAPI, WebSockets |
| **Generative AI** | Stable Diffusion XL, DepthAnythingV2 |
| **3D Engine** | Three.js + **Spark Renderer** |
| **Symbology** | milsymbol.js (NATO APP-6) |

## 🗺 Roadmap

- [x] **v0.1:** Three.js Prototyp mit Dual-Layer Architektur.
- [ ] **v0.2:** Integration von Spark Renderer für echte .splat Dateien.
- [ ] **v0.3:** Python Backend Integration für automatische Map-Generierung.
- [ ] **v0.4:** Multiplayer Sync (WebSocket) für Blue/Red Team.
- [ ] **v1.0:** Fog of War & KI-Gegenspieler (Ollama Integration).

## 🤝 Contributing

Beachte den `DOCS/` Ordner für Architektur-Entscheidungen (ADRs).
Pull Requests sind willkommen für Module: `Unit Logic`, `AI Agents`, `Splat Optimization`.

---
*Initiated by Rob / Project Skyfield Owner*
