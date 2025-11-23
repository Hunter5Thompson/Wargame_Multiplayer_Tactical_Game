import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SplatLoader } from './engine/SplatLoader';
import './style.css';

// --- Types ---
type UnitData = {
    id: string;
    name: string;
    team: 'blue' | 'red';
    x: number;
    z: number;
};

type NetworkMessage =
    | { type: 'MOVE'; id: string; x: number; z: number }
    | { type: 'INIT'; units: Record<string, any> };

type UnitMesh = THREE.Group & {
    targetPosition?: THREE.Vector3;
    userData: UnitData & { selectionRing: THREE.Mesh };
};

// --- Globals ---
const CONFIG = {
    wsUrl: 'ws://localhost:8000/ws/tactical',
    mapWidth: 200,
    mapHeight: 200,
};

let socket: WebSocket;
let scene: THREE.Scene, camera: THREE.PerspectiveCamera, renderer: THREE.WebGLRenderer;
let controls: OrbitControls;
let raycaster = new THREE.Raycaster();
let mouse = new THREE.Vector2();
let groundPlane: THREE.Mesh;
let splatLoader: SplatLoader;
let units: Map<string, UnitMesh> = new Map(); // Map ID -> Mesh
let selectedUnitId: string | null = null;
let isDragging = false;
let dragOffset = new THREE.Vector3();

// --- Network ---
function initNetwork() {
    socket = new WebSocket(CONFIG.wsUrl);

    socket.onopen = () => {
        console.log("[NET] Connected to Skyfield Core");
        const statusElement = document.getElementById('net-status-text');
        const dotElement = document.getElementById('net-status-dot');
        if (statusElement && dotElement) {
            statusElement.innerText = "ONLINE";
            statusElement.style.color = "var(--color-primary)";
            dotElement.className = "status-dot online";
        }
    };

    socket.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            handleNetworkMessage(msg);
        } catch (e) {
            console.error('[NET] Invalid JSON received:', event.data);
        }
    };

    socket.onerror = (error) => {
        console.error('[NET] WebSocket error:', error);
        const statusElement = document.getElementById('net-status-text');
        const dotElement = document.getElementById('net-status-dot');
        if (statusElement && dotElement) {
            statusElement.innerText = "ERROR";
            statusElement.style.color = "var(--color-danger)";
            dotElement.className = "status-dot error";
        }
    };

    socket.onclose = () => {
        console.log('[NET] Connection closed');
        const statusElement = document.getElementById('net-status-text');
        const dotElement = document.getElementById('net-status-dot');
        if (statusElement && dotElement) {
            statusElement.innerText = "OFFLINE";
            statusElement.style.color = "var(--color-text-muted)";
            dotElement.className = "status-dot offline";
        }
    };
}

function sendMove(id: string, x: number, z: number) {
    if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'MOVE', id, x, z }));
    }
}

function handleNetworkMessage(msg: NetworkMessage) {
    if (msg.type === 'INIT') {
        // Handle initial game state from server
        console.log('[NET] Received initial state:', msg.units);
        // Units are already spawned locally, just sync positions if needed
        for (const [unitId, unitData] of Object.entries(msg.units)) {
            const unit = units.get(unitId);
            if (unit && unitData.position) {
                const pos = unitData.position as { x: number; y: number; z: number };
                unit.position.set(pos.x, 0, pos.z);
            }
        }
    } else if (msg.type === 'MOVE') {
        const unit = units.get(msg.id);
        if (unit) {
            // Smooth interpolation using target position
            unit.targetPosition = new THREE.Vector3(msg.x, 0, msg.z);

            // Update UI if selected
            if (selectedUnitId === msg.id) updateUnitUI(msg.id, msg.x, msg.z);
        }
    }
}

// --- Engine ---
function init() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    // Camera
    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 80, 80);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('canvas-container')?.appendChild(renderer.domElement);

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI / 2.1;

    // Lights
    const dirLight = new THREE.DirectionalLight(0xffffff, 2);
    dirLight.position.set(50, 100, 50);
    scene.add(dirLight);
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));

    // Environment (Physics & Visuals)
    splatLoader = new SplatLoader(scene);
    setupEnvironment();
    setupUI();

    // Init Demo Units
    spawnUnit({ id: 'u1', name: 'Leopard 2A7', team: 'blue', x: -10, z: 10 });
    spawnUnit({ id: 'u2', name: 'T-90M', team: 'red', x: 20, z: -20 });

    // Inputs
    window.addEventListener('resize', onResize);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerUp);
    document.getElementById('btn-reset')?.addEventListener('click', () => controls.reset());

    initNetwork();
    animate();
}

function setupEnvironment() {
    // 1. Visual Layer (Fake Splat for now)
    const canvas = document.createElement('canvas');
    canvas.width = 512; canvas.height = 512;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('Failed to get 2D context');
        return;
    }
    ctx.fillStyle = '#2b332b'; ctx.fillRect(0, 0, 512, 512);
    // Noise
    for (let i = 0; i < 1000; i++) {
        ctx.fillStyle = Math.random() > 0.5 ? '#354035' : '#202820';
        ctx.beginPath(); ctx.arc(Math.random() * 512, Math.random() * 512, Math.random() * 5, 0, Math.PI * 2); ctx.fill();
    }
    // Roads
    ctx.strokeStyle = '#444'; ctx.lineWidth = 20;
    ctx.beginPath(); ctx.moveTo(0, 256); ctx.lineTo(512, 256); ctx.stroke();

    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.MeshBasicMaterial({ map: tex });
    const geo1 = new THREE.PlaneGeometry(CONFIG.mapWidth, CONFIG.mapHeight);
    geo1.rotateX(-Math.PI / 2);
    const visual = new THREE.Mesh(geo1, mat);
    visual.position.y = -0.1;
    scene.add(visual);

    // 2. Physics Layer (Invisible)
    const physMat = new THREE.MeshBasicMaterial({ visible: false }); // set true to debug
    const geo2 = new THREE.PlaneGeometry(CONFIG.mapWidth, CONFIG.mapHeight);
    geo2.rotateX(-Math.PI / 2);
    groundPlane = new THREE.Mesh(geo2, physMat);
    scene.add(groundPlane);

    const grid = new THREE.GridHelper(CONFIG.mapWidth, 20, 0x000000, 0x555555);
    grid.position.y = 0.05;
    grid.material.opacity = 0.2;
    grid.material.transparent = true;
    scene.add(grid);
}

function spawnUnit(data: UnitData): UnitMesh {
    const group = new THREE.Group() as UnitMesh;
    group.position.set(data.x, 0, data.z);

    // 3D Mesh
    const color = data.team === 'blue' ? 0x3388ff : 0xff3333;
    const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(4, 2.5, 6),
        new THREE.MeshStandardMaterial({ color: color })
    );
    mesh.position.y = 1.25;
    group.add(mesh);

    // Selection Ring
    const ring = new THREE.Mesh(
        new THREE.RingGeometry(5, 5.5, 32).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial({ color: 0x00ff00, visible: false })
    );
    ring.position.y = 0.1;
    group.add(ring);

    // Metadata
    group.userData = { ...data, selectionRing: ring };

    scene.add(group);
    units.set(data.id, group);
    return group;
}

// --- Interaction ---

function onPointerDown(e: PointerEvent) {
    // Only interact if clicking on the canvas (ignore UI)
    if (!renderer.domElement.contains(e.target as Node)) return;

    if (e.button !== 0) return;
    updateMouse(e);
    raycaster.setFromCamera(mouse, camera);

    // Check Units
    const intersectObjects = Array.from(units.values()).map(u => u.children[0]); // check box meshes
    const hits = raycaster.intersectObjects(intersectObjects);

    if (hits.length > 0) {
        const group = hits[0].object.parent as THREE.Group;
        selectUnit(group.userData.id);

        isDragging = true;
        controls.enabled = false;

        // Calc offset
        const groundHits = raycaster.intersectObject(groundPlane);
        if (groundHits.length > 0) {
            dragOffset.subVectors(group.position, groundHits[0].point);
        }
    } else {
        // Deselect if clicking ground
        const groundHits = raycaster.intersectObject(groundPlane);
        if (groundHits.length > 0) {
            selectUnit(null);
        }
    }
}

function onPointerMove(e: PointerEvent) {
    updateMouse(e);

    // Only interact if pointer is over the canvas (ignore UI)
    if (isDragging && !renderer.domElement.contains(e.target as Node)) {
        // Cancel drag if pointer moves over UI
        onPointerUp();
        return;
    }

    if (isDragging && selectedUnitId) {
        raycaster.setFromCamera(mouse, camera);
        const hits = raycaster.intersectObject(groundPlane);

        if (hits.length > 0) {
            const target = hits[0].point.add(dragOffset);
            const unit = units.get(selectedUnitId);

            if (unit) {
                // Local Update
                unit.position.set(target.x, 0, target.z);
                updateUnitUI(selectedUnitId, target.x, target.z);

                // Network Update (send to server)
                sendMove(selectedUnitId, target.x, target.z);
            }
        }
    }
}

function onPointerUp() {
    isDragging = false;
    controls.enabled = true;
}

function selectUnit(id: string | null) {
    // Old deselect
    if (selectedUnitId) {
        const u = units.get(selectedUnitId);
        if (u) u.userData.selectionRing.visible = false;
    }

    selectedUnitId = id;
    const ui = document.getElementById('unit-panel');
    if (!ui) return;

    if (id) {
        const u = units.get(id);
        if (!u) return;

        u.userData.selectionRing.visible = true;
        ui.style.display = 'block';

        const nameElement = document.getElementById('u-name');
        if (nameElement) {
            nameElement.innerText = u.userData.name;
        }

        updateUnitUI(id, u.position.x, u.position.z);
    } else {
        ui.style.display = 'none';
    }
}

function updateUnitUI(id: string, x: number, z: number) {
    if (selectedUnitId === id) {
        const coordsElement = document.getElementById('u-coords');
        if (coordsElement) {
            coordsElement.innerText = `GRID: ${x.toFixed(1)} / ${z.toFixed(1)}`;
        }
    }
}

function updateMouse(e: PointerEvent) {
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
}

function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();

    // Smooth movement interpolation for all units
    units.forEach((unit) => {
        if (unit.targetPosition) {
            const distance = unit.position.distanceTo(unit.targetPosition);

            // If close enough, snap to target
            if (distance < 0.1) {
                unit.position.copy(unit.targetPosition);
                unit.targetPosition = undefined;
            } else {
                // Smooth lerp with speed factor
                const lerpFactor = Math.min(0.1, distance * 0.05);
                unit.position.lerp(unit.targetPosition, lerpFactor);
            }
        }
    });

    renderer.render(scene, camera);
}

// --- UI & Map Gen ---
function setupUI() {
    // Create Main UI Layer
    const uiLayer = document.createElement('div');
    uiLayer.id = 'ui-layer';
    uiLayer.innerHTML = `
        <div class="top-bar">
            <!-- Map Controls -->
            <div id="map-controls" class="glass-panel">
                <h3>Tactical Map</h3>
                <select id="map-type">
                    <option value="quick">Quick (Perlin Noise)</option>
                    <option value="mixed">Mixed Terrain (AI)</option>
                    <option value="desert">Desert Ops (AI)</option>
                    <option value="urban">Urban Center (AI)</option>
                </select>
                <button id="btn-gen" class="btn">Initialize Sector</button>
                <div id="gen-status">> SYSTEM READY</div>
            </div>

            <!-- Network Status -->
            <div class="glass-panel status-indicator">
                <div id="net-status-dot" class="status-dot offline"></div>
                <span id="net-status-text">CONNECTING...</span>
            </div>
        </div>

        <!-- Unit Panel (Bottom) -->
        <div id="unit-panel" class="glass-panel">
            <div class="unit-header">
                <span id="u-name" class="unit-name">UNIT NAME</span>
                <span class="status-dot online"></span>
            </div>
            <div class="unit-stats">
                <div class="stat-row"><span>GRID</span> <span id="u-coords" class="stat-val">-- / --</span></div>
                <div class="stat-row"><span>STATUS</span> <span class="stat-val" style="color: var(--color-primary)">ACTIVE</span></div>
            </div>
        </div>
    `;
    document.body.appendChild(uiLayer);

    document.getElementById('btn-gen')?.addEventListener('click', generateMap);
}

async function generateMap() {
    const type = (document.getElementById('map-type') as HTMLSelectElement).value;
    const status = document.getElementById('gen-status');
    if (status) status.innerText = "> GENERATING SECTOR...";

    try {
        const res = await fetch(`http://localhost:8000/api/map/generate?type=${type}`, { method: 'POST' });
        const data = await res.json();

        if (data.status === 'success') {
            if (status) status.innerText = "> LOADING ASSETS...";

            // Load Splat (Visuals)
            if (data.splat_url) {
                // Let's update the ground plane texture at least
                const texLoader = new THREE.TextureLoader();
                texLoader.load(`http://localhost:8000${data.map_url}`, (tex) => {
                    (groundPlane.material as THREE.MeshBasicMaterial).map = tex;
                    (groundPlane.material as THREE.MeshBasicMaterial).needsUpdate = true;
                    (groundPlane.material as THREE.MeshBasicMaterial).visible = true; // Make visible
                });
            }

            if (status) status.innerText = "> SECTOR INITIALIZED";
        } else {
            if (status) status.innerText = "> ERROR: " + data.message;
        }
    } catch (e) {
        console.error(e);
        if (status) status.innerText = "> NETWORK ERROR";
    }
}

// Boot
init();