/**
 * Renderer - Abstraction layer for Three.js rendering
 * Manages scene, camera, and render loop configuration
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export interface RendererConfig {
    antialias?: boolean;
    shadowMap?: boolean;
    shadowMapType?: THREE.ShadowMapType;
}

export class Renderer {
    public scene: THREE.Scene;
    public camera: THREE.PerspectiveCamera;
    public renderer: THREE.WebGLRenderer;
    public controls: OrbitControls;

    constructor(container: HTMLElement, config: RendererConfig = {}) {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x111111);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            45,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 80, 80);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: config.antialias !== false
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        if (config.shadowMap) {
            this.renderer.shadowMap.enabled = true;
            this.renderer.shadowMap.type = config.shadowMapType || THREE.PCFSoftShadowMap;
        }

        container.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxPolarAngle = Math.PI / 2.1;

        // Lighting
        this.setupLighting();

        // Handle resize
        window.addEventListener('resize', () => this.onResize());
    }

    private setupLighting() {
        const dirLight = new THREE.DirectionalLight(0xffffff, 2);
        dirLight.position.set(50, 100, 50);
        dirLight.castShadow = true;
        this.scene.add(dirLight);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);
    }

    private onResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    public render() {
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    public dispose() {
        this.renderer.dispose();
        this.controls.dispose();
        window.removeEventListener('resize', () => this.onResize());
    }
}
