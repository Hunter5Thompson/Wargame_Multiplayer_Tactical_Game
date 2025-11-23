import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

export class SplatLoader {
    private loader: PLYLoader;
    private scene: THREE.Scene;

    constructor(scene: THREE.Scene) {
        this.scene = scene;
        this.loader = new PLYLoader();
    }

    public async load(url: string): Promise<THREE.Points> {
        return new Promise((resolve, reject) => {
            this.loader.load(
                url,
                (geometry) => {
                    // Create material for point cloud
                    const material = new THREE.PointsMaterial({
                        size: 0.1,
                        vertexColors: true,
                        sizeAttenuation: true,
                        transparent: true,
                        opacity: 0.8
                    });

                    const points = new THREE.Points(geometry, material);
                    
                    // Rotate to match coordinate system (if needed)
                    // points.rotation.x = -Math.PI / 2; 

                    this.scene.add(points);
                    console.log(`[SplatLoader] Loaded ${url}`);
                    resolve(points);
                },
                (xhr) => {
                    console.log(`[SplatLoader] ${(xhr.loaded / xhr.total * 100)}% loaded`);
                },
                (error) => {
                    console.error('[SplatLoader] Error loading splat:', error);
                    reject(error);
                }
            );
        });
    }
}
