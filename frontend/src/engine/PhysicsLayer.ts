/**
 * PhysicsLayer - Manages collision detection and raycasting
 * Provides invisible physics mesh for gameplay interactions
 */

import * as THREE from 'three';

export interface CollisionResult {
    hit: boolean;
    point?: THREE.Vector3;
    normal?: THREE.Vector3;
    distance?: number;
}

export class PhysicsLayer {
    private raycaster: THREE.Raycaster;
    private groundPlane: THREE.Mesh;
    private collisionObjects: THREE.Object3D[] = [];

    constructor(mapWidth: number, mapHeight: number) {
        this.raycaster = new THREE.Raycaster();

        // Create invisible ground plane for raycasting
        const geometry = new THREE.PlaneGeometry(mapWidth, mapHeight);
        geometry.rotateX(-Math.PI / 2);

        const material = new THREE.MeshBasicMaterial({
            visible: false // Can be set to true for debugging
        });

        this.groundPlane = new THREE.Mesh(geometry, material);
        this.groundPlane.name = 'PhysicsGround';
    }

    public getGroundPlane(): THREE.Mesh {
        return this.groundPlane;
    }

    public raycastGround(
        mouse: THREE.Vector2,
        camera: THREE.Camera
    ): CollisionResult {
        this.raycaster.setFromCamera(mouse, camera);
        const intersects = this.raycaster.intersectObject(this.groundPlane);

        if (intersects.length > 0) {
            const hit = intersects[0];
            return {
                hit: true,
                point: hit.point,
                normal: hit.face?.normal,
                distance: hit.distance
            };
        }

        return { hit: false };
    }

    public raycastObjects(
        mouse: THREE.Vector2,
        camera: THREE.Camera,
        objects: THREE.Object3D[]
    ): CollisionResult {
        this.raycaster.setFromCamera(mouse, camera);
        const intersects = this.raycaster.intersectObjects(objects, true);

        if (intersects.length > 0) {
            const hit = intersects[0];
            return {
                hit: true,
                point: hit.point,
                normal: hit.face?.normal,
                distance: hit.distance
            };
        }

        return { hit: false };
    }

    public checkCollision(
        position: THREE.Vector3,
        radius: number
    ): boolean {
        // Simple sphere collision check
        for (const obj of this.collisionObjects) {
            const distance = position.distanceTo(obj.position);
            if (distance < radius) {
                return true; // Collision detected
            }
        }
        return false;
    }

    public addCollisionObject(object: THREE.Object3D) {
        this.collisionObjects.push(object);
    }

    public removeCollisionObject(object: THREE.Object3D) {
        const index = this.collisionObjects.indexOf(object);
        if (index > -1) {
            this.collisionObjects.splice(index, 1);
        }
    }

    public isPositionValid(
        position: THREE.Vector3,
        mapBounds: { width: number; height: number }
    ): boolean {
        const halfWidth = mapBounds.width / 2;
        const halfHeight = mapBounds.height / 2;

        return (
            position.x >= -halfWidth &&
            position.x <= halfWidth &&
            position.z >= -halfHeight &&
            position.z <= halfHeight
        );
    }

    public dispose() {
        this.collisionObjects = [];
    }
}
