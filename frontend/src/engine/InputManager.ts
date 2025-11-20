/**
 * InputManager - Centralized input handling for the game
 * Manages keyboard, mouse, and touch input events
 */

export type InputEvent = {
    type: 'keydown' | 'keyup' | 'click' | 'mousemove';
    key?: string;
    mouseX?: number;
    mouseY?: number;
    timestamp: number;
};

export class InputManager {
    private keyStates: Map<string, boolean> = new Map();
    private listeners: Map<string, ((event: InputEvent) => void)[]> = new Map();
    private mousePosition = { x: 0, y: 0 };

    constructor() {
        this.setupEventListeners();
    }

    private setupEventListeners() {
        window.addEventListener('keydown', (e) => this.handleKeyDown(e));
        window.addEventListener('keyup', (e) => this.handleKeyUp(e));
        window.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    }

    private handleKeyDown(event: KeyboardEvent) {
        this.keyStates.set(event.key, true);
        this.emit({
            type: 'keydown',
            key: event.key,
            timestamp: Date.now()
        });
    }

    private handleKeyUp(event: KeyboardEvent) {
        this.keyStates.set(event.key, false);
        this.emit({
            type: 'keyup',
            key: event.key,
            timestamp: Date.now()
        });
    }

    private handleMouseMove(event: MouseEvent) {
        this.mousePosition = { x: event.clientX, y: event.clientY };
        this.emit({
            type: 'mousemove',
            mouseX: event.clientX,
            mouseY: event.clientY,
            timestamp: Date.now()
        });
    }

    public on(eventType: string, callback: (event: InputEvent) => void) {
        if (!this.listeners.has(eventType)) {
            this.listeners.set(eventType, []);
        }
        this.listeners.get(eventType)?.push(callback);
    }

    private emit(event: InputEvent) {
        const callbacks = this.listeners.get(event.type);
        if (callbacks) {
            callbacks.forEach(cb => cb(event));
        }
    }

    public isKeyPressed(key: string): boolean {
        return this.keyStates.get(key) || false;
    }

    public getMousePosition() {
        return { ...this.mousePosition };
    }

    public dispose() {
        this.listeners.clear();
        this.keyStates.clear();
    }
}
