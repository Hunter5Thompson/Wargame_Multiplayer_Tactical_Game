import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    host: true, // Erlaubt Zugriff im Netzwerk
    port: 5173,
  },
  build: {
    target: 'esnext' // Wichtig für Top-Level Await (Spark/Three)
  }
})