// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    strictPort: true,

    proxy: {
      "/ui": {
        target: "http://127.0.0.1:5050",
        changeOrigin: true,
        secure: false,
      },
      "/health": {
        target: "http://127.0.0.1:5050",
        changeOrigin: true,
        secure: false,
      },
      // Optional but often useful for debugging / future UI features
      "/jobs": {
        target: "http://127.0.0.1:5050",
        changeOrigin: true,
        secure: false,
      },
      "/options": {
        target: "http://127.0.0.1:5050",
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
