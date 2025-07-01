import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://192.168.0.46:8001', // FastAPI 백엔드 주소(포트 맞게 수정)
    },
  },
});
