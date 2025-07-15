import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    fs: {
      allow: ['.'], // !!프로젝트 루트 상위 폴더를 통째로 읽어오도록 허용(TODO: 운영에서는 제거할 것)
    },
    proxy: {
      '/ai-api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/ai-api/, '/api')
      }
    }
  },
  resolve: {
    alias: {
      '@': '/src', // root 경로를 '@'로 사용할 수 있도록 설정. 예시: @/components/common/AlertDialog
      '@tailwind-config': path.resolve(__dirname, 'tailwind.config.js'),
    },
  },
})
