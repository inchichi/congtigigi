import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitest/config'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [tailwindcss()],
  build: {
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        editor: fileURLToPath(new URL('./editor.html', import.meta.url))
      }
    }
  },
  server: {
    proxy: {
      '/api/openai': {
        target: 'https://api.openai.com',
        changeOrigin: true,
        secure: true,
        rewrite: (path) => path.replace(/^\/api\/openai/, '')
      },
      // 에디터의 Claude 호출을 서버사이드로 포워딩 → 브라우저 CORS 회피.
      '/api/anthropic': {
        target: 'https://api.anthropic.com',
        changeOrigin: true,
        secure: true,
        rewrite: (path) => path.replace(/^\/api\/anthropic/, '')
      },
      // AdaIN 스타일 트랜스퍼 로컬 Python 서비스 — style-service/server.py (포트는 그쪽 config.json).
      '/api/style': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/style/, '')
      }
    }
  },
  test: {
    include: ['src/**/*.test.ts']
  }
})
