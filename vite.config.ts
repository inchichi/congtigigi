import { defineConfig } from 'vitest/config'

export default defineConfig({
  server: {
    proxy: {
      '/api/openai': {
        target: 'https://api.openai.com',
        changeOrigin: true,
        secure: true,
        rewrite: (path) => path.replace(/^\/api\/openai/, '')
      }
    }
  },
  test: {
    include: ['src/**/*.test.ts']
  }
})
