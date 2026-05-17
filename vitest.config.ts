import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['nn-chess/**/*.test.ts', 'nn-chess/**/*.test.tsx'],
    environment: 'node',
    globals: false,
    coverage: {
      include: ['nn-chess/core/**/*.ts', 'nn-chess/adapters/**/*.ts'],
      reporter: ['text', 'html'],
    },
  },
});
