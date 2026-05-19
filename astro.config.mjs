import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import react from '@astrojs/react';

export default defineConfig({
  site: 'https://lucaswiman.github.io',
  integrations: [react()],
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
    },
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  vite: {
    resolve: {
      alias: {
        '@nn-chess': new URL('./nn-chess/', import.meta.url).pathname,
      },
    },
  },
});
