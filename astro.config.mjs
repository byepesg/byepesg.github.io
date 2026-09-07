import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  site: 'https://byepesg.github.io',
  integrations: [tailwind()],
  build: {
    dist: 'dist'
  }
});
