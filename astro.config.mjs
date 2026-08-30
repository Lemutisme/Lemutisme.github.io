import { defineConfig } from 'astro/config';
import vue from '@astrojs/vue';

export default defineConfig({
  site: 'https://www.duo-zhou.com',
  trailingSlash: 'always',
  integrations: [vue()],
});
