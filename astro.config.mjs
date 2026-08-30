import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import { rehypeReportMathErrors } from './src/lib/rehype-fail-on-math-error.mjs';

export default defineConfig({
  site: 'https://www.duo-zhou.com',
  trailingSlash: 'always',
  markdown: {
    // Astro 7 defaults to Sätteri; opt back into the unified pipeline
    // explicitly rather than relying on the deprecated top-level plugin keys.
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [[rehypeKatex, {
        // KaTeX's default: visual HTML plus MathML for screen readers.
        output: 'htmlAndMathml',
        strict: 'warn',
        // rehype-katex catches KaTeX's own errors regardless of this flag;
        // rehypeReportMathErrors below is what actually fails the build.
        throwOnError: true,
        macros: {
          '\\R': '\\mathbb{R}',
          '\\E': '\\mathbb{E}',
          '\\P': '\\mathbb{P}',
        },
      }], rehypeReportMathErrors],
    }),
  },
});
