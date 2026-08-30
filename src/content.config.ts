import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const blog = defineCollection({
  loader: glob({ base: './src/content/blog', pattern: '**/*.md' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    publishedAt: z.coerce.date(),
    updatedAt: z.coerce.date().optional(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
  }),
});

const projects = defineCollection({
  loader: glob({ base: './src/content/projects', pattern: '**/*.md' }),
  schema: z.object({
    title: z.string(),
    tagline: z.string(),
    description: z.string(),
    status: z.string(),
    period: z.string(),
    citationYear: z.number().int(),
    order: z.number(),
    tags: z.array(z.string()).default([]),
    links: z.array(z.object({
      label: z.string(),
      url: z.url(),
    })).default([]),
  }),
});

export const collections = { blog, projects };
