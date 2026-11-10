import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const blog = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    date: z.date(),
    categories: z.string().optional(),
    // Optional SEO summary; falls back to an excerpt of the post body.
    description: z.string().optional(),
  }),
});

export const collections = { blog };
