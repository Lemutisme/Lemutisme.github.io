import { getCollection } from 'astro:content';
import type { APIRoute } from 'astro';
import { buildBibTeX } from '../../../lib/citation';

interface CitationProps {
  id: string;
  title: string;
  year: number;
}

export async function getStaticPaths() {
  const posts = await getCollection('blog', ({ data }) => !data.draft);

  return posts.map((post) => ({
    params: { id: post.id },
    props: {
      id: post.id,
      title: post.data.title,
      year: post.data.publishedAt.getUTCFullYear(),
    } satisfies CitationProps,
  }));
}

export const GET: APIRoute = ({ props, site }) => {
  const { id, title, year } = props as CitationProps;
  const url = new URL(`/blog/${id}/`, site ?? 'https://www.duo-zhou.com');
  const bibtex = buildBibTeX({ id, title, year, url: url.toString(), note: 'Research note' });

  return new Response(bibtex, {
    headers: {
      'Content-Type': 'application/x-bibtex; charset=utf-8',
      'Content-Disposition': `attachment; filename="${id}.bib"`,
    },
  });
};

