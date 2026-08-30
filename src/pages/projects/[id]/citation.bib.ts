import { getCollection } from 'astro:content';
import type { APIRoute } from 'astro';
import { buildBibTeX } from '../../../lib/citation';

interface CitationProps {
  id: string;
  title: string;
  year: number;
}

export async function getStaticPaths() {
  const projects = await getCollection('projects');

  return projects.map((project) => ({
    params: { id: project.id },
    props: {
      id: project.id,
      title: project.data.title,
      year: project.data.citationYear,
    } satisfies CitationProps,
  }));
}

export const GET: APIRoute = ({ props, site }) => {
  const { id, title, year } = props as CitationProps;
  const url = new URL(`/projects/${id}/`, site ?? 'https://www.duo-zhou.com');
  const bibtex = buildBibTeX({ id, title, year, url: url.toString(), note: 'Project page' });

  return new Response(bibtex, {
    headers: {
      'Content-Type': 'application/x-bibtex; charset=utf-8',
      'Content-Disposition': `attachment; filename="${id}.bib"`,
    },
  });
};

