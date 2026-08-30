import { getCollection } from 'astro:content';
import type { APIRoute } from 'astro';
import { resolveCitation } from '../../../lib/citation';

interface CitationProps {
  id: string;
  title: string;
  year: number;
  bibtex?: string;
}

export async function getStaticPaths() {
  const projects = await getCollection('projects');

  return projects.map((project) => ({
    params: { id: project.id },
    props: {
      id: project.id,
      title: project.data.title,
      year: project.data.citationYear,
      bibtex: project.data.bibtex,
    } satisfies CitationProps,
  }));
}

export const GET: APIRoute = ({ props, site }) => {
  const { id, title, year, bibtex: paperBibTeX } = props as CitationProps;
  const url = new URL(`/projects/${id}/`, site ?? 'https://www.duo-zhou.com');
  const bibtex = resolveCitation(paperBibTeX, { id, title, year, url: url.toString(), note: 'Project page' });

  return new Response(bibtex, {
    headers: {
      'Content-Type': 'application/x-bibtex; charset=utf-8',
      'Content-Disposition': `attachment; filename="${id}.bib"`,
    },
  });
};

