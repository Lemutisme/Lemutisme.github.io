export interface CitationInput {
  id: string;
  title: string;
  year: number;
  url: string;
  note: 'Project page' | 'Research note';
}

const bibtexEscapes: Record<string, string> = {
  '\\': '\\textbackslash{}',
  '{': '\\{',
  '}': '\\}',
  '%': '\\%',
  '&': '\\&',
  '#': '\\#',
  '_': '\\_',
  '$': '\\$',
  '~': '\\textasciitilde{}',
  '^': '\\textasciicircum{}',
};

const escapeBibTeX = (value: string) => value.replace(
  /[\\{}%&#_$~^]/g,
  (character) => bibtexEscapes[character],
);

export const citationKey = (id: string, year: number) => {
  const slug = id
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-zA-Z0-9]/g, '')
    .toLowerCase();

  return `zhou${year}${slug || 'work'}`;
};

export const buildBibTeX = ({ id, title, year, url, note }: CitationInput) => `@misc{${citationKey(id, year)},
  author = {Zhou, Duo},
  title  = {{${escapeBibTeX(title)}}},
  year   = {${year}},
  url    = {${escapeBibTeX(url)}},
  note   = {${note}}
}
`;


/**
 * Prefer the real paper's BibTeX when a project supplies one, and fall back to
 * a generated citation for the project page otherwise.
 */
export const resolveCitation = (paperBibTeX: string | undefined, fallback: CitationInput) => {
  const paper = paperBibTeX?.trim();
  return paper ? `${paper}\n` : buildBibTeX(fallback);
};

export const countEntries = (bibtex: string) => (bibtex.match(/@\w+\s*\{/g) ?? []).length;
