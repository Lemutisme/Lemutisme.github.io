# Duo Zhou Academic Site

An Astro static site: a homepage rendered from `src/data.ts` and Markdown
content collections for research notes and project pages. No client-side
framework and no runtime JavaScript beyond the citation copy button.

## Local development

Requires Node.js 22.12 or later.

```sh
npm ci
npm run dev
```

Build the production site with:

```sh
npm run build
```

## Update the homepage

News, publications, honors, and service entries live in `src/data.ts`. Add a
new item to the relevant exported array; the homepage renders it on the next
build. `newsData` is ordered newest first, and only the first entry is shown
uncollapsed.

## Publish a blog post

Copy `src/content/blog/_template.md`, rename it with a URL-safe slug, complete
the frontmatter and body, then set `draft: false`.

```text
src/content/blog/my-new-research-note.md
```

The post will appear at `/blog/my-new-research-note/` after the next deployment.
Published posts automatically include a BibTeX panel and a downloadable
`citation.bib` file; the citation year comes from `publishedAt`.

## Add a project

Create a Markdown file in `src/content/projects/`. Required metadata is
validated by `src/content.config.ts`, and the project is automatically added to
the Projects index. Set `citationYear` to the year that should appear in the
generated BibTeX entry.

```text
src/content/projects/my-project.md
```

Each project detail page includes a copyable citation and exposes its BibTeX at
`/projects/<slug>/citation.bib`.

GitHub Actions builds and deploys the static `dist/` directory to GitHub Pages.
