# Duo Zhou Academic Site

An Astro-powered academic portfolio with a Vue homepage and Markdown content
collections for research notes and project pages.

## Local development

```sh
npm install
npm run dev
```

Build the production site with:

```sh
npm run build
```

## Publish a blog post

Copy `src/content/blog/_template.md`, rename it with a URL-safe slug, complete
the frontmatter and body, then set `draft: false`.

```text
src/content/blog/my-new-research-note.md
```

The post will appear at `/blog/my-new-research-note/` after the next deployment.

## Add a project

Create a Markdown file in `src/content/projects/`. Required metadata is
validated by `src/content.config.ts`, and the project is automatically added to
the Projects index.

```text
src/content/projects/my-project.md
```

GitHub Actions builds and deploys the static `dist/` directory to GitHub Pages.
