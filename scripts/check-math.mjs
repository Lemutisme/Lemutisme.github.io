/**
 * Fails the build when the generated site contains KaTeX rendering errors.
 *
 * rehype-katex renders malformed math in red rather than failing, so without
 * this check a research note can ship with broken mathematics. Run after
 * `astro build`; see src/lib/rehype-fail-on-math-error.mjs for why the check
 * lives here rather than inside the markdown pipeline.
 */
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

const DIST = 'dist';

// KaTeX marks failures two ways: a `katex-error` span for a parse error, and
// its `errorColor` (#cc0000) for an undefined control sequence.
const MARKERS = [
  ['katex-error', 'KaTeX parse error'],
  ['mathcolor="#cc0000"', 'undefined KaTeX command'],
  ['color:#cc0000', 'KaTeX error colour'],
];

async function* htmlFiles(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) yield* htmlFiles(path);
    else if (entry.name.endsWith('.html')) yield path;
  }
}

const failures = [];
for await (const file of htmlFiles(DIST)) {
  const html = await readFile(file, 'utf8');
  for (const [marker, label] of MARKERS) {
    if (html.includes(marker)) failures.push(`${file}: ${label}`);
  }
}

if (failures.length > 0) {
  console.error(`\n✗ ${failures.length} math rendering error(s) in ${DIST}:`);
  for (const failure of failures) console.error(`  ${failure}`);
  console.error('\nFix the LaTeX above; the build is failed so it is not published.\n');
  process.exit(1);
}

console.log('✓ no KaTeX rendering errors in the built site');
