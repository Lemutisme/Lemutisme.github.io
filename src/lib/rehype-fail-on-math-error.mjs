/**
 * rehype-katex swallows KaTeX parse errors: it renders the offending source as
 * a red `katex-error` span (or red inline text, for an undefined control
 * sequence) and lets the build succeed. Passing `throwOnError` to KaTeX does
 * not change that, because rehype-katex catches the error itself.
 *
 * Throwing from here does not work either - Astro's content loader catches it
 * and emits the page with an empty body, which ships a blank article instead
 * of a visibly broken one. So this plugin only reports where the problem is;
 * `scripts/check-math.mjs` scans the built output and fails the build.
 *
 * Must run after rehype-katex.
 */
export function rehypeReportMathErrors() {
  return (_tree, file) => {
    const errors = file.messages.filter((message) => message.source === 'rehype-katex');
    if (errors.length === 0) return;

    const where = file.path ?? file.history?.at(-1) ?? 'markdown';
    for (const message of errors) {
      const at = message.place?.start
        ? `${message.place.start.line}:${message.place.start.column}`
        : '?';
      console.error(`  [math] ${where}:${at}  ${message.reason}`);
    }
  };
}
