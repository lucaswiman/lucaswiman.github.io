/**
 * Derive a plain-text summary (for <meta name="description"> / og:description)
 * from a markdown post body: the first prose paragraph(s), with markdown
 * syntax stripped, truncated at a word boundary.
 */
export function excerpt(markdown: string, maxLength = 160): string {
  const withoutBlocks = markdown
    .replace(/```[\s\S]*?```/g, '\n\n') // fenced code
    .replace(/\$\$[\s\S]*?\$\$/g, '\n\n') // display math
    .replace(/<[^>]+>/g, ' '); // raw HTML tags

  const paragraphs = withoutBlocks
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(
      (p) =>
        p.length > 0 &&
        !/^(#|!\[|\||>|-{3,}|\* \* \*|\s{4})/.test(p) // headings, images, tables, quotes, rules
    );

  let text = '';
  for (const p of paragraphs) {
    text += (text ? ' ' : '') + p;
    if (text.length >= maxLength) break;
  }

  text = text
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '') // images
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1') // links -> link text
    .replace(/`([^`]+)`/g, '$1')
    .replace(/(\*\*|__|\*|_)(\S[^*_]*?)\1/g, '$2') // emphasis
    .replace(/\$([^$]+)\$/g, '$1') // inline math
    .replace(/\s+/g, ' ')
    .trim();

  if (text.length <= maxLength) return text;
  const cut = text.slice(0, maxLength - 1);
  const lastSpace = cut.lastIndexOf(' ');
  return (lastSpace > maxLength * 0.6 ? cut.slice(0, lastSpace) : cut).replace(/[,.;:\s]+$/, '') + '…';
}
