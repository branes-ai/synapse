// Generates the site's content from the repo's docs/ (docs/plan/docs-site).
//
//   docs/**/*.md        → src/content/docs/**  (README.md → index.md), except
//                         the records (docs/adr/, docs/plan/), which the
//                         architecture site publishes (branes-ai/architecture),
//                         and the session logs (docs/sessions/), which are not
//                         published
//   docs/index.mdx      → the home page
//   docs/README.md      → the documentation map, /overview/ (the home page
//                         when there is no docs/index.mdx)
//   sidebar             → src/generated/sidebar.json
//   the records' old pages → public/{adr,plan}/**/index.html, each a redirect to
//                         the record's page on the architecture site
//
// Nothing under src/content/docs/ is authored by hand; it is deleted and
// rebuilt on every run. Relative links are rewritten: to a page when the target
// is a document in docs/, to the architecture site when it is a record, to
// GitHub when it is any other file in the repo. A relative link whose target
// (or anchor) does not exist fails the run.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import GithubSlugger, { slug as slugify } from 'github-slugger';

const SITE_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const REPO_DIR = path.resolve(SITE_DIR, '..');
const DOCS_DIR = path.join(REPO_DIR, 'docs');
const OUT_DIR = path.join(SITE_DIR, 'src', 'content', 'docs');
const GEN_DIR = path.join(SITE_DIR, 'src', 'generated');

const GITHUB = 'https://github.com/branes-ai/synapse';
const BRANCH = 'main';

// The records: published on the architecture site, not here.
const RECORDS = new Set(['adr', 'plan']);
const ARCHITECTURE = 'https://branes-ai.github.io/architecture';
const REPO = 'synapse';

// Working records kept in the repo but not published: a link to one goes to
// GitHub, like a link to any other repo file.
const UNPUBLISHED = new Set(['sessions']);

// Sidebar labels for top-level folders whose name is not a good label.
const GROUP_LABELS = {
  arch: 'Architecture',
  'getting-started': 'Getting Started',
};

// The landing page: MDX, for Starlight's hero and components.
const LANDING = 'index.mdx';
const hasLanding = fs.existsSync(path.join(DOCS_DIR, LANDING));

const IMAGE_EXT = new Set(['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp']);

const toPosix = (p) => p.split(path.sep).join('/');
const isRecord = (docsRel) => RECORDS.has(docsRel.split('/')[0]);
const isUnpublished = (docsRel) => UNPUBLISHED.has(docsRel.split('/')[0]);

// A record's page on the architecture site: its path under docs/, each
// segment slugified, a README being its folder's page. The rule is the site's
// (branes-ai/architecture, scripts/lib/parse.mjs, routeOf).
function recordUrl(docsRel) {
  const parts = docsRel.split('/');
  const base = parts.pop().replace(/\.md$/, '');
  if (base !== 'README') parts.push(base);
  return `${ARCHITECTURE}/${[REPO, ...parts].map((x) => slugify(x)).join('/')}/`;
}

function walk(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(full));
    else out.push(full);
  }
  return out;
}

// docs-relative path of a document → its output path and its site route.
function pageOf(rel) {
  if (rel === LANDING) return { outRel: LANDING, slug: '', route: '/' };
  if (rel === 'README.md' && hasLanding) return { outRel: 'overview.md', slug: 'overview', route: '/overview/' };
  const dir = path.posix.dirname(rel);
  const base = path.posix.basename(rel, '.md');
  const isIndex = base === 'README';
  const outRel = path.posix.join(dir, isIndex ? 'index.md' : `${base}.md`);
  // The route as Starlight's loader derives it: each path segment slugified
  // (lowercased, spaces to hyphens, punctuation dropped).
  const segments = (isIndex ? dir : path.posix.join(dir, base)).split('/').filter((x) => x !== '.' && x !== '');
  const slug = segments.map((x) => slugify(x)).join('/');
  return { outRel, slug, route: slug === '' ? '/' : `/${slug}/` };
}

// Splits a document into lines tagged with whether they sit in a code fence.
// As in CommonMark, a fence closes only on a run of its own character at least
// as long as the opening one, with nothing after it, so a four-backtick fence
// can show a three-backtick example.
function lines(text) {
  let fence = null;
  return text.split('\n').map((line) => {
    const m = line.match(/^\s*(`{3,}|~{3,})(.*)$/);
    if (m) {
      if (fence === null) fence = m[1];
      else if (m[1][0] === fence[0] && m[1].length >= fence.length && m[2].trim() === '') fence = null;
      return { line, code: true };
    }
    return { line, code: fence !== null };
  });
}

// The anchors GitHub renders for a Markdown file: every heading's, the title's
// included (the site drops the title, so parse() leaves it out).
function githubAnchors(file) {
  const slugger = new GithubSlugger();
  const anchors = new Set();
  for (const t of lines(fs.readFileSync(file, 'utf8'))) {
    const m = !t.code && t.line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (m) anchors.add(slugger.slug(headingText(m[2])));
  }
  return anchors;
}

// The rendered text of a heading, as the slugger sees it: links become their
// text, and emphasis and code markers are dropped. An underscore is emphasis
// only at a word boundary; inside a word (Q8_0) or a code span it is text.
function headingText(raw) {
  return raw
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .split(/(`+[^`]*`+)/)
    .map((part, i) =>
      i % 2 === 1
        ? part.replace(/^`+|`+$/g, '')
        : part.replace(/\*/g, '').replace(/(^|[^\p{L}\p{N}])_+|_+(?=[^\p{L}\p{N}]|$)/gu, '$1'),
    )
    .join('')
    .trim();
}

function parse(file) {
  const text = fs.readFileSync(file, 'utf8');
  const tagged = lines(text);
  let title = null;
  let titleLine = -1;
  const slugger = new GithubSlugger();
  const anchors = new Set();
  tagged.forEach((t, i) => {
    if (t.code) return;
    const m = t.line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (!m) return;
    // The title is removed from the page, so it takes no slug: a later heading
    // with the same text renders as #text, not #text-1.
    if (m[1].length === 1 && title === null) {
      title = headingText(m[2]);
      titleLine = i;
    } else {
      anchors.add(slugger.slug(headingText(m[2])));
    }
  });
  return { text, tagged, title, titleLine, anchors };
}

const errors = [];

const allMd = walk(DOCS_DIR)
  .filter((f) => f.endsWith('.md'))
  .map((f) => toPosix(path.relative(DOCS_DIR, f)))
  .sort();
const mdFiles = allMd.filter((rel) => !isRecord(rel) && !isUnpublished(rel));

const docs = new Map();
for (const rel of mdFiles) {
  const d = parse(path.join(DOCS_DIR, rel));
  if (d.title === null) errors.push(`docs/${rel}: no "# " title heading`);
  docs.set(rel, { ...d, ...pageOf(rel) });
}

// The landing page: its front matter is kept as written (title required), and
// its body is parsed like any document's, for links and anchors.
let landingFront = null;
if (hasLanding) {
  const text = fs.readFileSync(path.join(DOCS_DIR, LANDING), 'utf8');
  const m = text.match(/^---\n([\s\S]*?)\n---\n/);
  if (!m) {
    errors.push(`docs/${LANDING}: no front matter`);
  } else {
    landingFront = m[1];
    const title = landingFront.match(/^title:\s*(.+)$/m);
    if (!title) errors.push(`docs/${LANDING}: no title in the front matter`);
    const body = parse(path.join(DOCS_DIR, LANDING));
    const offset = m[0].split('\n').length - 1;
    const tagged = lines(text.slice(m[0].length));
    docs.set(LANDING, { ...body, tagged, offset, title: title ? title[1] : null, titleLine: -1, ...pageOf(LANDING) });
  }
}

// Rewrites one link target found in docs/<rel>, line `lineNo`.
function rewrite(target, rel, lineNo, copies) {
  if (/^[a-z][a-z0-9+.-]*:/i.test(target) || target.startsWith('//')) return target;
  const where = `docs/${rel}:${lineNo}`;
  if (target.startsWith('#')) {
    const anchor = decodeURIComponent(target.slice(1));
    if (!docs.get(rel).anchors.has(anchor)) errors.push(`${where}: no heading for anchor ${target}`);
    return target;
  }
  if (target.startsWith('/')) {
    errors.push(`${where}: root-relative link ${target}; use a path relative to the document`);
    return target;
  }
  const [pathPart, ...rest] = target.split('#');
  const anchor = rest.length ? decodeURIComponent(rest.join('#')) : null;
  const abs = path.resolve(DOCS_DIR, path.posix.dirname(rel), decodeURIComponent(pathPart));
  const repoRel = toPosix(path.relative(REPO_DIR, abs));
  if (repoRel.startsWith('..')) {
    errors.push(`${where}: link ${target} leaves the repository`);
    return target;
  }
  if (!fs.existsSync(abs)) {
    errors.push(`${where}: link target ${target} does not exist`);
    return target;
  }
  const docsRel = toPosix(path.relative(DOCS_DIR, abs));
  const inDocs = !docsRel.startsWith('..');
  const isDir = fs.statSync(abs).isDirectory();
  const docRel = inDocs && isDir ? path.posix.join(docsRel, 'README.md').replace(/^\.\//, '') : docsRel;
  if (inDocs && docs.has(docRel)) {
    const page = docs.get(docRel);
    if (anchor !== null && !page.anchors.has(anchor)) {
      errors.push(`${where}: docs/${docRel} has no heading for anchor #${anchor}`);
    }
    return page.route + (anchor !== null ? `#${anchor}` : '');
  }
  if (inDocs && isRecord(docRel) && docRel.endsWith('.md') && fs.existsSync(path.join(DOCS_DIR, docRel))) {
    if (anchor !== null && !parse(path.join(DOCS_DIR, docRel)).anchors.has(anchor)) {
      errors.push(`${where}: docs/${docRel} has no heading for anchor #${anchor}`);
    }
    return recordUrl(docRel) + (anchor !== null ? `#${anchor}` : '');
  }
  if (inDocs && !isDir && IMAGE_EXT.has(path.extname(abs).toLowerCase())) {
    copies.add(docsRel);
    return target;
  }
  if (anchor !== null && !isDir && abs.endsWith('.md') && !githubAnchors(abs).has(anchor)) {
    errors.push(`${where}: ${repoRel} has no heading for anchor #${anchor}`);
  }
  const kind = isDir ? 'tree' : 'blob';
  return `${GITHUB}/${kind}/${BRANCH}/${repoRel}` + (anchor !== null ? `#${anchor}` : '');
}

// Rewrites the inline links of one line, leaving code spans alone.
function rewriteLine(line, rel, lineNo, copies) {
  return line
    .split(/(`+[^`]*`+)/)
    .map((part, i) => {
      if (i % 2 === 1) return part;
      return part
        .replace(/(\]\()(<[^>]+>|[^)\s]+)((?:\s+"[^"]*")?\))/g, (_, open, target, close) => {
          const bare = target.startsWith('<') ? target.slice(1, -1) : target;
          return open + rewrite(bare, rel, lineNo, copies) + close;
        })
        .replace(/^(\s*\[[^\]]+\]:\s+)(\S+)/, (_, open, target) => open + rewrite(target, rel, lineNo, copies));
    })
    .join('');
}

fs.rmSync(OUT_DIR, { recursive: true, force: true });
fs.mkdirSync(OUT_DIR, { recursive: true });
const copies = new Set();

for (const [rel, d] of docs) {
  if (d.title === null || rel === LANDING) continue;
  const body = d.tagged
    .map((t, i) => (i === d.titleLine ? null : t.code ? t.line : rewriteLine(t.line, rel, i + 1, copies)))
    .filter((l) => l !== null)
    .join('\n')
    .replace(/^\s*\n/, '');
  const front = [
    '---',
    `title: ${JSON.stringify(d.title)}`,
    `editUrl: ${JSON.stringify(`${GITHUB}/edit/${BRANCH}/docs/${rel}`)}`,
    '---',
    '',
  ].join('\n');
  const out = path.join(OUT_DIR, d.outRel);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, front + body);
}

if (landingFront !== null) {
  const d = docs.get(LANDING);
  // Each relative hero action (`link: ./x/`) must be the route of a page.
  const routes = new Set([...docs.values()].map((p) => p.route));
  for (const [i, line] of landingFront.split('\n').entries()) {
    const m = line.match(/^\s*link:\s*['"]?(\.\/[^'"\s]*)/);
    if (m && !routes.has(m[1].slice(1).replace(/\/?$/, '/'))) {
      errors.push(`docs/${LANDING}:${i + 2}: hero link ${m[1]} is not a page`);
    }
  }
  const body = d.tagged
    .map((t, i) => (t.code ? t.line : rewriteLine(t.line, LANDING, i + 1 + d.offset, copies)))
    .join('\n');
  const editUrl = `editUrl: ${JSON.stringify(`${GITHUB}/edit/${BRANCH}/docs/${LANDING}`)}`;
  fs.writeFileSync(path.join(OUT_DIR, LANDING), `---\n${landingFront}\n${editUrl}\n---\n${body}`);
}

for (const rel of copies) {
  const out = path.join(OUT_DIR, rel);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.copyFileSync(path.join(DOCS_DIR, rel), out);
}

// Sidebar: one group per top-level folder, in the order docs/README.md's table
// lists them, then any others alphabetically. Within a group: the documents
// the folder's table row links to, in that order; then the folder's README,
// its other documents, and its subfolders' documents.
const folders = [...new Set(mdFiles.filter((r) => r.includes('/')).map((r) => r.split('/')[0]))];
const order = [];
const rowLinks = new Map();
const index = docs.get('README.md');
if (index) {
  for (const { line, code } of index.tagged) {
    const m = !code && line.match(/^\|\s*\[?([A-Za-z0-9_-]+)\/\]?/);
    if (!m || !folders.includes(m[1]) || order.includes(m[1])) continue;
    order.push(m[1]);
    const cells = line.split('|').slice(2).join('|');
    const linked = [...cells.matchAll(/\]\(([^)#\s]+)/g)].map((x) => path.posix.normalize(x[1]));
    rowLinks.set(m[1], [...new Set(linked)]);
  }
}
for (const f of folders.sort()) if (!order.includes(f)) order.push(f);

// With a landing page, the documentation map is the last item of the first
// group; without one, it is the home page, linked first.
const sidebar = hasLanding ? [] : [{ label: 'Overview', link: '/' }];
for (const folder of order) {
  const pages = [...docs.entries()]
    .filter(([rel, d]) => rel.startsWith(`${folder}/`) && d.title !== null)
    .sort(([a], [b]) => {
      const links = rowLinks.get(folder) ?? [];
      const la = links.indexOf(a);
      const lb = links.indexOf(b);
      if (la !== lb) return la === -1 ? 1 : lb === -1 ? -1 : la - lb;
      const ia = a === `${folder}/README.md`;
      const ib = b === `${folder}/README.md`;
      if (ia !== ib) return ia ? -1 : 1;
      // The folder's own documents before its subfolders' (plan/test-plan.md
      // before plan/b0/README.md).
      const da = a.split('/').length;
      const db = b.split('/').length;
      return da !== db ? da - db : a.localeCompare(b);
    });
  sidebar.push({
    label: GROUP_LABELS[folder] ?? folder[0].toUpperCase() + folder.slice(1),
    items: pages.map(([, d]) => ({ label: d.title, slug: d.slug })),
  });
}
const map = docs.get('README.md');
if (hasLanding && map && map.title !== null && sidebar.length > 0) {
  sidebar[0].items.push({ label: map.title, slug: map.slug });
}

if (errors.length) {
  console.error(`sync-docs: ${errors.length} error(s):`);
  for (const e of errors) console.error(`  ${e}`);
  process.exit(1);
}

fs.mkdirSync(GEN_DIR, { recursive: true });
fs.writeFileSync(path.join(GEN_DIR, 'sidebar.json'), JSON.stringify(sidebar, null, 2) + '\n');

// The records were pages of this site until they moved to the architecture
// site. Each old page redirects there, keeping the URL's #fragment; the meta
// refresh is the fallback without JavaScript.
const PUBLIC_DIR = path.join(SITE_DIR, 'public');
for (const folder of RECORDS) fs.rmSync(path.join(PUBLIC_DIR, folder), { recursive: true, force: true });
const records = allMd.filter(isRecord);
for (const rel of records) {
  const url = JSON.stringify(recordUrl(rel));
  const html = [
    '<!doctype html>',
    '<html lang="en">',
    '<meta charset="utf-8">',
    '<title>Moved to the architecture site</title>',
    '<meta name="robots" content="noindex">',
    `<link rel="canonical" href=${url}>`,
    `<script>location.replace(${url} + location.hash);</script>`,
    `<meta http-equiv="refresh" content="0; url=${recordUrl(rel)}">`,
    `<p>This page moved to <a href=${url}>the architecture site</a>.</p>`,
    '</html>',
    '',
  ].join('\n');
  const out = path.join(PUBLIC_DIR, pageOf(rel).slug, 'index.html');
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, html);
}
console.log(`sync-docs: ${docs.size} pages from docs/, ${copies.size} images, ${order.length} sidebar groups, ${records.length} redirects to the architecture site`);
