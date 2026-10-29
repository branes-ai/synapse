---
status: complete
issue: branes-ai/synapse#65
---
# Plan: publish ./docs as a GitHub Pages site

Issue: #65. Implemented in #68.

## Goal

A published site for synapse's documentation at <https://branes-ai.github.io/synapse/>, with the same structure and look as the other Branes.AI sites ([cortex](https://branes-ai.github.io/cortex/), [reflex](https://branes-ai.github.io/reflex/), [racing-drone](https://branes-ai.github.io/racing-drone/)). As in racing-drone, the site is **always generated from `docs/`**: nothing is authored in `docs-site/`, and the documents we write and review stay in `docs/`, where GitHub renders them too.

## Design

racing-drone's setup (branes-ai/racing-drone#49, #53), copied from its `main` with only the synapse-specific parts changed:

- **`docs-site/scripts/sync-docs.mjs`** generates `docs-site/src/content/docs/` and the sidebar (both gitignored) from `docs/` before every `dev` and `build`. Changed: `GITHUB` and `REPO`, and `docs/sessions/` is skipped (below).
  - Every `docs/**/*.md` becomes a page, a folder's `README.md` its index, and its first `# ` heading its title. Each page's "Edit page" link opens its source in `docs/`.
  - `docs/index.mdx` is the splash landing page, with its front matter kept; `docs/README.md` becomes the "Documentation map" at `/overview/`.
  - Relative links to documents become page links, and links to any other repo file become GitHub links. **A relative link or anchor that does not resolve fails the build**, naming the file and line, and so does a hero link that isn't a page.
  - The sidebar has one group per top-level folder, in the order of `docs/README.md`'s table, and within a group the order of the links in that folder's row.
  - The records, `docs/plan/` and `docs/adr/`, are published on the [architecture site](https://branes-ai.github.io/architecture/synapse/), which already lists synapse in its `sources.yaml`. This site generates a redirect for each.
- **`docs-site/astro.config.mjs`**: title, description, `base: '/synapse/'`, GitHub links.
- **`docs-site/Doxyfile`**: the C++ API reference over `sdk/include`, at `<base>api/`.
- **Theme**: `src/styles/custom.css`, `public/favicon.svg` and `src/assets/hero.svg` in the same style as the other sites, with synapse's own accent: violet (reflex is amber, cortex blue, racing-drone teal). The hero is a small compute graph with one signal's path through it.
- **Deploy**: `.github/workflows/deploy-docs.yml`, on pushes to main that touch `docs/**`, `docs-site/**`, `sdk/include/**` or the workflow. Actions pinned to commit SHAs.
- **CI**: a `Docs site` job in `ci.yml` builds the site on every PR, so a broken link fails the PR instead of the deploy after merge.

## Content in `docs/`

- **`index.mdx`**: the hero, "What Synapse is", four feature cards, the platform at a glance, where to go next, and the status.
- **`getting-started/`**: `introduction.md`, `build-and-test.md` and `repository-layout.md`, from the top-level README.
- **`README.md`**: the documentation map, one table row per folder with `getting-started/` first, and its "Start here" list kept.
- **`arch/`**: the root-level `ggml-ecosystem.md` and `llama-model.md` move here, since the sidebar groups come from top-level folders. Their links, and the top-level README's links to them, are fixed.
- **`sessions/`: not published.** The issue left this open. The session logs are working records for the developers, not documentation for the library's users, and cortex does not publish its sessions either. `sync-docs.mjs` skips the folder (`UNPUBLISHED`, the one change to its logic from racing-drone's), and a link to a session log becomes a GitHub link, like a link to any other repo file.

## Acceptance

1. `npm run build` in `docs-site/` builds one page per `docs/**/*.md` outside the records and the session logs, plus the landing page.
2. Every internal link in the built site resolves.
3. A deliberately broken relative link in `docs/` fails the build, naming the file and line.
4. The landing page matches cortex's structure, at desktop and phone widths, in both themes.
5. CI's `Docs site` job passes on the PR.
6. After merge, the deploy publishes the site. Pages is already set to "GitHub Actions" (`build_type: workflow`), so no admin step is needed.

## Departures

- The Getting Started pages repeat sections of the top-level README, as cortex's, reflex's and racing-drone's do; a change to one should be made in the other.
- No KaTeX: no document uses math, and `remark-math` would read a dollar amount in prose as math.
