# Study — Unified Offline HTML Learning Site (Design Spec)

**Date:** 2026-06-14
**Status:** Approved (design), pending implementation
**Author:** Nate + Claude Code

## 1. Goal

Turn the three self-study courses (`embedded-mastery`, `ml-dl-curriculum`, `pid`) into
one cohesive, offline, dark-themed, iPad-friendly HTML site rooted at `study/`, built on
the proven `embedded-mastery` engine. The learner opens **one** page in Chrome
(`study/index.html`) and can reach everything — no Markdown, no Python, no build step.

### Locked decisions (from brainstorming)
1. **Bilingual, in-page toggle** `EN ⇄ 繁中` on every page. Reuse existing zh-TW source
   for ML & PID; **create zh-TW** for all 19 embedded-mastery chapters.
2. **Unified hub + shared assets.** One `study/index.html`; one `study/assets/`. The
   `app.js` MANIFEST becomes a `COURSES` registry of all courses.
3. **Interactives where they genuinely aid learning** (~8–12 ML lessons; PID gets a full
   live simulator). Use canvas via the `EM.sim()` framework.
4. **HTML-only.** After porting, delete `.md`, `.py`, and the PID `.png` figures.
5. **PID PNGs → live sims.** One parameterized Nomoto heading-hold simulator reproduces
   every "Case" interactively; the Python is shown read-only, then deleted.
6. **External libs allowed**, but anything new is **vendored locally** to preserve the
   100%-offline promise (consistent with the existing katex/highlight/mermaid vendoring).

## 2. Target structure

```
study/
├── index.html                      # master hub (3 course cards) — data-page="hub"
├── assets/                         # MOVED from embedded-mastery/assets/ (shared)
│   ├── app.js                      # extended: COURSES registry + language toggle + sim toggles
│   ├── styles.css                  # + course-accent classes, lang-toggle, hub-3-up rules
│   ├── katex/  vendor/             # KaTeX, highlight.js, Mermaid (unchanged)
├── embedded-mastery/
│   ├── Arduino/ 01..09.html              + Arduino/zh-tw/ 01..09.html   (NEW)
│   ├── STM32/   01..10.html              + STM32/zh-tw/   01..10.html   (NEW)
│   └── _AUTHORING_GUIDE.md          # kept (dev doc, updated for new paths)
├── ml-dl-curriculum/
│   ├── 00..21.html, appendix.html        (from .md, EN)
│   └── zh-tw/ 00..21.html, appendix.html, glossary.html   (from zh-tw/.md)
├── pid/
│   ├── 01..08.html, glossary.html        (EN)
│   └── zh-tw/ 01..08.html, glossary.html (繁中)
├── Dockerfile, docker-compose.yml, .dockerignore   # MOVED from embedded-mastery/, retargeted to study/ root
├── deploy/ nginx.conf, validate.mjs                # MOVED + extended to crawl all pages/langs
├── docs/superpowers/specs/                          # this spec
└── README.md                                        # updated: how to open & extend the site
```

## 3. Shared engine (`assets/app.js`)

Single source of truth becomes a `COURSES` registry:

```js
const COURSES = {
  embedded: { title:"Embedded Mastery", short:"Embedded", icon:"🔌",
              accent:"#33b1ff", accent2:"#6cc4ff", base:"embedded-mastery/",
              blurb:"…",
              tracks:{ arduino:{title:"Arduino", sub:"AVR · C++", dir:"Arduino/", chapters:[…]},
                       stm32:{title:"STM32",   sub:"ARM Cortex-M", dir:"STM32/", chapters:[…]} } },
  mldl:     { title:"Machine Learning & Deep Learning", short:"ML / DL", icon:"🧠",
              accent:"#a371f7", accent2:"#c8a6ff", base:"ml-dl-curriculum/",
              tracks:{ main:{title:"Course", sub:"22 lessons", dir:"", chapters:[…23…]} } },
  pid:      { title:"PID Control for a USV", short:"PID", icon:"🧭",
              accent:"#3fb950", accent2:"#56d364", base:"pid/",
              tracks:{ main:{title:"Course", sub:"8 chapters + glossary", dir:"", chapters:[…]} } }
};
```

Each `chapter` = `{ id, num, file, title, titleZh }` (one bilingual title pair keeps the
sidebar/nav labels correct in both languages without parsing the page).

### Page contract (data attributes on `<body>`)
- Chapter page: `data-course data-track data-chapter data-lang="en|zh" data-root="<rel path to study/>"`.
- Hub page: `data-page="hub" data-root="" data-lang="en|zh"`.
- `data-root` is the only path knowledge app.js needs to build **cross-page** links
  (home, hub cards, course/track jumps). Values: `"../"` (ML/PID EN), `"../../"`
  (embedded EN + ML/PID zh), `"../../../"` (embedded zh), `""` (hub). Asset `<link>`/
  `<script>` URLs are hardcoded per page as `<data-root>assets/...`.

### Link building
`chapterURL(course,track,chapter,lang) = data-root + base + dir + (lang==='zh'?'zh-tw/':'') + file`.
Sidebar, sub-TOC, prev/next, hub cards, and the language toggle all route through this so
the learner stays in their chosen language while browsing.

### Language toggle
- `EN | 繁中` pill in the topbar. Reflects the current page's `data-lang`; clicking it
  navigates to the current chapter's sibling in the other language and saves
  `localStorage["study-lang"]`.
- On load, app.js reads `study-lang` and rewrites forward-nav links to that language
  (no surprise auto-redirect of the page you explicitly opened).
- Title labels chosen by lang via `title`/`titleZh`.

### Progress
- Key `study-progress-v1`, keyed `course/track/chapter` (**language-agnostic** — finishing
  a lesson in either language counts once). Hub shows per-course completion bars.

### `EM.sim()` enhancement (backward compatible)
Add non-range param types so PID toggles work without hacks:
- `{type:'range', …}` (default, unchanged)
- `{type:'toggle', key, label, value:true|false}` → checkbox
- `{type:'select', key, label, options:[{label,value}], value}` → dropdown
`draw(ctx, vals, geom)` unchanged; `vals` now may hold booleans/strings. Existing embedded
sims keep working.

## 4. Master hub (`study/index.html`)

Mirrors the embedded hub aesthetic but lists **three** course cards (Embedded, ML/DL, PID),
each with icon, accent, blurb, chapter chips, and a progress bar fed by `initHub()` across
all courses. Includes the language toggle and the "100% offline / interactive / math+code /
progress / iPad" feature strip. `data-page="hub"`.

## 5. Per-course work

### 5.1 Embedded-mastery
- Repoint every chapter's asset `<link>`/`<script>` to the shared assets via `data-root="../../"`
  and add `data-course="embedded" data-root data-lang="en"`; brand/home → `<data-root>index.html`.
- Add the language toggle markup (app.js wires it).
- **Create 19 zh-TW chapters** (`Arduino/zh-tw/01..09`, `STM32/zh-tw/01..10`) — faithful
  繁中 translations preserving all code, math, diagrams, exercises, and any sim `<script>`s
  verbatim (translate prose/labels only, not code/identifiers). `data-lang="zh"`,
  `data-root="../../../"`.
- Delete `embedded-mastery/index.html` (master hub replaces it).

### 5.2 ML/DL
- Convert `00..21` + `appendix` to HTML in the embedded teaching rhythm (hero, layered
  `<h2 id>` sections, callouts, `.math-box` KaTeX, Mermaid, `.codeblock` highlighted
  **read-only** Python, quiz/numeric/reveal exercises, chapter recap). EN at
  `ml-dl-curriculum/*.html` (`data-root="../"`), 繁中 from `zh-tw/*.md` at
  `ml-dl-curriculum/zh-tw/*.html` (`data-root="../../"`). Convert `zh-tw/glossary` too.
- **Interactive demos where they illuminate** (canvas via `EM.sim`):
  - 02 Linear regression — draggable/slider line, live MSE on a scatter.
  - 03 Gradient descent — loss bowl + steps; learning-rate slider shows crawl vs diverge.
  - 04 Logistic regression — sigmoid + moving decision boundary on 2-class points.
  - 05 Overfitting — polynomial-degree slider; train vs val fit, bias–variance.
  - 08 K-means + PCA — k-means iterations on 2D clusters; PCA projection axes.
  - 09 Neural net (MLP) — small net decision boundary as weights/units change.
  - 13 CNN — convolution kernel slider over a small image (edge/blur).
  - 15 Attention — attention-weight heatmap over a toy token sequence.
  - (10 Backprop, 06 kNN regions — added only if a clean, honest demo fits.)

### 5.3 PID
- Convert `notes.md` (Cases 0–9 + control law + tuning + ROS2) into **8 chapters**:
  1. `01-plant` The Plant: Nomoto model, open-loop (Case 0), the discrete control law + worked example.
  2. `02-p-control` P: the spring & structural offset (Cases 1–2).
  3. `03-d-control` D: the damper (Case 3).
  4. `04-i-control` I: the trim tab (Case 4).
  5. `05-windup` Integral windup & anti-windup strategies (Case 5).
  6. `06-derivative-noise` Derivative kick & sensor noise (Cases 6–7).
  7. `07-wrapping-actuators` Angle wrapping & real-actuator lag (Cases 8–9).
  8. `08-tuning-ros2` Tuning walkthrough, Ziegler–Nichols, sim→USV (ROS2) mapping.
  Plus `glossary` (translate the existing 繁中 glossary to EN for parity). EN at `pid/*.html`,
  繁中 from `notes.zh-TW.md`/`GLOSSARY.zh-TW.md` at `pid/zh-tw/*.html`.
- **The PID simulator** (centerpiece, embedded per chapter, preset to the chapter's scenario):
  a faithful JS port of `usv_heading_sim.py` + `pid_controller.py`. Computes the full 60 s
  trajectory each redraw and plots heading-vs-setpoint (top) + rudder-with-limits (bottom),
  with a readout (overshoot, steady-state error, settle time). Controls:
  sliders `kp, ki, kd, disturbance(°), noise σ(°), setpoint(°)`; toggles
  `anti-windup, derivative-on-measurement, D low-pass`. Model constants match the Python
  (K=0.6, T=2.5, rudder ±35°, slew 25°/s, dt=0.05, 20 Hz). Validated numerically against the
  worked example in `notes.md` (kp=2,ki=0.2,kd=3 step → matches the by-hand timestep).
- Show `pid_controller.py` and `usv_heading_sim.py` verbatim in `.codeblock`s (the "read the
  code" layer) before deleting the files. A small angle-wrap dial visual supports Case 8.

## 6. Deploy & validation
- Move `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `deploy/` to `study/` root;
  set nginx root to the site root and confirm relative asset paths resolve at every depth.
- Extend `deploy/validate.mjs` to crawl **all** pages in both languages and assert:
  valid doc (`<!DOCTYPE html>`…`</html>`), required `data-*` attributes present and matching
  the registry, every referenced `assets/` URL resolves from that page's depth, every `<h2>`
  has a unique id, exercises have `data-id` (+ `data-correct` on quizzes), and the
  EN/zh sibling exists for every chapter. CI-style: exits non-zero on any failure.

## 7. Source files to delete (after porting & validation pass)
- `pid/*.py`, `pid/__pycache__/`, `pid/figures/*.png`, `pid/notes.md`, `pid/notes.zh-TW.md`,
  `pid/GLOSSARY.zh-TW.md`, stray `.DS_Store`.
- `ml-dl-curriculum/*.md` (all, incl. `README.md`, `appendix-math-cheatsheet.md`),
  `ml-dl-curriculum/zh-tw/*.md`, stray `.DS_Store`.
- `embedded-mastery/index.html`.
- Keep: `_AUTHORING_GUIDE.md`, top-level `README.md` (updated), `start-jupyter.sh`.

## 8. Phasing
- **P0** Shared engine: move assets, extend `app.js` (registry, toggle, sim types), styles,
  master hub, repoint + verify the existing embedded EN chapters still render.
- **P1** PID: 8 EN chapters + glossary + the simulator (highest interactive value, smallest
  page count — proves the new patterns end-to-end).
- **P2** ML/DL EN + 繁中 (44 pages) + interactive demos.
- **P3** Embedded 繁中 (19 translations).
- **P4** PID 繁中, deploy retarget, extend & run `validate.mjs`, delete sources, update README.
- **P5** Final: full validate pass, manual Chrome smoke test, `git add` + commit + push.

## 9. Verification (evidence required before "done")
- `node deploy/validate.mjs` exits 0 over the whole tree.
- Manual smoke in Chrome: hub loads and shows 3 cards; open one chapter per course; KaTeX,
  Mermaid, highlight, sidebar/TOC/prev-next, exercises, and the PID sim all work; the
  language toggle round-trips EN⇄繁中 and preserves position; progress bars update.
- `git status` clean after the deletes; no dangling references to removed `.md/.py/.png`.

## 10. Non-goals
- No server-side code, no framework/SPA, no build pipeline. Static files only.
- No new course content beyond faithfully porting what exists (plus the interactives).
- No change to the teaching voice or the per-chapter pedagogical structure.
