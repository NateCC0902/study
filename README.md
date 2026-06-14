# 我的學習課程 · My Study Courses

A personal collection of self-study courses, written for me by **Claude Code** — now a single
**offline, bilingual, interactive website**. Open one file in Chrome and learn everything: no
Markdown, no Python, no build step.

> 這是我個人的自學課程，由 Claude Code 為我撰寫，現在整合成一個**離線、雙語、可互動的網站**。
> 只要用 Chrome 打開一個檔案就能學所有東西：不需要 Markdown、不需要 Python、不需要建置流程。

## Open it

Just open **`index.html`** in a browser (double-click, or `file://…/study/index.html`).
Everything is bundled locally (KaTeX, Mermaid, syntax highlighting), so it works fully offline —
on a laptop or an iPad on the same Wi-Fi.

Every page has an **`EN / 繁中` toggle** in the top bar; your language and your chapter progress
are remembered in the browser.

## Courses

| Course | Chapters | Interactive |
|---|---|---|
| **[Embedded Mastery](embedded-mastery/Arduino/01-intro.html)** — Arduino (AVR) & STM32 (ARM) | 9 + 10 | live canvas sims in the chapters |
| **[Machine Learning & Deep Learning](ml-dl-curriculum/00-what-is-ml.html)** | 22 + appendix + glossary | 9 demos: gradient descent, regression, bias–variance, k-means, PCA, a neuron, CNN kernels, attention |
| **[PID Control for a USV](pid/01-plant.html)** | 8 + glossary | a full Nomoto heading-hold simulator, preset per chapter |

Each course exists in English and Traditional Chinese (繁中 pages live in `zh-tw/` subfolders).

## How it's built

- **One shared engine** in `assets/`: `app.js` (a `COURSES` registry that builds the hub, the
  sidebar/TOC, prev-next nav, the language toggle, exercises, and progress tracking),
  `styles.css` (the dark theme), and vendored KaTeX / highlight.js / Mermaid.
- **`assets/pid-sim.js`** — the PID heading-hold simulator (a JS port of the original Nomoto
  model, validated against the worked examples).
- **`assets/ml-demos.js`** — the nine ML/DL canvas demos.
- Each chapter is a self-contained HTML file declaring `data-course / data-track / data-chapter`;
  `app.js` self-locates and wires everything from there.

## Deploy on your LAN (optional)

```bash
docker compose up -d        # build + serve on :8080
# open http://<your-LAN-IP>:8080 on the iPad (same Wi-Fi)
docker compose down
```

`deploy/validate.mjs` (`node deploy/validate.mjs`) checks every page in both languages:
structure, offline-safety (no external URLs), asset paths, unique heading ids, exercises, and
EN/繁中 parity.

> The original Markdown notes and Python sources were merged into these HTML pages and removed;
> they remain in git history if ever needed. The ML/DL course assumes PyTorch examples would run
> in the conda env named `study`, but nothing here requires Python to read or use.

---
🤖 Made with [Claude Code](https://claude.com/claude-code).
