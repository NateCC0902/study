# Embedded Mastery 🔌⚙️

Two complete, interactive, **fully offline** courses — **Arduino (AVR)** and **STM32 (ARM Cortex-M)** — taught from beginner to master. Built as static HTML with a dark, iPad-friendly design, bundled math (KaTeX), diagrams (Mermaid), syntax highlighting, live canvas simulators, and in-page exercises with progress tracking.

```
embedded-mastery/
├── index.html              ← landing hub (start here)
├── Arduino/                ← 9 chapters: 01-intro … 09-capstone
├── STM32/                  ← 10 chapters: 01-intro … 10-capstone
├── assets/
│   ├── styles.css          ← shared dark-dashboard theme
│   ├── app.js              ← nav, exercises, sims, progress (the engine)
│   ├── katex/              ← bundled KaTeX (css + js + fonts)  [offline]
│   └── vendor/             ← mermaid.min.js, highlight.min.js + theme [offline]
├── Dockerfile              ← nginx static image
├── docker-compose.yml      ← one-command LAN serve
└── deploy/nginx.conf       ← MIME types, gzip, caching
```

Everything renders **with no internet** — the iPad only needs to reach this machine on your LAN.

---

## Run it with Docker (recommended for the iPad)

From inside the `embedded-mastery/` folder:

```bash
docker compose up -d --build
```

That serves the site on port **8080**. Now find this computer's LAN IP and open it on the iPad (same Wi-Fi):

```bash
# macOS — print your Wi-Fi IP address:
ipconfig getifaddr en0     # Wi-Fi    (try en1 if en0 is empty)
```

On the iPad's browser go to:

```
http://<that-IP>:8080
```

e.g. `http://192.168.1.42:8080`. Add it to your Home Screen ("Share → Add to Home Screen") for a full-screen, app-like reading experience.

Stop / restart:

```bash
docker compose down          # stop & remove
docker compose up -d         # start again (no rebuild needed)
```

> Port 8080 already in use? Edit the `ports:` line in `docker-compose.yml` (e.g. `"9090:80"`) and reopen on that port.

---

## Quick local preview (no Docker)

Any static file server works. The simplest:

```bash
cd embedded-mastery
python3 -m http.server 8080
# open http://localhost:8080  (or http://<LAN-IP>:8080 on the iPad)
```

Opening `index.html` directly via `file://` mostly works too, but a server is recommended so relative asset paths and fonts load cleanly.

---

## How it's built (for tinkering)

- **One source of truth for navigation.** `assets/app.js` holds a `MANIFEST` of both tracks and builds the sidebar, the in-page section TOC, and the prev/next links at runtime. To add or reorder chapters, edit `MANIFEST` and drop in the HTML file — no cross-links to maintain by hand.
- **Per-track theming.** A chapter sets `class="track-arduino"` or `track-stm32` on `<body>`; only the accent color changes.
- **Declarative exercises.** Quizzes/numeric/reveal challenges are pure HTML (`data-ex="quiz"`, etc.); `app.js` wires the behavior and saves results to `localStorage` (key `em-progress-v1`), which the hub reads for the progress bars.
- **Offline assets.** KaTeX, Mermaid, and highlight.js are vendored under `assets/`. Nothing is fetched at runtime.

The authoring conventions live in `_AUTHORING_GUIDE.md`, with `Arduino/01-intro.html` as the gold-standard reference page.

---

## Resetting progress

Progress is stored only in the browser. To clear it, open the site and run in the browser console:

```js
localStorage.removeItem('em-progress-v1')
```

Enjoy — and happy hacking. 🛠️
