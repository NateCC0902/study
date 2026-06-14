# Embedded Mastery — Chapter Authoring Guide (READ FIRST)

You are writing ONE self-contained chapter HTML file for a polished, offline,
dark-themed learning site read on an iPad. A complete gold-standard example
already exists at `Arduino/01-intro.html` — **open and mirror it exactly** for
structure, skeleton, tone, and component usage. This guide is the spec.

## Non-negotiable rules
1. Output a COMPLETE, valid HTML document. No truncation, no "...", no TODO.
2. Use ONLY the vendored assets via the exact `<link>`/`<script>` block below.
   Never reference a CDN or the internet — the iPad may be fully offline.
3. Paths are relative from inside `Arduino/` or `STM32/` → assets are `../assets/...`.
4. Set `<body class="track-arduino"...>` for Arduino, `track-stm32` for STM32,
   and `data-track` / `data-chapter` to the exact ids from the manifest.
5. Do NOT hand-write the sidebar list or prev/next nav — leave the empty
   `<nav class="sidebar" id="sidebar"></nav>` and `<nav class="chap-nav" id="chap-nav"></nav>`.
   `app.js` fills them automatically from the manifest + your `<h2 id>`s.
6. Every `<h2>` MUST have a unique `id` (kebab-case) — the in-page TOC is built
   from them. Number sections with `<span class="sec-no">N</span>`.

## Exact page skeleton (copy verbatim; change only title, track classes, ids, content)
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>CHAPTER TITLE — Embedded Mastery</title>
<link rel="stylesheet" href="../assets/katex/katex.min.css">
<link rel="stylesheet" href="../assets/vendor/atom-one-dark.min.css">
<link rel="stylesheet" href="../assets/styles.css">
</head>
<body class="track-arduino" data-track="arduino" data-chapter="02-gpio">
<div id="read-progress"></div>
<header class="topbar">
  <button class="menu-btn" id="menu-btn" aria-label="Open chapter menu">☰</button>
  <a class="brand" href="../index.html"><span class="dot"></span> Embedded&nbsp;Mastery</a>
  <span class="spacer"></span>
  <span class="track-pill">Arduino · Ch 02</span>
  <span class="done-chip" id="done-chip">In progress</span>
</header>
<div class="layout">
  <nav class="sidebar" id="sidebar"></nav>
  <div class="sb-backdrop" id="sb-backdrop"></div>
  <main class="main">
    <article class="content">
      <section class="chapter-hero">
        <div class="eyebrow">Arduino Track · Chapter 02 · Beginner→Intermediate</div>
        <h1>Chapter Title Here</h1>
        <p>One-paragraph hook explaining why this matters.</p>
        <div class="meta"><span>⏱ ~40 min</span><span>🧠 concept tag</span><span>⚡ hands-on tag</span></div>
      </section>

      <!-- ... your sections ... -->

      <nav class="chap-nav" id="chap-nav"></nav>
    </article>
  </main>
</div>
<button id="to-top" aria-label="Back to top">↑</button>
<script src="../assets/vendor/highlight.min.js"></script>
<script src="../assets/katex/katex.min.js"></script>
<script src="../assets/katex/contrib/auto-render.min.js"></script>
<script src="../assets/vendor/mermaid.min.js"></script>
<script src="../assets/app.js"></script>
<!-- optional chapter simulator <script> AFTER app.js, calling EM.sim(...) -->
</body>
</html>
```

## The teaching rhythm (REQUIRED for every major topic)
Each `<h2>` topic must move through these layers, marked with the badge element.
Not every layer needs equal length, but intuition → math → code → real case →
exercise must all appear across the chapter (math at least 3–5 real formulas).

```html
<div class="layer"><span class="step">①</span> Intuition <small>build the picture first</small></div>
<div class="layer"><span class="step">②</span> Math <small>derive it</small></div>
<div class="layer"><span class="step">③</span> Code <small>read every line</small></div>
<div class="layer"><span class="step">④</span> Real case <small>this ships in products</small></div>
<div class="layer"><span class="step">⑤</span> Practice <small>you decide</small></div>
```

## Component vocabulary (use these classes — they are already styled)

### Callouts (pick the hue that fits)
```html
<div class="callout intuition"><div class="c-title"><span class="ico">💡</span> Title</div><p>…</p></div>
<div class="callout key">      <div class="c-title"><span class="ico">🔑</span> Title</div><p>…</p></div>
<div class="callout math">      <div class="c-title"><span class="ico">∑</span> Title</div><p>…</p></div>
<div class="callout real">      <div class="c-title"><span class="ico">🏭</span> Real case — …</div><p>…</p></div>
<div class="callout warn">      <div class="c-title"><span class="ico">⚠️</span> Pitfall — …</div><p>…</p></div>
<div class="callout tip">       <div class="c-title"><span class="ico">🛠️</span> Title</div><p>…</p></div>
```

### Math — KaTeX auto-renders. Inline: wrap in `$ … $`. Display: `$$ … $$`.
Prefer a display block inside a `.math-box` for headline equations:
```html
<div class="math-box">$$ f = \frac{f_{clk}}{(PSC+1)(ARR+1)} $$</div>
```
KaTeX ignores anything inside `<code>`/`<pre>`, so `$` in code is safe.

### Code — always use the wrapper with a filename header. Escape `<` as `&lt;` and `&` as `&amp;` inside code.
```html
<div class="codeblock">
  <div class="code-head">file.ino <span class="lang">C++ / Arduino</span></div>
<pre><code class="language-cpp">// code here, &lt; and &amp; escaped
</code></pre>
</div>
```
Languages: `language-cpp` (Arduino/C/C++), `language-c`, `language-bash`. STM32 register code is `language-c`.

### Tables — wrap for horizontal scroll on iPad:
```html
<div class="tbl-wrap"><table><thead><tr><th>…</th></tr></thead><tbody><tr><td>…</td></tr></tbody></table></div>
```

### Diagrams — Mermaid (flowchart/sequence/stateDiagram). One diagram per `<div class="mermaid">`.
Keep node labels short. Use `<br/>` for line breaks inside labels. Example:
```html
<div class="mermaid">
flowchart LR
    A[Input] --> B{Decide}
    B -- yes --> C[Act]
    B -- no --> A
</div>
```
Theme is pre-configured (dark). Do NOT call mermaid.initialize yourself.

### Exercises (declarative — app.js wires them; give each a unique data-id like `ard04-pwm-duty`)
Multiple-choice quiz:
```html
<div class="exercise" data-ex="quiz" data-id="TRACKch-topic">
  <div class="ex-head"><span class="badge">Q1</span> Conceptual check</div>
  <div class="ex-q">Question text (may contain $math$ and <code>code</code>).</div>
  <ul class="ex-options">
    <li data-correct="false">wrong</li>
    <li data-correct="true">right</li>
    <li data-correct="false">wrong</li>
  </ul>
  <div class="ex-explain">Explanation shown after answering. Include the why.</div>
</div>
```
Numeric (tolerance check; data-tol is absolute):
```html
<div class="exercise" data-ex="numeric" data-id="…" data-answer="2.0" data-tol="0.05">
  <div class="ex-head"><span class="badge">Q2</span> Compute it</div>
  <div class="ex-q">Compute … (give the formula in $…$).</div>
  <input class="ex-input" type="number" step="any" placeholder="value"><span class="ex-unit">V</span>
  <button class="ex-check">Check</button>
  <div class="ex-explain">Worked solution.</div>
</div>
```
Reveal / coding challenge (hint + solution accordions):
```html
<div class="exercise" data-ex="reveal" data-id="…">
  <div class="ex-head"><span class="badge">Q3</span> Coding challenge — …</div>
  <div class="ex-q">Task description.</div>
  <details class="ex-acc"><summary>Hint</summary><div class="acc-body">…</div></details>
  <details class="ex-acc"><summary>Show one solution</summary><div class="acc-body">
    <div class="codeblock"><div class="code-head">file <span class="lang">…</span></div><pre><code class="language-cpp">…</code></pre></div>
  </div></details>
</div>
```

### Optional live simulator (great for one key concept per chapter)
A `<div class="sim">` with a `<canvas id="…">`, then AFTER app.js a script calling
`EM.sim(canvasId, {height, params:[{key,label,min,max,step,value,unit}], draw:function(ctx,v,g){…}})`.
`g` = {w,h,dpr}; canvas is sized in device pixels so multiply sizes by `g.dpr`.
Return an HTML string from `draw` to show a live readout. See `Arduino/01-intro.html` §6 for a full working example. Only add a sim if it genuinely illuminates the concept; otherwise skip it.

## Tone & depth
- Voice: a sharp senior engineer teaching a capable peer. Direct, concrete, a little wry. No fluff, no "in this section we will".
- Beginner→master: early chapters assume little; later chapters assume earlier ones. Always build from intuition before formalism.
- Be technically RIGOROUS and CORRECT: real register names, real formulas, real part numbers, realistic numbers. Show the derivation, not just the result.
- Every chapter: 6–9 `<h2>` sections, several callouts, ≥3 KaTeX formulas, ≥1 Mermaid diagram, ≥2 annotated code blocks, a "real case" or two, and an `⑤ Your turn` section with 3 exercises (one quiz, one numeric, one reveal/coding).
- End with a `<div class="callout key">` "Chapter recap" before the chap-nav.
- Length target: rich and complete — roughly 1500–2600 words of prose plus code/diagrams. Quality and correctness over padding.

## Final self-check before returning
- [ ] Doc starts with `<!DOCTYPE html>` and ends with `</html>`.
- [ ] Correct `track-*` class + `data-track` + `data-chapter`.
- [ ] Empty `#sidebar` and `#chap-nav` left for app.js.
- [ ] Every `<h2>` has a unique `id`.
- [ ] All `<` and `&` inside `<code>` are escaped.
- [ ] 3 exercises with unique `data-id`s; quiz options have `data-correct`.
- [ ] Scripts block present, app.js last, any sim script after it.
