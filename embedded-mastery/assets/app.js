/* ============================================================================
   Embedded Mastery — shared runtime
   - vendored libs: highlight.js, KaTeX (auto-render), Mermaid (all local)
   - builds sidebar + sub-TOC + prev/next from MANIFEST (single source of truth)
   - declarative exercises (quiz / numeric / reveal)
   - progress tracking in localStorage (read by the landing hub)
   Pages set:  <body data-track="arduino" data-chapter="04-pwm"> ... </body>
   ========================================================================== */
(function () {
  "use strict";

  /* ----------------------------------------------------------- MANIFEST ---
     Single source of truth for both tracks. file = path relative to track dir. */
  const MANIFEST = {
    arduino: {
      title: "Arduino",
      sub: "AVR / C++ · beginner → master",
      icon: "🔌",
      base: "Arduino/",
      chapters: [
        { id: "01-intro",              num: "01", file: "01-intro.html",              title: "Foundations: MCUs, the AVR & Your First Sketch" },
        { id: "02-gpio",               num: "02", file: "02-gpio.html",               title: "Digital I/O: GPIO, Ohm's Law, Buttons & Debouncing" },
        { id: "03-analog",             num: "03", file: "03-analog.html",             title: "The Analog World: ADC, Sensors & Signal Conditioning" },
        { id: "04-pwm",                num: "04", file: "04-pwm.html",                title: "PWM & Analog Output: Dimming, Servos & Motors" },
        { id: "05-timers-interrupts",  num: "05", file: "05-timers-interrupts.html",  title: "Timers & Interrupts: Real-Time Without delay()" },
        { id: "06-serial-i2c-spi",     num: "06", file: "06-serial-i2c-spi.html",     title: "Communication: UART, I²C & SPI" },
        { id: "07-sensors-actuators",  num: "07", file: "07-sensors-actuators.html",  title: "Sensors & Actuators: Real-World Interfacing" },
        { id: "08-architecture-power", num: "08", file: "08-architecture-power.html", title: "Robust Firmware: State Machines, Non-Blocking & Power" },
        { id: "09-capstone",           num: "09", file: "09-capstone.html",           title: "Master Capstone: A Complete Embedded System" }
      ]
    },
    stm32: {
      title: "STM32",
      sub: "ARM Cortex-M · HAL & registers",
      icon: "⚙️",
      base: "STM32/",
      chapters: [
        { id: "01-intro",       num: "01", file: "01-intro.html",       title: "From AVR to ARM: Cortex-M, Memory Map & Toolchains" },
        { id: "02-clocks",      num: "02", file: "02-clocks.html",      title: "The Clock Tree: HSI/HSE/PLL, RCC & SysTick" },
        { id: "03-gpio",        num: "03", file: "03-gpio.html",        title: "GPIO at the Register Level: MODER, BSRR & HAL" },
        { id: "04-nvic-exti",   num: "04", file: "04-nvic-exti.html",   title: "Interrupts Done Right: NVIC, EXTI & Priorities" },
        { id: "05-timers",      num: "05", file: "05-timers.html",      title: "Timer Mastery: PWM, Input Capture & Encoders" },
        { id: "06-adc-dac",     num: "06", file: "06-adc-dac.html",     title: "Analog: ADC, DMA Streaming, DAC & Oversampling" },
        { id: "07-uart-i2c-spi",num: "07", file: "07-uart-i2c-spi.html",title: "Serial Buses: USART, I²C, SPI & Baud Math" },
        { id: "08-dma",         num: "08", file: "08-dma.html",         title: "DMA Deep Dive: Channels, Circular Mode & Double Buffering" },
        { id: "09-rtos-lowpower",num:"09", file: "09-rtos-lowpower.html",title: "FreeRTOS & Low Power: Tasks, Scheduling & Sleep" },
        { id: "10-capstone",    num: "10", file: "10-capstone.html",    title: "Master Capstone: Real-Time Motor Control with PID" }
      ]
    }
  };

  /* ------------------------------------------------------------ progress --- */
  const PKEY = "em-progress-v1";
  const Progress = {
    load() { try { return JSON.parse(localStorage.getItem(PKEY)) || {}; } catch (e) { return {}; } },
    save(d) { try { localStorage.setItem(PKEY, JSON.stringify(d)); } catch (e) {} },
    key(track, ch) { return track + "/" + ch; },
    get(track, ch) { const d = this.load(); return d[this.key(track, ch)] || { visited: false, done: false, quiz: {} }; },
    set(track, ch, patch) {
      const d = this.load(); const k = this.key(track, ch);
      d[k] = Object.assign({ visited: false, done: false, quiz: {} }, d[k], patch);
      this.save(d); return d[k];
    },
    quiz(track, ch, id, ok) {
      const d = this.load(); const k = this.key(track, ch);
      const e = d[k] || { visited: true, done: false, quiz: {} };
      e.quiz = e.quiz || {}; e.quiz[id] = ok; d[k] = e; this.save(d);
    }
  };

  /* --------------------------------------------------------------- utils --- */
  const $  = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };

  const track   = document.body.dataset.track || null;
  const chapter = document.body.dataset.chapter || null;
  const onHub   = document.body.dataset.page === "hub";

  /* ============================================================ NAV BUILD === */
  function buildSidebar() {
    const host = $("#sidebar");
    if (!host || !track) return;
    const data = MANIFEST[track];

    const head = el("div", "sb-section-title", data.title + " Track");
    host.appendChild(head);

    data.chapters.forEach(ch => {
      const isCur = ch.id === chapter;
      const a = el("a", "sb-chapter" + (isCur ? " current" : ""));
      a.href = ch.file;
      a.innerHTML = `<span class="num">${ch.num}</span><span class="lbl">${ch.title.split(":")[0]}</span>`;
      host.appendChild(a);

      if (isCur) {
        // in-page section TOC, scanned from the content's h2[id]
        const toc = el("div", "sb-toc");
        $$(".content h2[id]").forEach(h => {
          const link = el("a", null, h.dataset.toc || h.textContent.replace(/^\s*[\d.]+\s*/, ""));
          link.href = "#" + h.id; link.dataset.target = h.id;
          toc.appendChild(link);
        });
        if (toc.children.length) host.appendChild(toc);
      }
    });

    const home = el("a", "sb-chapter", `<span class="num">↩</span><span class="lbl">All tracks</span>`);
    home.href = "../index.html"; home.style.marginTop = "18px";
    host.appendChild(home);
  }

  function buildChapNav() {
    const host = $("#chap-nav");
    if (!host || !track) return;
    const list = MANIFEST[track].chapters;
    const i = list.findIndex(c => c.id === chapter);
    const prev = list[i - 1], next = list[i + 1];

    host.innerHTML = "";
    const mk = (ch, dir, cls) => {
      if (!ch) { const d = el("a", "disabled " + cls); d.innerHTML = `<span class="dir">${dir}</span><span class="ttl">—</span>`; return d; }
      const a = el("a", cls); a.href = ch.file;
      a.innerHTML = `<span class="dir">${dir}</span><span class="ttl">${ch.title.split(":")[0]}</span>`;
      return a;
    };
    host.appendChild(mk(prev, "‹ Previous", "prev"));
    host.appendChild(mk(next, "Next ›", "next"));
  }

  /* =========================================================== LIB INIT ==== */
  function initHighlight() {
    if (window.hljs) $$("pre code").forEach(b => { try { hljs.highlightElement(b); } catch (e) {} });
  }
  function initKatex() {
    if (window.renderMathInElement) {
      renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false }
        ],
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"],
        throwOnError: false
      });
    }
  }
  function initMermaid() {
    if (!window.mermaid) return;
    try {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "loose",
        theme: "base",
        themeVariables: {
          darkMode: true,
          background: "#161b22",
          primaryColor: "#1c2230",
          primaryTextColor: "#e6edf3",
          primaryBorderColor: "#3a4458",
          lineColor: "#8b949e",
          secondaryColor: "#11161f",
          tertiaryColor: "#0d1117",
          fontSize: "15px",
          fontFamily: "var(--sans)"
        }
      });
      mermaid.run({ querySelector: ".mermaid" });
    } catch (e) { console.warn("mermaid", e); }
  }

  /* =========================================================== EXERCISES === */
  function initExercises() {
    // multiple choice quiz
    $$('.exercise[data-ex="quiz"]').forEach(ex => {
      const id = ex.dataset.id || "";
      const explain = $(".ex-explain", ex);
      $$(".ex-options li", ex).forEach(li => {
        li.addEventListener("click", () => {
          if (ex.dataset.answered) return;
          ex.dataset.answered = "1";
          const ok = li.dataset.correct === "true";
          $$(".ex-options li", ex).forEach(o => {
            o.classList.add("disabled");
            if (o.dataset.correct === "true") o.classList.add("correct");
          });
          if (!ok) li.classList.add("wrong");
          if (explain) explain.classList.add("show");
          if (track && chapter && id) Progress.quiz(track, chapter, id, ok);
          maybeFeedback(ex, ok);
        });
      });
    });

    // numeric answer within tolerance (handles hex like 0x40020014)
    const parseNum = (s) => {
      if (s == null) return NaN;
      s = String(s).trim().replace(/,/g, "").replace(/_/g, "");
      if (/^[-+]?0x[0-9a-f]+$/i.test(s)) return parseInt(s, 16);
      return parseFloat(s);
    };
    $$('.exercise[data-ex="numeric"]').forEach(ex => {
      const ans = parseNum(ex.dataset.answer);
      const tol = parseFloat(ex.dataset.tol || "0");
      const input = $(".ex-input", ex);
      // a number input can't accept "0x..."; widen it for hex answers
      if (input && /^[-+]?0x/i.test(ex.dataset.answer || "")) input.type = "text";
      const btn = $(".ex-check", ex);
      const explain = $(".ex-explain", ex);
      const fb = $(".ex-feedback", ex) || (() => { const f = el("div", "ex-feedback"); ex.insertBefore(f, explain || null); return f; })();
      const check = () => {
        const v = parseNum(input.value);
        if (isNaN(v)) { fb.className = "ex-feedback no"; fb.textContent = "Enter a number first."; return; }
        const ok = Math.abs(v - ans) <= (tol || Math.abs(ans) * 0.02 + 1e-9);
        fb.className = "ex-feedback " + (ok ? "ok" : "no");
        fb.textContent = ok ? "✓ Correct!" : "✗ Not quite — try again or reveal the explanation.";
        if (explain) explain.classList.add("show");
        if (track && chapter && ex.dataset.id) Progress.quiz(track, chapter, ex.dataset.id, ok);
      };
      if (btn) btn.addEventListener("click", check);
      if (input) input.addEventListener("keydown", e => { if (e.key === "Enter") check(); });
    });

    // reveal-style (hint / solution) just needs progress credit on open
    $$('.exercise[data-ex="reveal"] details.ex-acc').forEach(d => {
      d.addEventListener("toggle", () => {
        if (d.open && track && chapter) {
          const ex = d.closest(".exercise");
          if (ex && ex.dataset.id) Progress.quiz(track, chapter, ex.dataset.id, true);
        }
      });
    });
  }
  function maybeFeedback(ex, ok) {
    let fb = $(".ex-feedback", ex);
    if (!fb) { fb = el("div", "ex-feedback"); ex.appendChild(fb); }
    fb.className = "ex-feedback " + (ok ? "ok" : "no");
    fb.textContent = ok ? "✓ Correct!" : "✗ See the highlighted answer & explanation.";
  }

  /* ===================================================== SIM FRAMEWORK ===== */
  /* EM.sim(canvasId, {params, draw})  — params drive auto-built sliders.
     draw(ctx, vals, geom) is called on every change + on resize. */
  function makeSim(canvasId, cfg) {
    const cv = document.getElementById(canvasId);
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const wrap = cv.closest(".sim") || cv.parentElement;
    const vals = {};
    (cfg.params || []).forEach(p => vals[p.key] = p.value);

    // build controls
    let ctrls = wrap.querySelector(".controls");
    if (!ctrls) { ctrls = el("div", "controls"); wrap.appendChild(ctrls); }
    (cfg.params || []).forEach(p => {
      const lab = el("label");
      const out = el("output"); out.textContent = fmt(p.value, p.unit);
      lab.appendChild(el("span", null, p.label));
      const inp = el("input"); inp.type = "range"; inp.min = p.min; inp.max = p.max;
      inp.step = p.step != null ? p.step : 1; inp.value = p.value;
      inp.addEventListener("input", () => { vals[p.key] = parseFloat(inp.value); out.textContent = fmt(vals[p.key], p.unit); render(); });
      lab.appendChild(inp); lab.appendChild(out);
      ctrls.appendChild(lab);
    });

    let readout = wrap.querySelector(".readout");
    function render() {
      resize();
      ctx.clearRect(0, 0, cv.width, cv.height);
      const info = cfg.draw(ctx, vals, { w: cv.width, h: cv.height, dpr: window.devicePixelRatio || 1 }) || "";
      if (info) { if (!readout) { readout = el("div", "readout"); wrap.appendChild(readout); } readout.innerHTML = info; }
    }
    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const cssW = cv.clientWidth || 640;
      const cssH = cfg.height || 240;
      cv.style.height = cssH + "px";
      if (cv.width !== Math.round(cssW * dpr) || cv.height !== Math.round(cssH * dpr)) {
        cv.width = Math.round(cssW * dpr); cv.height = Math.round(cssH * dpr);
      }
    }
    function fmt(v, u) { const s = (Math.round(v * 1000) / 1000).toString(); return u ? s + " " + u : s; }
    window.addEventListener("resize", () => { clearTimeout(makeSim._t); makeSim._t = setTimeout(render, 120); });
    render();
    return { render, vals };
  }

  /* ===================================================== CHROME / UX ======= */
  function initChrome() {
    // mobile drawer
    const menu = $("#menu-btn"), backdrop = $("#sb-backdrop");
    if (menu) menu.addEventListener("click", () => document.body.classList.toggle("sb-open"));
    if (backdrop) backdrop.addEventListener("click", () => document.body.classList.remove("sb-open"));
    $$("#sidebar a").forEach(a => a.addEventListener("click", () => document.body.classList.remove("sb-open")));

    // reading progress + back-to-top + done marking
    const bar = $("#read-progress"), toTop = $("#to-top"), chip = $("#done-chip");
    const onScroll = () => {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      const pct = max > 0 ? (h.scrollTop / max) * 100 : 0;
      if (bar) bar.style.width = pct + "%";
      if (toTop) toTop.classList.toggle("show", h.scrollTop > 600);
      if (pct > 92 && track && chapter) {
        Progress.set(track, chapter, { visited: true, done: true });
        if (chip) { chip.classList.add("done"); chip.textContent = "✓ Completed"; }
      }
    };
    if (toTop) toTop.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
    window.addEventListener("scroll", onScroll, { passive: true });

    // mark visited & restore done chip
    if (track && chapter) {
      const st = Progress.get(track, chapter);
      Progress.set(track, chapter, { visited: true });
      if (chip && st.done) { chip.classList.add("done"); chip.textContent = "✓ Completed"; }
    }
    onScroll();

    // scrollspy for sub-TOC
    const tocLinks = $$(".sb-toc a");
    if (tocLinks.length) {
      const map = {};
      tocLinks.forEach(l => map[l.dataset.target] = l);
      const obs = new IntersectionObserver(entries => {
        entries.forEach(e => {
          if (e.isIntersecting) {
            tocLinks.forEach(l => l.classList.remove("active"));
            if (map[e.target.id]) map[e.target.id].classList.add("active");
          }
        });
      }, { rootMargin: "-70px 0px -70% 0px", threshold: 0 });
      $$(".content h2[id]").forEach(h => obs.observe(h));
    }
  }

  /* =============================================================== HUB ===== */
  function initHub() {
    if (!onHub) return;
    const data = Progress.load();
    Object.keys(MANIFEST).forEach(tk => {
      const chs = MANIFEST[tk].chapters;
      let done = 0;
      chs.forEach(c => { if ((data[tk + "/" + c.id] || {}).done) done++; });
      const pct = Math.round((done / chs.length) * 100);
      const bar = document.querySelector(`.track-card.${tk} .tc-progress > i`);
      const lab = document.querySelector(`.track-card.${tk} .tc-progress-label`);
      if (bar) bar.style.width = pct + "%";
      if (lab) lab.textContent = `${done} / ${chs.length} chapters complete`;
    });
  }

  /* =============================================================== BOOT ==== */
  function boot() {
    buildSidebar();
    buildChapNav();
    initHighlight();
    initKatex();
    initMermaid();
    initExercises();
    initChrome();
    initHub();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  // public API for chapter-specific simulators
  window.EM = { sim: makeSim, MANIFEST, Progress };
})();
