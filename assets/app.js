/* ============================================================================
   Study — shared runtime for all courses (Embedded Mastery · ML/DL · PID)
   - vendored libs: highlight.js, KaTeX (auto-render), Mermaid (all local)
   - self-locating: finds the site root from this script's own URL, so pages at
     any depth (and even file://) just work. No data-root needed.
   - COURSES registry is the single source of truth for nav + the hub.
   - bilingual: EN ⇄ 繁中. Language is derived from the path (/zh-tw/ = 繁中);
     the topbar toggle jumps to the sibling page (or swaps the hub in place).
   - declarative exercises (quiz / numeric / reveal); EM.sim() canvas framework.
   - progress in localStorage, keyed course/track/chapter (language-agnostic).
   Pages set on <body>:  data-course data-track data-chapter   (chapter pages)
                         data-page="hub"                        (the master hub)
   ========================================================================== */
(function () {
  "use strict";

  /* --- self-location: where am I served from? -------------------------------
     thisScript.src === ".../study/assets/app.js"  → SITE_ROOT === ".../study/" */
  const thisScript = document.currentScript ||
    (function () { const s = document.getElementsByTagName("script"); return s[s.length - 1]; })();
  const ASSET_BASE = thisScript.src.replace(/[^/]*$/, "");          // .../study/assets/
  const SITE_ROOT  = ASSET_BASE.replace(/assets\/$/, "");           // .../study/

  /* ----------------------------------------------------------- COURSES ----
     base = path under SITE_ROOT. track.dir = subdir under base ("" = none).
     Each chapter: { id, num, file, title (EN), zh (繁中) }.
     繁中 pages live in a `zh-tw/` subfolder mirroring the EN filenames.        */
  const COURSES = {
    embedded: {
      title: "Embedded Mastery", zh: "嵌入式精通", short: "Embedded",
      icon: "🔌", accent: "#33b1ff", accent2: "#6cc4ff", base: "embedded-mastery/",
      tracks: {
        arduino: {
          title: "Arduino", zh: "Arduino", sub: "AVR · C++", dir: "Arduino/",
          chapters: [
            { id: "01-intro",              num: "01", file: "01-intro.html",              title: "Foundations & First Sketch",      zh: "基礎與第一支草稿碼" },
            { id: "02-gpio",               num: "02", file: "02-gpio.html",               title: "Digital I/O, Ohm's Law & Buttons", zh: "數位 I/O、歐姆定律與按鈕" },
            { id: "03-analog",             num: "03", file: "03-analog.html",             title: "The Analog World: ADC & Sensors", zh: "類比世界：ADC 與感測器" },
            { id: "04-pwm",                num: "04", file: "04-pwm.html",                title: "PWM & Analog Output",             zh: "PWM 與類比輸出" },
            { id: "05-timers-interrupts",  num: "05", file: "05-timers-interrupts.html",  title: "Timers & Interrupts",             zh: "計時器與中斷" },
            { id: "06-serial-i2c-spi",     num: "06", file: "06-serial-i2c-spi.html",     title: "UART, I²C & SPI",                 zh: "UART、I²C 與 SPI" },
            { id: "07-sensors-actuators",  num: "07", file: "07-sensors-actuators.html",  title: "Sensors & Actuators",             zh: "感測器與致動器" },
            { id: "08-architecture-power", num: "08", file: "08-architecture-power.html", title: "Robust Firmware & Power",          zh: "穩健韌體與電源" },
            { id: "09-capstone",           num: "09", file: "09-capstone.html",           title: "Capstone: A Complete System",     zh: "結業專題：完整系統" }
          ]
        },
        stm32: {
          title: "STM32", zh: "STM32", sub: "ARM Cortex-M", dir: "STM32/",
          chapters: [
            { id: "01-intro",        num: "01", file: "01-intro.html",        title: "From AVR to ARM: Cortex-M",        zh: "從 AVR 到 ARM：Cortex-M" },
            { id: "02-clocks",       num: "02", file: "02-clocks.html",       title: "The Clock Tree & SysTick",        zh: "時脈樹與 SysTick" },
            { id: "03-gpio",         num: "03", file: "03-gpio.html",         title: "GPIO at the Register Level",      zh: "暫存器層級的 GPIO" },
            { id: "04-nvic-exti",    num: "04", file: "04-nvic-exti.html",    title: "NVIC, EXTI & Priorities",         zh: "NVIC、EXTI 與優先權" },
            { id: "05-timers",       num: "05", file: "05-timers.html",       title: "Timer Mastery",                   zh: "計時器精通" },
            { id: "06-adc-dac",      num: "06", file: "06-adc-dac.html",      title: "ADC, DMA Streaming & DAC",        zh: "ADC、DMA 串流與 DAC" },
            { id: "07-uart-i2c-spi", num: "07", file: "07-uart-i2c-spi.html", title: "Serial Buses & Baud Math",        zh: "串列匯流排與鮑率計算" },
            { id: "08-dma",          num: "08", file: "08-dma.html",          title: "DMA Deep Dive",                   zh: "DMA 深入" },
            { id: "09-rtos-lowpower",num: "09", file: "09-rtos-lowpower.html",title: "FreeRTOS & Low Power",            zh: "FreeRTOS 與低功耗" },
            { id: "10-capstone",     num: "10", file: "10-capstone.html",     title: "Capstone: Motor Control with PID",zh: "結業專題：PID 馬達控制" }
          ]
        }
      }
    },

    mldl: {
      title: "Machine Learning & Deep Learning", zh: "機器學習與深度學習", short: "ML / DL",
      icon: "🧠", accent: "#a371f7", accent2: "#c8a6ff", base: "ml-dl-curriculum/",
      tracks: {
        main: {
          title: "Course", zh: "課程", sub: "22 lessons + appendix", dir: "",
          chapters: [
            { id: "00-what-is-ml",                num: "00", file: "00-what-is-ml.html",                title: "What Is Machine Learning?",        zh: "什麼是機器學習？" },
            { id: "01-math-foundations",          num: "01", file: "01-math-foundations.html",          title: "The Math Toolbox",                 zh: "數學工具箱" },
            { id: "02-linear-regression",         num: "02", file: "02-linear-regression.html",         title: "Linear Regression",                zh: "線性迴歸" },
            { id: "03-gradient-descent",          num: "03", file: "03-gradient-descent.html",          title: "Gradient Descent",                 zh: "梯度下降" },
            { id: "04-logistic-regression",       num: "04", file: "04-logistic-regression.html",       title: "Logistic Regression",              zh: "邏輯迴歸與分類" },
            { id: "05-overfitting-evaluation",    num: "05", file: "05-overfitting-evaluation.html",    title: "Overfitting & Evaluation",         zh: "過度擬合、正則化與評估" },
            { id: "06-knn-trees-ensembles",       num: "06", file: "06-knn-trees-ensembles.html",       title: "k-NN, Trees & Ensembles",          zh: "k-最近鄰、決策樹與集成" },
            { id: "07-svm-kernels",               num: "07", file: "07-svm-kernels.html",               title: "SVMs & Kernels",                   zh: "支持向量機與核" },
            { id: "08-kmeans-pca",                num: "08", file: "08-kmeans-pca.html",                title: "k-Means & PCA",                    zh: "非監督式學習：k-平均與 PCA" },
            { id: "09-neural-networks-mlp",       num: "09", file: "09-neural-networks-mlp.html",       title: "Neural Networks & Forward Pass",   zh: "神經網路與前向傳播" },
            { id: "10-backpropagation",           num: "10", file: "10-backpropagation.html",           title: "Backpropagation from Scratch",     zh: "從零實作反向傳播" },
            { id: "11-pytorch-fundamentals",      num: "11", file: "11-pytorch-fundamentals.html",      title: "PyTorch Fundamentals",             zh: "PyTorch 基礎" },
            { id: "12-training-deep-nets",        num: "12", file: "12-training-deep-nets.html",        title: "Training Deep Networks",           zh: "訓練真正能收斂的深度網路" },
            { id: "13-cnns",                      num: "13", file: "13-cnns.html",                      title: "Convolutional Networks",           zh: "卷積神經網路" },
            { id: "14-rnns-lstms",                num: "14", file: "14-rnns-lstms.html",                title: "Sequence Models: RNNs & LSTMs",    zh: "序列模型：RNN 與 LSTM" },
            { id: "15-attention-transformers",    num: "15", file: "15-attention-transformers.html",    title: "Attention & Transformers",         zh: "注意力與 Transformer" },
            { id: "16-generative-models",         num: "16", file: "16-generative-models.html",         title: "Generative Models",                zh: "生成模型" },
            { id: "17-transfer-learning-llms-mlops",num:"17",file: "17-transfer-learning-llms-mlops.html",title:"Transfer Learning, LLMs & MLOps", zh: "遷移學習、LLM 與 MLOps" },
            { id: "18-reinforcement-learning",    num: "18", file: "18-reinforcement-learning.html",    title: "Reinforcement Learning",           zh: "強化學習" },
            { id: "19-detection-segmentation",    num: "19", file: "19-detection-segmentation.html",    title: "Detection & Segmentation",         zh: "物件偵測與分割" },
            { id: "20-data-feature-engineering",  num: "20", file: "20-data-feature-engineering.html",  title: "Data & Feature Engineering",       zh: "資料與特徵工程" },
            { id: "21-hyperparameter-optimization",num:"21",file: "21-hyperparameter-optimization.html",title:"Hyperparameter Optimization",      zh: "超參數最佳化與模型選擇" },
            { id: "appendix",                     num: "A",  file: "appendix.html",                     title: "Appendix: Math Cheatsheet",        zh: "附錄：數學速查表" },
            { id: "glossary",                     num: "G",  file: "glossary.html",                     title: "Glossary (EN/中)",                 zh: "中英術語對照表" }
          ]
        }
      }
    },

    pid: {
      title: "PID Control for a USV", zh: "USV 的 PID 控制", short: "PID",
      icon: "🧭", accent: "#3fb950", accent2: "#56d364", base: "pid/",
      tracks: {
        main: {
          title: "Course", zh: "課程", sub: "8 chapters + glossary", dir: "",
          chapters: [
            { id: "01-plant",              num: "01", file: "01-plant.html",              title: "The Plant & Open Loop",       zh: "受控體與開迴路" },
            { id: "02-p-control",          num: "02", file: "02-p-control.html",          title: "P Control: The Spring",       zh: "P 控制：彈簧" },
            { id: "03-d-control",          num: "03", file: "03-d-control.html",          title: "D Control: The Damper",       zh: "D 控制：阻尼器" },
            { id: "04-i-control",          num: "04", file: "04-i-control.html",          title: "I Control: The Trim Tab",     zh: "I 控制：配平片" },
            { id: "05-windup",             num: "05", file: "05-windup.html",             title: "Integral Windup",             zh: "積分飽和（Windup）" },
            { id: "06-derivative-noise",   num: "06", file: "06-derivative-noise.html",   title: "Derivative Kick & Noise",     zh: "微分衝擊與雜訊" },
            { id: "07-wrapping-actuators", num: "07", file: "07-wrapping-actuators.html", title: "Angle Wrapping & Actuators",  zh: "角度纏繞與致動器" },
            { id: "08-tuning-ros2",        num: "08", file: "08-tuning-ros2.html",        title: "Tuning & ROS 2 Deployment",   zh: "整定與 ROS 2 部署" },
            { id: "glossary",              num: "G",  file: "glossary.html",              title: "Glossary",                    zh: "術語表" }
          ]
        }
      }
    },

    io: {
      title: "Linux I/O for Robotics", zh: "機器人的 Linux I/O", short: "Linux I/O",
      icon: "🔀", accent: "#f0883e", accent2: "#ff9f5a", base: "io/",
      tracks: {
        main: {
          title: "Course", zh: "課程", sub: "12 lessons + glossary", dir: "",
          chapters: [
            { id: "01-io-model",                 num: "01", file: "01-io-model.html",                 title: "The Linux I/O Model",                  zh: "Linux I/O 模型" },
            { id: "02-devices-sysfs",            num: "02", file: "02-devices-sysfs.html",            title: "Devices as Files: /dev & sysfs",       zh: "裝置即檔案：/dev 與 sysfs" },
            { id: "03-udev-naming",              num: "03", file: "03-udev-naming.html",              title: "udev & Persistent Naming",             zh: "udev 與持久化命名" },
            { id: "04-drivers-modules-firmware", num: "04", file: "04-drivers-modules-firmware.html", title: "Drivers, Modules & Firmware",          zh: "驅動、模組與韌體" },
            { id: "05-buses-enumeration",        num: "05", file: "05-buses-enumeration.html",        title: "Buses & Enumeration",                  zh: "匯流排與列舉" },
            { id: "06-serial-tty",               num: "06", file: "06-serial-tty.html",               title: "Serial & the TTY Layer",               zh: "串列與 TTY 層" },
            { id: "07-interrupts-dma-mmio",      num: "07", file: "07-interrupts-dma-mmio.html",      title: "Interrupts, DMA & MMIO",               zh: "中斷、DMA 與 MMIO" },
            { id: "08-multiplexing-epoll",       num: "08", file: "08-multiplexing-epoll.html",       title: "Multiplexing: poll, epoll & io_uring", zh: "I/O 多工：poll、epoll 與 io_uring" },
            { id: "09-networking-io",            num: "09", file: "09-networking-io.html",            title: "Networking I/O: NIC to Socket",        zh: "網路 I/O：從網卡到 socket" },
            { id: "10-can-socketcan",            num: "10", file: "10-can-socketcan.html",            title: "CAN & SocketCAN",                      zh: "CAN 與 SocketCAN" },
            { id: "11-realtime-robust-io",       num: "11", file: "11-realtime-robust-io.html",       title: "Real-Time & Robust I/O",               zh: "即時與穩健 I/O" },
            { id: "12-capstone-robot-io",        num: "12", file: "12-capstone-robot-io.html",        title: "Capstone: A Robot's I/O Architecture", zh: "結業專題：機器人的 I/O 架構" },
            { id: "glossary",                    num: "G",  file: "glossary.html",                    title: "Glossary (EN/中)",                     zh: "術語表" }
          ]
        }
      }
    }
  };

  /* ----------------------------------------------------------------- i18n --- */
  const PREF_KEY = "study-lang";
  function prefLang() { try { return localStorage.getItem(PREF_KEY) || "en"; } catch (e) { return "en"; } }
  function setPref(l) { try { localStorage.setItem(PREF_KEY, l); } catch (e) {} }

  const T = {
    en: { courses: "All courses", track: "Track", prev: "‹ Previous", next: "Next ›",
          inprog: "In progress", done: "✓ Completed", toggle: "繁中", none: "—" },
    zh: { courses: "所有課程", track: "課程", prev: "‹ 上一章", next: "下一章 ›",
          inprog: "閱讀中", done: "✓ 已完成", toggle: "EN", none: "—" }
  };

  /* ------------------------------------------------------------ progress --- */
  const PKEY = "study-progress-v1";
  const Progress = {
    load() { try { return JSON.parse(localStorage.getItem(PKEY)) || {}; } catch (e) { return {}; } },
    save(d) { try { localStorage.setItem(PKEY, JSON.stringify(d)); } catch (e) {} },
    key(course, track, ch) { return course + "/" + track + "/" + ch; },
    get(course, track, ch) { const d = this.load(); return d[this.key(course, track, ch)] || { visited: false, done: false, quiz: {} }; },
    set(course, track, ch, patch) {
      const d = this.load(); const k = this.key(course, track, ch);
      d[k] = Object.assign({ visited: false, done: false, quiz: {} }, d[k], patch);
      this.save(d); return d[k];
    },
    quiz(course, track, ch, id, ok) {
      const d = this.load(); const k = this.key(course, track, ch);
      const e = d[k] || { visited: true, done: false, quiz: {} };
      e.quiz = e.quiz || {}; e.quiz[id] = ok; d[k] = e; this.save(d);
    }
  };

  /* --------------------------------------------------------------- utils --- */
  const $  = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };

  const course  = document.body.dataset.course || null;
  const track   = document.body.dataset.track || null;
  const chapter = document.body.dataset.chapter || null;
  const onHub   = document.body.dataset.page === "hub";
  // page language: chapter pages from the path; the hub follows the saved pref.
  const lang    = onHub ? prefLang() : (/\/zh-tw\//.test(location.pathname) ? "zh" : "en");
  const t       = T[lang] || T.en;

  function chTitle(ch) { return lang === "zh" ? (ch.zh || ch.title) : ch.title; }
  function chapterURL(courseId, trackId, chFile, l) {
    const C = COURSES[courseId], TR = C.tracks[trackId];
    return SITE_ROOT + C.base + TR.dir + (l === "zh" ? "zh-tw/" : "") + chFile;
  }

  /* ============================================================ NAV BUILD === */
  function buildSidebar() {
    const host = $("#sidebar");
    if (!host || !course || !track) return;
    const C = COURSES[course], TR = C.tracks[track];

    const head = el("div", "sb-section-title",
      (lang === "zh" ? (TR.zh || TR.title) : TR.title) + " " + t.track);
    host.appendChild(head);

    TR.chapters.forEach(ch => {
      const isCur = ch.id === chapter;
      const a = el("a", "sb-chapter" + (isCur ? " current" : ""));
      a.href = chapterURL(course, track, ch.file, lang);
      a.innerHTML = `<span class="num">${ch.num}</span><span class="lbl">${chTitle(ch)}</span>`;
      host.appendChild(a);

      if (isCur) {
        const toc = el("div", "sb-toc");
        $$(".content h2[id]").forEach(h => {
          const link = el("a", null, h.dataset.toc || h.textContent.replace(/^\s*[\d.]+\s*/, ""));
          link.href = "#" + h.id; link.dataset.target = h.id;
          toc.appendChild(link);
        });
        if (toc.children.length) host.appendChild(toc);
      }
    });

    const home = el("a", "sb-chapter", `<span class="num">↩</span><span class="lbl">${t.courses}</span>`);
    home.href = SITE_ROOT + "index.html"; home.style.marginTop = "18px";
    host.appendChild(home);
  }

  function buildChapNav() {
    const host = $("#chap-nav");
    if (!host || !course || !track) return;
    const list = COURSES[course].tracks[track].chapters;
    const i = list.findIndex(c => c.id === chapter);
    const prev = list[i - 1], next = list[i + 1];

    host.innerHTML = "";
    const mk = (ch, dir, cls) => {
      if (!ch) { const d = el("a", "disabled " + cls); d.innerHTML = `<span class="dir">${dir}</span><span class="ttl">${t.none}</span>`; return d; }
      const a = el("a", cls); a.href = chapterURL(course, track, ch.file, lang);
      a.innerHTML = `<span class="dir">${dir}</span><span class="ttl">${chTitle(ch)}</span>`;
      return a;
    };
    host.appendChild(mk(prev, t.prev, "prev"));
    host.appendChild(mk(next, t.next, "next"));
  }

  /* =========================================================== LANGUAGE ==== */
  function siblingURL() {
    // same page in the other language by toggling the /zh-tw/ path segment
    const u = new URL(location.href);
    if (/\/zh-tw\//.test(u.pathname)) u.pathname = u.pathname.replace("/zh-tw/", "/");
    else u.pathname = u.pathname.replace(/\/([^/]+)$/, "/zh-tw/$1");
    return u.href;
  }
  function applyI18nVisibility(activeLang) {
    // elements tagged .lang-en / .lang-zh (used on the hub) show only the active one
    $$(".lang-en").forEach(e => e.style.display = activeLang === "zh" ? "none" : "");
    $$(".lang-zh").forEach(e => e.style.display = activeLang === "zh" ? "" : "none");
    document.documentElement.lang = activeLang === "zh" ? "zh-Hant" : "en";
  }
  function injectLangToggle() {
    const bar = $(".topbar");
    if (!bar || $("#lang-toggle")) return;
    let cur = lang;                                  // live language — the hub toggles in place, so this must update
    const btn = el("button", "lang-toggle", T[cur].toggle);
    btn.id = "lang-toggle";
    btn.setAttribute("aria-label", "Switch language");
    btn.addEventListener("click", () => {
      const nextLang = cur === "zh" ? "en" : "zh";
      setPref(nextLang);
      if (onHub) { cur = nextLang; applyI18nVisibility(nextLang); renderHub(nextLang); btn.textContent = T[nextLang].toggle; }
      else { location.href = siblingURL(); }
    });
    const anchor = bar.querySelector(".track-pill") || bar.querySelector(".spacer");
    if (anchor && anchor.parentNode === bar) bar.insertBefore(btn, anchor.nextSibling || null);
    else bar.appendChild(btn);
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
        startOnLoad: false, securityLevel: "loose", theme: "base",
        themeVariables: {
          darkMode: true, background: "#161b22", primaryColor: "#1c2230",
          primaryTextColor: "#e6edf3", primaryBorderColor: "#3a4458", lineColor: "#8b949e",
          secondaryColor: "#11161f", tertiaryColor: "#0d1117", fontSize: "15px",
          fontFamily: "var(--sans)"
        }
      });
      mermaid.run({ querySelector: ".mermaid" });
    } catch (e) { console.warn("mermaid", e); }
  }

  /* =========================================================== EXERCISES === */
  function initExercises() {
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
          if (course && track && chapter && id) Progress.quiz(course, track, chapter, id, ok);
          maybeFeedback(ex, ok);
        });
      });
    });

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
      if (input && /^[-+]?0x/i.test(ex.dataset.answer || "")) input.type = "text";
      const btn = $(".ex-check", ex);
      const explain = $(".ex-explain", ex);
      const fb = $(".ex-feedback", ex) || (() => { const f = el("div", "ex-feedback"); ex.insertBefore(f, explain || null); return f; })();
      const check = () => {
        const v = parseNum(input.value);
        if (isNaN(v)) { fb.className = "ex-feedback no"; fb.textContent = lang === "zh" ? "請先輸入數字。" : "Enter a number first."; return; }
        const ok = Math.abs(v - ans) <= (tol || Math.abs(ans) * 0.02 + 1e-9);
        fb.className = "ex-feedback " + (ok ? "ok" : "no");
        fb.textContent = ok ? (lang === "zh" ? "✓ 正確！" : "✓ Correct!")
                            : (lang === "zh" ? "✗ 還不對 — 再試一次，或看解答。" : "✗ Not quite — try again or reveal the explanation.");
        if (explain) explain.classList.add("show");
        if (course && track && chapter && ex.dataset.id) Progress.quiz(course, track, chapter, ex.dataset.id, ok);
      };
      if (btn) btn.addEventListener("click", check);
      if (input) input.addEventListener("keydown", e => { if (e.key === "Enter") check(); });
    });

    $$('.exercise[data-ex="reveal"] details.ex-acc').forEach(d => {
      d.addEventListener("toggle", () => {
        if (d.open && course && track && chapter) {
          const ex = d.closest(".exercise");
          if (ex && ex.dataset.id) Progress.quiz(course, track, chapter, ex.dataset.id, true);
        }
      });
    });
  }
  function maybeFeedback(ex, ok) {
    let fb = $(".ex-feedback", ex);
    if (!fb) { fb = el("div", "ex-feedback"); ex.appendChild(fb); }
    fb.className = "ex-feedback " + (ok ? "ok" : "no");
    fb.textContent = ok ? (lang === "zh" ? "✓ 正確！" : "✓ Correct!")
                        : (lang === "zh" ? "✗ 見標示的答案與解說。" : "✗ See the highlighted answer & explanation.");
  }

  /* ===================================================== SIM FRAMEWORK =====
     EM.sim(canvasId, { height, params, draw })
     params: { key, label, value, unit,
               type:'range'(default) min/max/step | 'toggle' | 'select' options:[{label,value}] }
     draw(ctx, vals, geom) → optional HTML string for a live readout.            */
  function makeSim(canvasId, cfg) {
    const cv = document.getElementById(canvasId);
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const wrap = cv.closest(".sim") || cv.parentElement;
    const vals = {};
    (cfg.params || []).forEach(p => vals[p.key] = p.value);

    let ctrls = wrap.querySelector(".controls");
    if (!ctrls) { ctrls = el("div", "controls"); wrap.appendChild(ctrls); }

    (cfg.params || []).forEach(p => {
      const type = p.type || "range";
      const lab = el("label");
      if (type === "toggle") {
        lab.classList.add("toggle");
        const inp = el("input"); inp.type = "checkbox"; inp.checked = !!p.value;
        inp.addEventListener("change", () => { vals[p.key] = inp.checked; render(); });
        lab.appendChild(inp); lab.appendChild(el("span", null, p.label));
      } else if (type === "select") {
        lab.appendChild(el("span", null, p.label));
        const sel = el("select");
        (p.options || []).forEach(o => { const op = el("option", null, o.label); op.value = o.value; if (o.value === p.value) op.selected = true; sel.appendChild(op); });
        sel.addEventListener("change", () => { vals[p.key] = sel.value; render(); });
        lab.appendChild(sel);
      } else {
        const out = el("output"); out.textContent = fmt(p.value, p.unit);
        lab.appendChild(el("span", null, p.label));
        const inp = el("input"); inp.type = "range"; inp.min = p.min; inp.max = p.max;
        inp.step = p.step != null ? p.step : 1; inp.value = p.value;
        inp.addEventListener("input", () => { vals[p.key] = parseFloat(inp.value); out.textContent = fmt(vals[p.key], p.unit); render(); });
        lab.appendChild(inp); lab.appendChild(out);
      }
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
    const menu = $("#menu-btn"), backdrop = $("#sb-backdrop");
    if (menu) menu.addEventListener("click", () => document.body.classList.toggle("sb-open"));
    if (backdrop) backdrop.addEventListener("click", () => document.body.classList.remove("sb-open"));
    $$("#sidebar a").forEach(a => a.addEventListener("click", () => document.body.classList.remove("sb-open")));

    const bar = $("#read-progress"), toTop = $("#to-top"), chip = $("#done-chip");
    if (chip && !chip.classList.contains("done")) chip.textContent = t.inprog;
    const onScroll = () => {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      const pct = max > 0 ? (h.scrollTop / max) * 100 : 0;
      if (bar) bar.style.width = pct + "%";
      if (toTop) toTop.classList.toggle("show", h.scrollTop > 600);
      if (pct > 92 && course && track && chapter) {
        Progress.set(course, track, chapter, { visited: true, done: true });
        if (chip) { chip.classList.add("done"); chip.textContent = t.done; }
      }
    };
    if (toTop) toTop.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
    window.addEventListener("scroll", onScroll, { passive: true });

    if (course && track && chapter) {
      const st = Progress.get(course, track, chapter);
      Progress.set(course, track, chapter, { visited: true });
      if (chip && st.done) { chip.classList.add("done"); chip.textContent = t.done; }
    }
    onScroll();

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

  /* =============================================================== HUB =====
     The hub markup uses .track-card[data-course] with a .tc-progress > i bar and
     a .tc-progress-label. Card chapter links are (re)written for the active lang. */
  function renderHub(activeLang) {
    const data = Progress.load();
    Object.keys(COURSES).forEach(cid => {
      const C = COURSES[cid];
      let total = 0, done = 0, firstHref = null;
      Object.keys(C.tracks).forEach(tid => {
        C.tracks[tid].chapters.forEach((c, idx) => {
          total++;
          if ((data[cid + "/" + tid + "/" + c.id] || {}).done) done++;
          if (firstHref === null && idx === 0) firstHref = chapterURL(cid, tid, c.file, activeLang);
        });
      });
      const pct = total ? Math.round((done / total) * 100) : 0;
      const card = document.querySelector(`.track-card[data-course="${cid}"]`);
      if (!card) return;
      if (firstHref) card.href = firstHref;
      const bar = card.querySelector(".tc-progress > i");
      const lab = card.querySelector(".tc-progress-label");
      if (bar) bar.style.width = pct + "%";
      if (lab) lab.textContent = activeLang === "zh"
        ? `${done} / ${total} 章完成` : `${done} / ${total} chapters complete`;
    });
  }
  function initHub() {
    if (!onHub) return;
    applyI18nVisibility(lang);
    renderHub(lang);
  }

  /* =============================================================== BOOT ==== */
  function boot() {
    document.documentElement.lang = lang === "zh" ? "zh-Hant" : "en";
    injectLangToggle();
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
  window.EM = { sim: makeSim, COURSES, Progress, SITE_ROOT };
})();
