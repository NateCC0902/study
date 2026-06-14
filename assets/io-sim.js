/* ============================================================================
   Linux I/O for Robotics — interactive canvas demos on the shared EM.sim()
   framework. Each EM.ioDemo.<name>("canvas-id") builds sliders/selects + a live
   drawing. All math runs in the browser; values are deterministic (no data
   files, no animation loop — EM.sim redraws on each control change).
   Models marked "illustrative" use simplified but directionally-correct physics.
   ========================================================================== */
(function () {
  "use strict";
  if (!window.EM) { console.warn("io-sim: EM not ready"); return; }

  /* ---- helpers ------------------------------------------------------------- */
  const isZh = () => document.documentElement.lang === "zh-Hant";
  const L = (en, zh) => (isZh() ? zh : en);
  function cssVar(n, d) { const v = getComputedStyle(document.body).getPropertyValue(n); return (v && v.trim()) || d; }
  function line(ctx, x1, y1, x2, y2, color, w, dash) {
    ctx.strokeStyle = color; ctx.lineWidth = w; if (dash) ctx.setLineDash(dash); else ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.setLineDash([]);
  }
  const log10 = x => Math.log(x) / Math.LN10;
  const fmtBig = n => n >= 1e9 ? (n / 1e9).toFixed(n < 1e10 ? 1 : 0) + "G"
                    : n >= 1e6 ? (n / 1e6).toFixed(n < 1e7 ? 1 : 0) + "M"
                    : n >= 1e3 ? (n / 1e3).toFixed(n < 1e4 ? 1 : 0) + "k" : n.toFixed(0);
  const fmtRate = m => m >= 1000 ? (m / 1000).toFixed(m < 10000 ? 2 : 1) + " Gb/s" : m.toFixed(m < 1 ? 3 : 0) + " Mb/s";

  EM.ioDemo = {};

  /* ============================================================ 05 · buses ===
     Log-scale bus bandwidths vs a draggable "required throughput" line.       */
  EM.ioDemo.busBandwidth = function (id) {
    // capacity in Mb/s (typical maxima — illustrative)
    const BUS = [
      ["I²C 400k", 0.4], ["UART 1M", 1], ["CAN 1M", 1], ["CAN-FD 5M", 5],
      ["SPI 50M", 50], ["SDIO UHS", 200], ["USB 2.0", 480], ["GbE", 1000],
      ["USB 3.2", 5000], ["PCIe3 ×1", 7880], ["10GbE", 10000]
    ];
    const LO = 0.1, HI = 20000, lLO = log10(LO), lHI = log10(HI);
    return EM.sim(id, {
      height: 340,
      params: [{ key: "demand", label: L("required throughput", "需求吞吐量"), min: -1, max: 4, step: 0.05, value: 1.7, unit: "log Mb/s" }],
      draw: (ctx, v, g) => {
        const dpr = g.dpr, gut = 92 * dpr, x0 = gut, x1 = g.w - 16 * dpr, top = 14 * dpr, bot = g.h - 26 * dpr;
        const demand = Math.pow(10, v.demand);
        const X = cap => x0 + (log10(Math.max(LO, Math.min(HI, cap))) - lLO) / (lHI - lLO) * (x1 - x0);
        const acc = cssVar("--accent-2", "#ff9f5a"), dim = "#6e7681";
        const rowH = (bot - top) / BUS.length;
        ctx.font = `${11 * dpr}px ui-monospace, monospace`; ctx.textBaseline = "middle";
        // log gridlines at decades
        ctx.strokeStyle = "#1d242f"; ctx.fillStyle = dim; ctx.textAlign = "center";
        for (let e = 0; e <= 4; e++) { const x = X(Math.pow(10, e)); line(ctx, x, top, x, bot, "#1d242f", 1 * dpr); ctx.fillText(fmtBig(Math.pow(10, e)), x, bot + 12 * dpr); }
        let ok = 0;
        BUS.forEach((b, i) => {
          const y = top + i * rowH + rowH / 2, suff = b[1] >= demand; if (suff) ok++;
          ctx.fillStyle = suff ? acc : "#3a2a24";
          ctx.fillRect(x0, y - rowH * 0.3, X(b[1]) - x0, rowH * 0.6);
          ctx.fillStyle = suff ? cssVar("--fg-strong", "#e6edf3") : dim; ctx.textAlign = "right";
          ctx.fillText(b[0], gut - 8 * dpr, y);
        });
        line(ctx, X(demand), top - 4 * dpr, X(demand), bot, "#f04050", 2 * dpr, [5 * dpr, 4 * dpr]);
        return L(`need <b>${fmtRate(demand)}</b> → <b>${ok}/${BUS.length}</b> buses keep up (orange). Match the bus to the load.`,
                 `需求 <b>${fmtRate(demand)}</b> → <b>${ok}/${BUS.length}</b> 條匯流排足夠（橘色）。讓匯流排匹配負載。`);
      }
    });
  };

  /* ============================================================ 06 · UART ====
     One UART frame waveform; bit-time + byte-rate readout.                     */
  EM.ioDemo.uartFrame = function (id) {
    return EM.sim(id, {
      height: 260,
      params: [
        { key: "baud", type: "select", label: L("baud", "鮑率"), value: "115200",
          options: [{ label: "9600", value: "9600" }, { label: "115200", value: "115200" }, { label: "1000000", value: "1000000" }] },
        { key: "byte", label: L("data byte", "資料位元組"), min: 0, max: 255, step: 1, value: 65 },
        { key: "parity", type: "select", label: L("parity", "同位"), value: "none",
          options: [{ label: L("none", "無"), value: "none" }, { label: "even", value: "even" }, { label: "odd", value: "odd" }] },
        { key: "stop", type: "select", label: L("stop bits", "停止位元"), value: "1",
          options: [{ label: "1", value: "1" }, { label: "2", value: "2" }] }
      ],
      draw: (ctx, v, g) => {
        const dpr = g.dpr, baud = +v.baud, byte = v.byte | 0, stop = +v.stop;
        // build the bit stream: idle(1) start(0) d0..d7(LSB first) [parity] stop(1..2) idle(1)
        const data = []; for (let k = 0; k < 8; k++) data.push((byte >> k) & 1);
        const ones = data.reduce((a, b) => a + b, 0);
        let par = null;
        if (v.parity === "even") par = ones % 2; else if (v.parity === "odd") par = 1 - (ones % 2);
        const bits = [{ b: 1, t: "idle" }, { b: 0, t: "start" }];
        data.forEach((d, k) => bits.push({ b: d, t: "d" + k }));
        if (par !== null) bits.push({ b: par, t: "P" });
        for (let s = 0; s < stop; s++) bits.push({ b: 1, t: "stop" });
        bits.push({ b: 1, t: "idle" });
        const n = bits.length, x0 = 16 * dpr, x1 = g.w - 12 * dpr, yHi = 40 * dpr, yLo = g.h - 54 * dpr;
        const bw = (x1 - x0) / n, Y = b => (b ? yHi : yLo);
        const acc = cssVar("--accent-2", "#ff9f5a");
        // levels guide
        ctx.font = `${10 * dpr}px ui-monospace, monospace`; ctx.fillStyle = "#6e7681"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText("1", x0 - 2 * dpr, yHi - 12 * dpr); ctx.fillText("0", x0 - 2 * dpr, yLo + 12 * dpr);
        // waveform (step)
        ctx.strokeStyle = acc; ctx.lineWidth = 2.4 * dpr; ctx.beginPath();
        let px = x0, py = Y(bits[0].b); ctx.moveTo(px, py);
        for (let i = 0; i < n; i++) { const yb = Y(bits[i].b); if (yb !== py) ctx.lineTo(px, yb); px += bw; ctx.lineTo(px, yb); py = yb; }
        ctx.stroke();
        // bit cell separators + labels
        ctx.textAlign = "center"; ctx.textBaseline = "top";
        for (let i = 0; i < n; i++) {
          const cx = x0 + i * bw; line(ctx, cx, yHi - 6 * dpr, cx, yLo + 6 * dpr, "#1d242f", 1 * dpr);
          const t = bits[i].t; if (t === "idle") continue;
          ctx.fillStyle = t === "start" ? "#f0a020" : t === "stop" ? "#3fb950" : t === "P" ? "#bc8cff" : "#8b949e";
          ctx.fillText(t, cx + bw / 2, yLo + 10 * dpr);
        }
        const frameBits = 1 + 8 + (par !== null ? 1 : 0) + stop;       // start + data + parity + stop
        const bitT = 1e6 / baud, byteRate = baud / frameBits;
        return L(`bit time <b>${bitT.toFixed(2)} µs</b> · frame <b>${frameBits} bits</b> = <b>${(frameBits * bitT).toFixed(1)} µs</b> · max <b>${fmtBig(byteRate)} B/s</b> · efficiency <b>${(8 / frameBits * 100).toFixed(0)}%</b>`,
                 `位元時間 <b>${bitT.toFixed(2)} µs</b> · 一幀 <b>${frameBits} 位元</b> = <b>${(frameBits * bitT).toFixed(1)} µs</b> · 最高 <b>${fmtBig(byteRate)} B/s</b> · 效率 <b>${(8 / frameBits * 100).toFixed(0)}%</b>`);
      }
    });
  };

  /* ==================================================== 07 · data movement ===
     CPU operations for PIO vs IRQ(+FIFO coalescing) vs DMA over one transfer.  */
  EM.ioDemo.dataMovement = function (id) {
    return EM.sim(id, {
      height: 250,
      params: [
        { key: "kb", label: L("transfer size", "傳輸大小"), min: 1, max: 1024, step: 1, value: 64, unit: "KiB" },
        { key: "word", type: "select", label: L("word size", "字組大小"), value: "4",
          options: [{ label: "1 B", value: "1" }, { label: "2 B", value: "2" }, { label: "4 B", value: "4" }] },
        { key: "fifo", label: L("IRQ FIFO depth", "IRQ FIFO 深度"), min: 1, max: 64, step: 1, value: 16 }
      ],
      draw: (ctx, v, g) => {
        const dpr = g.dpr, bytes = v.kb * 1024, word = +v.word;
        const pio = Math.ceil(bytes / word);                 // CPU copies every word
        const irq = Math.ceil(bytes / (word * v.fifo));      // one interrupt per FIFO-full
        const dma = 1;                                        // one completion interrupt
        const rows = [["PIO (CPU copies)", pio, "#f04050"], ["IRQ (interrupts)", irq, "#f0a020"], ["DMA (interrupts)", dma, "#3fb950"]];
        const gut = 132 * dpr, x0 = gut, x1 = g.w - 60 * dpr, top = 18 * dpr, bot = g.h - 18 * dpr;
        const HI = Math.max(pio, 10), lHI = log10(HI);
        const X = c => x0 + (c <= 1 ? 0 : log10(c) / lHI) * (x1 - x0);
        const rowH = (bot - top) / rows.length;
        ctx.font = `${11 * dpr}px ui-monospace, monospace`; ctx.textBaseline = "middle";
        rows.forEach((r, i) => {
          const y = top + i * rowH + rowH / 2;
          ctx.fillStyle = r[2]; ctx.fillRect(x0, y - rowH * 0.26, Math.max(2 * dpr, X(r[1]) - x0), rowH * 0.52);
          ctx.fillStyle = cssVar("--fg-strong", "#e6edf3"); ctx.textAlign = "right"; ctx.fillText(r[0], gut - 8 * dpr, y);
          ctx.fillStyle = cssVar("--accent-2", "#ff9f5a"); ctx.textAlign = "left"; ctx.fillText(fmtBig(r[1]), X(r[1]) + 6 * dpr, y);
        });
        return L(`${v.kb} KiB transfer: PIO ties up the CPU for <b>${fmtBig(pio)}</b> copies; IRQ takes <b>${fmtBig(irq)}</b> interrupts; DMA just <b>1</b> — the CPU is free for the control loop.`,
                 `${v.kb} KiB 傳輸：PIO 佔用 CPU <b>${fmtBig(pio)}</b> 次複製；IRQ 需 <b>${fmtBig(irq)}</b> 次中斷；DMA 只要 <b>1</b> — CPU 可空出來跑控制迴圈。`);
      }
    });
  };

  /* ====================================================== 08 · multiplexing ==
     thread-per-fd vs epoll as the number of connections scales.               */
  EM.ioDemo.ioMultiplex = function (id) {
    const STACK_MB = 8;        // default pthread stack (virtual) per thread
    const EPOLL_KB = 0.3;      // ~ per-fd kernel epoll item
    return EM.sim(id, {
      height: 300,
      params: [
        { key: "logn", label: L("connections (fds)", "連線數 (fd)"), min: 0, max: 4.7, step: 0.05, value: 3, unit: "log N" },
        { key: "active", label: L("% active at once", "同時活躍 %"), min: 1, max: 100, step: 1, value: 5, unit: "%" }
      ],
      draw: (ctx, v, g) => {
        const dpr = g.dpr, box = { x: 44 * dpr, y: 14 * dpr, w: g.w - 60 * dpr, h: g.h - 44 * dpr };
        ctx.strokeStyle = "#262d3a"; ctx.lineWidth = 1 * dpr; ctx.strokeRect(box.x, box.y, box.w, box.h);
        const Nmax = Math.pow(10, 4.7), lMax = log10(Nmax);
        const X = n => box.x + log10(Math.max(1, n)) / lMax * box.w;
        const Y = c => box.y + box.h - Math.max(0, Math.min(1, c)) * box.h;   // c in 0..1 (normalized work)
        // curves: thread wakeups ∝ N (scheduler pressure); epoll wakeups ∝ active
        const norm = n => Math.min(1, n / Nmax);
        ctx.strokeStyle = "#f04050"; ctx.lineWidth = 2.4 * dpr; ctx.beginPath();
        for (let l = 0; l <= lMax; l += 0.05) { const n = Math.pow(10, l); const x = X(n), y = Y(norm(n)); l === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); } ctx.stroke();
        ctx.strokeStyle = cssVar("--accent-2", "#ff9f5a"); ctx.lineWidth = 2.4 * dpr; ctx.beginPath();
        for (let l = 0; l <= lMax; l += 0.05) { const n = Math.pow(10, l); const x = X(n), y = Y(norm(n * v.active / 100)); l === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); } ctx.stroke();
        const N = Math.round(Math.pow(10, v.logn)), act = Math.max(1, Math.round(N * v.active / 100));
        line(ctx, X(N), box.y, X(N), box.y + box.h, "#6e7681", 1.5 * dpr, [4 * dpr, 4 * dpr]);
        ctx.font = `${10 * dpr}px ui-monospace, monospace`; ctx.fillStyle = "#6e7681"; ctx.textAlign = "center"; ctx.textBaseline = "top";
        for (let e = 0; e <= 4; e++) ctx.fillText(fmtBig(Math.pow(10, e)), X(Math.pow(10, e)), box.y + box.h + 5 * dpr);
        const threadGB = N * STACK_MB / 1024;
        return L(`N=<b>${fmtBig(N)}</b> fds — thread-per-fd (red): ~<b>${threadGB < 1 ? (threadGB * 1024).toFixed(0) + " MB" : threadGB.toFixed(1) + " GB"}</b> stack VM, O(N) scheduler pressure. epoll (orange): 1 thread, ~<b>${fmtBig(N * EPOLL_KB * 1024)} B</b>, wakes only O(active=<b>${fmtBig(act)}</b>).`,
                 `N=<b>${fmtBig(N)}</b> 個 fd — 每連線一執行緒（紅）：約 <b>${threadGB < 1 ? (threadGB * 1024).toFixed(0) + " MB" : threadGB.toFixed(1) + " GB"}</b> 堆疊虛擬記憶體、O(N) 排程壓力。epoll（橘）：單執行緒、約 <b>${fmtBig(N * EPOLL_KB * 1024)} B</b>，只喚醒 O(活躍=<b>${fmtBig(act)}</b>)。`);
      }
    });
  };

  /* ========================================================= 09 · net link ===
     Latency vs offered load: Ethernet (full-duplex) vs WiFi (shared, jitter).
     Illustrative M/M/1-style queueing model.                                  */
  EM.ioDemo.netLink = function (id) {
    return EM.sim(id, {
      height: 300,
      params: [
        { key: "load", label: L("offered load", "提供負載"), min: 0, max: 950, step: 10, value: 300, unit: "Mb/s" },
        { key: "sta", label: L("WiFi stations", "WiFi 台數"), min: 1, max: 16, step: 1, value: 4 }
      ],
      draw: (ctx, v, g) => {
        const dpr = g.dpr, box = { x: 46 * dpr, y: 14 * dpr, w: g.w - 62 * dpr, h: g.h - 46 * dpr };
        ctx.strokeStyle = "#262d3a"; ctx.lineWidth = 1 * dpr; ctx.strokeRect(box.x, box.y, box.w, box.h);
        const ETH_CAP = 1000, WIFI_CAP = 600 / (1 + 0.18 * (v.sta - 1));   // contention cuts goodput
        const Lmax = 1000, latMax = 50;                                     // ms (clipped)
        const X = l => box.x + l / Lmax * box.w;
        const Y = ms => box.y + box.h - Math.min(latMax, ms) / latMax * box.h;
        const ethLat = l => 0.1 + 0.1 / Math.max(0.02, 1 - l / ETH_CAP);    // low base, low jitter
        const wifiLat = l => 2.0 + 1.5 / Math.max(0.02, 1 - l / WIFI_CAP);  // higher base + steeper
        const curve = (fn, col) => { ctx.strokeStyle = col; ctx.lineWidth = 2.4 * dpr; ctx.beginPath(); for (let l = 0; l <= Lmax; l += 8) { const x = X(l), y = Y(fn(l)); l === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); } ctx.stroke(); };
        ctx.font = `${10 * dpr}px ui-monospace, monospace`; ctx.fillStyle = "#6e7681"; ctx.textAlign = "center"; ctx.textBaseline = "top";
        for (let l = 0; l <= 1000; l += 250) ctx.fillText(l + "", X(l), box.y + box.h + 5 * dpr);
        curve(ethLat, "#3fb950"); curve(wifiLat, cssVar("--accent-2", "#ff9f5a"));
        line(ctx, X(v.load), box.y, X(v.load), box.y + box.h, "#6e7681", 1.5 * dpr, [4 * dpr, 4 * dpr]);
        const e = ethLat(v.load), w = wifiLat(v.load);
        const wsat = v.load >= WIFI_CAP ? L(" · WiFi <b>saturated</b>", " · WiFi <b>飽和</b>") : "";
        return L(`at <b>${v.load} Mb/s</b>, ${v.sta} WiFi station(s): Ethernet ≈ <b>${e.toFixed(2)} ms</b> (low jitter); WiFi ≈ <b>${w.toFixed(2)} ms</b>, goodput cap ~<b>${WIFI_CAP.toFixed(0)} Mb/s</b>${wsat}. Keep heavy data off WiFi.`,
                 `在 <b>${v.load} Mb/s</b>、${v.sta} 台 WiFi：乙太網 ≈ <b>${e.toFixed(2)} ms</b>（抖動小）；WiFi ≈ <b>${w.toFixed(2)} ms</b>，有效上限約 <b>${WIFI_CAP.toFixed(0)} Mb/s</b>${wsat}。重負載資料別走 WiFi。`);
      }
    });
  };

  /* ============================================================ 10 · CAN =====
     CAN bus-load gauge vs bitrate / frame-rate / payload (classic 11-bit frame).*/
  EM.ioDemo.canLoad = function (id) {
    return EM.sim(id, {
      height: 250,
      params: [
        { key: "bitrate", type: "select", label: L("bitrate", "位元率"), value: "500000",
          options: [{ label: "125k", value: "125000" }, { label: "250k (NMEA2000)", value: "250000" }, { label: "500k", value: "500000" }, { label: "1M", value: "1000000" }] },
        { key: "fps", label: L("frame rate", "幀率"), min: 0, max: 8000, step: 50, value: 1000, unit: "fps" },
        { key: "payload", label: L("payload", "資料"), min: 0, max: 8, step: 1, value: 8, unit: "B" }
      ],
      draw: (ctx, v, g) => {
        const dpr = g.dpr, bitrate = +v.bitrate, p = v.payload | 0;
        const nominal = 47 + 8 * p;                        // SOF+arb+ctrl+CRC+ACK+EOF+IFS, 11-bit ID
        const stuff = Math.floor((34 + 8 * p - 1) / 4);    // worst-case bit-stuffing
        const worst = nominal + stuff;
        const loadN = v.fps * nominal / bitrate * 100, loadW = v.fps * worst / bitrate * 100;
        // gauge bar 0..100%
        const x0 = 30 * dpr, x1 = g.w - 30 * dpr, y = 64 * dpr, h = 30 * dpr, W = x1 - x0;
        const zone = (a, b, col) => { ctx.fillStyle = col; ctx.fillRect(x0 + W * a / 100, y, W * (b - a) / 100, h); };
        zone(0, 50, "#13351f"); zone(50, 80, "#3a2f12"); zone(80, 100, "#3a1a1a");
        const col = loadN < 50 ? "#3fb950" : loadN < 80 ? "#f0a020" : "#f04050";
        ctx.fillStyle = col; ctx.fillRect(x0, y, W * Math.min(100, loadN) / 100, h);
        if (loadW > loadN) { line(ctx, x0 + W * Math.min(100, loadW) / 100, y - 6 * dpr, x0 + W * Math.min(100, loadW) / 100, y + h + 6 * dpr, "#e6edf3", 2 * dpr, [4 * dpr, 3 * dpr]); }
        ctx.strokeStyle = "#262d3a"; ctx.lineWidth = 1 * dpr; ctx.strokeRect(x0, y, W, h);
        ctx.font = `${11 * dpr}px ui-monospace, monospace`; ctx.fillStyle = "#6e7681"; ctx.textBaseline = "top"; ctx.textAlign = "left";
        ctx.fillText("0%", x0, y + h + 8 * dpr); ctx.textAlign = "center"; ctx.fillText("50%", x0 + W * .5, y + h + 8 * dpr); ctx.fillText("80%", x0 + W * .8, y + h + 8 * dpr);
        ctx.textAlign = "right"; ctx.fillText("100%", x1, y + h + 8 * dpr);
        const maxFps = Math.floor(bitrate / worst);
        return L(`${(bitrate / 1000)}k bit/s, ${p} B payload → <b>${nominal}–${worst} bits/frame</b>. Load ≈ <b>${loadN.toFixed(0)}%</b> (worst-case ${loadW.toFixed(0)}%, dashed). Keep < 50–70%; max ≈ <b>${fmtBig(maxFps)} fps</b>.`,
                 `${(bitrate / 1000)}k bit/s、${p} B 資料 → 每幀 <b>${nominal}–${worst} 位元</b>。負載 ≈ <b>${loadN.toFixed(0)}%</b>（最壞 ${loadW.toFixed(0)}%，虛線）。保持 < 50–70%；上限約 <b>${fmtBig(maxFps)} fps</b>。`);
      }
    });
  };
})();
