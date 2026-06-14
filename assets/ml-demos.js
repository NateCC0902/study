/* ============================================================================
   ML/DL interactive demos — canvas widgets on the shared EM.sim() framework.
   Each EM.mlDemo.<name>("canvas-id") builds sliders/toggles + a live drawing.
   All math runs in the browser (no data files); randomness is seeded so a given
   control state always draws the same picture.
   ========================================================================== */
(function () {
  "use strict";
  if (!window.EM) { console.warn("ml-demos: EM not ready"); return; }

  // ---- helpers -------------------------------------------------------------
  function rng(seed) { let s = seed >>> 0 || 1; return () => { s = (1103515245 * s + 12345) & 0x7fffffff; return s / 0x7fffffff; }; }
  function gaussFrom(r) { const u = Math.max(r(), 1e-9), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  const PAL = ["#a371f7", "#33b1ff", "#3fb950", "#f0a020", "#ff7b9c", "#2dd4bf"];
  function cssVar(n, d) { const v = getComputedStyle(document.body).getPropertyValue(n); return (v && v.trim()) || d; }
  const isZh = () => document.documentElement.lang === "zh-Hant";
  const L = (en, zh) => (isZh() ? zh : en);

  function frame(ctx, box, dpr) { ctx.strokeStyle = "#262d3a"; ctx.lineWidth = 1 * dpr; ctx.strokeRect(box.x, box.y, box.w, box.h); }
  function map(box, xr, yr) {
    return {
      X: v => box.x + (v - xr[0]) / (xr[1] - xr[0]) * box.w,
      Y: v => box.y + box.h - (v - yr[0]) / (yr[1] - yr[0]) * box.h
    };
  }
  function dot(ctx, x, y, r, color) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); }
  function line(ctx, x1, y1, x2, y2, color, w) { ctx.strokeStyle = color; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }

  // small dense linear solver (Gaussian elimination with partial pivot)
  function solve(A, b) {
    const n = b.length;
    for (let i = 0; i < n; i++) {
      let p = i; for (let k = i + 1; k < n; k++) if (Math.abs(A[k][i]) > Math.abs(A[p][i])) p = k;
      [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
      const piv = A[i][i] || 1e-12;
      for (let k = i + 1; k < n; k++) { const f = A[k][i] / piv; for (let j = i; j < n; j++) A[k][j] -= f * A[i][j]; b[k] -= f * b[i]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let j = i + 1; j < n; j++) s -= A[i][j] * x[j]; x[i] = s / (A[i][i] || 1e-12); }
    return x;
  }

  EM.mlDemo = {};

  /* ---- 02 · linear regression: drag the line, watch MSE ------------------- */
  EM.mlDemo.linearFit = function (id) {
    const r = rng(7), N = 30, X = [], Y = [];
    for (let i = 0; i < N; i++) { const x = -4 + 8 * r(); X.push(x); Y.push(0.8 * x + 1 + gaussFrom(r) * 1.1); }
    // OLS reference
    const mx = X.reduce((a, b) => a + b) / N, my = Y.reduce((a, b) => a + b) / N;
    let sxx = 0, sxy = 0; for (let i = 0; i < N; i++) { sxx += (X[i] - mx) ** 2; sxy += (X[i] - mx) * (Y[i] - my); }
    const wOls = sxy / sxx, bOls = my - wOls * mx;
    return EM.sim(id, {
      height: 300,
      params: [
        { key: "w", label: L("slope w", "斜率 w"), min: -2, max: 3, step: 0.05, value: 0.2 },
        { key: "b", label: L("intercept b", "截距 b"), min: -3, max: 4, step: 0.1, value: 0 }
      ],
      draw: (ctx, v, g) => {
        const dpr = g.dpr, box = { x: 40 * dpr, y: 12 * dpr, w: g.w - 56 * dpr, h: g.h - 40 * dpr };
        frame(ctx, box, dpr); const m = map(box, [-4.5, 4.5], [-5, 6]);
        let mse = 0;
        X.forEach((x, i) => {
          const yh = v.w * x + v.b; mse += (yh - Y[i]) ** 2;
          line(ctx, m.X(x), m.Y(Y[i]), m.X(x), m.Y(yh), "#8b949e55", 1 * dpr);
          dot(ctx, m.X(x), m.Y(Y[i]), 3 * dpr, PAL[1]);
        });
        mse /= N;
        line(ctx, m.X(-4.5), m.Y(wOls * -4.5 + bOls), m.X(4.5), m.Y(wOls * 4.5 + bOls), "#3fb95066", 1.5 * dpr);
        line(ctx, m.X(-4.5), m.Y(v.w * -4.5 + v.b), m.X(4.5), m.Y(v.w * 4.5 + v.b), cssVar("--accent-2", "#c8a6ff"), 2.4 * dpr);
        return L(`MSE <b>${mse.toFixed(2)}</b> · best-fit line w=<b>${wOls.toFixed(2)}</b>, b=<b>${bOls.toFixed(2)}</b> (green)`,
                 `MSE <b>${mse.toFixed(2)}</b> · 最佳擬合 w=<b>${wOls.toFixed(2)}</b>、b=<b>${bOls.toFixed(2)}</b>（綠線）`);
      }
    });
  };

  /* ---- 03 · gradient descent on a convex bowl ----------------------------- */
  EM.mlDemo.gradientDescent = function (id) {
    const J = w => (w - 3) ** 2, grad = w => 2 * (w - 3), w0 = -5;
    return EM.sim(id, {
      height: 300,
      params: [
        { key: "lr", label: L("learning rate α", "學習率 α"), min: 0.02, max: 1.1, step: 0.02, value: 0.1 },
        { key: "steps", label: L("steps", "步數"), min: 0, max: 40, step: 1, value: 8 }
      ],
      draw: (ctx, v, g) => {
        const dpr = g.dpr, box = { x: 40 * dpr, y: 12 * dpr, w: g.w - 56 * dpr, h: g.h - 40 * dpr };
        frame(ctx, box, dpr); const m = map(box, [-7, 9], [-2, 70]);
        ctx.strokeStyle = "#8b949e"; ctx.lineWidth = 1.6 * dpr; ctx.beginPath();
        for (let w = -7; w <= 9; w += 0.1) { const x = m.X(w), y = m.Y(J(w)); w === -7 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
        ctx.stroke();
        let w = w0, diverged = false;
        const acc = cssVar("--accent-2", "#c8a6ff");
        for (let i = 0; i <= v.steps; i++) {
          const Jw = J(w);
          if (Math.abs(w) > 60 || !isFinite(Jw)) { diverged = true; break; }
          dot(ctx, m.X(w), m.Y(Math.min(Jw, 70)), 4 * dpr, i === v.steps ? "#f0a020" : acc);
          w = w - v.lr * grad(w);
        }
        return diverged
          ? L(`α=${v.lr.toFixed(2)} is too big → <b>diverging</b> (stable needs α &lt; 1).`,
              `α=${v.lr.toFixed(2)} 太大 → <b>發散</b>（穩定需 α &lt; 1）。`)
          : L(`after ${v.steps} steps: w=<b>${w.toFixed(3)}</b>, J=<b>${J(w).toFixed(3)}</b> (min at w=3)`,
              `${v.steps} 步後：w=<b>${w.toFixed(3)}</b>、J=<b>${J(w).toFixed(3)}</b>（最小在 w=3）`);
      }
    });
  };

  /* ---- 04 · logistic regression decision boundary ------------------------- */
  EM.mlDemo.logistic = function (id) {
    const r = rng(11), P = [];
    for (let i = 0; i < 40; i++) { const c = i < 20 ? 0 : 1, cx = c ? 1.6 : -1.6, cy = c ? 1.1 : -1.1; P.push([cx + gaussFrom(r) * 1.0, cy + gaussFrom(r) * 1.0, c]); }
    const sig = z => 1 / (1 + Math.exp(-z));
    return EM.sim(id, {
      height: 320,
      params: [
        { key: "w1", label: "w₁", min: -3, max: 3, step: 0.1, value: 1 },
        { key: "w2", label: "w₂", min: -3, max: 3, step: 0.1, value: 1 },
        { key: "b", label: "b", min: -4, max: 4, step: 0.1, value: 0 }
      ],
      draw: (ctx, v, g) => {
        const dpr = g.dpr, box = { x: 30 * dpr, y: 12 * dpr, w: g.w - 44 * dpr, h: g.h - 40 * dpr };
        const m = map(box, [-4.5, 4.5], [-4, 4]);
        const STEP = 10 * dpr;
        for (let px = box.x; px < box.x + box.w; px += STEP) for (let py = box.y; py < box.y + box.h; py += STEP) {
          const x = (px - box.x) / box.w * 9 - 4.5, y = 4 - (py - box.y) / box.h * 8;
          const p = sig(v.w1 * x + v.w2 * y + v.b);
          ctx.fillStyle = `rgba(${Math.round(51 + p * 100)},${Math.round(177 - p * 80)},${Math.round(255 - p * 100)},0.13)`;
          ctx.fillRect(px, py, STEP, STEP);
        }
        frame(ctx, box, dpr);
        let correct = 0, ce = 0;
        P.forEach(([x, y, c]) => { const p = sig(v.w1 * x + v.w2 * y + v.b); if ((p > 0.5 ? 1 : 0) === c) correct++; ce += -(c * Math.log(p + 1e-9) + (1 - c) * Math.log(1 - p + 1e-9)); dot(ctx, m.X(x), m.Y(y), 4 * dpr, PAL[c ? 0 : 1]); });
        if (Math.abs(v.w2) > 1e-3) { const yA = (-v.b - v.w1 * -4.5) / v.w2, yB = (-v.b - v.w1 * 4.5) / v.w2; line(ctx, m.X(-4.5), m.Y(yA), m.X(4.5), m.Y(yB), cssVar("--accent-2", "#c8a6ff"), 2.4 * dpr); }
        return L(`accuracy <b>${(100 * correct / P.length).toFixed(0)}%</b> · cross-entropy <b>${(ce / P.length).toFixed(3)}</b>`,
                 `準確率 <b>${(100 * correct / P.length).toFixed(0)}%</b> · 交叉熵 <b>${(ce / P.length).toFixed(3)}</b>`);
      }
    });
  };

  /* ---- 05 · bias–variance: polynomial degree ------------------------------ */
  EM.mlDemo.biasVariance = function (id) {
    const r = rng(5), f = x => Math.sin(2.4 * x + 0.4); // x in [-1,1]
    const Xtr = [], Ytr = [], Xva = [], Yva = [];
    for (let i = 0; i < 14; i++) { const x = -1 + 2 * r(); Xtr.push(x); Ytr.push(f(x) + gaussFrom(r) * 0.18); }
    for (let i = 0; i < 40; i++) { const x = -1 + 2 * r(); Xva.push(x); Yva.push(f(x) + gaussFrom(r) * 0.18); }
    function fit(d) {
      const A = [], bb = [];
      for (let i = 0; i <= d; i++) { A.push(new Array(d + 1).fill(0)); bb.push(0); }
      Xtr.forEach((x, n) => { const pw = []; for (let j = 0; j <= d; j++) pw.push(Math.pow(x, j)); for (let i = 0; i <= d; i++) { for (let j = 0; j <= d; j++) A[i][j] += pw[i] * pw[j]; bb[i] += pw[i] * Ytr[n]; } });
      for (let i = 0; i <= d; i++) A[i][i] += 1e-6; // tiny ridge for stability
      return solve(A, bb);
    }
    const ev = (w, x) => w.reduce((s, c, j) => s + c * Math.pow(x, j), 0);
    const rmse = (w, Xs, Ys) => Math.sqrt(Xs.reduce((s, x, i) => s + (ev(w, x) - Ys[i]) ** 2, 0) / Xs.length);
    return EM.sim(id, {
      height: 300,
      params: [{ key: "d", label: L("polynomial degree", "多項式次數"), min: 1, max: 12, step: 1, value: 3 }],
      draw: (ctx, v, g) => {
        const dpr = g.dpr, box = { x: 36 * dpr, y: 12 * dpr, w: g.w - 50 * dpr, h: g.h - 40 * dpr };
        frame(ctx, box, dpr); const m = map(box, [-1.1, 1.1], [-1.8, 1.8]);
        const w = fit(v.d);
        ctx.strokeStyle = "#3fb95066"; ctx.lineWidth = 1.4 * dpr; ctx.beginPath();
        for (let x = -1.1; x <= 1.1; x += 0.02) { const px = m.X(x), py = m.Y(f(x)); x === -1.1 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); } ctx.stroke();
        ctx.strokeStyle = cssVar("--accent-2", "#c8a6ff"); ctx.lineWidth = 2.4 * dpr; ctx.beginPath();
        for (let x = -1.1; x <= 1.1; x += 0.01) { const px = m.X(x), py = m.Y(Math.max(-1.8, Math.min(1.8, ev(w, x)))); x === -1.1 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); } ctx.stroke();
        Xtr.forEach((x, i) => dot(ctx, m.X(x), m.Y(Ytr[i]), 3.4 * dpr, PAL[1]));
        return L(`degree ${v.d}: train RMSE <b>${rmse(w, Xtr, Ytr).toFixed(3)}</b> · val RMSE <b>${rmse(w, Xva, Yva).toFixed(3)}</b> (green = truth)`,
                 `次數 ${v.d}：訓練 RMSE <b>${rmse(w, Xtr, Ytr).toFixed(3)}</b> · 驗證 RMSE <b>${rmse(w, Xva, Yva).toFixed(3)}</b>（綠線=真實函數）`);
      }
    });
  };

  /* ---- 08a · k-means clustering ------------------------------------------- */
  EM.mlDemo.kmeans = function (id) {
    const r = rng(3), pts = [];
    const cen = [[-1.6, -1.2], [1.7, -0.6], [0.2, 1.7]];
    cen.forEach(c => { for (let i = 0; i < 22; i++) pts.push([c[0] + gaussFrom(r) * 0.6, c[1] + gaussFrom(r) * 0.6]); });
    return EM.sim(id, {
      height: 320,
      params: [
        { key: "k", label: "k", min: 1, max: 6, step: 1, value: 3 },
        { key: "iters", label: L("iterations", "迭代次數"), min: 0, max: 12, step: 1, value: 0 }
      ],
      draw: (ctx, v, g) => {
        const dpr = g.dpr, box = { x: 24 * dpr, y: 12 * dpr, w: g.w - 38 * dpr, h: g.h - 30 * dpr };
        frame(ctx, box, dpr); const m = map(box, [-3.2, 3.4], [-3, 3.4]);
        const ir = rng(99); let C = [];
        for (let i = 0; i < v.k; i++) { const p = pts[Math.floor(ir() * pts.length)]; C.push([p[0], p[1]]); }
        let assign = new Array(pts.length).fill(0);
        for (let it = 0; it < v.iters; it++) {
          pts.forEach((p, i) => { let bd = 1e9, bj = 0; C.forEach((c, j) => { const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2; if (d < bd) { bd = d; bj = j; } }); assign[i] = bj; });
          const sum = C.map(() => [0, 0, 0]); pts.forEach((p, i) => { const a = assign[i]; sum[a][0] += p[0]; sum[a][1] += p[1]; sum[a][2]++; });
          C = C.map((c, j) => sum[j][2] ? [sum[j][0] / sum[j][2], sum[j][1] / sum[j][2]] : c);
        }
        if (v.iters > 0) pts.forEach((p, i) => { let bd = 1e9, bj = 0; C.forEach((c, j) => { const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2; if (d < bd) { bd = d; bj = j; } }); assign[i] = bj; });
        let inertia = 0;
        pts.forEach((p, i) => { const col = v.iters > 0 ? PAL[assign[i] % PAL.length] : "#8b949e"; dot(ctx, m.X(p[0]), m.Y(p[1]), 3.4 * dpr, col); if (v.iters > 0) inertia += (p[0] - C[assign[i]][0]) ** 2 + (p[1] - C[assign[i]][1]) ** 2; });
        C.forEach((c, j) => { ctx.fillStyle = PAL[j % PAL.length]; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * dpr; ctx.beginPath(); ctx.arc(m.X(c[0]), m.Y(c[1]), 7 * dpr, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); });
        return L(`k=${v.k}, ${v.iters} iteration(s) · inertia (within-cluster SSE) <b>${inertia.toFixed(1)}</b>`,
                 `k=${v.k}、${v.iters} 次迭代 · 群內平方和 inertia <b>${inertia.toFixed(1)}</b>`);
      }
    });
  };

  /* ---- 08b · PCA principal axes ------------------------------------------- */
  EM.mlDemo.pca = function (id) {
    return EM.sim(id, {
      height: 300,
      params: [{ key: "corr", label: L("correlation", "相關性"), min: 0, max: 0.95, step: 0.05, value: 0.8 }],
      draw: (ctx, v, g) => {
        const dpr = g.dpr, box = { x: 24 * dpr, y: 12 * dpr, w: g.w - 38 * dpr, h: g.h - 30 * dpr };
        frame(ctx, box, dpr); const m = map(box, [-3.4, 3.4], [-3.4, 3.4]);
        const r = rng(21), P = [];
        for (let i = 0; i < 120; i++) { const a = gaussFrom(r), b = gaussFrom(r); P.push([a * 1.6, v.corr * a * 1.6 + Math.sqrt(1 - v.corr * v.corr) * b * 0.9]); }
        const mx = P.reduce((s, p) => s + p[0], 0) / P.length, my = P.reduce((s, p) => s + p[1], 0) / P.length;
        let cxx = 0, cyy = 0, cxy = 0; P.forEach(p => { cxx += (p[0] - mx) ** 2; cyy += (p[1] - my) ** 2; cxy += (p[0] - mx) * (p[1] - my); }); cxx /= P.length; cyy /= P.length; cxy /= P.length;
        const tr = cxx + cyy, det = cxx * cyy - cxy * cxy, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
        const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
        const ang = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
        P.forEach(p => dot(ctx, m.X(p[0]), m.Y(p[1]), 2.6 * dpr, "#33b1ff99"));
        const drawAxis = (len, col, a) => { const ux = Math.cos(a) * Math.sqrt(len) * 1.6, uy = Math.sin(a) * Math.sqrt(len) * 1.6; line(ctx, m.X(mx - ux), m.Y(my - uy), m.X(mx + ux), m.Y(my + uy), col, 3 * dpr); };
        drawAxis(l1, "#f0a020", ang); drawAxis(l2, "#a371f7", ang + Math.PI / 2);
        return L(`PC1 (orange) explains <b>${(100 * l1 / (l1 + l2)).toFixed(0)}%</b> of variance; PC2 (purple) the rest`,
                 `PC1（橘）解釋 <b>${(100 * l1 / (l1 + l2)).toFixed(0)}%</b> 的變異；PC2（紫）為其餘`);
      }
    });
  };

  /* ---- 09 · a single neuron / forward pass -------------------------------- */
  EM.mlDemo.neuron = function (id) {
    const acts = { sigmoid: z => 1 / (1 + Math.exp(-z)), tanh: z => Math.tanh(z), relu: z => Math.max(0, z) };
    return EM.sim(id, {
      height: 320,
      params: [
        { key: "w1", label: "w₁", min: -3, max: 3, step: 0.1, value: 1.2 },
        { key: "w2", label: "w₂", min: -3, max: 3, step: 0.1, value: -0.8 },
        { key: "b", label: "b", min: -4, max: 4, step: 0.1, value: 0 },
        { key: "act", type: "select", label: L("activation", "激活函數"), value: "sigmoid", options: [{ label: "sigmoid", value: "sigmoid" }, { label: "tanh", value: "tanh" }, { label: "ReLU", value: "relu" }] }
      ],
      draw: (ctx, v, g) => {
        const dpr = g.dpr, box = { x: 24 * dpr, y: 12 * dpr, w: g.w - 38 * dpr, h: g.h - 30 * dpr };
        const fn = acts[v.act] || acts.sigmoid; const S = 9 * dpr;
        let lo = 1e9, hi = -1e9;
        for (let px = box.x; px < box.x + box.w; px += S) for (let py = box.y; py < box.y + box.h; py += S) {
          const x = (px - box.x) / box.w * 8 - 4, y = 4 - (py - box.y) / box.h * 8;
          const a = fn(v.w1 * x + v.w2 * y + v.b); lo = Math.min(lo, a); hi = Math.max(hi, a);
          const tt = (a - (v.act === "tanh" ? -1 : 0)) / ((v.act === "relu" ? Math.max(1, a) : (v.act === "tanh" ? 2 : 1)) || 1);
          const c = Math.max(0, Math.min(1, v.act === "relu" ? a / 8 : (v.act === "tanh" ? (a + 1) / 2 : a)));
          ctx.fillStyle = `rgba(${Math.round(40 + c * 120)},${Math.round(30 + c * 90)},${Math.round(80 + c * 160)},0.5)`;
          ctx.fillRect(px, py, S, S);
        }
        frame(ctx, box, dpr);
        return L(`unit output a = ${v.act}(w₁x₁ + w₂x₂ + b); brighter = higher activation`,
                 `神經元輸出 a = ${v.act}(w₁x₁ + w₂x₂ + b)；越亮代表激活值越大`);
      }
    });
  };

  /* ---- 13 · convolution kernels ------------------------------------------- */
  EM.mlDemo.conv = function (id) {
    const G = 16, img = [];
    for (let y = 0; y < G; y++) { img.push([]); for (let x = 0; x < G; x++) { let val = 0.15; if (x + y > 14 && x + y < 18) val = 0.95; if (x > 9 && x < 14 && y > 2 && y < 7) val = 0.8; img[y].push(val); } }
    const K = {
      identity: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
      blur: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map(r => r.map(c => c / 9)),
      sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
      edge: [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
      sobelx: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
      sobely: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    };
    return EM.sim(id, {
      height: 280,
      params: [{ key: "k", type: "select", label: L("kernel", "卷積核"), value: "edge",
        options: [{ label: L("edge (Laplacian)", "邊緣 (Laplacian)"), value: "edge" }, { label: L("blur", "模糊"), value: "blur" }, { label: L("sharpen", "銳化"), value: "sharpen" }, { label: "Sobel-x", value: "sobelx" }, { label: "Sobel-y", value: "sobely" }, { label: L("identity", "原圖"), value: "identity" }] }],
      draw: (ctx, v, g) => {
        const dpr = g.dpr, ker = K[v.k] || K.edge;
        const cell = Math.floor(Math.min((g.w / 2 - 30 * dpr) / G, (g.h - 30 * dpr) / G));
        const gap = 24 * dpr, x0 = 14 * dpr, y0 = 14 * dpr, x1 = x0 + G * cell + gap;
        const drawGrid = (ox, vals) => { for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) { const c = Math.max(0, Math.min(1, vals[y][x])); ctx.fillStyle = `rgb(${(c * 255) | 0},${(c * 255) | 0},${(c * 255) | 0})`; ctx.fillRect(ox + x * cell, y0 + y * cell, cell - dpr, cell - dpr); } };
        const out = [];
        for (let y = 0; y < G; y++) { out.push([]); for (let x = 0; x < G; x++) { let s = 0; for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) { const yy = Math.max(0, Math.min(G - 1, y + j)), xx = Math.max(0, Math.min(G - 1, x + i)); s += img[yy][xx] * ker[j + 1][i + 1]; } out[y].push(v.k === "blur" || v.k === "identity" ? s : 0.5 + s); } }
        drawGrid(x0, img); drawGrid(x1, out);
        ctx.fillStyle = cssVar("--fg-faint", "#6e7681"); ctx.font = `${11 * dpr}px ui-monospace`; ctx.textAlign = "center";
        ctx.fillText(L("input", "輸入"), x0 + G * cell / 2, y0 + G * cell + 14 * dpr);
        ctx.fillText(L("output", "輸出"), x1 + G * cell / 2, y0 + G * cell + 14 * dpr);
        return L(`kernel <b>${v.k}</b> slides over every pixel — a 3×3 weighted sum. This is one CNN filter.`,
                 `卷積核 <b>${v.k}</b> 滑過每個像素 — 一個 3×3 加權和。這就是一個 CNN 濾波器。`);
      }
    });
  };

  /* ---- 15 · self-attention weights ---------------------------------------- */
  EM.mlDemo.attention = function (id) {
    const toks = ["The", "cat", "sat", "on", "the", "mat"], D = 4, r = rng(13);
    const Q = toks.map(() => Array.from({ length: D }, () => gaussFrom(r)));
    const Kk = toks.map(() => Array.from({ length: D }, () => gaussFrom(r)));
    return EM.sim(id, {
      height: 260,
      params: [
        { key: "q", label: L("query token #", "查詢詞 #"), min: 0, max: 5, step: 1, value: 1 },
        { key: "temp", label: L("temperature", "溫度"), min: 0.2, max: 3, step: 0.1, value: 1 }
      ],
      draw: (ctx, v, g) => {
        const dpr = g.dpr, qi = v.q | 0;
        const score = Kk.map(k => k.reduce((s, kv, d) => s + kv * Q[qi][d], 0) / Math.sqrt(D) / v.temp);
        const mx = Math.max(...score), ex = score.map(s => Math.exp(s - mx)), Z = ex.reduce((a, b) => a + b);
        const w = ex.map(e => e / Z);
        const n = toks.length, bw = (g.w - 40 * dpr) / n, base = g.h - 40 * dpr, top = 20 * dpr;
        let best = 0; w.forEach((wv, i) => { if (wv > w[best]) best = i; });
        ctx.font = `${13 * dpr}px ui-monospace`; ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
        toks.forEach((tk, i) => {
          const x = 20 * dpr + i * bw, h = w[i] * (base - top);
          ctx.fillStyle = i === best ? "#f0a020" : cssVar("--accent-2", "#c8a6ff");
          ctx.fillRect(x + bw * 0.18, base - h, bw * 0.64, h);
          ctx.fillStyle = cssVar("--fg-faint", "#6e7681"); ctx.fillText(w[i].toFixed(2), x + bw / 2, base - h - 5 * dpr);
          ctx.fillStyle = i === qi ? "#f0a020" : cssVar("--fg", "#c9d1d9");
          ctx.fillText(tk, x + bw / 2, base + 18 * dpr);
        });
        return L(`query = "<b>${toks[qi]}</b>" attends most to "<b>${toks[best]}</b>" — bars are softmax weights (sum=1)`,
                 `查詢 = 「<b>${toks[qi]}</b>」最關注「<b>${toks[best]}</b>」 — 長條為 softmax 權重（總和=1）`);
      }
    });
  };
})();
