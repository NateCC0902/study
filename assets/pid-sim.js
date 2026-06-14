/* ============================================================================
   PID heading-hold simulator — faithful JS port of usv_heading_sim.py +
   pid_controller.py (1st-order Nomoto boat). Drives the shared EM.sim() canvas.

   Boat constants match the Python exactly:
     K = 0.6 (rad/s)/rad · T = 2.5 s · rudder ±35° · servo slew 25°/s · dt = 0.05 s

   Usage in a chapter, AFTER app.js + this file:
     EM.pidSim("sim-id", PRESET);
   where PRESET picks which sliders/toggles to show and the scenario (setpoint
   schedule, disturbance schedule, initial heading, open-loop mode).
   ========================================================================== */
(function () {
  "use strict";
  if (!window.EM) { console.warn("pid-sim: EM not ready"); return; }

  const DT = 0.05, T_END = 60, N = Math.round(T_END / DT);
  const K = 0.6, T = 2.5;
  const RMAX = 35 * Math.PI / 180;     // rudder saturation (rad)
  const RRATE = 25 * Math.PI / 180;    // servo slew limit (rad/s)
  const rad = d => d * Math.PI / 180;
  const deg = r => r * 180 / Math.PI;
  const wrap = a => (a + Math.PI) % (2 * Math.PI) - Math.PI;   // → (-π, π]

  // deterministic gaussian noise (seeded) so a given slider state always draws
  // the same trace — no flicker between renders.
  function rng(seed) {
    let s = seed >>> 0 || 1;
    const next = () => { s = (1103515245 * s + 12345) & 0x7fffffff; return s / 0x7fffffff; };
    return () => { const u = Math.max(next(), 1e-9), v = next(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  }

  /* Run the closed loop for 60 s and return the trajectories. opt holds the
     resolved gains/flags; sc holds scenario functions. */
  function simulate(opt, sc) {
    const gauss = rng(12345);
    let integral = 0, prevErr = null, prevMeas = null, dFilt = 0;
    let psi = rad(sc.psi0 || 0), r = 0, rudder = 0;
    const t = new Float64Array(N), H = new Float64Array(N), SP = new Float64Array(N), RUD = new Float64Array(N);
    let saturated = false;

    for (let i = 0; i < N; i++) {
      const ti = i * DT;
      const spDeg = sc.spFn ? sc.spFn(ti, opt) : opt.sp;
      const distDeg = sc.distFn ? sc.distFn(ti, opt) : opt.dist;
      const sp = rad(spDeg);
      const meas = psi + (opt.noise ? rad(opt.noise) * gauss() : 0);

      let cmd;
      if (sc.openLoop) {
        cmd = rad(opt.openRudder != null ? opt.openRudder : 10);   // Case 0: hold a fixed rudder
      } else {
        const error = opt.wrap ? wrap(sp - meas) : (sp - meas);    // Case 8: wrap (or the naive bug)
        const P = opt.kp * error;
        integral += error * DT;
        if (opt.antiwindup && Math.abs(opt.ki) > 1e-9) {           // Case 5: clamp the integral
          const lim = RMAX / opt.ki;
          integral = Math.max(-lim, Math.min(lim, integral));
        }
        const I = opt.ki * integral;
        let dRaw;
        if (opt.dMeas) dRaw = prevMeas == null ? 0 : -(meas - prevMeas) / DT;  // Case 6: derivative on measurement
        else dRaw = prevErr == null ? 0 : (error - prevErr) / DT;
        dFilt = opt.dLP ? (0.85 * dFilt + 0.15 * dRaw) : dRaw;     // Case 7: low-pass the derivative
        const D = opt.kd * dFilt;
        cmd = P + I + D;
        prevErr = error; prevMeas = meas;
      }

      const u = Math.max(-RMAX, Math.min(RMAX, cmd));
      if (Math.abs(cmd) >= RMAX - 1e-6) saturated = true;
      const step = RRATE * DT;                                     // Case 9: servo can't teleport
      rudder += Math.max(-step, Math.min(step, u - rudder));

      const rdot = (K * (rudder + rad(distDeg)) - r) / T;          // Nomoto plant
      r += rdot * DT;
      psi += r * DT;

      t[i] = ti; H[i] = deg(psi); SP[i] = spDeg; RUD[i] = deg(rudder);
    }
    return { t, H, SP, RUD, saturated };
  }

  /* metrics for the readout (overshoot, steady-state error, settle time) */
  function metrics(sim) {
    const { H, SP, t } = sim;
    const finalSp = SP[SP.length - 1];
    const tail = Math.round(H.length * 0.95);
    let ssSum = 0; for (let i = tail; i < H.length; i++) ssSum += H[i];
    const ssHeading = ssSum / (H.length - tail);
    const sse = finalSp - ssHeading;
    let peak = -Infinity; for (let i = 0; i < H.length; i++) peak = Math.max(peak, H[i]);
    const overshoot = Math.max(0, peak - finalSp);
    // settle time: last time |H - finalSp| leaves a ±2° band
    let settle = NaN;
    for (let i = H.length - 1; i >= 0; i--) { if (Math.abs(H[i] - finalSp) > 2) { settle = t[Math.min(i + 1, t.length - 1)]; break; } }
    return { sse, overshoot, settle };
  }

  /* ---- the plot: two stacked panels (heading + rudder) -------------------- */
  function plot(ctx, g, sim, scLabel) {
    const dpr = g.dpr, W = g.w, H = g.h;
    const padL = 46 * dpr, padR = 12 * dpr, padT = 14 * dpr, gap = 26 * dpr, padB = 26 * dpr;
    const panelH = (H - padT - padB - gap) / 2;
    const x0 = padL, x1 = W - padR;
    const tMax = T_END;
    const css = getComputedStyle(document.body);
    const cAcc = (css.getPropertyValue("--accent-2") || "#56d364").trim();
    const cMut = (css.getPropertyValue("--fg-faint") || "#6e7681").trim();
    const cGrid = "#1d242f";
    ctx.font = `${11 * dpr}px ui-monospace, monospace`;
    ctx.textBaseline = "middle";

    function panel(top, series, yLabel, yMin, yMax, limit) {
      const yb = top + panelH, yt = top;
      const X = ti => x0 + (ti / tMax) * (x1 - x0);
      const Y = v => yb - ((v - yMin) / (yMax - yMin)) * (yb - yt);
      // grid + axes
      ctx.strokeStyle = cGrid; ctx.lineWidth = 1 * dpr; ctx.fillStyle = cMut;
      ctx.textAlign = "right";
      const ticks = 4;
      for (let k = 0; k <= ticks; k++) {
        const v = yMin + (yMax - yMin) * k / ticks; const y = Y(v);
        ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
        ctx.fillText(Math.round(v).toString(), x0 - 6 * dpr, y);
      }
      ctx.textAlign = "center";
      for (let s = 0; s <= 60; s += 15) { const x = X(s); ctx.fillText(s + "s", x, yb + 12 * dpr); }
      // limit lines (rudder ±35)
      if (limit != null) {
        ctx.strokeStyle = "#f0506050"; ctx.setLineDash([4 * dpr, 4 * dpr]);
        [limit, -limit].forEach(L => { const y = Y(L); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); });
        ctx.setLineDash([]);
      }
      // series
      series.forEach(ser => {
        ctx.strokeStyle = ser.color; ctx.lineWidth = (ser.w || 2) * dpr;
        if (ser.dash) ctx.setLineDash([6 * dpr, 5 * dpr]); else ctx.setLineDash([]);
        ctx.beginPath();
        for (let i = 0; i < ser.data.length; i++) { const x = X(sim.t[i]), y = Y(ser.data[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
        ctx.stroke(); ctx.setLineDash([]);
      });
      ctx.fillStyle = cMut; ctx.textAlign = "left"; ctx.save();
      ctx.translate(12 * dpr, (yt + yb) / 2); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center"; ctx.fillText(yLabel, 0, 0); ctx.restore();
    }

    // heading panel range from data
    let hmin = 0, hmax = 0;
    for (let i = 0; i < sim.H.length; i++) { hmin = Math.min(hmin, sim.H[i], sim.SP[i]); hmax = Math.max(hmax, sim.H[i], sim.SP[i]); }
    const pad = Math.max(8, (hmax - hmin) * 0.1); hmin -= pad; hmax += pad;
    panel(padT, [
      { data: sim.SP, color: cMut, w: 1.5, dash: true },
      { data: sim.H, color: cAcc, w: 2.2 }
    ], "heading °", hmin, hmax, null);
    panel(padT + panelH + gap, [
      { data: sim.RUD, color: "#f0a020", w: 2 }
    ], "rudder °", -40, 40, 35);
  }

  /* ---- public: build a configured sim ------------------------------------- */
  EM.pidSim = function (canvasId, preset) {
    preset = preset || {};
    const fixed = preset.fixed || {};
    const sc = {
      psi0: preset.psi0 || 0,
      openLoop: !!preset.openLoop,
      spFn: preset.spFn || null,
      distFn: preset.distFn || null
    };
    return EM.sim(canvasId, {
      height: preset.height || 340,
      params: preset.params || [],
      draw: function (ctx, vals, g) {
        const pick = (k, d) => (k in vals ? vals[k] : (k in fixed ? fixed[k] : d));
        const opt = {
          kp: pick("kp", 0), ki: pick("ki", 0), kd: pick("kd", 0),
          dist: pick("dist", 0), noise: pick("noise", 0), sp: pick("sp", 90),
          openRudder: pick("rudder", 10),
          antiwindup: !!pick("antiwindup", true),
          dMeas: !!pick("dMeas", true),
          dLP: !!pick("dLP", false),
          wrap: !!pick("wrap", true)
        };
        const sim = simulate(opt, sc);
        plot(ctx, g, sim);
        const m = metrics(sim);
        const lang = document.documentElement.lang === "zh-Hant" ? "zh" : "en";
        if (sc.openLoop) {
          return lang === "zh"
            ? `開迴路：固定舵角，航向是偏航率的<b>積分</b> → 永遠不會穩定。`
            : `Open loop: fixed rudder, heading is the <b>integral</b> of yaw rate → never settles.`;
        }
        const sat = sim.saturated ? (lang === "zh" ? " · 舵已<b>飽和</b>" : " · rudder <b>saturated</b>") : "";
        return lang === "zh"
          ? `穩態誤差 <b>${m.sse.toFixed(1)}°</b> · 超越量 <b>${m.overshoot.toFixed(1)}°</b> · 安定時間 <b>${isNaN(m.settle) ? "—" : m.settle.toFixed(1) + " s"}</b>${sat}`
          : `steady-state error <b>${m.sse.toFixed(1)}°</b> · overshoot <b>${m.overshoot.toFixed(1)}°</b> · settle <b>${isNaN(m.settle) ? "—" : m.settle.toFixed(1) + " s"}</b>${sat}`;
      }
    });
  };

  /* ---- ready-made presets per chapter (EN/zh labels via a helper) ---------- */
  const L = (en, zh) => (document.documentElement.lang === "zh-Hant" ? zh : en);
  EM.pidPresets = {
    // ch01 — open loop: hold a fixed rudder, watch heading ramp forever
    openLoop: () => ({
      openLoop: true, height: 300,
      params: [{ key: "rudder", label: L("fixed rudder", "固定舵角"), min: -20, max: 20, step: 1, value: 10, unit: "°" }]
    }),
    // ch02 — P only: spring + structural offset from the current
    pOnly: () => ({
      psi0: 0, fixed: { ki: 0, kd: 0, sp: 90 },
      params: [
        { key: "kp", label: "kp", min: 0, max: 8, step: 0.5, value: 2 },
        { key: "dist", label: L("current", "水流"), min: 0, max: 8, step: 1, value: 4, unit: "°" }
      ]
    }),
    // ch03 — PD: add the damper
    pd: () => ({
      psi0: 0, fixed: { ki: 0, sp: 90, dist: 0 },
      params: [
        { key: "kp", label: "kp", min: 0, max: 6, step: 0.5, value: 2 },
        { key: "kd", label: "kd", min: 0, max: 8, step: 0.5, value: 0 }
      ]
    }),
    // ch04 — PID: disturbance switches on at t=20 s, I trims it out
    pid: () => ({
      psi0: 0, fixed: { kp: 2, kd: 3, sp: 90 },
      distFn: (t, o) => (t >= 20 ? 4 : 0),
      params: [{ key: "ki", label: "ki", min: 0, max: 2.0, step: 0.05, value: 0.2 }]
    }),
    // ch05 — windup: hard 180° turn, integral winds up unless clamped
    windup: () => ({
      psi0: 0, fixed: { kp: 2, kd: 3, ki: 1, sp: 180, dist: 4 },
      params: [{ key: "antiwindup", type: "toggle", label: L("anti-windup", "抗飽和"), value: true }]
    }),
    // ch06 — derivative kick + noise: setpoint bump at t=30, compare options
    derivative: () => ({
      psi0: 0, fixed: { kp: 2, ki: 0.2, kd: 3 },
      spFn: (t) => (t < 30 ? 90 : 110),
      params: [
        { key: "noise", label: L("noise σ", "雜訊 σ"), min: 0, max: 1.5, step: 0.1, value: 0, unit: "°" },
        { key: "dMeas", type: "toggle", label: L("D on measurement", "對量測微分"), value: true },
        { key: "dLP", type: "toggle", label: L("D low-pass", "D 低通"), value: false }
      ]
    }),
    // ch07 — angle wrap: start at 350°, target 10°; toggle the wrap bug
    wrap: () => ({
      psi0: 350, fixed: { kp: 2, ki: 0.2, kd: 3, sp: 10, dist: 0 },
      params: [{ key: "wrap", type: "toggle", label: L("wrap error", "纏繞修正"), value: true }]
    }),
    // ch08 — full playground: every knob, apply the tuning recipe
    full: () => ({
      psi0: 0, fixed: { sp: 90 },
      params: [
        { key: "kp", label: "kp", min: 0, max: 8, step: 0.5, value: 2 },
        { key: "ki", label: "ki", min: 0, max: 1.0, step: 0.05, value: 0.2 },
        { key: "kd", label: "kd", min: 0, max: 8, step: 0.5, value: 3 },
        { key: "dist", label: L("current", "水流"), min: 0, max: 8, step: 1, value: 4, unit: "°" },
        { key: "noise", label: L("noise σ", "雜訊 σ"), min: 0, max: 1.5, step: 0.1, value: 0, unit: "°" },
        { key: "antiwindup", type: "toggle", label: L("anti-windup", "抗飽和"), value: true },
        { key: "dMeas", type: "toggle", label: L("D on meas.", "對量測微分"), value: true }
      ]
    })
  };
})();
