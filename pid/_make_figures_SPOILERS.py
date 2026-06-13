"""Generates the figures embedded in notes.md.

!! SPOILER WARNING !!
This file contains a complete PID implementation. If you haven't written
PID.update() in pid_controller.py yet, don't read past this docstring —
run it blind:  conda run -n study python _make_figures_SPOILERS.py
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K, T = 0.6, 2.5                      # Nomoto parameters (same as the simulator)
RUDDER_MAX = math.radians(35)        # rudder hard stop
RUDDER_RATE = math.radians(25)       # servo slew limit, rad/s
DIST = math.radians(4)               # current/wind as equivalent stuck rudder
DT = 0.05
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def simulate(kp=0.0, ki=0.0, kd=0.0, t_end=60.0, psi0_deg=0.0, dist=DIST,
             noise=0.0, anti_windup=True, d_on_meas=False, d_alpha=0.0,
             use_wrap=True, setpoint_fn=None, rudder_rate=RUDDER_RATE, seed=0):
    """Closed-loop sim. `dist` may be a float or a function of t.
    Returns dict of arrays, angles in degrees. u = actual rudder, cmd = PID output."""
    if setpoint_fn is None:
        setpoint_fn = lambda t: math.radians(90)
    dist_fn = dist if callable(dist) else (lambda t, d=dist: d)
    rng = np.random.default_rng(seed)
    n = int(t_end / DT)
    t = np.arange(n) * DT
    psi = np.zeros(n)
    psi[0] = math.radians(psi0_deg)
    r, act = 0.0, 0.0
    u, cmd = np.zeros(n), np.zeros(n)
    uP, uI, uD = np.zeros(n), np.zeros(n), np.zeros(n)
    sp = np.array([setpoint_fn(ti) for ti in t])
    integral, prev_e, prev_m, d_state = 0.0, None, None, 0.0

    for i in range(1, n):
        meas = psi[i - 1] + rng.normal(0.0, noise)
        e = wrap(sp[i] - meas) if use_wrap else (sp[i] - meas)

        P = kp * e
        integral += e * DT
        if anti_windup and ki > 0:
            lim = RUDDER_MAX / ki          # clamp so |ki*integral| <= RUDDER_MAX
            integral = max(-lim, min(lim, integral))
        I = ki * integral
        if d_on_meas:
            d_raw = 0.0 if prev_m is None else -wrap(meas - prev_m) / DT
        else:
            d_raw = 0.0 if prev_e is None else (e - prev_e) / DT
        if d_alpha > 0:
            d_state = d_alpha * d_state + (1 - d_alpha) * d_raw
            d_raw = d_state
        D = kd * d_raw
        prev_e, prev_m = e, meas

        c = max(-RUDDER_MAX, min(RUDDER_MAX, P + I + D))
        if rudder_rate is not None:                       # servo can't teleport
            step = rudder_rate * DT
            act += max(-step, min(step, c - act))
        else:
            act = c
        cmd[i], u[i] = c, act
        uP[i], uI[i], uD[i] = P, I, D

        r_dot = (K * (act + dist_fn(t[i])) - r) / T
        r += r_dot * DT
        psi[i] = psi[i - 1] + r * DT

    deg = np.degrees
    return dict(t=t, psi=deg(psi), sp=deg(sp), u=deg(u), cmd=deg(cmd),
                uP=deg(uP), uI=deg(uI), uD=deg(uD))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, name), dpi=110)
    plt.close(fig)
    print("wrote", name)


# ---- fig 0: open loop — the boat with no controller ------------------------
def fig0_openloop():
    n = int(30 / DT)
    t = np.arange(n) * DT
    r_arr, psi_arr = np.zeros(n), np.zeros(n)
    r = 0.0
    delta = math.radians(10)             # rudder held at 10 deg, no feedback
    for i in range(1, n):
        r += (K * delta - r) / T * DT
        r_arr[i] = r
        psi_arr[i] = psi_arr[i - 1] + r * DT
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    a1.plot(t, np.degrees(r_arr))
    a1.axhline(math.degrees(K * delta), color="r", ls=":", label="K·δ = 6 °/s")
    a1.axvline(T, color="gray", ls="--", lw=1, label="t = T = 2.5 s")
    a1.set_ylabel("yaw rate r (°/s)")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.plot(t, np.degrees(psi_arr))
    a2.set_ylabel("heading ψ (°)")
    a2.set_xlabel("time (s)")
    a2.grid(alpha=0.3)
    fig.suptitle("Open loop: rudder fixed at 10° — rate settles, heading ramps forever")
    save(fig, "fig0_openloop.png")


# ---- fig 1: P-only gain sweep ----------------------------------------------
def fig1_p_sweep():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for kp, c in [(0.5, "tab:blue"), (2, "tab:green"), (5, "tab:orange"), (8, "tab:red")]:
        s = simulate(kp=kp, t_end=90)
        ax.plot(s["t"], s["psi"], color=c, label=f"kp={kp}")
    ax.axhline(90, color="k", ls="--", lw=1, label="setpoint")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("heading (°)")
    ax.legend(loc="lower right", ncols=2)
    ax.grid(alpha=0.3)
    ax.set_title("P only: kp=0.5 sluggish, kp=2 good, kp=5 never stops oscillating, kp=8 worse")
    save(fig, "fig1_p_sweep.png")


# ---- fig 2: the steady-state offset, zoomed --------------------------------
def fig2_offset():
    fig, ax = plt.subplots(figsize=(9, 4))
    for kp in [1, 2, 4]:
        s = simulate(kp=kp, t_end=90)
        ax.plot(s["t"], s["psi"], label=f"kp={kp}  →  settles at {s['psi'][-1]:.1f}°")
    ax.axhline(90, color="k", ls="--", lw=1, label="setpoint 90°")
    ax.set_xlim(20, 90)
    ax.set_ylim(85, 100)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("heading (°)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("P only under a 4°-equivalent current: offset = 4°/kp, never zero")
    save(fig, "fig2_offset.png")


# ---- fig 3: adding D — damping ----------------------------------------------
def fig3_pd():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for kd in [0, 1, 3]:
        s = simulate(kp=2, kd=kd, t_end=40)
        ax.plot(s["t"], s["psi"], label=f"kp=2, kd={kd}")
    ax.axhline(90, color="k", ls="--", lw=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("heading (°)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Adding D: same kp, overshoot shrinks then disappears")
    save(fig, "fig3_pd.png")


# ---- fig 4: disturbance rejection — what I is FOR ----------------------------
def fig4_integral():
    dist = lambda t: 0.0 if t < 20 else DIST     # current switches on at t=20
    pd_ = simulate(kp=2, kd=3, t_end=120, psi0_deg=90, dist=dist)
    pid = simulate(kp=2, ki=0.2, kd=3, t_end=120, psi0_deg=90, dist=dist)
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    a1.plot(pd_["t"], pd_["psi"], label="PD only — bent forever")
    a1.plot(pid["t"], pid["psi"], label="PID — I trims it out")
    a1.axhline(90, color="k", ls="--", lw=1, label="setpoint")
    a1.axvline(20, color="gray", ls=":", label="current switches on")
    a1.set_ylabel("heading (°)")
    a1.legend(loc="upper right")
    a1.grid(alpha=0.3)
    a2.plot(pid["t"], pid["uI"], color="tab:orange", label="PID's I term")
    a2.plot(pd_["t"], pd_["uP"], color="tab:blue", ls="--", label="PD's P term (stuck holding)")
    a2.axhline(-4, color="gray", ls=":", label="−4° = what the current needs")
    a2.axvline(20, color="gray", ls=":")
    a2.set_ylabel("rudder contribution (°)")
    a2.set_xlabel("time (s)")
    a2.legend(loc="upper right")
    a2.grid(alpha=0.3)
    fig.suptitle("Boat is on heading; a current appears at t=20 s")
    save(fig, "fig4_integral.png")


# ---- fig 5: integral windup --------------------------------------------------
def fig5_windup():
    sp = lambda t: math.radians(180)
    a = simulate(kp=2, ki=1.0, kd=3, t_end=90, setpoint_fn=sp, anti_windup=True)
    b = simulate(kp=2, ki=1.0, kd=3, t_end=90, setpoint_fn=sp, anti_windup=False)
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    a1.plot(a["t"], a["psi"], label="with anti-windup (integral clamped)")
    a1.plot(b["t"], b["psi"], label="NO anti-windup")
    a1.axhline(180, color="k", ls="--", lw=1)
    a1.set_ylabel("heading (°)")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.plot(a["t"], a["uI"], label="I term, clamped")
    a2.plot(b["t"], b["uI"], label="I term, unclamped")
    a2.set_ylabel("I-term (° rudder)")
    a2.set_xlabel("time (s)")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.suptitle("180° turn with ki=1: rudder saturates for seconds, the integral 'winds up'")
    save(fig, "fig5_windup.png")


# ---- fig 6: derivative kick ---------------------------------------------------
def fig6_dkick():
    sp = lambda t: math.radians(90 if t < 30 else 100)   # small 10° step
    a = simulate(kp=2, ki=0.2, kd=3, t_end=45, setpoint_fn=sp, d_on_meas=False)
    b = simulate(kp=2, ki=0.2, kd=3, t_end=45, setpoint_fn=sp, d_on_meas=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(a["t"], a["cmd"], label="derivative of ERROR → kick!")
    ax.plot(b["t"], b["cmd"], label="derivative of MEASUREMENT")
    ax.set_xlim(28, 40)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("commanded rudder (°)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Setpoint steps 90°→100° at t=30 s: one-sample slam vs smooth command")
    save(fig, "fig6_dkick.png")


# ---- fig 7: noise amplification by D ------------------------------------------
def fig7_noise():
    noise = math.radians(0.5)
    a = simulate(kp=2, ki=0.2, kd=3, t_end=60, noise=noise, d_alpha=0.0)
    b = simulate(kp=2, ki=0.2, kd=3, t_end=60, noise=noise, d_alpha=0.85)
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(9, 5.5))
    a1.plot(a["t"], a["cmd"], lw=0.5, alpha=0.6, label="commanded")
    a1.plot(a["t"], a["u"], lw=1.2, label="actual rudder (servo grinding)")
    a1.set_ylabel("rudder (°)")
    a1.legend(loc="lower right")
    a1.set_title("kd=3, compass noise σ=0.5°, raw derivative")
    a1.grid(alpha=0.3)
    a2.plot(b["t"], b["cmd"], lw=0.5, alpha=0.6, label="commanded")
    a2.plot(b["t"], b["u"], lw=1.2, label="actual rudder")
    a2.set_ylabel("rudder (°)")
    a2.set_xlabel("time (s)")
    a2.legend(loc="lower right")
    a2.set_title("same, derivative low-pass filtered (α=0.85)")
    a2.grid(alpha=0.3)
    save(fig, "fig7_noise.png")


# ---- fig 8: angle wrapping -----------------------------------------------------
def fig8_wrap():
    sp = lambda t: math.radians(10)
    a = simulate(kp=2, kd=3, t_end=40, psi0_deg=350, dist=0.0, setpoint_fn=sp,
                 use_wrap=True)
    b = simulate(kp=2, kd=3, t_end=40, psi0_deg=350, dist=0.0, setpoint_fn=sp,
                 use_wrap=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(a["t"], a["psi"], label="wrapped error → +20° turn (correct)")
    ax.plot(b["t"], b["psi"], label="naive error = −340° → the long way round")
    ax.axhline(370, color="k", ls=":", lw=1)
    ax.axhline(10, color="k", ls=":", lw=1)
    ax.annotate("370° ≡ 10°", (1, 374))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("heading, unwrapped (°)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Start 350°, setpoint 10°: with vs without angle wrapping")
    save(fig, "fig8_wrap.png")


# ---- fig 9: WHY high kp oscillates — the rudder can't keep up ------------------
def fig9_ratelimit():
    s = simulate(kp=8, t_end=60)
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    a1.plot(s["t"], s["psi"])
    a1.axhline(90, color="k", ls="--", lw=1)
    a1.set_ylabel("heading (°)")
    a1.set_title("kp=8: heading limit-cycles forever")
    a1.grid(alpha=0.3)
    a2.plot(s["t"], s["cmd"], lw=0.8, alpha=0.7, label="commanded (slams rail to rail)")
    a2.plot(s["t"], s["u"], lw=1.5, label="actual rudder (25 °/s slew limit)")
    a2.set_xlim(30, 60)
    a2.set_ylabel("rudder (°)")
    a2.set_xlabel("time (s)")
    a2.legend(loc="lower right")
    a2.set_title("the servo lags the command — that lag is what destabilizes the loop")
    a2.grid(alpha=0.3)
    save(fig, "fig9_ratelimit.png")


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    fig0_openloop()
    fig1_p_sweep()
    fig2_offset()
    fig3_pd()
    fig4_integral()
    fig5_windup()
    fig6_dkick()
    fig7_noise()
    fig8_wrap()
    fig9_ratelimit()
    print("all figures written to", FIGDIR)
