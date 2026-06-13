"""USV heading-hold simulator (1st-order Nomoto model).

Run:  conda run -n study python usv_heading_sim.py

Scenario: boat starts at heading 0 deg. At t=0 the setpoint is 90 deg, at
t=30s it steps to 180 deg. A constant disturbance (current/wind acting like
~4 deg of stuck rudder) pushes the bow the whole time — that's what the
I term must trim out.

Until PID.update() is implemented this exits with NotImplementedError.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from pid_controller import PID

# ---- boat (Nomoto) parameters -------------------------------------------
K_NOMOTO = 0.6     # turning gain: steady yaw rate (rad/s) per rad of rudder
T_NOMOTO = 2.5     # time constant (s): how sluggish the yaw response is
RUDDER_MAX = math.radians(35)   # actuator saturation
RUDDER_RATE = math.radians(25)  # servo slew limit (rad/s) — see notes Case 9
DISTURBANCE = math.radians(4)   # current/wind, as equivalent stuck rudder

# ---- sensor ---------------------------------------------------------------
NOISE_STD = 0.0    # compass noise, radians. Try math.radians(0.5) -> watch D

# ---- simulation -----------------------------------------------------------
DT = 0.05          # 20 Hz controller, typical for a hobby USV
T_END = 60.0


def wrap(angle):
    """Wrap angle to (-pi, pi]. ALWAYS do this to heading errors (notes 4.1)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def setpoint_at(t):
    return math.radians(90) if t < 30.0 else math.radians(180)


def simulate(pid, seed=0):
    rng = np.random.default_rng(seed)
    n = int(T_END / DT)
    t = np.arange(n) * DT
    psi = np.zeros(n)      # heading (rad)
    r = 0.0                # yaw rate (rad/s)
    rudder = np.zeros(n)
    sp = np.array([setpoint_at(ti) for ti in t])

    for i in range(1, n):
        measured = psi[i - 1] + rng.normal(0.0, NOISE_STD)
        error = wrap(sp[i] - measured)
        cmd = pid.update(error, DT)

        # the servo can't teleport: slew toward the command at RUDDER_RATE max
        step = RUDDER_RATE * DT
        rudder[i] = rudder[i - 1] + np.clip(cmd - rudder[i - 1], -step, step)

        # Nomoto:  T*r_dot + r = K*(rudder + disturbance)
        r_dot = (K_NOMOTO * (rudder[i] + DISTURBANCE) - r) / T_NOMOTO
        r += r_dot * DT
        psi[i] = psi[i - 1] + r * DT

    return t, np.degrees(psi), np.degrees(sp), np.degrees(rudder)


def main():
    # ---- tune these ------------------------------------------------------
    pid = PID(kp=2.0, ki=0.0, kd=0.0,
              output_limits=(-RUDDER_MAX, RUDDER_MAX))
    # ----------------------------------------------------------------------

    t, psi, sp, rudder = simulate(pid)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    ax1.plot(t, sp, "k--", label="setpoint")
    ax1.plot(t, psi, label="heading")
    ax1.set_ylabel("heading (deg)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(t, rudder, color="tab:orange")
    ax2.axhline(math.degrees(RUDDER_MAX), color="r", ls=":", lw=1)
    ax2.axhline(-math.degrees(RUDDER_MAX), color="r", ls=":", lw=1)
    ax2.set_ylabel("rudder (deg)")
    ax2.set_xlabel("time (s)")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f"kp={pid.kp}  ki={pid.ki}  kd={pid.kd}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


# ---- exercises (after update() works; case numbers refer to notes.md) -------
# 1. (Case 1) P only, ki=kd=0: try kp = 0.5, 2, 5, 8. You should find
#    sustained oscillation near kp=5 (that's Ku; period Tu ~ 9 s). Back off
#    to kp ~ 2.
# 2. (Case 2) With kp=2 note the boat settles ~2 deg PAST the setpoint —
#    the DISTURBANCE. Verify offset = 4/kp deg by trying kp=1 and kp=4.
# 3. (Case 3) Add kd: 1, then 3. Find the kd that just kills the overshoot.
# 4. (Case 4) Add ki=0.2: offset gone in ~30 s. Then try ki=2: slow weaving.
# 5. (Case 5) Windup: ki=1, change setpoint_at() to return 180 deg always.
#    Run with your anti-windup, then temporarily disable it. Donuts?
# 6. (Case 6) The setpoint steps at t=30 s. If you differentiated the ERROR,
#    the rudder command slams for one sample. Refactor to differentiate the
#    measurement and compare.
# 7. (Case 7) NOISE_STD = math.radians(0.5): rudder grinds nonstop. Add a
#    low-pass filter on the D term (alpha ~ 0.85) and compare.
