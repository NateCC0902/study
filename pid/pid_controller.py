"""A PID controller, built for understanding — and later, for the USV.

See notes.md sections 2 and 4 before implementing.
"""


class PID:
    def __init__(self, kp, ki, kd, output_limits=(-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min, self.out_max = output_limits

        # Controller state (mutable across calls)
        self.integral = 0.0
        self.prev_error = None  # None = "no previous sample yet"

    def reset(self):
        """Call when the controller is re-engaged after being off/overridden."""
        self.integral = 0.0
        self.prev_error = None

    def update(self, error, dt):
        """One control step. Returns the actuator command u.

        error : setpoint - measurement, already angle-wrapped (radians)
        dt    : seconds since the last call (> 0)

        TODO(you) — implement the discrete PID law:
          1. P term: kp * error
          2. I term: accumulate `self.integral`, multiply by ki
          3. D term: slope of the error, (error - prev_error) / dt.
             On the very first call prev_error is None — the derivative is
             undefined, so use 0 for that sample.
          4. Sum the terms, clamp to [self.out_min, self.out_max], return it.

        Design choice (yours): anti-windup. If you only do steps 1-4, the
        integral keeps growing while the output is pinned at its limit
        (notes.md 4.2). Pick a strategy:
          a) clamp self.integral so ki*integral can't exceed the output range
          b) conditional: skip integrating on samples where output saturated
          c) back-calculation: integral -= (u_unclamped - u_clamped) * gain
        (a) is the usual first implementation. You can compare strategies
        in the simulator later — exercise 4.
        """
        raise NotImplementedError("implement me, then run usv_heading_sim.py")
