import numpy as np

from pendulum_simulation.core.numerical_solver import AbsSolver


class RKF45Solver(AbsSolver):
    """Adaptive Runge-Kutta-Fehlberg 4(5) with a zero-order-hold control.

    Takes as many internal steps as needed to cross `time_step` while keeping
    the estimated local error within rtol/atol.
    """

    SAFETY = 0.9
    MIN_FACTOR = 0.2  # never shrink the step by more than 5x at once
    MAX_FACTOR = 5.0  # never grow it by more than 5x at once

    def __init__(self, derivative, rtol=1e-6, atol=1e-8, max_steps=10_000):
        super().__init__(derivative, rtol, atol)
        self.max_steps = max_steps

    def _attempt(self, t, y, h, u):
        """One trial step. Returns the 5th-order estimate and the error vector."""

        def d(tt, yy):
            return np.asarray(self.derivative(tt, yy, u), dtype=float)

        k1 = d(t, y)
        k2 = d(t + h / 4, y + h * (k1 / 4))
        k3 = d(t + 3 * h / 8, y + h * (3 * k1 / 32 + 9 * k2 / 32))
        k4 = d(
            t + 12 * h / 13,
            y + h * (1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197),
        )
        k5 = d(
            t + h,
            y + h * (439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104),
        )
        k6 = d(
            t + h / 2,
            y
            + h
            * (
                -8 * k1 / 27
                + 2 * k2
                - 3544 * k3 / 2565
                + 1859 * k4 / 4104
                - 11 * k5 / 40
            ),
        )

        y5 = y + h * (
            16 * k1 / 135
            + 6656 * k3 / 12825
            + 28561 * k4 / 56430
            - 9 * k5 / 50
            + 2 * k6 / 55
        )
        # y5 - y4, written out so the 4th-order solution is never formed
        error = h * (
            k1 / 360 - 128 * k3 / 4275 - 2197 * k4 / 75240 + k5 / 50 + 2 * k6 / 55
        )
        return y5, error

    def solve(
        self,
        start_values: np.ndarray,
        start_time: float,
        time_step: float,
        input_vec: np.ndarray,
    ) -> np.ndarray:
        """Advance the state by `time_step`, holding `input_vec` constant."""
        if time_step < 0:
            raise ValueError("time_step must be non-negative")

        y = np.asarray(start_values, dtype=float)
        if time_step == 0:
            return y

        t = start_time
        t_end = start_time + time_step
        h = time_step  # first trial: try to cross the whole interval at once

        for _ in range(self.max_steps):
            # Clip the final step so we land exactly on t_end
            last = h >= t_end - t
            if last:
                h = t_end - t

            y_new, error = self._attempt(t, y, h, input_vec)

            # Scaled RMS error: <= 1 means within tolerance
            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y_new))
            err_norm = float(np.sqrt(np.mean((error / scale) ** 2)))

            if err_norm <= 1.0:
                if last:
                    return y_new
                t += h
                y = y_new

            # Next step size from the error (exponent 1/5 for a 4(5) pair)
            if err_norm == 0.0:
                factor = self.MAX_FACTOR
            else:
                factor = self.SAFETY * err_norm ** (-0.2)
            factor = min(self.MAX_FACTOR, max(self.MIN_FACTOR, factor))
            if err_norm > 1.0:
                factor = min(factor, 1.0)  # a rejected step must shrink
            h *= factor

            if h <= 16 * np.finfo(float).eps * max(abs(t), 1.0):
                raise RuntimeError(f"Step size underflow at t={t}")

        raise RuntimeError(f"Exceeded {self.max_steps} internal steps")
