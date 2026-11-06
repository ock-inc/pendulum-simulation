import numpy as np

from ..core.numerical_solver import AbsSolver  # adjust path

# Dormand-Prince coefficients
C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
B = np.array(
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
)  # 5th order
E = np.array(  # 5th minus 4th order: the error estimate
    [71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40]
)


class RK45Solver(AbsSolver):
    """Adaptive Dormand-Prince RK45 with a zero-order-hold control."""

    def _attempt(self, t, y, h, u):
        """One trial step: returns the new state and an error estimate."""
        k = np.empty((7, y.size))
        for i in range(7):
            k[i] = self.derivative(t + C[i] * h, y + h * (np.array(A[i]) @ k[:i]), u)
        return y + h * (B @ k), h * (E @ k)

    def solve(self, start_values, start_time, time_step, input_vec):
        y = np.asarray(start_values, dtype=float)
        t, t_end = start_time, start_time + time_step
        h = time_step

        while t < t_end:
            final = h >= t_end - t
            if final:
                h = t_end - t  # land exactly on the end time

            y_new, error = self._attempt(t, y, h, input_vec)

            # Error relative to tolerance: <= 1 means acceptable
            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y_new))
            err_norm = max(float(np.sqrt(np.mean((error / scale) ** 2))), 1e-10)

            if err_norm <= 1.0:  # accept
                t = t_end if final else t + h
                y = y_new

            # Resize the step: shrinks after a rejection, grows when error is small
            h *= min(5.0, max(0.2, 0.9 * err_norm**-0.2))

        return y
