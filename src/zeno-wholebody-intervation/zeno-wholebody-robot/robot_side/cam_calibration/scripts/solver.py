"""AX=YB solver encapsulated in a class."""

from __future__ import annotations

import numpy as np
from pymlg import SE3
from scipy.optimize import least_squares


class AXYSolver:
    """Minimal AX=YB solver on SE(3)."""

    def __init__(self, noise_std: float = 1e-3):
        self.noise_std = noise_std

    @staticmethod
    def pack_params(p: np.ndarray) -> tuple[SE3, SE3]:
        x_vec = p[:6]
        y_vec = p[6:]
        return SE3.Exp(x_vec), SE3.Exp(y_vec)

    def generate_data(self, n: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        X_true = SE3.random()
        Y_true = SE3.random()

        A_list, B_list = [], []
        for _ in range(n):
            A = SE3.random()
            B = SE3.inverse(Y_true) @ A @ X_true  # A X = Y B -> B = Y^-1 A X
            noise = SE3.Exp(rng.normal(0.0, self.noise_std, 6))  # small perturbation
            B_noisy = noise @ B
            A_list.append(A)
            B_list.append(B_noisy)
        return X_true, Y_true, A_list, B_list

    def residuals(self, p: np.ndarray, A_list: list[SE3], B_list: list[SE3]) -> np.ndarray:
        X, Y = self.pack_params(p)
        res = []
        for A, B in zip(A_list, B_list):
            err = A @ X @ SE3.inverse(B) @ SE3.inverse(Y)
            log_err = np.asarray(SE3.Log(err)).reshape(-1)
            res.append(log_err)
        return np.concatenate(res)

    def solve(self, A_list: list[SE3], B_list: list[SE3], p0: np.ndarray | None = None):
        p0 = np.zeros(12) if p0 is None else p0
        sol = least_squares(self.residuals, p0, args=(A_list, B_list))
        X_hat, Y_hat = self.pack_params(sol.x)
        return sol, X_hat, Y_hat

    def run_demo(self, n: int = 10, seed: int = 0):
        X_true, Y_true, A_list, B_list = self.generate_data(n=n, seed=seed)
        sol, X_hat, Y_hat = self.solve(A_list, B_list)
        err_x = np.linalg.norm(SE3.Log(SE3.inverse(X_hat) @ X_true))
        err_y = np.linalg.norm(SE3.Log(SE3.inverse(Y_hat) @ Y_true))
        return {
            "success": sol.success,
            "cost": sol.cost,
            "err_x": err_x,
            "err_y": err_y,
        }


if __name__ == "__main__":
    solver = AXYSolver()
    stats = solver.run_demo()
    print("Optimization success:", stats["success"])
    print("Final cost:", stats["cost"])
    print("||log(X_hat^-1 X_true)||:", stats["err_x"])
    print("||log(Y_hat^-1 Y_true)||:", stats["err_y"])
