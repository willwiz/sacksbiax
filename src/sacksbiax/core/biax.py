from typing import Final
import numpy as np
from scipy.linalg import lstsq
from ..types import MatV, Mat, Vec, f64


dfdr: Vec[f64] = 0.25 * np.array([-1, -1, 1, 1], dtype=float)
dfds: Vec[f64] = 0.25 * np.array([-1, 1, -1, 1], dtype=float)
gradv: Mat[f64] = 0.25 * np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]], dtype=float)


EX: Final[Vec[f64]] = np.array([1, 0], dtype=float)
EY: Final[Vec[f64]] = np.array([0, 1], dtype=float)
A: Final[Mat[f64]] = np.zeros((4, 3), dtype=float)
# dfdr: Vec[f64] = 0.25 * np.array([1, -1, -1, 1], dtype=float)
# dfds: Vec[f64] = 0.25 * np.array([1, 1, -1, -1], dtype=float)
# gradv: Mat[f64] = 0.25 * np.array([[1, -1, -1, 1], [1, 1, -1, -1]], dtype=float)


class BiaxialKinematics:
    ref: Mat[f64]

    def __init__(self, x0: Vec[f64], y0: Vec[f64]) -> None:
        dXdr = x0 @ dfdr
        dXds = x0 @ dfds
        dYdr = y0 @ dfdr
        dYds = y0 @ dfds
        det = dXdr * dYds - dXds * dYdr
        self.ref_tensor = np.array([[dYds, -dXds], [-dYdr, dXdr]], dtype=float) / det

    def deformation_gradient(self, coord: MatV[f64]) -> MatV[f64]:
        grad = np.einsum("mij,kj->mik", coord, gradv)
        def_grad = np.einsum("mij,jk->mik", grad, self.ref_tensor)
        return def_grad


def stress_homogenous(tFinv: Mat[f64], f1: float, f2: float) -> Vec[f64]:
    n1 = EX @ tFinv
    n2 = EY @ tFinv
    A[0, :2] = n1
    A[1, 1:] = n1
    A[2, :2] = n2
    A[3, 1:] = n2
    b = np.array([f1, 0, 0, f2], dtype=float)
    return lstsq(A, b, lapack_driver="gelsy", check_finite=False)[0]
