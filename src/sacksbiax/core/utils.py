from .biax import BiaxialKinematics
from ..data import *
from ..types import *
import numpy as np


def mat_vec_contraction(tensor_list: MatV[f64], vec: Vec[f64]) -> float:
    mi = np.einsum("mij,j->mi", tensor_list, vec)
    m = np.einsum("mi,mi->m", mi, mi)
    return np.sqrt(m)


def RVE_analysis(
    spec: SpecimenInfo, bx: SacksFormat
) -> tuple[Vec[f64], Vec[f64], Vec[f64], Mat[f64]]:
    Lx0 = spec.dim[0] * bx.stretch_x[0]
    Ly0 = spec.dim[1] * bx.stretch_y[0]
    Lz0 = spec.dim[2] / bx.stretch_x[0] / bx.stretch_y[1]
    # Compute Deformation Gradient from initial state
    x_dirt = np.array([1, 0], dtype=float)
    y_dirt = np.array([0, 1], dtype=float)
    kin_origin = BiaxialKinematics(spec.x_iff, spec.y_iff)
    DG_origin = kin_origin.deformation_gradient(bx.coord)
    J_origin = (
        DG_origin[:, 0, 0] * DG_origin[:, 1, 1]
        - DG_origin[:, 0, 1] * DG_origin[:, 1, 0]
    )
    invGrad_origin = np.empty_like(DG_origin, dtype=float)
    invGrad_origin[:, 0, 0] = DG_origin[:, 1, 1] / J_origin
    invGrad_origin[:, 0, 1] = -DG_origin[:, 0, 1] / J_origin
    invGrad_origin[:, 1, 0] = -DG_origin[:, 1, 0] / J_origin
    invGrad_origin[:, 1, 1] = DG_origin[:, 0, 0] / J_origin
    stretch_x = mat_vec_contraction(DG_origin, x_dirt)
    stretch_y = mat_vec_contraction(DG_origin, y_dirt)
    # Approximation by projecting dimensions of RVE to full specimen assuming homogeneity
    Lz = Lz0 / J_origin
    Ly = Ly0 * stretch_y
    Lx = Lx0 * stretch_x
    return Lx, Ly, Lz, invGrad_origin
