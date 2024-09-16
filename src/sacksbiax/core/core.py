from .utils import RVE_analysis
from .biax import BiaxialKinematics, stress_homogenous
from ..data import SACKS_NODE_ORDER, CycleState, Kinematics, Kinetics
from ..io import *
from ..types import *
from ..data import *
import numpy as np
import pandas as pd


def import_ref_markers(p: str) -> tuple[Vec[f64], Vec[f64]]:
    ref_markers = np.loadtxt(p)
    return (
        ref_markers[[SACKS_NODE_ORDER[i] for i in range(4)]],
        ref_markers[[SACKS_NODE_ORDER[i] + 4 for i in range(4)]],
    )


def compute_kinematics(def_grad: BiaxialKinematics, bx: SacksFormat) -> Kinematics:
    DefGrad = def_grad.deformation_gradient(bx.coord)
    jacobian = DefGrad[:, 0, 0] * DefGrad[:, 1, 1] - DefGrad[:, 0, 1] * DefGrad[:, 1, 0]
    invGrad = np.empty_like(DefGrad, dtype=float)
    invGrad[:, 0, 0] = DefGrad[:, 1, 1] / jacobian
    invGrad[:, 0, 1] = -DefGrad[:, 0, 1] / jacobian
    invGrad[:, 1, 0] = -DefGrad[:, 1, 0] / jacobian
    invGrad[:, 1, 1] = DefGrad[:, 0, 0] / jacobian
    rightCG = np.einsum("mji,mjk->mik", DefGrad, DefGrad)
    return Kinematics(DefGrad, invGrad, rightCG, jacobian)


def compute_kinetics(spec: SpecimenInfo, kin: Kinematics, bx: SacksFormat) -> Kinetics:
    # n_rows = keys.End[-1]
    Lx, Ly, Lz, invGrad_origin = RVE_analysis(spec, bx)
    T1 = 9.80665 * bx.load_x / Ly / Lz
    T2 = 9.80665 * bx.load_y / Lx / Lz
    vVals = np.empty((len(T1), 3), dtype=float)
    for k, (f, i, j) in enumerate(zip(invGrad_origin, T1, T2)):
        vVals[k] = stress_homogenous(f, i, j)
    cauchy = np.empty_like(kin.F, dtype=float)
    cauchy[:, 0, 0] = vVals[:, 0]
    cauchy[:, 0, 1] = vVals[:, 1]
    cauchy[:, 1, 0] = vVals[:, 1]
    cauchy[:, 1, 1] = vVals[:, 2]
    pk1 = np.einsum("mjk,mlk->mjl", cauchy, kin.Finv)
    pk2 = np.einsum("mij,mjk->mik", kin.Finv, pk1)
    return Kinetics(cauchy, pk1, pk2)


def compute_shear_angle(kin: Kinematics) -> Vec[f64]:
    a = kin.F @ [1.0, 0.0]
    b = kin.F @ [0.0, 1.0]
    abs_a = np.linalg.norm(a, axis=1)
    abs_b = np.linalg.norm(b, axis=1)
    return 180.0 * (
        0.5 - np.arccos(np.einsum("mi, mi -> m", a, b) / abs_a / abs_b) / np.pi
    )


def parse_cycle(kin: Kinematics) -> Vec[i32]:
    max_index = np.argmax(kin.J) + 1
    min_index = np.argmin(kin.J)
    if min_index > 20:
        min_index = 0
    label = np.zeros(len(kin.J), dtype=int)
    label[:min_index] = CycleState.Preload
    label[min_index:max_index] = CycleState.Stretch
    label[max_index:] = CycleState.Recover
    return label


def export_kamenskiy_format(
    spec: SpecimenInfo,
    data: SacksFormat,
    cycle: Vec[i32],
    kin: Kinematics,
    sig: Kinetics,
    shear: Vec[f64],
    tag: int,
) -> pd.DataFrame:
    df = dict()
    df["Cycle"] = [f"{tag}-{CycleState(i).name}" for i in cycle]
    df["Time_S"] = data.time
    df["XSize_um"] = ((1000.0 * spec.dim[0]) * data.stretch_x).astype(int)
    df["YSize_um"] = ((1000.0 * spec.dim[1]) * data.stretch_y).astype(int)
    df["XDisplacement_um"] = df["XSize_um"] - df["XSize_um"][0]
    df["YDisplacement_um"] = df["YSize_um"] - df["YSize_um"][0]
    df["XForce_mN"] = 9.80665 * data.load_x
    df["YForce_mN"] = 9.80665 * data.load_y
    df["Temperature"] = 37
    for i in range(4):
        df[f"X{i+1}"] = data.coord[:, 0, i]
        df[f"Y{i+1}"] = data.coord[:, 1, i]
    df["lx"] = kin.F[:, 0, 0]
    df["kx"] = kin.F[:, 0, 1]
    df["ky"] = kin.F[:, 1, 0]
    df["ly"] = kin.F[:, 1, 1]
    df["txx"] = sig.P[:, 0, 0]
    df["txy"] = sig.P[:, 0, 1]
    df["tyx"] = sig.P[:, 1, 0]
    df["tyy"] = sig.P[:, 1, 1]
    df["ShearAngleDeg"] = shear
    return pd.DataFrame.from_dict(df)


def fix_time(time: Vec[f64]) -> Vec[f64]:
    dt = np.diff(time, prepend=[0.0])
    for i in range(1, len(dt)):
        if dt[i] < 0.0:
            dt[i] = dt[i - 1]
    return 0.001 * np.add.accumulate(dt)
