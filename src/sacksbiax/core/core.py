import re
from .utils import RVE_analysis
from .biax import BiaxialKinematics, stress_homogenous
from ..datatypes import SACKS_NODE_ORDER, CycleState, Kinematics, Kinetics
from ..parsers import *
from ..types import *
from ..datatypes import *
import numpy as np
import pandas as pd


def import_ref_markers(p: str) -> tuple[Vec[f64], Vec[f64]]:
    ref_markers = np.loadtxt(p)
    return (
        ref_markers[[SACKS_NODE_ORDER[i] for i in range(4)]],
        ref_markers[[SACKS_NODE_ORDER[i] + 4 for i in range(4)]],
    )


TEMP = re.compile(r"(\d+)-.+")


def match_n_s(vals: list[str] | Vec[char]):
    for v in vals:
        yield TEMP.match(v)


def find_last_cycle_name(data: pd.DataFrame):
    cycles = {
        int(g.group(1)): g.group(0) for g in match_n_s(data["Cycle"].unique()) if g
    }
    return cycles[max(cycles.keys())]


def find_last_cycle(raw: pd.DataFrame, setname: str):
    this_set = raw["SetName"] == setname
    return (this_set) & (raw["Cycle"] == find_last_cycle_name(raw[this_set]))


def sort_data_cycle_types(raw: pd.DataFrame) -> CycleTypes:
    setnames = raw["SetName"].unique()
    precond = raw["SetName"].str.contains("precond", case=False)
    preload = raw["Cycle"].str.contains("preload", case=False)
    stretch = raw["Cycle"].str.contains("stretch", case=False)
    recover = raw["Cycle"].str.contains("recover", case=False)
    equibx = raw["SetName"].str.contains("equib", case=False) & ~preload
    relax = raw["SetName"].str.contains("relax", case=False) & ~preload
    creep = raw["SetName"].str.contains("creep", case=False) & ~preload
    last_cycle = find_last_cycle(raw, setnames[0])
    for s in setnames[1:]:
        last_cycle = last_cycle | find_last_cycle(raw, s)
    last_eb = raw["SetName"] == (equibx.unique()[-1])
    plotting = ~precond & ~preload & last_cycle
    fitting = plotting & ~(equibx ^ last_eb)
    return CycleTypes(
        precond.to_numpy(int),
        equibx.to_numpy(int),
        relax.to_numpy(int),
        creep.to_numpy(int),
        preload.to_numpy(int),
        stretch.to_numpy(int),
        recover.to_numpy(int),
        last_cycle.to_numpy(int),
        last_eb.to_numpy(int),
        plotting.to_numpy(int),
        fitting.to_numpy(int),
    )


def compute_kinematics(def_grad: BiaxialKinematics, bx: RawBiaxFormat) -> Kinematics:
    DefGrad = def_grad.deformation_gradient(bx.coord)
    jacobian = DefGrad[:, 0, 0] * DefGrad[:, 1, 1] - DefGrad[:, 0, 1] * DefGrad[:, 1, 0]
    invGrad = np.empty_like(DefGrad, dtype=float)
    invGrad[:, 0, 0] = DefGrad[:, 1, 1] / jacobian
    invGrad[:, 0, 1] = -DefGrad[:, 0, 1] / jacobian
    invGrad[:, 1, 0] = -DefGrad[:, 1, 0] / jacobian
    invGrad[:, 1, 1] = DefGrad[:, 0, 0] / jacobian
    rightCG = np.einsum("mji,mjk->mik", DefGrad, DefGrad)
    return Kinematics(DefGrad, invGrad, rightCG, jacobian)


def compute_kinetics_cauchy(
    spec: SpecimenInfo, kin: Kinematics, bx: RawBiaxFormat
) -> Kinetics:
    # n_rows = keys.End[-1]
    Lx, Ly, Lz, invGrad_origin = RVE_analysis(spec, bx)
    T1 = bx.XForce_mN / Ly / Lz
    T2 = bx.YForce_mN / Lx / Lz
    vVals = np.empty((len(T1), 3), dtype=float)
    for k, (f, i, j) in enumerate(zip(invGrad_origin, T1, T2)):
        vVals[k] = stress_homogenous(f, i, j)
    cauchy = np.empty_like(kin.F, dtype=float)
    cauchy[:, 0, 0] = vVals[:, 0]
    cauchy[:, 0, 1] = vVals[:, 1]
    cauchy[:, 1, 0] = vVals[:, 1]
    cauchy[:, 1, 1] = vVals[:, 2]
    pk1 = np.einsum("mij,mkj->mik", cauchy, kin.Finv)
    pk2 = np.einsum("mij,mjk->mik", kin.Finv, pk1)
    return Kinetics(cauchy, pk1, pk2, Lz)


def compute_kinetics_pk1(
    spec: SpecimenInfo, kin: Kinematics, bx: RawBiaxFormat
) -> Kinetics:
    # n_rows = keys.End[-1]
    _, _, Lz, _ = RVE_analysis(spec, bx)
    pk1 = np.zeros_like(kin.F, dtype=float)
    pk1[:, 0, 0] = bx.XForce_mN / spec.dim[1] / spec.dim[2]
    pk1[:, 1, 1] = bx.YForce_mN / spec.dim[0] / spec.dim[2]
    cauchy = np.einsum("mij,mkj->mik", pk1, kin.F)
    pk2 = np.einsum("mij,mjk->mik", kin.Finv, pk1)
    return Kinetics(cauchy, pk1, pk2, Lz)


def compute_kinetics_nominal(
    spec: SpecimenInfo, kin: Kinematics, bx: RawBiaxFormat
) -> Kinetics:
    # n_rows = keys.End[-1]
    _, _, Lz, _ = RVE_analysis(spec, bx)
    nominal = np.zeros_like(kin.F, dtype=float)
    nominal[:, 0, 0] = bx.XForce_mN / spec.dim[1] / spec.dim[2]
    nominal[:, 1, 1] = bx.YForce_mN / spec.dim[0] / spec.dim[2]
    cauchy = np.einsum("mij,mjk->mik", kin.F, nominal)
    pk1 = nominal.swapaxes(1, 2)
    pk2 = np.einsum("mij,mjk->mik", kin.Finv, pk1)
    return Kinetics(cauchy, pk1, pk2, Lz)


def compute_energy(kin: Kinematics, sig: Kinetics) -> Energy:
    dE: MatV[f64] = np.diff(kin.C, axis=0, prepend=[[[0, 0], [0, 0]]])
    dH = np.zeros_like(dE)
    dH[:-1] = dE[1:] * sig.S[:-1]
    dH[1:] = dH[1:] + dE[1:] * sig.S[1:]
    dW = dH[:, 0, 0] + dH[:, 0, 1] + dH[:, 1, 0] + dH[:, 1, 1]
    psi = np.add.accumulate(dW)
    return Energy(dE, dH, dW, psi)


def compute_shear_angle(kin: Kinematics) -> Vec[f64]:
    a = kin.F @ [1.0, 0.0]
    b = kin.F @ [0.0, 1.0]
    abs_a = np.linalg.norm(a, axis=1)
    abs_b = np.linalg.norm(b, axis=1)
    return 180.0 * (
        0.5 - np.arccos(np.einsum("mi, mi -> m", a, b) / abs_a / abs_b) / np.pi
    )


def parse_cycle(kin: Kinematics, tag: str | int) -> list[str]:
    max_index = np.argmax(kin.J) + 1
    min_index = np.argmin(kin.J)
    if min_index > 20:
        min_index = 0
    label = np.zeros(len(kin.J), dtype=int)
    label[:min_index] = CycleState.Preload
    label[min_index:max_index] = CycleState.Stretch
    label[max_index:] = CycleState.Recover
    return [f"{tag}-{CycleState(i).name}" for i in label]


def fix_time(time: Vec[f64]) -> Vec[f64]:
    dt = np.diff(time, prepend=[0.0])
    for i in range(1, len(dt)):
        if dt[i] < 0.0:
            dt[i] = dt[i - 1]
    return np.add.accumulate(dt)


def export_kamenskiy_format(
    data: RawBiaxFormat,
    kin: Kinematics,
    sig: Kinetics,
    shear: Vec[f64],
) -> pd.DataFrame:
    df = dict()
    df["Time_S"] = data.time
    df["XSize_um"] = data.XSize_um.astype(int)
    df["YSize_um"] = data.YSize_um.astype(int)
    df["XForce_mN"] = data.XForce_mN
    df["YForce_mN"] = data.YForce_mN
    df["Temperature"] = data.Temperature
    df["XDisplacement_um"] = df["XSize_um"] - df["XSize_um"][0]
    df["YDisplacement_um"] = df["YSize_um"] - df["YSize_um"][0]
    for i in range(4):
        df[f"X{i+1}"] = data.coord[:, 0, i]
        df[f"Y{i+1}"] = data.coord[:, 1, i]
    df["lx"] = kin.F[:, 0, 0]
    df["txx"] = sig.sigma[:, 0, 0]
    df["ly"] = kin.F[:, 1, 1]
    df["tyy"] = sig.sigma[:, 1, 1]
    df["ShearAngleDeg"] = shear
    df["kx"] = kin.F[:, 0, 1]
    df["ky"] = kin.F[:, 1, 0]
    df["txy"] = sig.sigma[:, 0, 1]
    df["tyx"] = sig.sigma[:, 1, 0]
    df["Pxx"] = sig.P[:, 0, 0]
    df["Pxy"] = sig.P[:, 0, 1]
    df["Pyx"] = sig.P[:, 1, 0]
    df["Pyy"] = sig.P[:, 1, 1]
    return pd.DataFrame.from_dict(df)


def export_prepped_format(
    data: RawBiaxFormat,
    tag: CycleTypes,
    kin: Kinematics,
    sig: Kinetics,
    erg: Energy,
    shear: Vec[f64],
) -> pd.DataFrame:
    df = dict()
    df["Time_S"] = data.time
    df["XSize_um"] = data.XSize_um.astype(int)
    df["YSize_um"] = data.YSize_um.astype(int)
    df["ZSize_um"] = 1000.0 * sig.ZSize_mm
    df["XDisplacement_um"] = df["XSize_um"] - df["XSize_um"][0]
    df["YDisplacement_um"] = df["YSize_um"] - df["YSize_um"][0]
    df["XForce_mN"] = data.XForce_mN
    df["YForce_mN"] = data.YForce_mN
    df["Temperature"] = data.Temperature
    for i in range(4):
        df[f"X{i+1}"] = data.coord[:, 0, i]
    for i in range(4):
        df[f"Y{i+1}"] = data.coord[:, 1, i]
    df["ShearAngleDeg"] = shear
    df["J"] = kin.J
    df["F11"] = kin.F[:, 0, 0]
    df["F12"] = kin.F[:, 0, 1]
    df["F21"] = kin.F[:, 1, 0]
    df["F22"] = kin.F[:, 1, 1]
    df["C11"] = kin.F[:, 0, 0]
    df["C12"] = kin.F[:, 0, 1]
    df["C21"] = kin.F[:, 1, 0]
    df["C22"] = kin.F[:, 1, 1]
    df["t11"] = sig.sigma[:, 0, 0]
    df["t12"] = sig.sigma[:, 0, 1]
    df["t21"] = sig.sigma[:, 1, 0]
    df["t22"] = sig.sigma[:, 1, 1]
    df["S11"] = sig.S[:, 0, 0]
    df["S12"] = sig.S[:, 0, 1]
    df["S21"] = sig.S[:, 1, 0]
    df["S22"] = sig.S[:, 1, 1]
    df["dE11"] = erg.dE[:, 0, 0]
    df["dE12"] = erg.dE[:, 0, 1]
    df["dE21"] = erg.dE[:, 1, 0]
    df["dE22"] = erg.dE[:, 1, 1]
    df["dH11"] = erg.dH[:, 0, 0]
    df["dH12"] = erg.dH[:, 0, 1]
    df["dH21"] = erg.dH[:, 1, 0]
    df["dH22"] = erg.dH[:, 1, 1]
    df["dW"] = erg.dW
    df["W"] = erg.W
    df["precond"] = tag.precond
    df["preload"] = tag.preload
    df["stretch"] = tag.stretch
    df["recover"] = tag.recover
    df["equibx"] = tag.equibx
    df["last_cycle"] = tag.last_cycle
    df["last_eb"] = tag.last_eb
    df["fitting"] = tag.fitting
    df["plotting"] = tag.plotting
    df["relax"] = tag.relax
    df["creep"] = tag.creep
    return pd.DataFrame.from_dict(df)
