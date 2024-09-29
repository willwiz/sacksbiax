from ..data import *
import pandas as pd
import numpy as np


def convert_df_2_bx(raw: pd.DataFrame):
    coord = np.empty((len(raw["Time_S"]), 2, 4), dtype=float)
    for i in range(4):
        coord[:, 0, i] = raw[f"X{i+1}"]
        coord[:, 1, i] = raw[f"Y{i+1}"]
    return RawBiaxFormat(
        raw["Time_S"].to_numpy(dtype=float),
        raw["XSize_um"].to_numpy(dtype=float),
        raw["YSize_um"].to_numpy(dtype=float),
        raw["XForce_mN"].to_numpy(dtype=float),
        raw["YForce_mN"].to_numpy(dtype=float),
        raw["Temperature"].to_numpy(dtype=float),
        raw["ShearAngleDeg"].to_numpy(dtype=float),
        coord,
    )


def guess_dim(raw: pd.DataFrame, trys: int = 20):
    for k in range(trys):
        if np.allclose(raw[["XDisplacement_um", "YDisplacement_um"]].iloc[k], [0, 0]):
            Lx0 = raw["XSize_um"].iat[0] / 1000.0
            Ly0 = raw["YSize_um"].iat[0] / 1000.0
            x_iff = raw[[f"X{i+1}" for i in range(4)]].iloc[k].to_numpy(dtype=float)
            y_iff = raw[[f"Y{i+1}" for i in range(4)]].iloc[k].to_numpy(dtype=float)
            break
    else:
        raise ValueError(f"Couldn't not find reference point from {trys} time points.")
    for k in range(trys):
        if raw["XForce_mN"].iat[k] != 0:
            if raw["txx"].iat[k] == 0:
                raise ValueError(f"XForce is None zero but txx is 0")
            Lz0 = raw["XForce_mN"].iat[k] * raw["lx"].iat[k] / Ly0 / raw["txx"].iat[k]
            break
        elif raw["YForce_mN"].iat[k] != 0:
            if raw["tyy"].iat[k] == 0:
                raise ValueError(f"YForce is None zero but tyy is 0")
            Lz0 = raw["YForce_mN"].iat[k] * raw["ly"].iat[k] / Lx0 / raw["tyy"].iat[k]
            break
    else:
        raise ValueError(f"Cannot guess initial thickness from {trys} time points.")
    return (Lx0, Ly0, Lz0), x_iff, y_iff


def get_specimen_info(raw: pd.DataFrame):
    prots = {
        i: BXProtocol("None", i, 1, 1, s) for i, s in enumerate(raw["SetName"].unique())
    }
    dim, x_iff, y_iff = guess_dim(raw)
    return SpecimenInfo(len(prots), prots, dim, x_iff, y_iff)


def find_reference_markers_every(
    raw: pd.DataFrame, trys: int = 100
) -> tuple[Vec[f64], Vec[f64]]:
    trys = min(trys, len(raw) // 2)
    for k in range(1, trys):
        if np.allclose(raw[["lx", "kx", "ky", "ly"]].iloc[k], [1, 0, 0, 1]):
            return (
                raw[[f"X{i+1}" for i in range(4)]].iloc[k].to_numpy(dtype=float),
                raw[[f"Y{i+1}" for i in range(4)]].iloc[k].to_numpy(dtype=float),
            )
    return (
        raw[[f"X{i+1}" for i in range(4)]].iloc[0].to_numpy(dtype=float),
        raw[[f"Y{i+1}" for i in range(4)]].iloc[0].to_numpy(dtype=float),
    )


def find_reference_markers_auto(raw: pd.DataFrame) -> tuple[Vec[f64], Vec[f64]]:
    preconditioning = raw["SetName"].str.contains("precond", case=False)
    preload = raw["Cycle"].str.contains("preload", case=False)
    valid_pts = ~preconditioning & ~preload
    k = raw[valid_pts].index[0]
    return (
        raw[[f"X{i+1}" for i in range(4)]].iloc[k].to_numpy(dtype=float),
        raw[[f"Y{i+1}" for i in range(4)]].iloc[k].to_numpy(dtype=float),
    )


def find_reference_markers_first(raw: pd.DataFrame) -> tuple[Vec[f64], Vec[f64]]:
    return (
        raw[[f"X{i+1}" for i in range(4)]].iloc[0].to_numpy(dtype=float),
        raw[[f"Y{i+1}" for i in range(4)]].iloc[0].to_numpy(dtype=float),
    )
