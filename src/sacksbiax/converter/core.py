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


def guess_dim(raw: pd.DataFrame):
    for i in range(10):
        if raw["txx"].iat[i] != 0:
            Lz0 = (
                raw["XForce_mN"].iat[i]
                * raw["lx"].iat[i]
                / raw["YSize_um"].iat[i]
                / raw["txx"].iat[i]
            )
            return (
                raw["XSize_um"].iat[0] / 1000.0,
                raw["YSize_um"].iat[0] / 1000.0,
                Lz0 * 1000.0,
            )
        elif raw["tyy"].iat[i] != 0:
            Lz0 = (
                raw["YForce_mN"].iat[i]
                * raw["ly"].iat[i]
                / raw["XSize_um"].iat[i]
                / raw["tyy"].iat[i]
            )
            return (
                raw["XSize_um"].iat[0] / 1000.0,
                raw["YSize_um"].iat[0] / 1000.0,
                Lz0 * 1000.0,
            )
    raise ValueError("Cannot guess initial thickness from data")


def get_specimen_info(raw: pd.DataFrame):
    prots = {
        i: SacksProtocol("None", i, 1, 1, s)
        for i, s in enumerate(raw["SetName"].unique())
    }
    dim = guess_dim(raw)
    x_iff = raw[[f"X{i+1}" for i in range(4)]].iloc[0].to_numpy(dtype=float)
    y_iff = raw[[f"Y{i+1}" for i in range(4)]].iloc[0].to_numpy(dtype=float)
    return SpecimenInfo(len(prots), prots, dim, x_iff, y_iff)


def find_reference_markers(raw: pd.DataFrame) -> tuple[Vec[f64], Vec[f64]]:
    for k in range(20):
        if (raw[["lx", "ly"]].iloc[k].to_numpy(float) == [1, 1]).all():
            return (
                raw[[f"X{i+1}" for i in range(4)]].iloc[k].to_numpy(dtype=float),
                raw[[f"Y{i+1}" for i in range(4)]].iloc[k].to_numpy(dtype=float),
            )
    return (
        raw[[f"X{i+1}" for i in range(4)]].iloc[0].to_numpy(dtype=float),
        raw[[f"Y{i+1}" for i in range(4)]].iloc[0].to_numpy(dtype=float),
    )
