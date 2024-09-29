from dataclasses import fields
import numpy as np
from ..data import *


def import_bxfile(name: str):
    raw = np.loadtxt(name)
    return BXStruct(**{k.name: raw[:, i] for i, k in enumerate(fields(BXStruct))})


def convert_bxfile(spec: SpecimenInfo, name: str):
    raw = import_bxfile(name)
    coord = np.empty((len(raw.x1), 2, 4), dtype=float)
    for i in range(4):
        coord[:, 0, i] = getattr(raw, f"x{SACKS_NODE_ORDER[i]+1}")
        coord[:, 1, i] = getattr(raw, f"y{SACKS_NODE_ORDER[i]+1}")
    return RawBiaxFormat(
        raw.time,
        (1000.0 * spec.dim[0]) * raw.stretch_x,
        (1000.0 * spec.dim[1]) * raw.stretch_y,
        9.80665 * raw.load_x,
        9.80665 * raw.load_y,
        np.full_like(raw.time, 37),
        raw.shear,
        coord,
    )
