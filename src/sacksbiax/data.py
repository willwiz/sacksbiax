__all__ = [
    "BXStruct",
    "KamenskiyFormat",
    "SpecimenInfo",
    "SacksProtocol",
    "InputArgs",
    "path",
    "SacksFormat",
    "SACKS_NODE_ORDER",
]
import os
import dataclasses as dc
import enum
from .types import *
from typing import Final, Literal, get_args


def path(*names: str) -> str:
    return os.path.join(*[s for s in names if s])


RISK_OPTIONS = Literal["Sex", "HTN", "DM", "CAD", "DYS"]
RISK_FACTORS: Final[list[str]] = list(get_args(RISK_OPTIONS))


SACKS_NODE_ORDER: Final[dict[int, int]] = {0: 2, 1: 1, 2: 3, 3: 0}


class CycleState(enum.IntEnum):
    Preload = 0
    Stretch = 1
    Hold = 2
    Recover = 3


@dc.dataclass(slots=True)
class InputArgs:
    directory: list[str]


@dc.dataclass(slots=True)
class SacksProtocol:
    d: str
    i: int
    x: int
    y: int
    name: str


@dc.dataclass(slots=True)
class SpecimenInfo:
    n_test: int
    tests: dict[int, SacksProtocol]
    dim: tuple[float, float, float]
    x_iff: Vec[f64]
    y_iff: Vec[f64]


@dc.dataclass(slots=True)
class BXStruct:
    time: Vec[f64]
    stretch_x: Vec[f64]
    stretch_y: Vec[f64]
    shear: Vec[f64]
    load_x: Vec[f64]
    load_y: Vec[f64]
    t_x: Vec[f64]
    t_y: Vec[f64]
    stress_x: Vec[f64]
    stress_y: Vec[f64]
    x1: Vec[f64]
    x2: Vec[f64]
    x3: Vec[f64]
    x4: Vec[f64]
    x5: Vec[f64]
    x6: Vec[f64]
    x7: Vec[f64]
    x8: Vec[f64]
    x9: Vec[f64]
    y1: Vec[f64]
    y2: Vec[f64]
    y3: Vec[f64]
    y4: Vec[f64]
    y5: Vec[f64]
    y6: Vec[f64]
    y7: Vec[f64]
    y8: Vec[f64]
    y9: Vec[f64]


@dc.dataclass(slots=True)
class SacksFormat:
    time: Vec[f64]
    stretch_x: Vec[f64]
    stretch_y: Vec[f64]
    shear: Vec[f64]
    load_x: Vec[f64]
    load_y: Vec[f64]
    t_x: Vec[f64]
    t_y: Vec[f64]
    stress_x: Vec[f64]
    stress_y: Vec[f64]
    coord: MatV[f64]


@dc.dataclass(slots=True)
class KamenskiyFormat:
    SetName: Vec[char]
    Cycle: Vec[char]
    Time_S: Vec[f64]
    XSize_um: Vec[f64]
    YSize_um: Vec[f64]
    XDisplacement_um: Vec[f64]
    YDisplacement_um: Vec[f64]
    XForce_mN: Vec[f64]
    YForce_mN: Vec[f64]
    Temperature: Vec[f64]
    X1: Vec[f64]
    Y1: Vec[f64]
    X2: Vec[f64]
    Y2: Vec[f64]
    X3: Vec[f64]
    Y3: Vec[f64]
    X4: Vec[f64]
    Y4: Vec[f64]
    lx: Vec[f64]
    txx: Vec[f64]
    ly: Vec[f64]
    tyy: Vec[f64]
    ShearAngleDeg: Vec[f64]
    kx: Vec[f64]
    txy: Vec[f64]
    ky: Vec[f64]
    tyx: Vec[f64]


@dc.dataclass(slots=True)
class Kinematics:
    F: MatV[f64]
    Finv: MatV[f64]
    C: MatV[f64]
    J: Vec[f64]


@dc.dataclass(slots=True)
class Kinetics:
    sigma: MatV[f64]
    P: MatV[f64]
    S: MatV[f64]
