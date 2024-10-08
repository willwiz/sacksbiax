# __all__ = [
#     "LogLevel",
#     "ExportFormat",
#     "BXStruct",
#     "KamenskiyFormat",
#     "SpecimenInfo",
#     "SacksProtocol",
#     "InputArgs",
#     "path",
#     "SacksFormat",
#     "RawBiaxFormat",
#     "SACKS_NODE_ORDER",
# ]
import os
import dataclasses as dc
import enum
from .types import *
from typing import Final, Literal


def path(*names: str) -> str:
    return os.path.join(*[s for s in names if s])


# RISK_OPTIONS = Literal["Sex", "HTN", "DM", "CAD", "DYS"]
# RISK_FACTORS: Final[list[str]] = list(get_args(RISK_OPTIONS))


SACKS_NODE_ORDER: Final[dict[int, int]] = {0: 2, 1: 1, 2: 3, 3: 0}

BIAX_DATA_ALIASES: dict[str, str] = {
    "Shear_angle_deg": "ShearAngleDeg",
    "StressXX_kPa": "txx",
    "StressXY_kPa": "txy",
    "StressYX_kPa": "tyx",
    "StressYY_kPa": "tyy",
}


class LogLevel(enum.IntEnum):
    NULL = 0
    FATAL = 1
    ERROR = 2
    WARN = 3
    INFO = 4
    DEBUG = 5


class CycleState(enum.IntEnum):
    Preload = 0
    Stretch = 1
    Hold = 2
    Recover = 3


class FileFormat(enum.StrEnum):
    EXCEL = "EXCEL"
    CSV = "CSV"
    AUTO = "AUTO"


class WriteMode(enum.StrEnum):
    w = "w"
    wb = "wb"


class StressMethodOption(enum.StrEnum):
    CAUCHY = "CAUCHY"
    PK1 = "PK1"
    NOMINAL = "NOMINAL"


class ReferenceStateOption(enum.StrEnum):
    AUTO = "AUTO"
    EVERY = "EVERY"
    FIRST = "FIRST"


@dc.dataclass(slots=True)
class ProgramSettings:
    input_format: FileFormat
    export_format: FileFormat
    export_mode: WriteMode
    stress_method: StressMethodOption
    ref_state: ReferenceStateOption
    tag: str
    cores: int
    overwrite: bool


@dc.dataclass(slots=True)
class InputArgs:
    directory: list[str]
    loglevel: LogLevel
    settings: ProgramSettings


@dc.dataclass(slots=True)
class BXProtocol:
    d: str
    i: int
    x: int
    y: int
    name: str


@dc.dataclass(slots=True)
class SpecimenInfo:
    n_test: int
    tests: dict[int, BXProtocol]
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
    Pxx: Vec[f64]
    Pxy: Vec[f64]
    Pyx: Vec[f64]
    Pyy: Vec[f64]


@dc.dataclass(slots=True)
class RawBiaxFormat:
    time: Vec[f64]
    XSize_um: Vec[f64]
    YSize_um: Vec[f64]
    XForce_mN: Vec[f64]
    YForce_mN: Vec[f64]
    Temperature: Vec[f64]
    ShearAngleDeg: Vec[f64]
    coord: MatV[f64]


@dc.dataclass(slots=True)
class CycleTypes:
    precond: Vec[i32]
    equibx: Vec[i32]
    relax: Vec[i32]
    creep: Vec[i32]
    preload: Vec[i32]
    stretch: Vec[i32]
    recover: Vec[i32]
    last_cycle: Vec[i32]
    last_eb: Vec[i32]
    plotting: Vec[i32]
    fitting: Vec[i32]


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
    ZSize_mm: Vec[f64]


@dc.dataclass(slots=True)
class Energy:
    dE: MatV[f64]
    dH: MatV[f64]
    dW: Vec[f64]
    W: Vec[f64]


@dc.dataclass(slots=True)
class SpecMetaData:
    Name: str
    data: str
    duration: float
    size: tuple[float, float, float]
    alpha: tuple[float, float]
    relax: bool
    creep: bool
    cidx: int
    cmax: Vec[f64]


@dc.dataclass(slots=True)
class SpecHealthData:
    SetName: Vec[char]
    Cycles: Vec[char]
    Start: Vec[char]
    End: Vec[char]
    Fit: Vec[i32]
    Plot: Vec[i32]
    tear: Vec[i32]
    dift: Vec[i32]
    hyst: Vec[f64]
    cidx: int
    cmax: Vec[f64]


@dc.dataclass(slots=True)
class SpecDataFormat:
    SetName: Vec[char]
    Cycle: Vec[char]
    Time_S: Vec[f64]
    XSize_um: Vec[f64]
    YSize_um: Vec[f64]
    ZSize_um: Vec[f64]
    XDisplacement_um: Vec[f64]
    YDisplacement_um: Vec[f64]
    XForce_mN: Vec[f64]
    YForce_mN: Vec[f64]
    Temperature: Vec[f64]
    X1: Vec[f64]
    X2: Vec[f64]
    X3: Vec[f64]
    X4: Vec[f64]
    Y1: Vec[f64]
    Y2: Vec[f64]
    Y3: Vec[f64]
    Y4: Vec[f64]
    ShearAngleDeg: Vec[f64]
    J: Vec[f64]
    F11: Vec[f64]
    F12: Vec[f64]
    F21: Vec[f64]
    F22: Vec[f64]
    C11: Vec[f64]
    C12: Vec[f64]
    C21: Vec[f64]
    C22: Vec[f64]
    t11: Vec[f64]
    t12: Vec[f64]
    t21: Vec[f64]
    t22: Vec[f64]
    S11: Vec[f64]
    S12: Vec[f64]
    S21: Vec[f64]
    S22: Vec[f64]
    dE11: Vec[f64]
    dE12: Vec[f64]
    dE21: Vec[f64]
    dE22: Vec[f64]
    dH11: Vec[f64]
    dH12: Vec[f64]
    dH21: Vec[f64]
    dH22: Vec[f64]
    dW: Vec[f64]
    W: Vec[f64]
    precond: Vec[i32]
    preload: Vec[i32]
    stretch: Vec[i32]
    recover: Vec[i32]
    equibx: Vec[i32]
    last_cycle: Vec[i32]
    last_eb: Vec[i32]
    fitting: Vec[i32]
    plotting: Vec[i32]
    relax: Vec[i32]
    creep: Vec[i32]
