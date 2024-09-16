from dataclasses import fields
import numpy as np
from ..data import *
from ..data import LogLevel
from .parser import parser


def parse_cmdline_args(cmd_args: list[str] | None):
    args = parser.parse_args(cmd_args)
    return InputArgs(args.names, LogLevel[args.log_level])


def import_bxfile(name: str):
    raw = np.loadtxt(name)
    return BXStruct(**{k.name: raw[:, i] for i, k in enumerate(fields(BXStruct))})


def convert_bxfile(name: str):
    raw = import_bxfile(name)
    coord = np.empty((len(raw.x1), 2, 4), dtype=float)
    for i in range(4):
        coord[:, 0, i] = getattr(raw, f"x{SACKS_NODE_ORDER[i]+1}")
        coord[:, 1, i] = getattr(raw, f"y{SACKS_NODE_ORDER[i]+1}")
    return SacksFormat(
        raw.time,
        raw.stretch_x,
        raw.stretch_y,
        raw.shear,
        raw.load_x,
        raw.load_y,
        raw.t_x,
        raw.t_y,
        raw.stress_x,
        raw.stress_y,
        coord,
    )
