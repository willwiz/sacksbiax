from dataclasses import fields
from glob import glob
import numpy as np
import pandas as pd
from ..data import *
from ..parsers.parser import parser


def parse_cmdline_args(cmd_args: list[str] | None):
    args = parser.parse_args(cmd_args)
    return InputArgs(
        [s for name in args.names for s in glob(name) if s],
        LogLevel[args.log_level],
        ProgramSettings(
            FileFormat[args.input_format],
            FileFormat[args.export_format],
            MethodOption[args.method],
            args.overwrite,
        ),
    )


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
