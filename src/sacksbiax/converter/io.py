import numpy as np
import pandas as pd
from glob import glob
from ..data import *
from ..parsers import parser


def import_bx_dataframe(name: str, fmt: FileFormat) -> pd.DataFrame:
    match fmt:
        case FileFormat.CSV:
            raw = pd.read_csv(name)
        case FileFormat.EXCEL:
            raw = pd.concat(pd.read_excel(name, sheet_name=None), ignore_index=True)
    return raw.rename(columns=BIAX_DATA_ALIASES)


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
