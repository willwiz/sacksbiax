from glob import glob
from typing import Literal
import numpy as np
import pandas as pd
from scipy import interpolate
from ..datatypes import *
from ..parsers.parser import parser


def parse_cmdline_args(
    cmd_args: list[str] | None, method: Literal["file", "dir"] = "file"
):
    args = parser.parse_args(cmd_args)
    match method:
        case "file":
            names = [s for name in args.names for s in glob(name) if os.path.isfile(s)]
        case "dir":
            names = [s for name in args.names for s in glob(name) if os.path.isdir(s)]
    return InputArgs(
        names,
        LogLevel[args.log_level],
        ProgramSettings(
            FileFormat[args.input_format],
            FileFormat[args.export_format],
            WriteMode[args.write_mode],
            StressMethodOption[args.method],
            ReferenceStateOption[args.ref],
            args.tag,
            args.n_cores,
            args.overwrite,
        ),
    )


def repair_array_by_interpolation(time: Vec[f64], serie: pd.Series):
    x = serie.apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if ~np.isnan(x).any():
        return x
    valid_pos = np.where(~np.isnan(x))
    fi = interpolate.interp1d(time[valid_pos], x[valid_pos])
    return fi(time)


def create_export_name(
    name: str, setting: ProgramSettings, arg_type: Literal["file", "dir"] = "file"
) -> str | None:
    match arg_type:
        case "file":
            folder = os.path.dirname(name)
        case "dir":
            folder = name
    match setting.stress_method:
        case StressMethodOption.CAUCHY:
            ex_name = path(folder, f"{setting.tag} - corrected")
        case StressMethodOption.PK1 | StressMethodOption.NOMINAL:
            ex_name = path(folder, f"{setting.tag} - raw")
    match setting.export_format:
        case FileFormat.CSV | FileFormat.AUTO:
            ex_name = ex_name + ".csv"
        case FileFormat.EXCEL:
            ex_name = ex_name + ".xlsx"
    if os.path.isfile(ex_name) and not setting.overwrite:
        return None
    return ex_name


def import_bx_dataframe(name: str, fmt: FileFormat) -> pd.DataFrame:
    if fmt is FileFormat.AUTO:
        ext = os.path.splitext(name)[1]
        match ext:
            case ".xlsx" | ".xls":
                fmt = FileFormat.EXCEL
            case ".csv":
                fmt = FileFormat.CSV
            case _:
                raise ValueError(f"File extension {ext} not recognized.")
    match fmt:
        case FileFormat.CSV:
            raw = pd.read_csv(name)
        case FileFormat.EXCEL:
            raw = pd.concat(pd.read_excel(name, sheet_name=None), ignore_index=True)
    time = raw["Time_S"].to_numpy(dtype=np.float64)
    for k in raw.columns[3:]:
        raw[k] = repair_array_by_interpolation(time, raw[k])
    return raw.rename(columns=BIAX_DATA_ALIASES)


def export_bx_dataframe(
    ex_name: str,
    df: pd.DataFrame,
    setting: ProgramSettings,
) -> None:
    match setting.export_format:
        case FileFormat.CSV | FileFormat.AUTO:
            df.to_csv(ex_name, index=False, mode=setting.export_mode.value)
        case FileFormat.EXCEL:
            df.to_excel(ex_name, index=False, engine="xlsxwriter")
