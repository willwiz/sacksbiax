__all__ = ["parser"]
import argparse
from ..data import (
    FileFormat,
    LogLevel,
    ReferenceStateOption,
    StressMethodOption,
    WriteMode,
)


parser = argparse.ArgumentParser(
    "biaxpp", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("names", type=str, nargs="+")
parser.add_argument(
    "--log-level", type=str.upper, default="INFO", choices=list(LogLevel.__members__)
)
parser.add_argument(
    "--input-format",
    type=str.upper,
    default="AUTO",
    choices=list(FileFormat.__members__),
)
parser.add_argument(
    "--export-format",
    type=str.upper,
    default="CSV",
    choices=list(FileFormat.__members__),
)
parser.add_argument(
    "--write-mode",
    "-w",
    type=str.lower,
    default="w",
    choices=list(WriteMode.__members__),
)
parser.add_argument(
    "--tag",
    type=str,
    default="All data",
)
parser.add_argument(
    "--method",
    type=str.upper,
    default="CAUCHY",
    choices=list(StressMethodOption.__members__),
)
parser.add_argument(
    "--ref",
    type=str.upper,
    default="EVERY",
    choices=list(ReferenceStateOption.__members__),
)
parser.add_argument("--n-cores", "-n", type=int, default=1)
parser.add_argument("--overwrite", action="store_true")
