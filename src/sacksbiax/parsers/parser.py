__all__ = ["parser"]
import argparse
from ..datatypes import (
    FileFormat,
    LogLevel,
    ReferenceStateOption,
    StressMethodOption,
    WriteMode,
)


parser = argparse.ArgumentParser(
    "main", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("names", type=str, nargs="+")
parser.add_argument(
    "--log-level",
    type=str.upper,
    default="INFO",
    choices=list(LogLevel.__members__),
    help="Logging details",
)
parser.add_argument(
    "--input-format",
    type=str.upper,
    default="AUTO",
    choices=list(FileFormat.__members__),
    help="Auto is based on extension",
)
parser.add_argument(
    "--export-format",
    type=str.upper,
    default="CSV",
    choices=list(FileFormat.__members__),
    help="Auto is CSV",
)
parser.add_argument(
    "--write-mode",
    "-w",
    type=str.lower,
    default="w",
    choices=list(WriteMode.__members__),
    help="wb is binary",
)
parser.add_argument(
    "--tag",
    type=str,
    default="All data",
    help="Prefix for export filename",
)
parser.add_argument(
    "--method",
    type=str.upper,
    default="CAUCHY",
    choices=list(StressMethodOption.__members__),
    help="For processing stress",
)
parser.add_argument(
    "--ref",
    type=str.upper,
    default="EVERY",
    choices=list(ReferenceStateOption.__members__),
    help="For kinematics",
)
parser.add_argument("--n-cores", "-n", type=int, default=1, help="Parallelization")
parser.add_argument(
    "--overwrite", action="store_true", help="Do not skip if export file is found"
)
