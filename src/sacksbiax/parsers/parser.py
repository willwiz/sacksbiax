__all__ = ["parser"]
from argparse import ArgumentParser
from ..data import FileFormat, LogLevel, MethodOption


parser = ArgumentParser("biaxpp")
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
    "--method", type=str.upper, default="CAUCHY", choices=list(MethodOption.__members__)
)
parser.add_argument("--n-cores", "-n", type=int, default=1)
parser.add_argument("--overwrite", action="store_true")
