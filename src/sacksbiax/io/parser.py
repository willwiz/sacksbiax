__all__ = ["parser"]
from argparse import ArgumentParser

from ..data import LogLevel


parser = ArgumentParser("biaxpp")
parser.add_argument("names", type=str, nargs="+")
parser.add_argument(
    "--log-level", type=str.upper, default="INFO", choices=list(LogLevel.__members__)
)
