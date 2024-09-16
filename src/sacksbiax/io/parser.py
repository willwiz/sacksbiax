__all__ = ["parser"]
from argparse import ArgumentParser


parser = ArgumentParser("biaxpp")
parser.add_argument("specimen", type=str, nargs="+")
