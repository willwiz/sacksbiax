__all__ = ["parse_specimen"]
from glob import glob
import os
import re
from ..parsers import *
from ..datatypes import BXProtocol, SpecimenInfo
from .fio import *


def get_dir_structure(name: str):
    return {
        i: s
        for i, s in enumerate(
            [w for w in sorted(glob(f"{name}/*")) if os.path.isdir(w)]
        )
    }


def parse_directory_name(name: str):
    words = os.path.basename(name).split("-")
    if len(words) == 3:
        i, x, y = words
        return BXProtocol(
            name, int(i), int(x), int(y), f"{int(x)}-{int(y)} tension save cycle"
        )
    elif len(words) == 4:
        i, x, y, t = words
        return BXProtocol(
            name, int(i), int(x), int(y), f"{int(x)}-{int(y)}-{t} tension save cycle"
        )
    else:
        raise ValueError(
            f">>>[FATAL]: protocol founder name ({name}) does not match i-x-y, or i-x-y-t"
        )


def get_specimen_dim(name: str):
    dim_file = os.path.join(name, "size.txt")
    if not os.path.isfile(dim_file):
        raise ValueError(f">>>[FATAL]: {dim_file} not found!")
    with open(dim_file, "r") as f:
        lines = [line.strip() for line in f if line]
    if len(lines) != 4:
        raise ValueError(f">>>[FATAL]: number of lines in {dim_file} != 4")
    alignment = re.match(r"x-(\w+)", lines[3])
    if not alignment:
        raise ValueError(f">>>[FATAL]: line 4 does not match pattern x-[axis]")
    if alignment.group(1) != "axial":
        raise NotImplementedError(
            f">>>[FATAL]: x-axis is not axial, code not implemented for this"
        )
    parsed_dims = [re.match(r"(\w)=(\d+(?:\.\d+)?)", s) for s in lines[:3]]
    dims: dict[str, float] = {d.group(1): float(d.group(2)) for d in parsed_dims if d}
    if set(dims) != {"x", "y", "h"}:
        raise ValueError(
            f">>>[FATAL]: dimensions found, {set(dims)}, not consistent with assumption"
        )
    return (dims["x"], dims["y"], dims["h"])


def find_first_test(prot: dict[int, BXProtocol]) -> str:
    temp = re.compile(r"(\w+)_1(\s+?).bx")
    for _, p in sorted(prot.items()):
        first_test = [
            s for s in glob(f"{p.d}/*_1*.bx") if temp.match(os.path.basename(s))
        ]
        if len(first_test) == 1:
            return first_test[0]
        elif len(first_test) >= 1:
            raise ValueError(
                f">>>[FATAL]: multiple files found that could be the first test: {first_test}"
            )
    raise ValueError(f">>>[FATAL]: No file found that could be the first test")


def get_initial_free_floating(
    prot: dict[int, BXProtocol], dims: tuple[float, float, float]
):
    first_test = find_first_test(prot)
    data = import_bxfile(first_test)
    dims = (
        dims[0] * data.stretch_x[0],
        dims[1] * data.stretch_y[0],
        dims[2] / data.stretch_x[0] / data.stretch_y[0],
    )
    x_ref = np.array(
        [getattr(data, f"x{SACKS_NODE_ORDER[i] + 1}")[0] for i in range(4)], dtype=float
    )
    y_ref = np.array(
        [getattr(data, f"y{SACKS_NODE_ORDER[i] + 1}")[0] for i in range(4)], dtype=float
    )
    return dims, x_ref, y_ref


def parse_specimen(name: str):
    prots = {i: parse_directory_name(k) for i, k in get_dir_structure(name).items()}
    dims = get_specimen_dim(name)
    dims, x_ref, y_ref = get_initial_free_floating(prots, dims)
    return SpecimenInfo(len(prots), prots, dims, x_ref, y_ref)
