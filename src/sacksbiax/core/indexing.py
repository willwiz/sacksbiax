__all__ = ["parse_specimen"]
from glob import glob
import os
import re
from ..io import *
from ..data import SacksProtocol, SpecimenInfo


def get_dir_structure(name: str):
    return {
        i: s
        for i, s in enumerate(
            [w for w in sorted(glob(f"{name}/*")) if os.path.isdir(w)]
        )
    }


def parse_directory_name(name: str):
    words = os.path.basename(name).split("-")
    if len(words) != 3:
        raise ValueError(
            f">>>[FATAL]: protocol founder name ({name}) does not match i-x-y"
        )
    i, x, y = [int(s) for s in words]
    return SacksProtocol(name, i, x, y, f"{x}-{y} tension save cycle")


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


def get_initial_free_floating(
    prot: dict[int, SacksProtocol], dims: tuple[float, float, float]
):
    first_test = glob(f"{prot[0].d}/*_1*.bx")
    if len(first_test) != 1:
        raise ValueError(
            f">>>[FATAL]: multiple files found that could be the first test: {first_test}"
        )
    first_test = first_test[0]
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
