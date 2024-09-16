from glob import glob
import dataclasses as dc
import pandas as pd
from .data import *
from .io import parse_cmdline_args
from .core.indexing import parse_specimen
from .core.core import *


def process_protocol(
    test: SacksProtocol, spec: SpecimenInfo, loglevel: str = "INFO"
) -> pd.DataFrame:
    print(f"[{loglevel:5}]>>Working on protocol {test.name}")
    x_ref, y_ref = import_ref_markers(path(test.d, "marker.ref"))
    def_grad = BiaxialKinematics(x_ref, y_ref)
    # cycles = [f"{test.d}/t_1 .bx"]
    cycles = sorted(glob(rf"{test.d}/t_*.bx"))
    df = pd.concat(
        [core_loop(c, def_grad, spec, i) for i, c in enumerate(cycles)],
        ignore_index=True,
    )
    df["SetName"] = test.name
    df = df[[s.name for s in dc.fields(KamenskiyFormat)]]
    print(f"[{loglevel:5}]>>Finished processing protocol!")
    return df


def core_loop(
    name: str,
    def_grad: BiaxialKinematics,
    spec: SpecimenInfo,
    cycle: int,
    loglevel: str = "INFO",
):
    print(f"[{loglevel:5}]>>Working on cycle {name}")
    data = convert_bxfile(name)
    print(f"[{loglevel:5}]>>Computing kinematics")
    kinematics = compute_kinematics(def_grad, data)
    print(f"[{loglevel:5}]>>Computing kinetics")
    kinetics = compute_kinetics(spec, kinematics, data)
    print(f"[{loglevel:5}]>>Computing shear angle")
    shear = compute_shear_angle(kinematics)
    print(f"[{loglevel:5}]>>Finding loading and and unloading points")
    cycle_state = parse_cycle(kinematics)
    print(f"[{loglevel:5}]>>Compiling data from cycle")
    df = export_kamenskiy_format(
        spec, data, cycle_state, kinematics, kinetics, shear, cycle
    )
    print(f"[{loglevel:5}]>>Finished processing cycle!")
    return df


def main_loop(name: str, loglevel: str = "INFO"):
    spec = parse_specimen(name)
    print(f"[{loglevel:5}]>>Working on specimen {name}")
    df = pd.concat(
        [process_protocol(t, spec) for _, t in sorted(spec.tests.items())],
        ignore_index=True,
    )
    print(f"[{loglevel:5}]>>Fixing Time array to always increasing")
    df["Time_S"] = fix_time(df["Time_S"].to_numpy(dtype=float))
    print(f"[{loglevel:5}]>>Exporting results to excel")
    df.to_excel(path(name, "01 - All data.xlsx"), index=False)
    print(f"[{loglevel:5}]>>Processing complete!!!\n\n")


def main(args: InputArgs):
    for name in args.directory:
        main_loop(name)


def main_cli(cmd_args: list[str] | None = None):
    args = parse_cmdline_args(cmd_args)
    main(args)


if __name__ == "__main__":
    main_cli()
