from glob import glob
import dataclasses as dc
import pandas as pd
from .tools.logging import BasicLogger
from .data import *
from .io import parse_cmdline_args
from .core.indexing import parse_specimen
from .core.core import *


def core_loop(
    name: str,
    def_grad: BiaxialKinematics,
    spec: SpecimenInfo,
    cycle: int,
    log: BasicLogger,
):
    log.debug(f"Working on cycle {name}")
    data = convert_bxfile(name)
    log.debug(f"Computing kinematics")
    kinematics = compute_kinematics(def_grad, data)
    log.debug(f"Computing kinetics")
    kinetics = compute_kinetics(spec, kinematics, data)
    log.debug(f"Computing shear angle")
    shear = compute_shear_angle(kinematics)
    log.debug(f"Finding loading and and unloading points")
    cycle_state = parse_cycle(kinematics)
    log.debug(f"Compiling data from cycle")
    df = export_kamenskiy_format(
        spec, data, cycle_state, kinematics, kinetics, shear, cycle
    )
    log.debug(f"Finished processing cycle!")
    return df


def process_protocol(
    test: SacksProtocol, spec: SpecimenInfo, log: BasicLogger
) -> pd.DataFrame:
    log.info(f"Working on protocol {test.name}")
    x_ref, y_ref = import_ref_markers(path(test.d, "marker.ref"))
    def_grad = BiaxialKinematics(x_ref, y_ref)
    # cycles = [f"{test.d}/t_1 .bx"]
    cycles = sorted(glob(rf"{test.d}/t_*.bx"))
    df = pd.concat(
        [core_loop(c, def_grad, spec, i, log) for i, c in enumerate(cycles, start=1)],
        ignore_index=True,
    )
    df["SetName"] = test.name
    df = df[[s.name for s in dc.fields(KamenskiyFormat)]]
    log.debug(f"Finished processing protocol!")
    return df


def main_loop(name: str, log: BasicLogger):
    log.info(f"Working on specimen {name}")
    spec = parse_specimen(name)
    df = pd.concat(
        [process_protocol(t, spec, log) for _, t in sorted(spec.tests.items())],
        ignore_index=True,
    )
    log.debug(f"Fixing Time array to always increasing")
    df["Time_S"] = fix_time(df["Time_S"].to_numpy(dtype=float))
    log.info(f"Exporting results to excel")
    df.to_excel(path(name, "01 - All data.xlsx"), index=False)
    log.info(f"Processing complete!!!\n\n")


def main(args: InputArgs, log: BasicLogger):
    for name in args.directory:
        main_loop(name, log)


def main_cli(cmd_args: list[str] | None = None):
    args = parse_cmdline_args(cmd_args)
    log = BasicLogger(args.loglevel)
    try:
        main(args, log)
    except Exception as e:
        log.exception(e)


if __name__ == "__main__":
    main_cli()
