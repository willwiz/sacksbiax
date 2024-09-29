from glob import glob
import dataclasses as dc
import pandas as pd
from .tools.logging import BasicLogger
from .data import *
from .core.core import *
from .core.io import *
from .sacks import parse_specimen, convert_bxfile


def core_loop(
    name: str,
    def_grad: BiaxialKinematics,
    spec: SpecimenInfo,
    setting: ProgramSettings,
    cycle: int,
    log: BasicLogger,
):
    log.debug(f"Working on cycle {name}")
    data = convert_bxfile(spec, name)
    log.debug(f"Computing kinematics")
    kinematics = compute_kinematics(def_grad, data)
    log.debug(f"Computing kinetics")
    match setting.stress_method:
        case StressMethodOption.CAUCHY:
            kinetics = compute_kinetics_cauchy(spec, kinematics, data)
        case StressMethodOption.PK1:
            kinetics = compute_kinetics_pk1(spec, kinematics, data)
        case StressMethodOption.NOMINAL:
            kinetics = compute_kinetics_nominal(spec, kinematics, data)
    log.debug(f"Computing shear angle")
    shear = compute_shear_angle(kinematics)
    log.debug(f"Finding loading and and unloading points")
    cycle_state = parse_cycle(kinematics, cycle)
    log.debug(f"Compiling data from cycle")
    df = export_kamenskiy_format(data, cycle_state, kinematics, kinetics, shear)
    log.debug(f"Finished processing cycle!")
    return df


def process_protocol(
    test: BXProtocol, spec: SpecimenInfo, setting: ProgramSettings, log: BasicLogger
) -> pd.DataFrame | None:
    cycles = sorted(glob(rf"{test.d}/t_*.bx"))
    if len(cycles) == 0:
        log.info(f"No BX file found in {test.name}. Protocol is skipped.")
        return None
    log.info(f"Working on protocol {test.name}")
    x_ref, y_ref = import_ref_markers(path(test.d, "marker.ref"))
    def_grad = BiaxialKinematics(x_ref, y_ref)
    # cycles = [f"{test.d}/t_1 .bx"]
    df = pd.concat(
        [
            core_loop(c, def_grad, spec, setting, i, log)
            for i, c in enumerate(cycles, start=1)
        ],
        ignore_index=True,
    )
    df["SetName"] = test.name
    df = df[[s.name for s in dc.fields(KamenskiyFormat)]]
    log.debug(f"Finished processing protocol!")
    return df


def main_loop(
    name: str,
    setting: ProgramSettings,
    log: BasicLogger,
):
    ex_name = create_export_name(name, setting)
    if ex_name is None:
        log.info(f"{name} already processed, skipped.")
        return
    log.info(f"Working on specimen {name}")
    spec = parse_specimen(name)
    df = pd.concat(
        [
            process_protocol(t, spec, setting, log)
            for _, t in sorted(spec.tests.items())
        ],
        ignore_index=True,
    )
    log.debug(f"Fixing Time array to always increasing")
    df["Time_S"] = fix_time(df["Time_S"].to_numpy(dtype=float))
    log.info(f"Exporting results to {setting.export_format}: {ex_name}")
    export_bx_dataframe(ex_name, df, setting)
    log.info(f"Processing complete!!!\n")


def main(args: InputArgs, log: BasicLogger):
    for name in args.directory:
        main_loop(name, args.settings, log)


def main_cli(cmd_args: list[str] | None = None):
    args = parse_cmdline_args(cmd_args, method="dir")
    log = BasicLogger(args.loglevel)
    try:
        main(args, log)
    except Exception as e:
        log.exception(e)


if __name__ == "__main__":
    main_cli()
