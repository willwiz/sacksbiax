from .core.biax import BiaxialKinematics
from .tools.logging import BasicLogger
from .datatypes import *
from .core import *
from .converter.core import (
    convert_df_2_bx,
    find_reference_markers_auto,
    find_reference_markers_every,
    find_reference_markers_first,
    get_specimen_info,
)
import pandas as pd
from concurrent import futures


def compile_protocol_data(
    t: BXProtocol,
    raw: pd.DataFrame,
    spec: SpecimenInfo,
    setting: ProgramSettings,
    log: BasicLogger,
) -> pd.DataFrame | None:
    log.debug(f"Working on cycle {t.name}")
    cycle = raw[raw["SetName"] == t.name]
    data = convert_df_2_bx(cycle)
    log.debug(f"Computing kinematics")
    match setting.ref_state:
        case ReferenceStateOption.AUTO:
            x_ref, y_ref = find_reference_markers_auto(raw)
        case ReferenceStateOption.EVERY:
            x_ref, y_ref = find_reference_markers_every(cycle)
        case ReferenceStateOption.FIRST:
            x_ref, y_ref = find_reference_markers_first(raw)
    def_grad = BiaxialKinematics(x_ref, y_ref)
    kinematics = compute_kinematics(def_grad, data)
    log.debug(f"Computing kinetics using {setting.stress_method}")
    match setting.stress_method:
        case StressMethodOption.CAUCHY:
            kinetics = compute_kinetics_cauchy(spec, kinematics, data)
        case StressMethodOption.PK1:
            kinetics = compute_kinetics_pk1(spec, kinematics, data)
        case StressMethodOption.NOMINAL:
            kinetics = compute_kinetics_nominal(spec, kinematics, data)
    log.debug(f"Computing shear angle")
    shear = compute_shear_angle(kinematics)
    log.debug(f"Compiling data from cycle")
    df = export_kamenskiy_format(data, kinematics, kinetics, shear)
    df["SetName"] = t.name
    df["Cycle"] = cycle["Cycle"]
    df = df[[s.name for s in dc.fields(KamenskiyFormat)]]
    log.debug(f"Finished processing cycle!")
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
    raw = import_bx_dataframe(name, setting.input_format)
    spec = get_specimen_info(raw)
    df = pd.concat(
        [
            compile_protocol_data(t, raw, spec, setting, log)
            for _, t in sorted(spec.tests.items())
        ],
        ignore_index=True,
    )
    log.debug(f"Fixing Time array to always increasing")
    df["Time_S"] = fix_time(df["Time_S"].to_numpy(dtype=float))
    log.info(f"Exporting results to {setting.export_format}: {ex_name}")
    export_bx_dataframe(ex_name, df, setting)
    log.info(f"{name} complete!!!\n")


def main(args: InputArgs, log: BasicLogger):
    if args.settings.cores > 1:
        future_pool = dict()
        with futures.ProcessPoolExecutor(args.settings.cores) as exec:
            for name in args.directory:
                future_pool[
                    exec.submit(main_loop, name=name, setting=args.settings, log=log)
                ] = name
            for future in futures.as_completed(future_pool):
                try:
                    future.result()
                except KeyboardInterrupt:
                    print("canceling jobs, please wait")
                    exec.shutdown(wait=True, cancel_futures=True)
                except Exception as e:
                    print(f"Error on {future_pool[future]}")
                    raise e
    else:
        for name in args.directory:
            main_loop(name, args.settings, log)


def main_cli(cmd_args: list[str] | None = None):
    args = parse_cmdline_args(cmd_args)
    log = BasicLogger(args.loglevel)
    try:
        main(args, log)
    except Exception as e:
        log.exception(e)


if __name__ == "__main__":
    main_cli()
