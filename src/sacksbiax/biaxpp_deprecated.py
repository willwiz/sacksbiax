from .core.biax import BiaxialKinematics
from .tools.logging import BasicLogger
from .data import *
from .core import *
from .converter.core import convert_df_2_bx, get_specimen_info
import pandas as pd
from concurrent import futures


def compile_protocol_data(
    t: BXProtocol,
    raw: pd.DataFrame,
    spec: SpecimenInfo,
    def_grad: BiaxialKinematics,
    setting: ProgramSettings,
    log: BasicLogger,
) -> pd.DataFrame | None:
    cycle = raw[raw["SetName"] == t.name]
    data = convert_df_2_bx(cycle)
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
    log.debug(f"Finding loading and and unloading points")
    cycle_state = cycle["Cycle"]
    log.debug(f"Compiling data from cycle")
    df = export_kamenskiy_format(data, cycle_state, kinematics, kinetics, shear)
    df["SetName"] = t.name
    df = df[[s.name for s in dc.fields(KamenskiyFormat)]]
    log.debug(f"Finished processing cycle!")
    return df


def find_reference_markers(raw: pd.DataFrame) -> tuple[Vec[f64], Vec[f64]]:
    preconditioning = raw["SetName"].str.contains("precond", case=False)
    preload = raw["Cycle"].str.contains("preload", case=False)
    valid_pts = ~preconditioning & ~preload
    k = raw[valid_pts].index[0]
    return (
        raw[[f"X{i+1}" for i in range(4)]].iloc[k].to_numpy(dtype=float),
        raw[[f"Y{i+1}" for i in range(4)]].iloc[k].to_numpy(dtype=float),
    )


def main_loop(
    name: str,
    setting: ProgramSettings,
    log: BasicLogger,
):
    ex_name = create_export_name(name, setting)
    if not ex_name:
        log.info(f"{name} already processed, skipped.")
        return
    log.info(f"Working on specimen {name}")
    raw = import_bx_dataframe(name, setting.input_format)
    spec = get_specimen_info(raw)
    def_grad = BiaxialKinematics(*find_reference_markers(raw))
    df = pd.concat(
        [
            compile_protocol_data(t, raw, spec, def_grad, setting, log)
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
