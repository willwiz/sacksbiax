from .core.biax import BiaxialKinematics
from .converter.io import import_bx_dataframe
from .tools.logging import BasicLogger
from .data import *
from .core.core import (
    compute_kinematics,
    compute_kinetics_cauchy,
    compute_kinetics_pk1,
    compute_shear_angle,
    export_kamenskiy_format,
    fix_time,
)
from .converter.io import parse_cmdline_args
import pandas as pd
import numpy as np
from concurrent import futures


def convert_df_2_bx(raw: pd.DataFrame):
    coord = np.empty((len(raw["Time_S"]), 2, 4), dtype=float)
    for i in range(4):
        coord[:, 0, i] = raw[f"X{i+1}"]
        coord[:, 1, i] = raw[f"Y{i+1}"]
    return RawBiaxFormat(
        raw["Time_S"].to_numpy(dtype=float),
        raw["XSize_um"].to_numpy(dtype=float),
        raw["YSize_um"].to_numpy(dtype=float),
        raw["XForce_mN"].to_numpy(dtype=float),
        raw["YForce_mN"].to_numpy(dtype=float),
        raw["Temperature"].to_numpy(dtype=float),
        raw["ShearAngleDeg"].to_numpy(dtype=float),
        coord,
    )


def guess_dim(raw: pd.DataFrame):
    for i in range(10):
        if raw["txx"].iat[i] != 0:
            Lz0 = (
                raw["XForce_mN"].iat[i]
                * raw["lx"].iat[i]
                / raw["YSize_um"].iat[i]
                / raw["txx"].iat[i]
            )
            return (
                raw["XSize_um"].iat[0] / 1000.0,
                raw["YSize_um"].iat[0] / 1000.0,
                Lz0 * 1000.0,
            )
        elif raw["tyy"].iat[i] != 0:
            Lz0 = (
                raw["YForce_mN"].iat[i]
                * raw["ly"].iat[i]
                / raw["XSize_um"].iat[i]
                / raw["tyy"].iat[i]
            )
            return (
                raw["XSize_um"].iat[0] / 1000.0,
                raw["YSize_um"].iat[0] / 1000.0,
                Lz0 * 1000.0,
            )
    raise ValueError("Cannot guess initial thickness from data")


def get_specimen_info(raw: pd.DataFrame):
    prots = {
        i: SacksProtocol("None", i, 1, 1, s)
        for i, s in enumerate(raw["SetName"].unique())
    }
    dim = guess_dim(raw)
    x_iff = raw[[f"X{i+1}" for i in range(4)]].iloc[0].to_numpy(dtype=float)
    y_iff = raw[[f"Y{i+1}" for i in range(4)]].iloc[0].to_numpy(dtype=float)
    return SpecimenInfo(len(prots), prots, dim, x_iff, y_iff)


def find_reference_markers(raw: pd.DataFrame) -> tuple[Vec[f64], Vec[f64]]:
    for k in range(20):
        if (raw[["lx", "ly"]].iloc[k].to_numpy(float) == [1, 1]).all():
            return (
                raw[[f"X{i+1}" for i in range(4)]].iloc[k].to_numpy(dtype=float),
                raw[[f"Y{i+1}" for i in range(4)]].iloc[k].to_numpy(dtype=float),
            )
    return (
        raw[[f"X{i+1}" for i in range(4)]].iloc[0].to_numpy(dtype=float),
        raw[[f"Y{i+1}" for i in range(4)]].iloc[0].to_numpy(dtype=float),
    )


def compile_protocol_data(
    t: SacksProtocol,
    raw: pd.DataFrame,
    spec: SpecimenInfo,
    setting: ProgramSettings,
    log: BasicLogger,
) -> pd.DataFrame | None:
    cycle = raw[raw["SetName"] == t.name]
    data = convert_df_2_bx(cycle)
    x_ref, y_ref = find_reference_markers(cycle)
    def_grad = BiaxialKinematics(x_ref, y_ref)
    kinematics = compute_kinematics(def_grad, data)
    log.debug(f"Computing kinetics using {setting.method.name}")
    match setting.method:
        case MethodOption.CAUCHY:
            kinetics = compute_kinetics_cauchy(spec, kinematics, data)
        case MethodOption.PK1:
            kinetics = compute_kinetics_pk1(spec, kinematics, data)
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


def main_loop(
    name: str,
    setting: ProgramSettings,
    log: BasicLogger,
):
    match setting.method:
        case MethodOption.CAUCHY:
            ex_name = path(os.path.dirname(name), "All data - corrected")
        case MethodOption.PK1:
            ex_name = path(os.path.dirname(name), "All data - raw")
    match setting.export_format:
        case FileFormat.CSV | FileFormat.AUTO:
            ex_name = ex_name + ".csv"
        case FileFormat.EXCEL:
            ex_name = ex_name + ".xlsx"
    if os.path.isfile(ex_name) and not setting.overwrite:
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
    match setting.export_format:
        case FileFormat.CSV | FileFormat.AUTO:
            log.info(f"Exporting results to CSV: {ex_name}")
            df.to_csv(ex_name, index=False)
        case FileFormat.EXCEL:
            log.info(f"Exporting results to excel")
            df.to_excel(ex_name, index=False, engine="xlsxwriter")
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
