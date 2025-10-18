import argparse
import os
import time
from collections import defaultdict
from importlib import import_module
from typing import Dict, List, Tuple

import numpy as np

from ..config import load_yaml
from ..evaluation.metrics import save_metrics
from ..io.dataset import split_traj
from ..models import ensure_models_imported
from ..models.base import MODEL_REGISTRY

if os.environ.get("JETEDMD_DISABLE_PLOTS", "").lower() in {"1", "true", "yes"}:
    plot_model_comparison = None
    plot_model_comparison_timeseries = None
else:
    from ..io.viz import plot_model_comparison, plot_model_comparison_timeseries
SYSTEMS = {
    "dry_friction_oscillator": (
        "dynid_benchmark.systems.a1_dry_friction",
        "DryFrictionOscillator",
    ),
    "bouncing_ball": ("dynid_benchmark.systems.a2_bouncing_ball", "BouncingBall"),
    "ou": ("dynid_benchmark.systems.b1_ou", "OrnsteinUhlenbeck"),
    "doublewell": ("dynid_benchmark.systems.b2_doublewell", "DoubleWellSDE"),
    "mass_spring_input": (
        "dynid_benchmark.systems.c1_lti_mass_spring",
        "MassSpringInput",
    ),
    "duffing_forced": ("dynid_benchmark.systems.c1_duffing", "DuffingForced"),
    "burgers1d": ("dynid_benchmark.systems.d1_burgers", "Burgers1D"),
    "kuramoto_sivashinsky": (
        "dynid_benchmark.systems.d2_kuramoto_sivashinsky",
        "KuramotoSivashinsky",
    ),
    "lorenz63": ("dynid_benchmark.systems.lorenz63", "Lorenz63"),
}


def _resolve_system(system_key: str):
    module_path, class_name = SYSTEMS[system_key]
    module = import_module(module_path)
    return getattr(module, class_name)


def _prepare_observations(
    system_key: str,
    cfg,
    total_T: float,
    r: float,
    snr_db: float,
    seed: int,
) -> Tuple[Dict, Dict]:
    SystemCls = _resolve_system(system_key)
    system = SystemCls(cfg.params)
    true = system.simulate_true(total_T, cfg.dt_true, seed=seed)
    obs = system.sample_observations(
        true,
        cfg.Tfast,
        r,
        snr_db=snr_db,
        jitter_pct=cfg.sampling.get("jitter_pct", 0.0),
        missing_pct=cfg.sampling.get("missing_pct", 0.0),
        outlier_rate=cfg.noise.get("outlier_rate", 0.0),
        seed=seed,
    )
    return true, obs


def _mask_eval_horizon(t: np.ndarray, duration: float) -> np.ndarray:
    if duration is None:
        return slice(None)
    t0 = t[0]
    mask = t - t0 <= duration
    if not np.any(mask):
        return slice(None)
    last_idx = np.where(mask)[0][-1] + 1
    return slice(0, last_idx)


def main():
    ap = argparse.ArgumentParser(
        description="Run multiple models and compare metrics as a function of training data size."
    )
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--outdir", default="runs", help="Directory for outputs")
    ap.add_argument(
        "--models",
        default="edmd,pykoopman_edmd,zero",
        help="Comma separated list of model keys registered in MODEL_REGISTRY",
    )
    ap.add_argument(
        "--time",
        default=None,
        help="Override total simulation time (seconds)"
    )
    ap.add_argument(
        "--eval_duration",
        default=None,
        type=float,
        help="Limit rollout evaluation to this duration (seconds) from the start of the test split",
    )
    ap.add_argument(
        "--plot_dims",
        default=2,
        type=int,
        help="Number of leading state dimensions to plot in the time-series comparison",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ensure_models_imported()
    os.makedirs(args.outdir, exist_ok=True)

    default_T_by_system = {
        "dry_friction_oscillator": 30.0,
        "bouncing_ball": 10.0,
        "ou": 50.0,
        "doublewell": 500.0,
        "mass_spring_input": 120.0,
        "duffing_forced": 600.0,
        "burgers1d": 5.0,
        "kuramoto_sivashinsky": 200.0,
        "lorenz63": 50.0,
    }
    total_T = (
        float(args.time)
        if args.time is not None
        else default_T_by_system.get(cfg.system, 30.0)
    )

    seeds = cfg.seeds or [101]
    r_list = cfg.r_list
    snr_list = cfg.noise.get("SNR_dB", [30])

    tag_root = os.path.splitext(os.path.basename(args.config))[0]
    results = []

    for r in r_list:
        for snr_db in snr_list:
            for seed in seeds:
                tag = f"r{r}_SNR{snr_db}_seed{seed}"
                outdir = os.path.join(args.outdir, tag_root, tag)
                os.makedirs(outdir, exist_ok=True)

                _, obs = _prepare_observations(
                    cfg.system,
                    cfg,
                    total_T,
                    r,
                    snr_db,
                    seed,
                )

                splits = split_traj(
                    obs,
                    cfg.splits["train"],
                    cfg.splits["val"],
                    cfg.splits["test"],
                )

                train = splits["train"]
                test = splits["test"]
                t_train, y_train = train["t"], train["y"]
                u_train = train.get("u")
                t_test, y_test = test["t"], test["y"]
                u_test = test.get("u")

                eval_slice = _mask_eval_horizon(t_test, args.eval_duration)
                t_eval = t_test[eval_slice]
                y_eval = y_test[eval_slice]
                u_eval = u_test[eval_slice] if u_test is not None else None

                if len(t_eval) == 0:
                    print(
                        f"[WARN] evaluation horizon empty for r={r}, SNR={snr_db}, seed={seed}; skipped"
                    )
                    continue

                np.savez(
                    os.path.join(outdir, "data_train.npz"),
                    t=t_train,
                    y=y_train,
                    u=u_train if u_train is not None else [],
                )
                np.savez(
                    os.path.join(outdir, "data_test.npz"),
                    t=t_eval,
                    y=y_eval,
                    u=u_eval if u_eval is not None else [],
                )

                predictions = {}

                for mkey in args.models.split(","):
                    mkey = mkey.strip()
                    if mkey not in MODEL_REGISTRY:
                        print(f"[WARN] model '{mkey}' not registered. Skipped.")
                        continue
                    ModelCls = MODEL_REGISTRY[mkey]
                    model = ModelCls()

                    start_fit = time.perf_counter()
                    try:
                        model.fit(t_train, y_train, u_train)
                    except Exception as err:
                        if mkey.startswith("pykoopman"):
                            err_path = os.path.join(outdir, f"error_{mkey}.txt")
                            with open(err_path, "w", encoding="utf-8") as fh:
                                fh.write(str(err))
                        print(f"[ERROR] model '{mkey}' failed during fit: {err}")
                        continue
                    fit_time = time.perf_counter() - start_fit

                    start_rollout = time.perf_counter()
                    try:
                        x0 = y_eval[0]
                        y_pred = model.rollout(t_eval, x0, u_eval)
                    except Exception as err:
                        print(f"[ERROR] model '{mkey}' failed during rollout: {err}")
                        continue
                    rollout_time = time.perf_counter() - start_rollout

                    rmse = float(np.sqrt(np.mean((y_eval - y_pred) ** 2)))
                    metrics = {
                        "model": mkey,
                        "r": float(r),
                        "snr_db": float(snr_db),
                        "seed": int(seed),
                        "n_train": int(len(t_train)),
                        "n_eval": int(len(t_eval)),
                        "fit_time_sec": fit_time,
                        "rollout_time_sec": rollout_time,
                        "rollout_rmse": rmse,
                    }
                    save_metrics(os.path.join(outdir, f"metrics_{mkey}.json"), metrics)
                    results.append(metrics)
                    predictions[mkey] = y_pred

                if predictions and plot_model_comparison_timeseries is not None:
                    plot_path = os.path.join(outdir, "timeseries_comparison.png")
                    plot_model_comparison_timeseries(
                        t_eval,
                        y_eval,
                        predictions,
                        plot_path,
                        max_dims=max(1, args.plot_dims),
                    )

    if not results:
        print("[WARN] no successful model runs recorded; skipping aggregation")
        return

    summary_dir = os.path.join(args.outdir, tag_root)
    os.makedirs(summary_dir, exist_ok=True)

    by_snr: Dict[float, Dict[str, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for entry in results:
        by_snr[entry["snr_db"]][entry["model"]].append((entry["n_train"], entry["rollout_rmse"]))

    for snr_db, model_series in by_snr.items():
        summary = {}
        for model_name, rows in model_series.items():
            rows.sort(key=lambda x: x[0])
            counts = {}
            for n_train, rmse in rows:
                if n_train not in counts:
                    counts[n_train] = []
                counts[n_train].append(rmse)
            summary[model_name] = [
                {
                    "n_train": n_train,
                    "rmse_mean": float(np.mean(values)),
                    "rmse_std": float(np.std(values)),
                }
                for n_train, values in sorted(counts.items())
            ]

        if plot_model_comparison is not None:
            plot_path = os.path.join(summary_dir, f"comparison_SNR{snr_db}.png")
            plot_model_comparison(summary, plot_path)

    log_path = os.path.join(summary_dir, "comparison_results.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        import json

        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
