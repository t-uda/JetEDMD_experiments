import argparse, os, json, time
import numpy as np

from ..config import load_yaml
from ..systems.a1_dry_friction import DryFrictionOscillator
from ..systems.a2_bouncing_ball import BouncingBall
from ..systems.b1_ou import OrnsteinUhlenbeck
from ..systems.b2_doublewell import DoubleWellSDE
from ..systems.c1_lti_mass_spring import MassSpringInput
from ..systems.c1_duffing import DuffingForced
from ..systems.d1_burgers import Burgers1D
from ..systems.d2_kuramoto_sivashinsky import KuramotoSivashinsky

from ..systems.base import DynamicalSystem
from ..io.dataset import split_traj
from ..models.base import MODEL_REGISTRY
from ..models import null  # register
from ..models import sindy_stlsq  # register
from ..evaluation.metrics import rollout_rmse, save_metrics
from ..io.viz import plot_rollout

SYSTEMS = {
    "dry_friction_oscillator": DryFrictionOscillator,
    "bouncing_ball": BouncingBall,
    "ou": OrnsteinUhlenbeck,
    "doublewell": DoubleWellSDE,
    "mass_spring_input": MassSpringInput,
    "duffing_forced": DuffingForced,
    "burgers1d": Burgers1D,
    "kuramoto_sivashinsky": KuramotoSivashinsky,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML")
    ap.add_argument("--outdir", default="runs", help="Output directory")
    ap.add_argument("--models", default="sindy_stlsq,zero", help="Comma-separated model keys")
    ap.add_argument("--time", default=None, help="Override total simulation time (seconds)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    os.makedirs(args.outdir, exist_ok=True)

    # instantiate system
    SystemCls = SYSTEMS[cfg.system]
    Tfast = cfg.Tfast
    seeds = cfg.seeds or [101]
    r_list = cfg.r_list
    snr_list = cfg.noise.get("SNR_dB", [30])
    jitter_pct = cfg.sampling.get("jitter_pct", 0.0)
    missing_pct = cfg.sampling.get("missing_pct", 0.0)
    outlier_rate = cfg.noise.get("outlier_rate", 0.0)

    # default total time per system
    default_T_by_system = {
        "dry_friction_oscillator": 30.0,
        "bouncing_ball": 10.0,
        "ou": 50.0,
        "doublewell": 500.0,
        "mass_spring_input": 120.0,
        "duffing_forced": 600.0,
        "burgers1d": 5.0,
        "kuramoto_sivashinsky": 200.0,
    }

    total_T = float(args.time) if args.time is not None else default_T_by_system.get(cfg.system, 30.0)

    for r in r_list:
        for snr_db in snr_list:
            for seed in seeds:
                tag = f"r{r}_SNR{snr_db}_seed{seed}"
                outdir = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.config))[0], tag)
                os.makedirs(outdir, exist_ok=True)

                # Simulate true
                system = SystemCls(cfg.params)
                true = system.simulate_true(total_T, cfg.dt_true, seed=seed)
                obs = system.sample_observations(true, cfg.Tfast, r,
                                                 snr_db=snr_db,
                                                 jitter_pct=cfg.sampling.get("jitter_pct", 0.0),
                                                 missing_pct=cfg.sampling.get("missing_pct", 0.0),
                                                 outlier_rate=cfg.noise.get("outlier_rate", 0.0),
                                                 seed=seed)

                # Split
                splits = split_traj(obs, cfg.splits["train"], cfg.splits["val"], cfg.splits["test"])
                t_train = splits["train"]["t"]; y_train = splits["train"]["y"]; u_train = splits["train"].get("u")
                t_test = splits["test"]["t"]; y_test = splits["test"]["y"]; u_test = splits["test"].get("u")

                # Save raw data
                np.savez(os.path.join(outdir, "data_train.npz"), t=t_train, y=y_train, u=u_train if u_train is not None else [])
                np.savez(os.path.join(outdir, "data_test.npz"),  t=t_test,  y=y_test,  u=u_test  if u_test  is not None else [])

                # Train & evaluate models
                for mkey in args.models.split(","):
                    mkey = mkey.strip()
                    if mkey not in MODEL_REGISTRY:
                        print(f"[WARN] model '{mkey}' not registered. Skipped.")
                        continue
                    ModelCls = MODEL_REGISTRY[mkey]
                    model = ModelCls()
                    model.fit(t_train, y_train, u_train)

                    # Rollout from test initial condition on test times
                    x0 = y_test[0]
                    y_pred = model.rollout(t_test, x0, u_test)
                    # Metrics
                    metrics = {
                        "rollout_rmse": float(np.sqrt(np.mean((y_test - y_pred)**2))),
                        "n_test": int(len(t_test)),
                        "dt_obs": float(np.mean(np.diff(t_test))),
                        "model": mkey,
                    }
                    save_metrics(os.path.join(outdir, f"metrics_{mkey}.json"), metrics)
                    # Plot
                    plot_rollout(t_test, y_test, y_pred, os.path.join(outdir, f"rollout_{mkey}.png"))

if __name__ == "__main__":
    main()
