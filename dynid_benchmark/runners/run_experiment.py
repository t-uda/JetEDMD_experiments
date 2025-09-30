import argparse, os, numpy as np
from ..config import load_yaml
from ..systems.a1_dry_friction import DryFrictionOscillator
from ..systems.a2_bouncing_ball import BouncingBall
from ..systems.b1_ou import OrnsteinUhlenbeck
from ..systems.b2_doublewell import DoubleWellSDE
from ..systems.c1_lti_mass_spring import MassSpringInput
from ..systems.c1_duffing import DuffingForced
from ..systems.d1_burgers import Burgers1D
from ..systems.d2_kuramoto_sivashinsky import KuramotoSivashinsky
from ..io.dataset import split_traj
from ..models import sindy_stlsq  # register
from ..models import edmd         # register
from ..models import implicit_pi  # register (SINDy-PI & implicit)
from ..models import neural_ode   # register (PyTorch optional)
from ..models.base import MODEL_REGISTRY
from ..evaluation.metrics import save_metrics, rmse_robust, psd_welch, frf_welch, bode_metrics, lyapunov_rosenstein, event_timing_error
from ..io.viz import plot_rollout as plot_acc

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
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML")
    ap.add_argument("--outdir", default="runs", help="Output directory")
    ap.add_argument("--models", default="sindy_stlsq,edmd,sindy_pi,implicit_sindy,neural_ode", help="Comma-separated model keys")
    ap.add_argument("--time", default=None, help="Override total simulation time (seconds)")
    ap.add_argument("--eval-frf", action="store_true", help="If set and input is available, compute FRF(Bode) metrics/plots")
    ap.add_argument("--eval-psd", action="store_true", help="Compute PSD of true vs predicted (dim 0)")
    ap.add_argument("--eval-lyap", action="store_true", help="Estimate largest Lyapunov exponent (Rosenstein) from dim 0")
    ap.add_argument("--eval-events", action="store_true", help="For bouncing_ball: compute event timing errors")
    args=ap.parse_args()

    cfg=load_yaml(args.config)
    os.makedirs(args.outdir, exist_ok=True)
    SystemCls = SYSTEMS[cfg.system]
    default_T = {
        "dry_friction_oscillator": 30.0, "bouncing_ball": 10.0, "ou": 50.0,
        "doublewell": 500.0, "mass_spring_input": 120.0, "duffing_forced": 600.0,
        "burgers1d": 5.0, "kuramoto_sivashinsky": 200.0,
    }
    total_T = float(args.time) if args.time is not None else default_T.get(cfg.system, 30.0)

    for r in cfg.r_list:
        # Accept either "SNR_dB" or yaml-escaped key
        snr_values = cfg.noise.get("SNR_dB", cfg.noise.get("SNR_D\\B".replace("\\",""), [30]))
        if not isinstance(snr_values, (list,tuple)):
            snr_values = [snr_values]
        for snr_db in snr_values:
            for seed in (cfg.seeds or [101]):
                tag = f"r{r}_SNR{snr_db}_seed{seed}"
                outdir = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.config))[0], tag)
                os.makedirs(outdir, exist_ok=True)

                system = SystemCls(cfg.params)
                true = system.simulate_true(total_T, cfg.dt_true, seed=seed)
                obs  = system.sample_observations(true, cfg.Tfast, r, snr_db=snr_db,
                                                  jitter_pct=cfg.sampling.get("jitter_pct",0.0),
                                                  missing_pct=cfg.sampling.get("missing_pct",0.0),
                                                  outlier_rate=cfg.noise.get("outlier_rate",0.0),
                                                  seed=seed)

                splits = split_traj(obs, cfg.splits["train"], cfg.splits["val"], cfg.splits["test"])
                t_tr=splits["train"]["t"]; y_tr=splits["train"]["y"]; u_tr=splits["train"].get("u")
                t_te=splits["test"]["t"];  y_te=splits["test"]["y"];  u_te=splits["test"].get("u")

                np.savez(os.path.join(outdir,"data_train.npz"), t=t_tr, y=y_tr, u=(u_tr if u_tr is not None else []))
                np.savez(os.path.join(outdir,"data_test.npz"),  t=t_te, y=y_te, u=(u_te if u_te is not None else []))

                preds=[]; labels=[]
                for mkey in args.models.split(","):
                    mkey=mkey.strip()
                    if mkey not in MODEL_REGISTRY or MODEL_REGISTRY[mkey] is None:
                        print(f"[WARN] model '{mkey}' not available. Skipped."); continue
                    model = MODEL_REGISTRY[mkey]()
                    try:
                        model.fit(t_tr, y_tr, u_tr)
                        y_hat = model.rollout(t_te, y_te[0], u_te)

                        # Robust metrics
                        metrics={"model": mkey}
                        metrics["rollout_rmse"] = rmse_robust(y_te, y_hat)
                        metrics["diverged"] = bool(np.any(~np.isfinite(y_hat)) or np.max(np.abs(y_hat))>1e6)

                        # Optional PSD (dim 0)
                        if args.eval_psd and y_te.shape[1] >= 1:
                            fT, PT = psd_welch(t_te, y_te[:,0])
                            fH, PH = psd_welch(t_te, y_hat[:,0])
                            PHi = np.interp(fT, fH, PH[:,0] if PH.ndim==2 else PH)
                            PTc = np.clip(PT[:,0] if PT.ndim==2 else PT, 1e-12, None)
                            PHc = np.clip(PHi, 1e-12, None)
                            metrics["psd_log_l2"] = float(np.mean((np.log(PTc) - np.log(PHc))**2))

                        # Optional FRF
                        if args.eval_frf and u_te is not None:
                            fT, HT = frf_welch(t_te, y_te, u_te)
                            fH, HH = frf_welch(t_te, y_hat, u_te)
                            bm = bode_metrics(fT, HT, fH, HH)
                            metrics.update(bm)

                        # Optional Lyapunov (on dim 0 of prediction)
                        if args.eval_lyap and y_hat.shape[1] >= 1:
                            ly = lyapunov_rosenstein(t_te, y_hat[:,0])
                            metrics.update({f"lyap_{k}": v for k,v in ly.items()})

                        # Optional event timing (bouncing_ball on position=dim 0)
                        if args.eval_events and "bouncing_ball" in cfg.system and y_hat.shape[1] >= 1:
                            ev = event_timing_error(t_te, y_te[:,0], t_te, y_hat[:,0], level=0.0, direction="down")
                            metrics.update(ev)

                        save_metrics(os.path.join(outdir, f"metrics_{mkey}.json"), metrics)
                        preds.append(y_hat); labels.append(mkey)
                    except Exception as e:
                        print(f"[WARN] model '{mkey}' failed: {e}")

                # Plot accessible overlay if any predictions
                if preds:
                    plot_acc(t_te, y_te, preds, os.path.join(outdir, "rollout_compare.png"))
                else:
                    plot_acc(t_te, y_te, y_te, os.path.join(outdir, "rollout_compare.png"))

if __name__ == "__main__":
    main()
