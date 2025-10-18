import os
import numpy as np
import matplotlib.pyplot as plt


_STYLE_CYCLE = [
    ("o", "#1f77b4", "-"),
    ("s", "#ff7f0e", "--"),
    ("^", "#2ca02c", "-."),
    ("D", "#d62728", ":"),
    ("v", "#9467bd", "-"),
    ("X", "#8c564b", "--"),
    ("P", "#e377c2", "-."),
]


def plot_rollout(t, y_true, y_pred, out_png):
    plt.figure()
    for d in range(y_true.shape[1]):
        plt.plot(t, y_true[:, d], label=f"true_{d}")
        plt.plot(t, y_pred[:, d], linestyle="--", label=f"pred_{d}")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("state")
    plt.title("Rollout: true vs pred")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_psd(y, fs, out_png):
    plt.figure()
    n = len(y)
    Y = np.fft.rfft(y - np.mean(y))
    f = np.fft.rfftfreq(n, d=1 / fs)
    P = (np.abs(Y) ** 2) / n
    plt.semilogy(f, P)
    plt.xlabel("frequency")
    plt.ylabel("PSD")
    plt.title("Power Spectral Density")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_model_comparison(summary, out_png):
    plt.figure()
    valid_items = [(name, rows) for name, rows in sorted(summary.items()) if rows]
    if not valid_items:
        plt.close()
        return
    for idx, (model_name, rows) in enumerate(valid_items):
        n_train = np.array([item["n_train"] for item in rows], dtype=float)
        mean = np.array([item["rmse_mean"] for item in rows], dtype=float)
        std = np.array([item["rmse_std"] for item in rows], dtype=float)
        marker, color, linestyle = _STYLE_CYCLE[idx % len(_STYLE_CYCLE)]
        markevery = max(1, len(n_train) // 6)
        plt.plot(
            n_train,
            mean,
            marker=marker,
            color=color,
            linestyle=linestyle,
            label=model_name,
            linewidth=1.5,
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=color,
            markevery=markevery,
            zorder=3 + idx,
        )
        lower = mean - std
        upper = mean + std
        plt.fill_between(
            n_train,
            lower,
            upper,
            color=color,
            alpha=0.18,
            zorder=1 + idx,
        )
    plt.xlabel("number of training observations")
    plt.ylabel("rollout RMSE")
    plt.title("Model comparison vs data volume")
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_png)
    plt.close()


def plot_model_comparison_timeseries(t, y_true, predictions, out_png, max_dims=2):
    y_true = np.asarray(y_true)
    if y_true.ndim == 1:
        y_true = y_true[:, None]

    plot_dims = max(1, min(int(max_dims), y_true.shape[1]))
    fig, axes = plt.subplots(plot_dims, 1, sharex=True, figsize=(7, 2.5 * plot_dims))
    axes = np.atleast_1d(axes)

    model_items = [(name, predictions[name]) for name in sorted(predictions.keys())]
    for dim in range(plot_dims):
        ax = axes[dim]
        ax.plot(t, y_true[:, dim], label="true", color="k", linewidth=1.5)
        for idx, (name, values) in enumerate(model_items):
            series = np.asarray(values)
            if series.ndim == 1:
                series = series[:, None]
            if series.shape[1] <= dim:
                continue
            marker, color, linestyle = _STYLE_CYCLE[idx % len(_STYLE_CYCLE)]
            ax.plot(
                t,
                series[:, dim],
                label=name,
                color=color,
                linestyle=linestyle,
                linewidth=1.3,
            )
        ax.set_ylabel(f"state[{dim}]")
        if dim == 0:
            ax.legend(loc="best")

    axes[-1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
