import numpy as np
import matplotlib.pyplot as plt

# Predefined style cycles (no explicit colors): line styles + markers
LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "x", "s", "^", "d", "v", ">", "<", "p", "*"]

def plot_rollout(t, y_true, y_pred, out_png):
    plt.figure()
    # Use paired solid vs dashed with distinct markers for accessibility
    for d in range(y_true.shape[1]):
        ls_true = LINESTYLES[d % len(LINESTYLES)]
        ls_pred = LINESTYLES[(d+1) % len(LINESTYLES)]
        mk_true = MARKERS[d % len(MARKERS)]
        mk_pred = MARKERS[(d+3) % len(MARKERS)]
        # true
        plt.plot(t, y_true[:,d], linestyle=ls_true, marker=mk_true, markevery=max(len(t)//25,1),
                 label=f"true[{d}]")
        # pred
        plt.plot(t, y_pred[:,d], linestyle=ls_pred, marker=mk_pred, markevery=max(len(t)//25,1),
                 label=f"pred[{d}]")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("state")
    plt.title("Rollout: true vs pred (linestyle+marker encoded)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_psd(y, fs, out_png):
    plt.figure()
    n = len(y)
    Y = np.fft.rfft(y - np.mean(y))
    f = np.fft.rfftfreq(n, d=1/fs)
    P = (np.abs(Y)**2)/n
    # style with markers
    plt.semilogy(f, P, linestyle="-", marker="o", markevery=max(len(f)//40,1))
    plt.xlabel("frequency")
    plt.ylabel("PSD")
    plt.title("Power Spectral Density (marker encoded)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
