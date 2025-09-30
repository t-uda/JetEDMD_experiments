import os
import numpy as np
import matplotlib.pyplot as plt


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
