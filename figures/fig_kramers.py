#!/usr/bin/env python3
"""
escape statistics under threshold kicks.

Usage
    python fig_kramers.py ../data_gen/kramers_eps-0.070--0.050--0.030_tau0.30_K2.0-3.0.pkl --criterion R_late --level 0.05
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ----------------------------------------------------------------------
# separatrix
#
# Eq. (13) of the notes. If sparse_kick_formulas.py exports an equivalent
# routine, import it here instead and delete this function.
# ----------------------------------------------------------------------
def separatrix(eps, K, delta, Delta, tau):
    """Closed-form separatrix R*(delta), Eq. (13)."""
    lam = np.exp((K - 2.0 * Delta) * tau / 2.0)
    s = np.sqrt(1.0 - delta ** 2)
    den = (lam * np.cos(eps) - 1.0
           + 2.0 * lam * (1.0 - np.cos(eps)) / np.pi
           * (delta * s + np.arcsin(delta)))
    return 2.0 * abs(np.sin(eps)) * s / (np.pi * den)


# ----------------------------------------------------------------------
# escape probability and Kramers slope
# ----------------------------------------------------------------------
def escape_probability(record, criterion, level):
    """Fraction of realizations counted as escaped."""
    if criterion == "R_late":
        vals = np.asarray(record["R_finals"], dtype=float)
    elif criterion in record:
        vals = np.asarray(record[criterion], dtype=float)
    else:
        raise KeyError(
            "criterion %r not stored in this pkl; available keys: %s"
            % (criterion, sorted(record.keys()))
        )
    return float(np.mean(vals >= level))


def kramers_slope(results, eps, K_ratio, delta, Ns, criterion, level,
                  floor_guard=None, min_events=0):
    """
    Weighted straight-line fit of log[-log(1 - P_esc)] against N.

    Returns (slope, intercept, N_used, y_used, sigma_y_used, n_dropped) or
    None when fewer than three sizes have a resolved escape probability.

    floor_guard: drop sizes below the level-dependent finite-size floor,
    N < pi / (4 * level^2), where the mean order parameter of random phases
    already exceeds the level.

    min_events: require at least this many escapes and at least this many
    non-escapes.  With M realizations the smallest measurable probability is
    1 / M, so at large N the estimate is censored rather than noisy.  On the
    data reported in the paper the cut changes no slope ratio by more than
    0.5% and raises the rms residual of the collapse, so the default is 0,
    which keeps every size with 0 < P < 1; the option is kept as a check.
    """
    N_used, y_used, w_used, sy_used = [], [], [], []
    n_dropped = 0
    N_floor = np.pi / (4.0 * level ** 2) if floor_guard else 0.0

    for N in Ns:
        if N < N_floor:
            continue
        rec = results.get((eps, K_ratio, N, delta))
        if rec is None:
            continue
        P = escape_probability(rec, criterion, level)
        n = int(rec.get("n_realizations", len(rec["R_finals"])))
        if P <= 0.0 or P >= 1.0:
            continue
        if min(P * n, (1.0 - P) * n) < min_events:
            n_dropped += 1
            continue
        y = np.log(-np.log(1.0 - P))
        sP = np.sqrt(P * (1.0 - P) / n)
        sy = sP / ((1.0 - P) * abs(np.log(1.0 - P)))
        N_used.append(N)
        y_used.append(y)
        sy_used.append(sy)
        w_used.append(1.0 / sy ** 2)

    if len(N_used) < 3:
        return None

    x = np.asarray(N_used, float)
    y = np.asarray(y_used, float)
    w = np.asarray(w_used, float)
    W = w.sum()
    mx = (w * x).sum() / W
    my = (w * y).sum() / W
    Sxx = np.sum(w * (x - mx) ** 2)
    slope = np.sum(w * (x - mx) * (y - my)) / Sxx
    intercept = my - slope * mx
    slope_err = 1.0 / np.sqrt(Sxx)
    return (slope, intercept, x, y, np.asarray(sy_used, float), n_dropped,
            slope_err)


# ----------------------------------------------------------------------
# panels
# ----------------------------------------------------------------------
def panel_linearity(ax, fits, K_ratio, eps_list, deltas, colours, markers):
    """Panel (a): Kramers plot at one coupling."""
    for (eps, dl), f in sorted(fits.items()):
        slope, intercept, x, y, sy = f[:5]
        colour = colours[dl]
        marker = markers[eps]
        ax.errorbar(x, y, yerr=sy, fmt=marker, ms=4, mfc="none",
                    color=colour, lw=0, elinewidth=0.7, capsize=1.5)
        xf = np.linspace(x.min(), x.max(), 2)
        ax.plot(xf, intercept + slope * xf, "-", color=colour, lw=0.9,
                alpha=0.8)

    handles = [plt.Line2D([], [], color=colours[dl], lw=1.4,
                          label=r"$\delta = %.1f$" % dl) for dl in deltas]
    handles += [plt.Line2D([], [], color="0.35", lw=0, marker=markers[e],
                           mfc="none", ms=5,
                           label=r"$\epsilon = %.2f$" % e) for e in eps_list]
    ax.legend(handles=handles, fontsize=7, frameon=False, ncol=2,
              loc="upper right")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\log[-\log(1 - P_{\rm esc})]$")
    ax.set_title(r"(a) Kramers plot")

    # the fitted lines are drawn across the full width of each condition and
    # run far below the data; bound the axis by the measurements instead, and
    # leave headroom at the top for the legend
    lo = min((f[3] - f[4]).min() for f in fits.values())
    hi = max((f[3] + f[4]).max() for f in fits.values())
    span = hi - lo
    ax.set_ylim(lo - 0.06 * span, hi + 0.06 * span)


def panel_collapse(ax, fits, rstars, K_ratio, eps_list, deltas, colours,
                   markers):
    """Panel (b): fitted slopes against R*(delta)^2 at one coupling."""
    xs, ys = [], []
    for (eps, dl), f in sorted(fits.items()):
        Rs = rstars[(eps, dl)]
        x, y = Rs ** 2, abs(f[0])
        xs.append(x)
        ys.append(y)
        ax.plot(x, y, markers[eps], ms=6, mfc=colours[dl], mec=colours[dl])

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    c = float((xs * ys).sum() / (xs * xs).sum())
    local = ys / xs
    rms_rel = float(np.std(local) / np.mean(local))

    xf = np.linspace(0.0, xs.max() * 1.08, 2)
    ax.plot(xf, c * xf, "k--", lw=1.0,
            label="slope $= %.2f\\,R^{*2}$" % c)

    handles = [plt.Line2D([], [], lw=0, marker="o", mfc=colours[dl],
                          mec=colours[dl], ms=6,
                          label=r"$\delta = %.1f$" % dl) for dl in deltas]
    handles += [plt.Line2D([], [], color="0.35", lw=0, marker=markers[e],
                           mfc="none", ms=5,
                           label=r"$\epsilon = %.2f$" % e) for e in eps_list]
    handles += [plt.Line2D([], [], color="k", ls="--", lw=1.0,
                           label="fit at this $K$")]
    ax.legend(handles=handles, fontsize=7, frameon=False, ncol=2,
              loc="upper left")
    ax.set_xlabel(r"$R^{*}(\delta)^{2}$")
    ax.set_ylabel("Kramers slope")
    ax.set_title(r"(b) Collapse")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    # four or five ticks, so the four-decimal labels do not run together
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    return c, rms_rel


def panel_ratio(ax, fits, K_ratio, eps_list, deltas, colours, markers,
                inset=False):
    """Slope ratio against 1 - delta^2, Eq. (14), with no fitted constant."""
    grid = np.linspace(0.0, max(deltas) * 1.05, 200)
    ax.plot(grid, 1.0 - grid ** 2, "k--", lw=1.0, zorder=1,
            label=r"$1-\delta^{2}$")

    for eps in eps_list:
        base = fits.get((eps, 0.0))
        if base is None:
            continue
        s0, e0 = abs(base[0]), base[6]
        xs, ys, es = [], [], []
        for dl in deltas:
            f = fits.get((eps, dl))
            if f is None or dl == 0.0:
                continue
            r = abs(f[0]) / s0
            xs.append(dl)
            ys.append(r)
            es.append(r * np.sqrt((f[6] / abs(f[0])) ** 2 + (e0 / s0) ** 2))
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt=markers[eps], ms=5, mfc="none",
                        color="0.25", lw=0, elinewidth=0.8, capsize=1.8,
                        zorder=3, label=r"$\epsilon = %.2f$" % eps)

    ax.set_xlabel(r"$\delta$", fontsize=9 if inset else None)
    ax.set_ylabel(r"$\kappa(\delta)/\kappa(0)$", fontsize=9 if inset else None)
    ax.set_xlim(0.0, max(deltas) * 1.05)
    ax.set_ylim(0.0, 1.15)
    if inset:
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=5.5, frameon=False, loc="lower left",
                  handlelength=1.2, borderpad=0.1, labelspacing=0.25)
    else:
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        ax.set_title(r"(c) Slope ratio against $1-\delta^{2}$")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", help="Kramers escape pkl")
    ap.add_argument("--K-ratio", type=float, default=3.0,
                    help="coupling used for panels (b) and (c)")
    ap.add_argument("--criterion", default="R_late",
                    choices=["R_late", "R_max", "t_cross"])
    ap.add_argument("--level", type=float, default=0.05)
    ap.add_argument("--min-events", type=int, default=0,
                    help="require at least this many escapes and non-escapes "
                         "per size; 0 keeps every size with 0 < P < 1")
    ap.add_argument("--floor-guard", action="store_true",
                    help="drop N below the finite-size floor pi/(4 level^2)")
    ap.add_argument("--ratio-inset", action="store_true",
                    help="put the slope-ratio test inside panel (b) instead of "
                         "giving it a panel of its own")
    ap.add_argument("--no-ratio", action="store_true",
                    help="omit the slope-ratio test altogether")
    ap.add_argument("--collapse-only", action="store_true",
                    help="emit the collapse panel alone, for the supplement")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = pickle.load(open(args.pkl, "rb"))
    results = d["results"]
    Ns = d["Ns"]
    deltas = list(d["deltas"])
    eps_list = list(d["eps_list"])
    Delta = d["Delta"]
    tau = d["tau"]
    Kc = d["Kc"]
    Kr = args.K_ratio

    if Kr not in d["K_ratios"]:
        raise SystemExit("K/Kc = %s not in this dataset (have %s)"
                         % (Kr, d["K_ratios"]))

    cmap = plt.get_cmap("viridis")
    colours = {dl: cmap(0.08 + 0.75 * i / max(1, len(deltas) - 1))
               for i, dl in enumerate(deltas)}
    marker_cycle = ["o", "s", "^", "D", "v", "P"]
    markers = {e: marker_cycle[i % len(marker_cycle)]
               for i, e in enumerate(eps_list)}

    fits, rstars = {}, {}
    for eps in eps_list:
        for dl in deltas:
            f = kramers_slope(results, eps, Kr, dl, Ns, args.criterion,
                              args.level, floor_guard=args.floor_guard,
                              min_events=args.min_events)
            if f is None:
                print("  skipped: eps=%+.2f delta=%.1f (fewer than three "
                      "resolved sizes)" % (eps, dl))
                continue
            fits[(eps, dl)] = f
            rstars[(eps, dl)] = separatrix(eps, Kr * Kc, dl, Delta, tau)

    if not fits:
        raise SystemExit("no condition had a resolved escape probability")

    figs = []  # list of (suffix, fig), one entry per output figure

    if args.collapse_only:
        fig_b, ax_c = plt.subplots(1, 1, figsize=(4.2, 3.6))
        c, rms_rel = panel_collapse(ax_c, fits, rstars, Kr, eps_list, deltas,
                                    colours, markers)
        figs.append(('collapse', fig_b))
    else:
        fig_a, ax_a = plt.subplots(figsize=(4.6, 3.8))
        panel_linearity(ax_a, fits, Kr, eps_list, deltas, colours, markers)
        figs.append(('linearity', fig_a))

        fig_b, ax_c = plt.subplots(figsize=(4.6, 3.8))
        c, rms_rel = panel_collapse(ax_c, fits, rstars, Kr, eps_list, deltas,
                                    colours, markers)
        figs.append(('collapse', fig_b))

        if not args.no_ratio:
            if args.ratio_inset:
                ax_r = ax_c.inset_axes([0.56, 0.12, 0.40, 0.40])
                panel_ratio(ax_r, fits, Kr, eps_list, deltas, colours, markers,
                           inset=True)
            else:
                fig_c, ax_r = plt.subplots(figsize=(4.6, 3.8))
                panel_ratio(ax_r, fits, Kr, eps_list, deltas, colours, markers)
                figs.append(('ratio', fig_c))

    print("\nK = %.1f Kc, criterion %s >= %.2f%s"
          % (Kr, args.criterion, args.level,
             ", floor guard on" if args.floor_guard else ""))
    print("  conditions fitted : %d" % len(fits))
    print("  sizes dropped     : %d (fewer than %d escapes or non-escapes)"
          % (sum(f[5] for f in fits.values()), args.min_events))
    print("  slope / R*^2      : %.3f  (rms relative residual %.3f)"
          % (c, rms_rel))
    print("  R* range          : %.4f to %.4f"
          % (min(rstars.values()), max(rstars.values())))
    print("  slope ratios against 1 - delta^2:")
    for eps in eps_list:
        if (eps, 0.0) not in fits:
            continue
        s0 = fits[(eps, 0.0)][0]
        parts = []
        for dl in deltas:
            if dl == 0.0 or (eps, dl) not in fits:
                continue
            ratio = fits[(eps, dl)][0] / s0
            pred = 1.0 - dl ** 2
            parts.append("d=%.1f %.3f vs %.3f (%+.1f%%)"
                         % (dl, ratio, pred, 100.0 * (ratio - pred) / pred))
        print("    eps=%+.2f  %s" % (eps, " | ".join(parts)))

    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.pkl))[0]
        tag = "collapse" if args.collapse_only else "fig"
        out_base = "%s_%s_K%.1f_%s%.2f" % (stem, tag, Kr, args.criterion,
                                           args.level)
    else:
        out_base = os.path.splitext(args.out)[0]

    print()
    for suffix, fig in figs:
        fig.tight_layout()
        # a single output figure keeps out_base as-is; several figures get
        # a distinguishing suffix each
        stem_i = out_base if len(figs) == 1 else "%s_%s" % (out_base, suffix)
        for ext in (".pdf", ".png", ".eps"):
            fig.savefig(stem_i + ext, bbox_inches="tight")
        plt.close(fig)
        print("wrote %s.{pdf,png,eps}" % stem_i)


if __name__ == "__main__":
    main()
