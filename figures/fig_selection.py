"""Figure: selection rules at a fixed budget (theory only).

No simulation data is needed; everything is evaluated from the closed forms, so
this figure can be produced before any run finishes.

Usage:
    python fig_selection.py [--eps 0.05] [--tau 0.3] [--out-dir figures_selection]
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sparse_kick_formulas import kick, kicked_fraction, S_delta

Delta = 0.5

# ------------------------------------------------------------------ #
# All-to-all machinery
# ------------------------------------------------------------------ #
def phi_tau(R0, K, tau):
    R0 = max(min(R0, 1.0), 0.0)
    if R0 < 1e-14:
        return 0.0
    beta = K - 2*Delta
    u0 = R0**2
    if abs(beta) < 1e-14:
        u = u0/(1 + K*u0*tau)
    else:
        E = np.exp(beta*tau)
        u = beta*u0*E/(beta + K*u0*(E - 1))
    return np.sqrt(max(u, 0.0))


def delta_for_budget(R, c):
    """Threshold that selects exactly the fraction c of the population at state R."""
    if c >= 1.0:
        return 0.0
    f = lambda d: kicked_fraction(R, d) - c
    if f(0.0) <= 0:
        return 0.0
    return brentq(f, 0.0, 1.0 - 1e-12, xtol=1e-14)


def gain(c, R):
    """First-order gain of the threshold rule over random selection at budget c."""
    d = delta_for_budget(R, c)
    S = S_delta(R, 0.0)
    return S_delta(R, d)/(kicked_fraction(R, d)*S)


def gain_exact(c, R, eps):
    """Per-kick ratio from the exact map rather than its first-order form.

    Panel (b) compares this with the steady state so that both sides use the
    same map and the comparison isolates the saturation of the fixed point.
    The exact ratio can sit a few percent above the first-order ceiling of
    panel (a), which is the O(eps^2) term that the first-order argument drops.
    """
    d = delta_for_budget(R, c)
    num = kick(R, eps, d) - R
    den = c*(kick(R, eps, 0.0) - R)
    return num/den if abs(den) > 1e-15 else np.nan


def ceiling(R):
    """Largest gain any selection rule can achieve: |sin phi| <= 1 over the mean 2S(R)."""
    return 1.0/(2.0*S_delta(R, 0.0))


def fixed_point(step, R0=0.6, tol=1e-13, itmax=20000):
    R = R0
    for _ in range(itmax):
        Rn = min(max(step(R), 0.0), 1.0)
        if abs(Rn - R) < tol:
            return Rn
        R = Rn
    return R


def steady(K, eps, tau, c, mode):
    """Pre-kick steady state for one rule. mode in {none, adaptive, random, full}.

    'full' is no longer used by any panel; it is kept because the outcome panel
    that used it may be wanted for the supplement.
    """
    if mode == 'none':
        R = fixed_point(lambda R: phi_tau(R, K, tau))
        return phi_tau(R, K, tau)
    if mode == 'full':
        step = lambda R: kick(phi_tau(R, K, tau), eps, 0.0)
    elif mode == 'adaptive':
        def step(R):
            Rf = phi_tau(R, K, tau)
            return kick(Rf, eps, delta_for_budget(Rf, c))
    else:                                   # random static at budget p = c
        def step(R):
            Rf = phi_tau(R, K, tau)
            return (1 - c)*Rf + c*kick(Rf, eps, 0.0)
    R = fixed_point(step)
    return phi_tau(R, K, tau)


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--tau', type=float, default=0.3)
    ap.add_argument('--out-dir', default='figures_selection')
    ap.add_argument('--name', default='fig_selection')
    args = ap.parse_args()
    eps, tau = args.eps, args.tau
    Kc = 2*Delta

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    cmap = plt.get_cmap('viridis')

    # ---------------- (a) gain and its ceiling ----------------
    ax = axes[0]
    Rs = [0.2, 0.4, 0.7, 0.9]
    cs = np.geomspace(0.01, 1.0, 200)
    for i, R in enumerate(Rs):
        col = cmap(0.12 + 0.72*i/(len(Rs)-1))
        ax.plot(cs, [gain(c, R) for c in cs], '-', color=col, lw=1.8,
                label=fr'$R={R}$')
        ax.axhline(ceiling(R), ls=':', color=col, lw=1.1)
    ax.axhline(1.0, ls='--', color='0.45', lw=1.2)
    ax.text(0.985, 1.0, 'random static', ha='right', va='bottom',
            fontsize=8, color='0.35', transform=ax.get_yaxis_transform())
    ax.set_xscale('log')
    ax.set_xlabel(r'budget $c$', fontsize=12)
    ax.set_ylabel(r'gain over random, $S_\delta/(\sigma_\delta S)$', fontsize=11)
    ax.set_title('(a) Per-kick gain and its ceiling', fontsize=12)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.set_ylim(bottom=0.9)

    # ---------------- (b) erosion at the fixed point ----------------
    ax = axes[1]
    Krs = [0.9, 1.2, 2.0, 3.0]
    cs_b = np.geomspace(0.01, 1.0, 45)
    for i, Kr in enumerate(Krs):
        K = Kr*Kc
        col = cmap(0.12 + 0.72*i/(len(Krs)-1))
        Rn = steady(K, eps, tau, 1.0, 'none')
        per, ss = [], []
        for c in cs_b:
            Ra = steady(K, eps, tau, c, 'adaptive')
            Rr = steady(K, eps, tau, c, 'random')
            per.append(gain_exact(c, Ra, eps))
            ss.append((Ra - Rn)/(Rr - Rn) if abs(Rr - Rn) > 1e-12 else np.nan)
        ax.plot(cs_b, per, ':', color=col, lw=1.4)
        ax.plot(cs_b, ss, '-', color=col, lw=1.8, label=fr'$K={Kr}K_c$')
    ax.axhline(1.0, ls='--', color='0.45', lw=1.2)
    ax.set_xscale('log')
    ax.set_xlabel(r'budget $c$', fontsize=12)
    ax.set_ylabel('advantage over random', fontsize=11)
    ax.set_title('(b) Per-kick (dotted) vs steady state (solid)', fontsize=12)
    # Both curves come from the exact map, evaluated at the pre-kick state of
    # the threshold rule's own fixed point; they must coincide as c -> 0, where
    # the response is linear and the advantage is not yet saturated.
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.set_ylim(bottom=0.9)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f'{args.name}_eps{eps:+.3f}_tau{tau:.2f}')
    plt.tight_layout()
    for ext in ('pdf', 'png', 'eps'):
        fig.savefig(f'{stem}.{ext}', dpi=140, bbox_inches='tight')
    print(f'Saved {stem}.{{pdf,png,eps}}')

    # Numbers worth quoting in the text
    print('\nceiling 1/(2S(R)):')
    for R in Rs:
        print(f'  R={R}: {ceiling(R):.3f}   (gain at c=0.3: {gain(0.3, R):.3f})')
    print('\nerosion of the per-kick advantage at the fixed point:')
    for Kr in Krs:
        K = Kr*Kc
        Rn = steady(K, eps, tau, 1.0, 'none')
        for c in (0.3, 0.03):
            Ra = steady(K, eps, tau, c, 'adaptive')
            Rr = steady(K, eps, tau, c, 'random')
            g = gain_exact(c, Ra, eps)
            s = (Ra - Rn)/(Rr - Rn)
            er = (g - s)/(g - 1) if g > 1.0001 else float('nan')
            print(f'  K={Kr}Kc, c={c}: per-kick {g:.3f}, steady {s:.3f}, '
                  f'{100*er:.0f}% erased')


if __name__ == '__main__':
    main()
