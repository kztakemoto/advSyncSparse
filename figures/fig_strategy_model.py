"""Figure: strategy comparison on model networks.

Rows are enhancement and suppression. Columns are the budgets p when a single
network is given, or the networks themselves when several are given together
with a single p, which is the layout used for the main text.

Usage:
    # supplementary layout: one network, columns are p
    python fig_strategy_model.py er_enh.pkl er_sup.pkl

    # main-text layout: columns are networks at one budget
    python fig_strategy_model.py er_enh.pkl er_sup.pkl ba_enh.pkl ba_sup.pkl --p 0.3

    python fig_strategy_model.py ../data_gen/strategy_ba_eps+0.050_tau0.30_N1000_deg12.pkl ../data_gen/strategy_ba_eps-0.050_tau0.30_N1000_deg12.pkl --p 0.3

    python fig_strategy_model.py ../data_gen/strategy_er_eps+0.050_tau0.30_N1000_deg12.pkl ../data_gen/strategy_er_eps-0.050_tau0.30_N1000_deg12.pkl ../data_gen/strategy_ba_eps+0.050_tau0.30_N1000_deg12.pkl ../data_gen/strategy_ba_eps-0.050_tau0.30_N1000_deg12.pkl --p 0.3
"""
import sys, os, pickle, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STRATEGY_STYLES = {
    'none':          ('#888888', 'x', ':',  'no kick'),
    'all':           ('#444444', '+', '-',  'all ($p=1$)'),
    'adaptive':      ('#D55E00', 'o', '-',  'top-$p$ adaptive'),
    'random_static': ('#0072B2', 'v', '--', 'random $p$ static'),
    'hub':           ('#882255', 'D', '-.', 'hub $p$ (high degree)'),
    'low_degree':    ('#117733', 'p', '-.', 'low-degree $p$'),
}
STRAT_ORDER = ['all', 'adaptive', 'hub', 'low_degree', 'random_static', 'none']


# ------------------------------------------------------------------ #
# Schema access
#
# Current files store a dict per condition, with one value per realization in
# R_pre_runs / R_post_runs plus summary statistics. Older files stored a bare
# (mean, std) tuple whose R was read immediately after the kick. Both are
# handled, but note that the two are not on the same convention: reading R
# after the kick adds one kick step, which is larger for a strategy that
# selects well, so it inflates the gap between adaptive and random by roughly
# 17% at p=0.3. Analyses default to the pre-kick sample.
# ------------------------------------------------------------------ #
def value(rec, conv):
    """(mean, standard error) for one condition."""
    if isinstance(rec, dict):
        k = 'R_pre' if conv == 'pre' else 'R_post'
        return float(rec[k]), float(rec.get(k + '_sem', 0.0))
    return float(rec[0]), 0.0


def runs(rec, conv):
    """Per-realization values, or None for legacy files."""
    if isinstance(rec, dict):
        k = 'R_pre_runs' if conv == 'pre' else 'R_post_runs'
        if k in rec:
            return np.asarray(rec[k], dtype=float)
    return None


def efficiency(results, key_of, conv, n_boot=4000, seed=0):
    """Random-static efficiency eta = (R_rand - R_none)/(R_adapt - R_none).

    Seeds in the current generator depend on the run index alone, so the three
    strategies share realizations and the ratio can be bootstrapped over run
    indices with the pairing preserved. That is worth doing: eta is a ratio of
    two small differences, and the unpaired error on it is several times larger.
    Returns (eta, lo, hi) with a 68% interval, or (eta, nan, nan) when the
    per-run values are unavailable.
    """
    try:
        rec_n, rec_a, rec_r = (results[key_of(s)] for s in
                               ('none', 'adaptive', 'random_static'))
    except KeyError:
        return float('nan'), float('nan'), float('nan')
    mn, ma, mr = (value(x, conv)[0] for x in (rec_n, rec_a, rec_r))
    den = ma - mn
    eta = (mr - mn)/den if abs(den) > 1e-9 else float('nan')

    vn, va, vr = (runs(x, conv) for x in (rec_n, rec_a, rec_r))
    if vn is None or va is None or vr is None:
        return eta, float('nan'), float('nan')
    m = min(len(vn), len(va), len(vr))
    vn, va, vr = vn[:m], va[:m], vr[:m]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_boot, m))
    d_a = va[idx].mean(axis=1) - vn[idx].mean(axis=1)
    d_r = vr[idx].mean(axis=1) - vn[idx].mean(axis=1)
    ok = np.abs(d_a) > 1e-9
    if ok.sum() < 100:
        return eta, float('nan'), float('nan')
    boot = d_r[ok]/d_a[ok]
    return eta, float(np.percentile(boot, 16)), float(np.percentile(boot, 84))


def draw_panel(ax, results, key_of, xs, conv, p_val):
    """One panel: R against the x axis for every strategy."""
    for strat in STRAT_ORDER:
        col, mk, ls, lbl = STRATEGY_STYLES[strat]
        ys, es = [], []
        for x in xs:
            rec = results.get(key_of(x, strat, p_val))
            if rec is None:
                ys.append(np.nan); es.append(0.0)
            else:
                v, e = value(rec, conv)
                ys.append(v); es.append(e)
        ax.errorbar(xs, ys, yerr=es, marker=mk, ls=ls, color=col,
                    ms=6, lw=1.3, mfc='white', mew=1.3, capsize=2, label=lbl)

def load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    results, p_values = data['results'], data['p_values']
    # Older files did not propagate none/all across p; harmless if already done.
    for (Kr, strat, p), val in list(results.items()):
        if strat in ('none', 'all') and p == 0.3:
            for p_other in p_values:
                results.setdefault((Kr, strat, p_other), val)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkls', nargs='+',
                    help='enhancement/suppression pkl pairs, one pair per network')
    ap.add_argument('--p', type=float, nargs='+', default=None,
                    help='budgets to show (default: all in the file)')
    ap.add_argument('--convention', choices=['pre', 'post'], default='pre')
    ap.add_argument('--out-dir', default='figures_strategy')
    ap.add_argument('--name', default=None, help='output basename')
    args = ap.parse_args()

    if len(args.pkls) % 2:
        raise SystemExit('give pkls in enhancement/suppression pairs')
    pairs = [(load(args.pkls[i]), load(args.pkls[i+1]))
             for i in range(0, len(args.pkls), 2)]
    for enh, sup in pairs:
        if enh['eps'] < 0 or sup['eps'] > 0:
            raise SystemExit('each pair must be given as enhancement then suppression')

    p_values = args.p if args.p is not None else pairs[0][0]['p_values']
    if len(pairs) > 1 and len(p_values) != 1:
        raise SystemExit('with more than one network, select a single --p')

    columns = ([(f"{e.get('network','?').upper()}", i) for i, (e, _) in enumerate(pairs)]
               if len(pairs) > 1 else [(fr'$p={p}$', p) for p in p_values])
    ncol = len(columns)

    fig, axes = plt.subplots(2, ncol, figsize=(4.6*ncol, 7),
                             sharex=True, sharey='row', squeeze=False)
    conv = args.convention

    print(f'random-static efficiency, {conv}-kick convention '
          f'(68% interval from a paired bootstrap):')
    for row, which in enumerate(('enhancement', 'suppression')):
        for ci, (label, sel) in enumerate(columns):
            data = pairs[sel][row] if len(pairs) > 1 else pairs[0][row]
            p_val = p_values[0] if len(pairs) > 1 else sel
            results, K_ratios = data['results'], data['K_ratios']
            ax = axes[row][ci]
            draw_panel(ax, results, lambda x, s, p: (x, s, p), K_ratios, conv, p_val)
            ax.axvline(1, ls=':', color='gray', lw=0.6)
            ax.set_ylim(-0.05, 0.9)
            ax.set_title(label if row == 0 else '', fontsize=11)
            if row == 1:
                ax.set_xlabel(r'$K/K_c$', fontsize=12)
            if ci == 0:
                ax.set_ylabel(r'$R$', fontsize=12)
                ax.text(0.04, 0.95, f"{which} ($\\varepsilon={data['eps']:+.3f}$)",
                        transform=ax.transAxes, va='top', fontsize=9,
                        fontweight='bold')
            if row == 0 and ci == 0:
                ax.legend(frameon=False, fontsize=8, loc='lower right')

            net = data.get('network', '?')
            for Kr in K_ratios:
                e, lo, hi = efficiency(results,
                                       lambda s: (Kr, s, p_val), conv)
                if np.isfinite(e):
                    band = f' [{lo:.2f},{hi:.2f}]' if np.isfinite(lo) else ''
                    print(f'  {which[:3]} {net} K={Kr}Kc p={p_val}: '
                          f'eta={e:.3f}{band}')

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or ('fig_strategy_' +
                         '-'.join(e.get('network', '?') for e, _ in pairs) +
                         f'_p{"-".join(str(p) for p in p_values)}_{conv}')
    stem = os.path.join(args.out_dir, name)
    plt.tight_layout()
    for ext in ('pdf', 'png', 'eps'):
        fig.savefig(f'{stem}.{ext}', dpi=140, bbox_inches='tight')
    print(f'Saved {stem}.{{pdf,png,eps}}')


if __name__ == '__main__':
    main()
