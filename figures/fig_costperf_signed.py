"""Figure: cost-performance with enhancement and suppression overlaid.

A signed variant of fig_costperf.py.  Both signs of the kick are drawn against
a single unperturbed reference curve, so that curve is not duplicated across
two figures.

    eps > 0 : solid line, filled marker
    eps < 0 : dashed line, open marker
    no kick : plus, outside the filled/open contrast
    delta   : colour in the transition panels, as in fig_costperf.py

Two modes, chosen by how many networks are supplied.

  One network (the all-to-all figure)
      (a) transition curves    (b) R vs delta    (c) R vs sigma
      colour in (b) and (c) encodes K, as in fig_costperf.py, so a coupling
      used on both sides gets one colour and is told apart by line style

  Several networks (the network figure)
      (a) ER   (b) BA   ...   (last) cost-performance
      colour encodes the network and marker shape encodes K

Usage:
    # all-to-all, several couplings on each side
    python fig_costperf_signed.py \
        --pos ../data_gen/a2a_costperf_eps0.050_tau0.30_N1000.pkl \
        --neg ../data_gen/a2a_costperf_eps-0.050_tau0.30_N1000.pkl \
        --cp-K-pos 0.7 0.9 --cp-K-neg 1.2 2.0

    # ER + BA
    python fig_costperf_signed.py \
        --pos ../data_gen/er_costperf_eps0.050_tau0.30_N1000_deg12.pkl \
              ../data_gen/ba_costperf_eps0.050_tau0.30_N1000_deg12.pkl \
        --neg ../data_gen/er_costperf_eps-0.050_tau0.30_N1000_deg12.pkl \
              ../data_gen/ba_costperf_eps-0.050_tau0.30_N1000_deg12.pkl \
        --cp-K-pos 0.9 --cp-K-neg 2.0

Options:
    --cp-K-pos KR [KR ...]   couplings for the eps > 0 curves (default 0.9)
    --cp-K-neg KR [KR ...]   couplings for the eps < 0 curves (default 2.0)
    --split-cost             separate cost-performance panels for the two signs
    --out-dir DIR            default: figures_costperf
    --convention C           'pre' (default) or 'post'

Datasets are paired by data['network'], so the order of the two lists does not
have to match.
"""
import os, pickle, argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Accessors, imported from fig_costperf.py so the two scripts read the same
# schema versions.  Keep them in sync.
from fig_costperf import (sim_R, sim_R_err, sim_sigma, sim_sigma_err,
                          th_R, th_sigma, palette)

FIXED_D = {0.0: '#222222', 0.5: '#0072B2', 0.8: '#D55E00', 1.0: '#999999'}
FIXED_K = {0.7: '#56B4E9', 0.9: '#009E73', 1.2: '#E69F00', 2.0: '#882255',
           1.5: '#009E73', 3.0: '#882255'}
NET_COL = ['#3b3b98', '#2e8b57', '#b8860b', '#8e44ad']
K_MARKERS = ['o', 's', '^', 'D', 'v']
NOKICK = 1.0

# line style and marker fill carry the sign
SIGN = {'pos': dict(ls='-', mfc=None, mew=1.4),
        'neg': dict(ls='--', mfc='white', mew=1.4)}


def kick_deltas(data):
    """deltas_main without the no-kick entry."""
    return [d for d in data['deltas_main'] if d != NOKICK]


def sign_handles():
    return [Line2D([], [], color='0.3', lw=1.6, ls='-', marker='o', ms=6,
                   mfc='0.3', mew=1.4, label=r'$\epsilon>0$'),
            Line2D([], [], color='0.3', lw=1.6, ls='--', marker='o', ms=6,
                   mfc='white', mew=1.4, label=r'$\epsilon<0$')]


# ------------------------------------------------------------------ #
# Panels
# ------------------------------------------------------------------ #
def transition_panel(ax, pos, neg, letter, conv, show_legend, title=None):
    deltas = kick_deltas(pos)
    colors_d = palette(pos['deltas_main'], fixed=FIXED_D)
    Krs_th = sorted(set(Kr for (Kr, d) in pos['theory_trans']))

    # unperturbed reference, drawn once from the positive-eps file
    if NOKICK in pos['deltas_main']:
        R_th = [th_R(pos['theory_trans'][(float(Kr), NOKICK)], conv)
                for Kr in Krs_th]
        ax.plot(Krs_th, R_th, '-', color=colors_d[NOKICK], lw=1.8, zorder=2)
        recs = [pos['sim_trans'][(Kr, NOKICK)] for Kr in pos['K_ratios_sim']]
        # a plus, so the no-kick points sit outside the filled/open contrast
        # that carries the sign of the kick
        ax.errorbar(pos['K_ratios_sim'], [sim_R(r, conv) for r in recs],
                    yerr=[sim_R_err(r) for r in recs], fmt='+',
                    color=colors_d[NOKICK], ms=7, mew=1.4, capsize=2,
                    ls='none', zorder=3)

    for sign, data in (('pos', pos), ('neg', neg)):
        if data is None:
            continue
        st = SIGN[sign]
        Krs = sorted(set(Kr for (Kr, d) in data['theory_trans']))
        for d in deltas:
            col = colors_d[d]
            R_th = [th_R(data['theory_trans'][(float(Kr), float(d))], conv)
                    for Kr in Krs]
            ax.plot(Krs, R_th, st['ls'], color=col, lw=1.8, zorder=4)
            recs = [data['sim_trans'][(Kr, d)] for Kr in data['K_ratios_sim']]
            ax.errorbar(data['K_ratios_sim'], [sim_R(r, conv) for r in recs],
                        yerr=[sim_R_err(r) for r in recs], fmt='o', color=col,
                        ms=6, mfc=(st['mfc'] or col), mew=st['mew'],
                        capsize=2, ls='none', zorder=5)

    ax.axvline(1, ls=':', color='gray', lw=0.8)
    ax.set_xlabel(r'$K/K_c$', fontsize=12)
    ax.set_title('(%s) %s' % (letter, title or pos.get('network', '?').upper()),
                 fontsize=12)
    ax.set_ylim(0, 0.95)

    if show_legend:
        ax.set_ylabel(r'$R$', fontsize=12)
        h_d = [Line2D([], [], color=colors_d[d], lw=1.8, label=r'$\delta=%g$' % d)
               for d in deltas]
        h_d.append(Line2D([], [], color=colors_d[NOKICK], lw=1.8, marker='+',
                          ms=7, mew=1.4, label='no kick'))
        leg = ax.legend(handles=h_d, frameon=False, fontsize=9, loc='upper left')
        ax.add_artist(leg)
        ax.legend(handles=sign_handles(), frameon=False, fontsize=9,
                 loc='upper center')


def _cp_series(data, sign, Krs, colour_of, marker_of, label_of):
    """Expand one dataset into per-K drawing instructions."""
    out = []
    for j, Kr in enumerate(Krs):
        if Kr not in data['Kr_cp']:
            raise SystemExit('K=%g is not among Kr_cp=%s for %s'
                             % (Kr, data['Kr_cp'], data.get('network', '?')))
        out.append(dict(data=data, sign=sign, Kr=Kr, colour=colour_of(j, Kr),
                        marker=marker_of(j, Kr), label=label_of(data, sign, Kr)))
    return out


def _draw_cp(ax, series, conv, xkey):
    """xkey: 'sigma' for the trade-off panel, 'delta' for R vs delta."""
    for s in series:
        data, st = s['data'], SIGN[s['sign']]
        col, mk = s['colour'], s['marker']

        ds = sorted(d for (k, d) in data['theory'] if k == s['Kr'])
        recs_th = [data['theory'][(s['Kr'], float(d))] for d in ds]
        x_th = ds if xkey == 'delta' else [th_sigma(r, conv) for r in recs_th]
        ax.plot(x_th, [th_R(r, conv) for r in recs_th], st['ls'], color=col,
                lw=1.8, label=s['label'], zorder=2)

        d_s = sorted(d for (k, d) in data['sim_cp'] if k == s['Kr'])
        recs = [data['sim_cp'][(s['Kr'], d)] for d in d_s]
        x_sim = d_s if xkey == 'delta' else [sim_sigma(r) for r in recs]
        xerr = None if xkey == 'delta' else [sim_sigma_err(r) for r in recs]
        ax.errorbar(x_sim, [sim_R(r, conv) for r in recs], xerr=xerr,
                    yerr=[sim_R_err(r) for r in recs], fmt=mk, color=col, ms=6,
                    mfc=(st['mfc'] or col), mew=st['mew'], capsize=2,
                    ls='none', zorder=3)


def delta_panel(ax, series, letter, conv):
    _draw_cp(ax, series, conv, xkey='delta')
    ax.set_xlabel(r'$\delta$', fontsize=12)
    ax.set_ylabel(r'$R$', fontsize=12)
    ax.set_title(r'(%s) Effect of threshold $\delta$' % letter, fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 0.95)
    ax.legend(frameon=False, fontsize=8.5, loc='best')


def cost_panel(ax, series, letter, conv, title='Cost-performance',
               show_ylabel=True):
    _draw_cp(ax, series, conv, xkey='sigma')
    ax.set_xlabel(r'Kicked fraction $\sigma$ (at the kick instant)', fontsize=11)
    if show_ylabel:
        ax.set_ylabel(r'$R$', fontsize=12)
    ax.set_title('(%s) %s' % (letter, title), fontsize=12)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0, 0.95)
    ax.legend(frameon=False, fontsize=8.5, loc='best')


# ------------------------------------------------------------------ #
# Figures
# ------------------------------------------------------------------ #
def make_signed_single(pairs, conv, K_pos, K_neg, split):
    """One network: transition curves, R vs delta, R vs sigma.

    Returns a list of (name_suffix, fig) pairs, one figure per panel.
    """
    pos, neg = pairs[0]
    colors_K = palette(sorted(set(K_pos) | set(K_neg)), fixed=FIXED_K)

    def col_of(j, Kr):
        return colors_K[Kr]

    def mk_of(j, Kr):
        return 'o'

    def lab_of(data, sign, Kr):
        return r'$K=%gK_c$, $\epsilon%s0$' % (Kr, '>' if sign == 'pos' else '<')

    enh = _cp_series(pos, 'pos', K_pos, col_of, mk_of, lab_of)
    sup = _cp_series(neg, 'neg', K_neg, col_of, mk_of, lab_of)

    figs = []

    fig_t, ax_t = plt.subplots(figsize=(5.4, 4.4))
    transition_panel(ax_t, pos, neg, 'a', conv, show_legend=True,
                     title='Transition curves')
    figs.append(('transition', fig_t))

    fig_d, ax_d = plt.subplots(figsize=(5.4, 4.4))
    delta_panel(ax_d, enh + sup, 'b', conv)
    figs.append(('delta', fig_d))

    if split:
        fig_c1, ax_c1 = plt.subplots(figsize=(5.4, 4.4))
        cost_panel(ax_c1, enh, 'c', conv,
                   title=r'Cost-performance, $\epsilon>0$')
        figs.append(('cost_pos', fig_c1))

        fig_c2, ax_c2 = plt.subplots(figsize=(5.4, 4.4))
        cost_panel(ax_c2, sup, 'd', conv,
                   title=r'Cost-performance, $\epsilon<0$')
        figs.append(('cost_neg', fig_c2))
    else:
        fig_c, ax_c = plt.subplots(figsize=(5.4, 4.4))
        cost_panel(ax_c, enh + sup, 'c', conv)
        figs.append(('cost', fig_c))

    return figs


def make_signed_combined(pairs, conv, K_pos, K_neg, split):
    """Several networks: one transition panel each, then cost-performance.

    Returns a list of (name_suffix, fig) pairs, one figure per panel.
    """
    n = len(pairs)

    def series_for(i, data, sign, Krs):
        def lab(d, s, Kr):
            return (r'%s, $\epsilon%s0$, $K=%gK_c$'
                    % (d.get('network', '?').upper(),
                       '>' if s == 'pos' else '<', Kr))
        return _cp_series(data, sign, Krs,
                          lambda j, Kr: NET_COL[i % len(NET_COL)],
                          lambda j, Kr: K_MARKERS[j % len(K_MARKERS)], lab)

    enh, sup = [], []
    for i, (p, q) in enumerate(pairs):
        enh += series_for(i, p, 'pos', K_pos)
        sup += series_for(i, q, 'neg', K_neg)

    figs = []
    for i, (p, q) in enumerate(pairs):
        fig_t, ax_t = plt.subplots(figsize=(5.4, 4.4))
        transition_panel(ax_t, p, q, chr(97 + i), conv, show_legend=True)
        net_name = p.get('network', chr(97 + i))
        figs.append(('transition_%s' % net_name, fig_t))

    if split:
        fig_c1, ax_c1 = plt.subplots(figsize=(5.4, 4.4))
        cost_panel(ax_c1, enh, chr(97 + n), conv,
                   title=r'Cost-performance, $\epsilon>0$')
        figs.append(('cost_pos', fig_c1))

        fig_c2, ax_c2 = plt.subplots(figsize=(5.4, 4.4))
        cost_panel(ax_c2, sup, chr(98 + n), conv,
                   title=r'Cost-performance, $\epsilon<0$')
        figs.append(('cost_neg', fig_c2))
    else:
        fig_c, ax_c = plt.subplots(figsize=(5.4, 4.4))
        cost_panel(ax_c, enh + sup, chr(97 + n), conv)
        figs.append(('cost', fig_c))

    return figs


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(
        description='Cost-performance figure with both signs of the kick.')
    ap.add_argument('--pos', nargs='+', required=True, help='eps > 0 pkls')
    ap.add_argument('--neg', nargs='+', required=True, help='eps < 0 pkls')
    ap.add_argument('--cp-K-pos', type=float, nargs='+', default=[0.9],
                    metavar='KR', help='couplings for the eps > 0 curves')
    ap.add_argument('--cp-K-neg', type=float, nargs='+', default=[2.0],
                    metavar='KR', help='couplings for the eps < 0 curves')
    ap.add_argument('--split-cost', action='store_true')
    ap.add_argument('--out-dir', default='figures_costperf')
    ap.add_argument('--convention', choices=['pre', 'post'], default='pre')
    args = ap.parse_args()

    def load(paths):
        out = {}
        for p in paths:
            with open(p, 'rb') as f:
                d = pickle.load(f)
            out[d.get('network', os.path.basename(p))] = d
        return out

    pos, neg = load(args.pos), load(args.neg)
    only_one = set(pos) ^ set(neg)
    if only_one:
        raise SystemExit('networks present for only one sign: %s'
                         % sorted(only_one))

    order = [k for k in ('a2a', 'er', 'ba') if k in pos] + \
            [k for k in sorted(pos) if k not in ('a2a', 'er', 'ba')]
    pairs = [(pos[k], neg[k]) for k in order]

    for p, q in pairs:
        if p['eps'] < 0 or q['eps'] > 0:
            raise SystemExit('sign mismatch: --pos takes eps>0 files (got %g) '
                             'and --neg eps<0 (got %g)' % (p['eps'], q['eps']))

    build = make_signed_single if len(pairs) == 1 else make_signed_combined
    figs = build(pairs, args.convention, args.cp_K_pos, args.cp_K_neg,
                args.split_cost)

    os.makedirs(args.out_dir, exist_ok=True)
    eps_pos, eps_neg = pairs[0][0]['eps'], pairs[0][1]['eps']
    stem_base = os.path.join(
        args.out_dir,
        'fig_costperf_signed_%s_epspos%.3f_epsneg%.3f'
        % ('-'.join(order), eps_pos, eps_neg))
    if args.split_cost:
        stem_base += '_split'

    for suffix, fig in figs:
        fig.tight_layout()
        stem = '%s_%s' % (stem_base, suffix)
        for ext in ('pdf', 'png', 'eps'):
            fig.savefig('%s.%s' % (stem, ext), dpi=140, bbox_inches='tight')
        plt.close(fig)
        print('Saved %s.{pdf,png,eps}' % stem)


if __name__ == '__main__':
    main()
