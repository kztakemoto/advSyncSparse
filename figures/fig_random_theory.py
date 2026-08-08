"""Figure: Random-static annealed theory vs simulation (Sec 8 supporting)."""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    with open('../data_gen/random_theory_vs_sim.pkl', 'rb') as f:
        data = pickle.load(f)
    results = data['results']
    
    K_ratios_th = sorted(set(k[1] for k in results if k[0] == 'theory_fps'))
    ps = data['p_values']
    colors_p = {0.1: '#56B4E9', 0.3: '#0072B2', 0.5: '#332288'}
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    
    # (a) Enhancement
    ax = axes[0]
    for p in ps:
        # Theory: take maximum FP (only one for enh)
        K_th = []; R_th = []
        for K in K_ratios_th:
            fps = results.get(('theory_fps', K, 0.05, p), [])
            if fps:
                K_th.append(K); R_th.append(max(fps))
        ax.plot(K_th, R_th, '-', color=colors_p[p], lw=2, label=fr'theory, $p={p}$')
        K_sim = []; R_sim = []; E_sim = []
        for K_ratio in sorted(set(k[1] for k in results if k[0] == 'sim')):
            if ('sim', K_ratio, 0.05, p) in results:
                R, E = results[('sim', K_ratio, 0.05, p)]
                K_sim.append(K_ratio); R_sim.append(R); E_sim.append(E)
        ax.errorbar(K_sim, R_sim, yerr=E_sim, fmt='o', color=colors_p[p],
                    ms=6, mfc='white', mew=1.5, capsize=3, label=fr'sim, $p={p}$')
    ax.axvline(1, ls=':', color='gray', lw=0.8)
    ax.set_xlabel(r'$K/K_c$'); ax.set_ylabel(r'$R_{\rm ss}$')
    ax.set_title(r'(a) Random static, enhancement ($\epsilon=+0.05$)')
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='lower right')
    ax.set_ylim(0, 1); ax.set_xlim(0.3, 3.5)
    
    # (b) Suppression with branches
    ax = axes[1]
    for p in ps:
        K_th = sorted([K for K in K_ratios_th if results.get(('theory_fps', K, -0.05, p))])
        lower_K = []; lower_R = []
        middle_K = []; middle_R = []
        upper_K = []; upper_R = []
        single_K = []; single_R = []
        for K in K_th:
            fps = sorted(results[('theory_fps', K, -0.05, p)])
            if len(fps) == 1:
                single_K.append(K); single_R.append(fps[0])
            elif len(fps) == 3:
                lower_K.append(K); lower_R.append(fps[0])
                middle_K.append(K); middle_R.append(fps[1])
                upper_K.append(K); upper_R.append(fps[2])
            elif len(fps) == 2:
                lower_K.append(K); lower_R.append(fps[0])
                upper_K.append(K); upper_R.append(fps[1])
        if single_K:
            ax.plot(single_K, single_R, '-', color=colors_p[p], lw=2,
                    label=fr'theory, $p={p}$')
        if upper_K:
            label = fr'theory, $p={p}$' if not single_K else None
            ax.plot(upper_K, upper_R, '-', color=colors_p[p], lw=2, label=label)
        if middle_K:
            ax.plot(middle_K, middle_R, '--', color=colors_p[p], lw=1, alpha=0.6)
        if lower_K:
            ax.plot(lower_K, lower_R, '-', color=colors_p[p], lw=2)
        K_sim = []; R_sim = []; E_sim = []
        for K_ratio in sorted(set(k[1] for k in results if k[0] == 'sim')):
            if ('sim', K_ratio, -0.05, p) in results:
                R, E = results[('sim', K_ratio, -0.05, p)]
                K_sim.append(K_ratio); R_sim.append(R); E_sim.append(E)
        ax.errorbar(K_sim, R_sim, yerr=E_sim, fmt='s', color=colors_p[p],
                    ms=6, mfc='white', mew=1.5, capsize=3)
    ax.axvline(1, ls=':', color='gray', lw=0.8)
    ax.set_xlabel(r'$K/K_c$'); ax.set_ylabel(r'$R_{\rm ss}$')
    ax.set_title(r'(b) Random static, suppression ($\epsilon=-0.05$)')
    ax.legend(frameon=False, fontsize=8, loc='center right')
    ax.set_ylim(0, 1); ax.set_xlim(0.3, 3.5)
    ax.text(0.18, 0.94, 'dashed: unstable middle branch\n'
            'sim escapes upper branch via finite-N Kramers',
            transform=ax.transAxes, fontsize=8, color='gray', va='top')
    
    plt.tight_layout()
    plt.savefig('fig_random_theory.pdf', bbox_inches='tight')
    plt.savefig('fig_random_theory.png', dpi=140, bbox_inches='tight')
    print("Saved fig_random_theory")


if __name__ == '__main__':
    main()
