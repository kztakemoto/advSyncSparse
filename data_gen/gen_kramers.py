"""
Data generation: All-to-all suppression Kramers escape.

Starting from random initial phases (R ~ 1/sqrt(N)), we measure whether
the system escapes to the synchronized state (R >= threshold) under
suppression kicks (eps < 0).

R_finals (per-run R_late averages) are saved so that threshold can be
adjusted at plot time without rerunning simulations.

P_escape   = fraction of runs where R_final >= threshold (reached sync)
P_survival = 1 - P_escape (stayed incoherent)

Kramers prediction: log(-log(1 - P_escape)) ~ -c R*(delta)^2 N

Usage:
    python gen_kramers.py [--eps -0.05 -0.07] [--K_ratios 2.0 3.0] [--tmax 80]

    # extra thresholds, own N window, one coupling; writes its own pkl
    python gen_kramers.py --eps -0.03 -0.05 -0.07 --K_ratios 3.0 \
        --deltas 0.8 --Ns 400 600 800 1000 1400 2000 2800 4000 5000 6000

Output example: kramers_eps-0.050--0.070_tau0.30_K2.0-3.0_d0.0-0.3-0.5-0.7.pkl
"""
import sys, os, time, pickle, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
from multiprocessing import Pool, cpu_count

# Fixed parameters
Delta = 0.5
tau   = 0.3
dt    = 0.01


def sim_kramers(args):
    """
    All-to-all suppression run starting from random initial phases.
    Returns R_final = mean of R over second half of simulation.
    """
    N, K, eps, delta, tmax, dt, tau, seed = args
    rng = np.random.default_rng(seed)
    omega = np.clip(rng.standard_cauchy(N)*Delta, -50*Delta, 50*Delta)
    # Start from random (incoherent) initial condition
    theta = rng.uniform(0, 2*np.pi, N)

    def rhs(th):
        z = np.mean(np.exp(1j*th))
        return omega + K*np.abs(z)*np.sin(np.angle(z) - th)

    n_sub = max(int(tau/dt), 1)
    h = tau/n_sub
    t = 0.0
    R_late = []
    while t < tmax:
        for _ in range(n_sub):
            k1 = rhs(theta); k2 = rhs(theta + 0.5*h*k1)
            k3 = rhs(theta + 0.5*h*k2); k4 = rhs(theta + h*k3)
            theta = theta + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        theta = np.mod(theta, 2*np.pi)
        t += tau
        if t > tmax*0.5:
            R_late.append(float(np.abs(np.mean(np.exp(1j*theta)))))
        psi = np.angle(np.mean(np.exp(1j*theta)))
        sk = np.sin(psi - theta)
        mask = np.abs(sk) > delta
        theta = theta + eps*np.sign(sk)*mask

    return float(np.mean(R_late)) if R_late else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps',       type=float, nargs='+', default=[-0.05],
                        help='Kick strengths (negative for suppression), e.g. --eps -0.05 -0.07')
    parser.add_argument('--K_ratios',  type=float, nargs='+', default=[2.0, 3.0],
                        help='K/Kc values to simulate')
    parser.add_argument('--tmax',      type=float, default=80,
                        help='Simulation time per run')
    parser.add_argument('--n_runs',    type=int,   default=10000,
                        help='Number of realisations per (eps, N, delta, K) point')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Default escape threshold for R (saved but adjustable at plot time)')
    parser.add_argument('--deltas',    type=float, nargs='+',
                        default=[0.0, 0.3, 0.5, 0.7],
                        help='Thresholds to simulate. The barrier goes as '
                             'R*^2 N with R*^2 proportional to 1 - delta^2, so '
                             'a larger delta needs a larger N window; give it '
                             'with --Ns.')
    parser.add_argument('--Ns',        type=int,   nargs='+', default=None,
                        help='System sizes. Default is the 24-point grid '
                             '200-4000 used for delta <= 0.7.')
    parser.add_argument('--out',       default=None,
                        help='Output filename (default encodes eps, K and delta)')
    parser.add_argument('--seed-mode',  choices=['condition', 'paired'],
                        default='condition',
                        help="'condition' gives every (eps, K, delta, N) its "
                             "own block of realizations. 'paired' keys the "
                             "seed on (N, run index) alone, so one realization "
                             "index means the same frequencies and initial "
                             "phases at every eps, K and delta; comparisons "
                             "across delta are then paired and the slope "
                             "ratio has a smaller variance. Mixing the two "
                             "across a merged dataset gives up the pairing, "
                             "so 'paired' is only useful if delta = 0 is "
                             "regenerated the same way.")
    args = parser.parse_args()

    eps_list  = sorted(args.eps)
    K_ratios  = sorted(args.K_ratios)
    tmax      = args.tmax
    n_runs    = args.n_runs
    threshold = args.threshold
    n_proc    = min(cpu_count(), n_runs)
    Kc        = 2*Delta

    Ns     = args.Ns if args.Ns else [200, 220, 240, 260, 280, 300, 320, 350,
                                      400, 450, 500, 550, 600, 650, 700, 800,
                                      900, 1000, 1200, 1500, 2000, 2500, 3000,
                                      4000]
    Ns     = sorted(set(int(n) for n in Ns))
    deltas = sorted(set(float(d) for d in args.deltas))

    # Output filename. The delta tag keeps a run with extra thresholds from
    # overwriting an earlier pkl that shares eps, K and tau.
    K_tag   = '-'.join(f'{Kr}' for Kr in K_ratios)
    eps_tag = '-'.join(f'{e:.3f}' for e in eps_list)
    d_tag   = '-'.join(f'{d:g}' for d in deltas)
    fname   = (args.out or
               f'kramers_eps{eps_tag}_tau{tau:.2f}_K{K_tag}_d{d_tag}.pkl')

    print(f"eps={eps_list}, K/Kc={K_ratios}, tmax={tmax}, n_runs={n_runs}, "
          f"deltas={deltas}, seed-mode={args.seed_mode}", flush=True)
    print(f"Output: {fname}", flush=True)

    results = {}
    t0 = time.time()

    for eps in eps_list:
        print(f"\n=== eps={eps} ===", flush=True)
        for Kr in K_ratios:
            K = Kr * Kc
            print(f"  K/Kc={Kr}", flush=True)
            for delta in deltas:
                for N in Ns:
                    # Seeds. The old scheme spaced conditions by 1e3-1e6
                    # while the run index spanned n_runs*131, so different
                    # delta and K shared realizations without meaning to,
                    # and delta was resolved only to one decimal. Both modes
                    # below draw a fresh omega and a fresh initial phase
                    # vector for every one of the n_runs realizations; they
                    # differ only in whether two conditions reuse them.
                    if args.seed_mode == 'paired':
                        key = (int(N),)
                    else:
                        key = (int(round(abs(eps)*1000)),
                               int(round(Kr*100)),
                               int(round(delta*1000)),
                               int(N))
                    base_seed = int(np.random.SeedSequence(
                        list(key)).generate_state(1, dtype=np.uint32)[0])
                    args_list = [
                        (N, K, eps, delta, tmax, dt, tau,
                         (base_seed + i) % (2**32))
                        for i in range(n_runs)
                    ]
                    with Pool(n_proc) as pool:
                        R_finals = pool.map(sim_kramers, args_list)
                    R_finals = [float(r) for r in R_finals]
                    P_escape   = float(np.mean(np.array(R_finals) >= threshold))
                    P_survival = 1.0 - P_escape
                    results[(float(eps), Kr, N, float(delta))] = {
                        'R_finals':       R_finals,
                        'P_escape':       P_escape,
                        'P_survival':     P_survival,
                        'n_realizations': n_runs,
                    }
                    print(f"    N={N}, delta={delta}: "
                          f"P_escape={P_escape:.3f}, "
                          f"P_survival={P_survival:.3f}, "
                          f"t={time.time()-t0:.0f}s", flush=True)

    out = {
        'results':   results,
        'Ns':        Ns,
        'deltas':    deltas,
        'K_ratios':  K_ratios,
        'eps_list':  eps_list,
        'Kc':        Kc,
        'tau':       tau,
        'tmax':      tmax,
        'threshold': threshold,  # default, adjustable at plot time
        'Delta':     Delta,
        'seed_mode': args.seed_mode,
    }
    with open(fname, 'wb') as f:
        pickle.dump(out, f)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min, saved {fname}")


if __name__ == '__main__':
    main()