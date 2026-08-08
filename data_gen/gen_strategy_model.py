"""
Data generation for the strategy comparison on model networks.

Compares six strategies across:
  - network: er or ba (--network)
  - eps: enhancement (+) or suppression (-) (--eps)
  - 8 K/Kc values
  - 3 budgets p (0.1, 0.3, 0.5)
  - n_runs realizations

Strategies:
  none          - no kick (baseline)
  all           - full kick (baseline)
  adaptive      - top-p by |sin(psi-theta)| at each kick event
  random_static - random p selected once at t=0, fixed thereafter
  hub           - top-p by degree (fixed)
  low_degree    - bottom-p by degree (fixed)

Resumable: re-running continues from where it left off.

Output, e.g.:
  strategy_er_eps+0.050_tau0.30_N1000_deg12.pkl

Usage:
    python gen_strategy_model.py --network er --eps 0.05
    python gen_strategy_model.py --network ba --eps -0.05
"""
import sys, os, time, pickle, argparse, zlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import networkx as nx
from multiprocessing import Pool, cpu_count

# ------------------------------------------------------------------ #
# Fixed parameters
# ------------------------------------------------------------------ #
Delta    = 0.5
tau      = 0.3
dt       = 0.01
tmax     = 160
N        = 1000
mean_deg = 12
n_runs   = 100

FORMAT = 3   # per-run arrays, both conventions, paired seeds


def build_graph(network, seed):
    if network == 'er':
        return nx.gnm_random_graph(N, int(mean_deg*N/2), seed=seed)
    return nx.barabasi_albert_graph(N, mean_deg//2, seed=seed)

# ------------------------------------------------------------------ #
# Seeding and pairing
#
# The seed depends on the run index and nothing else, matching the convention
# of gen_costperf.py (seed_base + j*97 there). Run j therefore supplies the
# same network, natural frequencies and initial phases to every strategy, every
# budget p and every K, so any comparison along those axes is a within-
# realization difference. Different j still gives a different network, so the
# reported values remain averages over n_runs distinct graphs.
#
# Pairing across strategies is what matters most here, since the strategy
# difference is the measured quantity: on BA at K=2Kc the standard deviation of
# the adaptive-minus-random difference falls from 0.024 to 0.010, worth about a
# five-fold increase in n_runs at no cost.
#
# The previous scheme used seed = base + j*97 + hash(strat)%1000 + int(p*100).
# Python randomises str hashing per process, so those seeds changed on every
# invocation, which broke both reproducibility and the resume path, and gave
# each strategy a different network.
# ------------------------------------------------------------------ #
def make_seed(j):
    return int(np.random.SeedSequence((20260731, int(j))).generate_state(1)[0])


# ------------------------------------------------------------------ #
# Measurement convention
#
# R is recorded twice per cycle, after the free flow and again after the kick.
# The two differ by the kick step itself, which is larger for a strategy that
# selects well: on BA at K=2Kc, p=0.3 the step is +0.014 for adaptive and
# +0.007 for random static. Reading R immediately after the kick therefore adds
# one kick's worth of the selection advantage on top of the steady-state
# difference and inflates the adaptive-minus-random gap by about 17%. The
# pre-kick sample is the one used elsewhere in this package, and the analysis
# defaults to it; both are stored.
# ------------------------------------------------------------------ #
def run_strategy(A, deg, K, eps, p, strat, tmax, dt, tau, rng, Delta):
    """One realization. Returns (R_pre, R_post) time-averaged over the second half."""
    n = A.shape[0]
    # Dynamics drawn before any strategy-specific draw, so the random_static
    # mask cannot shift the stream and break the pairing.
    omega = np.clip(rng.standard_cauchy(n)*Delta, -50*Delta, 50*Delta)
    theta = rng.uniform(0, 2*np.pi, n)

    n_kick = max(1, int(p*n))
    fixed_mask = None
    if strat == 'random_static':
        fixed_mask = np.zeros(n, dtype=bool)
        fixed_mask[rng.choice(n, n_kick, replace=False)] = True
    elif strat == 'hub':
        fixed_mask = np.zeros(n, dtype=bool)
        fixed_mask[np.argsort(deg)[-n_kick:]] = True
    elif strat == 'low_degree':
        fixed_mask = np.zeros(n, dtype=bool)
        fixed_mask[np.argsort(deg)[:n_kick]] = True

    def rhs(th):
        c = np.cos(th); s = np.sin(th)
        return omega + K*(c*A.dot(s) - s*A.dot(c))

    n_sub = max(int(tau/dt), 1)
    h = tau/n_sub
    pre, post = [], []
    t = 0.0
    while t < tmax - 0.5*tau:
        for _ in range(n_sub):
            k1 = rhs(theta); k2 = rhs(theta + 0.5*h*k1)
            k3 = rhs(theta + 0.5*h*k2); k4 = rhs(theta + h*k3)
            theta = theta + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        theta = np.mod(theta, 2*np.pi)
        t += tau
        rec = t > tmax*0.5
        if rec:
            pre.append(float(np.abs(np.mean(np.exp(1j*theta)))))
        if strat != 'none':
            psi = np.angle(np.mean(np.exp(1j*theta)))
            sk = np.sin(psi - theta)
            if strat == 'all':
                mask = np.ones(n, dtype=bool)
            elif strat == 'adaptive':
                mask = np.zeros(n, dtype=bool)
                mask[np.argsort(np.abs(sk))[-n_kick:]] = True
            else:
                mask = fixed_mask
            theta = theta + eps*np.sign(sk)*mask
        if rec:
            post.append(float(np.abs(np.mean(np.exp(1j*theta)))))
    if not pre:
        return 0.0, 0.0
    return float(np.mean(pre)), float(np.mean(post))


def summarize(res):
    """Per-run arrays plus summary statistics, on both conventions."""
    a = np.asarray(res, dtype=float)          # (n_runs, 2)
    out = {'R_pre_runs': a[:, 0].tolist(), 'R_post_runs': a[:, 1].tolist(),
           'n_runs': int(len(a))}
    for i, name in ((0, 'R_pre'), (1, 'R_post')):
        m = float(a[:, i].mean())
        sd = float(a[:, i].std(ddof=1)) if len(a) > 1 else 0.0
        out[name] = m
        out[name + '_std'] = sd
        out[name + '_sem'] = sd/np.sqrt(len(a))
    return out

def sim_one(args):
    """One realization on a freshly generated network."""
    network, K, eps, p, strat, tmax_, dt_, tau_, seed = args
    rng = np.random.default_rng(seed)
    net_seed = int(rng.integers(1_000_000))
    G = build_graph(network, net_seed)
    A = nx.to_scipy_sparse_array(G, format='csr', dtype=float)
    deg = np.array([d for _, d in G.degree()])
    return run_strategy(A, deg, K, eps, p, strat, tmax_, dt_, tau_, rng, Delta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--network', choices=['er', 'ba'], default='er')
    ap.add_argument('--eps', type=float, default=0.05,
                    help='Kick strength (positive=enhancement, negative=suppression)')
    ap.add_argument('--n_runs', type=int, default=n_runs)
    ap.add_argument('--n_proc', type=int, default=None)
    ap.add_argument('--out_dir', type=str, default='.')
    args = ap.parse_args()

    network, eps = args.network, args.eps
    nr = args.n_runs
    n_proc = args.n_proc or min(cpu_count(), nr)

    deg_samples = []
    for idx in range(20):
        deg_samples.extend(d for _, d in build_graph(network, idx*137).degree())
    deg_arr = np.array(deg_samples)
    Kc = 2*Delta*deg_arr.mean()/(deg_arr**2).mean()
    print(f"Network: {network.upper()}, N={N}, <k>={deg_arr.mean():.2f}, Kc={Kc:.4f}",
          flush=True)

    fname = os.path.join(
        args.out_dir,
        f'strategy_{network}_eps{eps:+.3f}_tau{tau:.2f}_N{N}_deg{mean_deg}.pkl')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output: {fname}", flush=True)

    strategies = ['none', 'all', 'adaptive', 'random_static', 'hub', 'low_degree']
    K_ratios = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    p_values = [0.1, 0.3, 0.5]

    results = {}
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            saved = pickle.load(f)
        if saved.get('format') == FORMAT:
            results = saved['results']
            print(f"Loaded {len(results)} existing entries", flush=True)
        else:
            print("Existing file uses an older format (different seeding and "
                  "measurement convention); starting fresh rather than mixing "
                  "the two. Move it aside if you want to keep it.", flush=True)

    todo = []
    for Kr in K_ratios:
        for strat in strategies:
            for p_val in ([0.3] if strat in ('none', 'all') else p_values):
                key = (Kr, strat, p_val)
                if key not in results:
                    todo.append((key, Kr*Kc, p_val, strat))
    print(f"TODO: {len(todo)} configurations, {nr} runs each on {n_proc} workers",
          flush=True)

    t0 = time.time()
    with Pool(n_proc) as pool:
        for i, (key, K, p_val, strat) in enumerate(todo):
            # Seed depends on the run index only, so strategies, budgets and
            # K values all share realizations and differences can be taken
            # pairwise.
            arglist = [(network, K, eps, p_val, strat, tmax, dt, tau,
                        make_seed(j)) for j in range(nr)]
            results[key] = summarize(pool.map(sim_one, arglist))
            if (i+1) % 10 == 0 or (i+1) == len(todo):
                _propagate(results, p_values)
                _save(fname, results, K_ratios, p_values, strategies, Kc, eps, nr, network)
                print(f"  {i+1}/{len(todo)}, t={time.time()-t0:.0f}s, saved",
                      flush=True)

    _propagate(results, p_values)
    _save(fname, results, K_ratios, p_values, strategies, Kc, eps, nr, network)
    print(f"\nDONE: {len(results)} entries, total {(time.time()-t0)/60:.1f} min")


def _propagate(results, p_values):
    """none and all do not depend on p; copy the p=0.3 entry to the others."""
    for (Kr, strat, p), val in list(results.items()):
        if strat in ('none', 'all') and p == 0.3:
            for p_other in [pp for pp in p_values if pp != 0.3]:
                results.setdefault((Kr, strat, p_other), val)


def _save(fname, results, K_ratios, p_values, strategies, Kc, eps, nr, network):
    with open(fname, 'wb') as f:
        pickle.dump({
            'results': results, 'Kc': Kc, 'K_ratios': K_ratios,
            'network': network,
            'p_values': p_values, 'strategies': strategies, 'N': N,
            'mean_deg': mean_deg, 'eps': eps, 'tau': tau, 'dt': dt,
            'tmax': tmax, 'n_runs': nr, 'Delta': Delta, 'format': FORMAT,
            'convention': 'pre_kick',
            'note': ('Each entry stores R_pre_runs and R_post_runs, one value '
                     'per realization. Seeds depend on the run index only, so '
                     'run j supplies the same network, frequencies and initial '
                     'phases to every strategy, budget and K; take differences '
                     'pairwise. Analyses should '
                     'use R_pre, matching the rest of the package: reading R '
                     'after the kick adds one kick step, which is larger for a '
                     'better-selecting strategy and inflates the gap.'),
        }, f)


if __name__ == '__main__':
    main()
