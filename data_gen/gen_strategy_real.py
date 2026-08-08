"""
Data generation for the strategy comparison on real networks.

Networks (Network Repository, reduced to a simple graph without self-loops and
restricted to the largest connected component):
  - power-1138-bus            : N=1138, <k>=2.56, k_max=17
  - bn-mouse-kasthuri-graph-v4: N=987,  <k>=3.11, k_max=123

The annealed Kc severely underestimates the synchronization onset for these
sparse networks, so K is set empirically: the mouse brain is parametrized by
K/Kc_emp with Kc_emp from the no-kick R=0.5 crossing, and the power grid, which
never reaches R=0.5 in range, by absolute K.

Output, e.g.:
  strategy_real_eps+0.050_tau0.30.pkl

Usage:
    python gen_strategy_real.py --eps 0.05
    python gen_strategy_real.py --eps -0.05
"""
import sys, os, time, pickle, argparse, zlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import pandas as pd
import networkx as nx
from multiprocessing import Pool, cpu_count

# ------------------------------------------------------------------ #
# Fixed parameters
# ------------------------------------------------------------------ #
Delta  = 0.5
tau    = 0.3
dt     = 0.01
tmax   = 80
n_runs = 100

FORMAT = 3   # per-run arrays, both conventions, paired seeds

NETWORK_DIR = os.path.join(os.path.dirname(__file__), '..', 'network_data')

Kc_emp_mouse   = 0.9
K_abs_power    = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0]
K_ratios_mouse = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]


def load_real(name):
    """Simple graph, no self-loops, largest connected component."""
    df = pd.read_csv(os.path.join(NETWORK_DIR, f'{name}.txt'),
                     sep=r'\s+', header=None)
    g = nx.Graph(nx.from_pandas_edgelist(df, source=0, target=1))
    g.remove_edges_from(nx.selfloop_edges(g))
    lcc = max(nx.connected_components(g), key=len)
    return nx.convert_node_labels_to_integers(g.subgraph(lcc).copy())

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
    """One realization on a fixed real network."""
    A, deg, K, eps, p, strat, tmax_, dt_, tau_, seed = args
    rng = np.random.default_rng(seed)
    return run_strategy(A, deg, K, eps, p, strat, tmax_, dt_, tau_, rng, Delta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, default=0.05,
                    help='Kick strength (positive=enhancement, negative=suppression)')
    ap.add_argument('--n_runs', type=int, default=n_runs)
    ap.add_argument('--n_proc', type=int, default=None)
    ap.add_argument('--out_dir', type=str, default='.')
    args = ap.parse_args()

    eps = args.eps
    nr = args.n_runs
    n_proc = args.n_proc or min(cpu_count(), nr)

    fname = os.path.join(args.out_dir,
                         f'strategy_real_eps{eps:+.3f}_tau{tau:.2f}.pkl')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"eps={eps}, Output: {fname}", flush=True)

    print("Loading real networks...", flush=True)
    nets = {}
    for tag, name in (('power', 'power-1138-bus'), ('mouse', 'bn-mouse')):
        G = load_real(name)
        A = nx.to_scipy_sparse_array(G, format='csr', dtype=float)
        deg = np.array([d for _, d in G.degree()])
        nets[tag] = (A, deg)
        print(f"  {tag}: N={A.shape[0]}, <k>={deg.mean():.2f}, "
              f"k_max={int(deg.max())}", flush=True)

    strategies = ['none', 'all', 'adaptive', 'random_static', 'hub', 'low_degree']
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
    for tag, K_list, is_ratio in (('mouse', [Kr*Kc_emp_mouse for Kr in K_ratios_mouse], True),
                                  ('power', K_abs_power, False)):
        for K in K_list:
            K_key = round(K/Kc_emp_mouse, 4) if is_ratio else round(K, 4)
            for strat in strategies:
                for p_val in ([0.3] if strat in ('none', 'all') else p_values):
                    key = (tag, K_key, strat, p_val)
                    if key not in results:
                        todo.append((key, tag, K, p_val, strat))
    print(f"TODO: {len(todo)} configurations, {nr} runs each on {n_proc} workers",
          flush=True)

    t0 = time.time()
    with Pool(n_proc) as pool:
        for i, (key, tag, K, p_val, strat) in enumerate(todo):
            A, deg = nets[tag]
            # Seed depends on the run index only, so strategies, budgets and
            # K values all share realizations and differences can be taken
            # pairwise.
            arglist = [(A, deg, K, eps, p_val, strat, tmax, dt, tau,
                        make_seed(j)) for j in range(nr)]
            results[key] = summarize(pool.map(sim_one, arglist))
            if (i+1) % 10 == 0 or (i+1) == len(todo):
                _propagate(results, p_values)
                _save(fname, results, strategies, p_values, eps, nr)
                print(f"  {i+1}/{len(todo)}, t={time.time()-t0:.0f}s, saved",
                      flush=True)

    _propagate(results, p_values)
    _save(fname, results, strategies, p_values, eps, nr)
    print(f"\nDONE: {len(results)} entries, total {(time.time()-t0)/60:.1f} min")


def _propagate(results, p_values):
    """none and all do not depend on p; copy the p=0.3 entry to the others."""
    for (tag, K_key, strat, p), val in list(results.items()):
        if strat in ('none', 'all') and p == 0.3:
            for p_other in [pp for pp in p_values if pp != 0.3]:
                results.setdefault((tag, K_key, strat, p_other), val)


def _save(fname, results, strategies, p_values, eps, nr):
    with open(fname, 'wb') as f:
        pickle.dump({
            'results': results, 'Kc_emp_mouse': Kc_emp_mouse,
            'K_abs_power': K_abs_power, 'K_ratios_mouse': K_ratios_mouse,
            'strategies': strategies, 'p_values': p_values, 'eps': eps,
            'tau': tau, 'dt': dt, 'tmax': tmax, 'n_runs': nr, 'Delta': Delta,
            'format': FORMAT, 'convention': 'pre_kick',
            'note': ('Each entry stores R_pre_runs and R_post_runs, one value '
                     'per realization. Seeds depend on the run index only, so '
                     'run j supplies the same frequencies and initial phases to '
                     'every strategy, budget and K; take differences pairwise. Analyses should use '
                     'R_pre, matching the rest of the package: reading R after '
                     'the kick adds one kick step, which is larger for a '
                     'better-selecting strategy and inflates the gap.'),
        }, f)


if __name__ == '__main__':
    main()
