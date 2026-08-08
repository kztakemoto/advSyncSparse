"""
Data generation for the cost-performance trade-off.
"""
import sys, os, time, pickle, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import networkx as nx
from multiprocessing import Pool, cpu_count
from sparse_kick_formulas import kick, kicked_fraction, free_flow_tau

# Reference thresholds at which the empirical fraction |sin phi| > t is
# recorded every cycle, independently of the run's own delta.  Comparing
# these with sigma_delta(R_pre, t) tests the Poisson-kernel closure directly
# and costs nothing extra.
TAIL_REF = (0.5, 0.8, 0.95)

# ------------------------------------------------------------------ #
# Fixed parameters
# ------------------------------------------------------------------ #
Delta = 0.5
tau   = 0.3
dt    = 0.01
tmax  = 80
n_theory_nets = 20   # network samples pooled for the theory degree distribution


# ------------------------------------------------------------------ #
# All-to-all free-flow propagator
# ------------------------------------------------------------------ #
def Phi_tau_a2a(R0, K, tau):
    """Exact free flow of dR/dt = -Delta R + (K/2) R (1 - R^2) over time tau."""
    R0 = max(min(R0, 1), 0)
    if R0 < 1e-14:
        return 0.0
    beta = K - 2*Delta
    u0 = R0**2
    if abs(beta) < 1e-14:
        u = u0 / (1 + K*u0*tau)
    else:
        E = np.exp(beta*tau)
        u = beta*u0*E / (beta + K*u0*(E - 1))
    return np.sqrt(max(u, 0))


# ------------------------------------------------------------------ #
# Averaged degree distribution for theory
# ------------------------------------------------------------------ #
def avg_deg_array(network, N, mean_deg, n_samples):
    """Pool the degree sequences of n_samples independent network draws."""
    all_deg = []
    for idx in range(n_samples):
        G = build_graph(network, N, mean_deg, seed=idx * 137)
        all_deg.extend(d for _, d in G.degree())
    return np.array(all_deg, dtype=int)


def build_graph(network, N, mean_deg, seed):
    """Single point of truth for network construction, shared by theory and simulation."""
    if network == 'er':
        return nx.gnm_random_graph(N, int(mean_deg * N / 2), seed=seed)
    elif network == 'ba':
        return nx.barabasi_albert_graph(N, mean_deg // 2, seed=seed)
    raise ValueError(f"unknown network type: {network}")


# ------------------------------------------------------------------ #
# Theory: fixed-point solvers
#
# Each solver returns a dict with both conventions:
#   R_star, sigma_star : at the post-kick map fixed point
#   R_obs,  sigma_obs  : after one further free-flow interval (pre-kick)
# ------------------------------------------------------------------ #
# Seed floor for warm starts.  At delta = 1 the kick is the identity, so the
# iteration reduces to repeated free flow and R = 0 is a fixed point of it:
# unstable for K > Kc, but an iteration seeded at a numerically negligible R
# cannot escape it within a finite number of steps.  Sweeping K upward with a
# warm start would otherwise carry the collapsed subcritical solution into the
# supercritical regime and report R = 0 there.  This applies whatever the sign
# of eps, since it is a property of the free flow and not of the kick.
#
# The floor differs by sign because it also decides which branch is found.
# For enhancement the fixed point is unique, so any small positive seed does.
# For suppression the map is bistable above Kc and the branch reached depends
# on where the iteration starts, so the seed is set to N^{-1/2}, the order
# parameter of the random initial condition the simulations actually use.  The
# theory then selects the same branch as the simulation: the incoherent one
# while the separatrix R*(delta) stays above N^{-1/2}, and the synchronized one
# once thresholding has pushed R*(delta) below it.
SEED_FLOOR = 1e-3


def seed_floor(eps, N):
    return SEED_FLOOR if eps >= 0 else 1.0/np.sqrt(N)

# Convergence is tested relatively, not absolutely: |R_{n+1} - R_n| shrinks in
# proportion to R itself near a fixed point, so an absolute tolerance fires
# spuriously whenever R happens to be tiny, terminating the iteration before it
# has grown.
def _converged(Rn, R):
    return abs(Rn - R) < 1e-11 * max(R, 1e-4)


def find_ss_a2a(K, eps, delta, N, R_init=None):
    """All-to-all hybrid map fixed point plus its pre-kick observables."""
    R = R_init if R_init is not None else (1.0/np.sqrt(N) if eps < 0 else 0.6)
    R = min(max(R, seed_floor(eps, N)), 1.0)
    for _ in range(20000):
        Rn = min(max(kick(Phi_tau_a2a(R, K, tau), eps, delta), 0), 1)
        if _converged(Rn, R):
            R = Rn
            break
        R = Rn
    R_obs = Phi_tau_a2a(R, K, tau)
    return {'R_star':     R,
            'sigma_star': kicked_fraction(R, delta),
            'R_obs':      R_obs,
            'sigma_obs':  kicked_fraction(R_obs, delta),
            'H_star':     R}


def find_ss_network(K, eps, delta, deg_array, N, H_init=None):
    """
    Annealed OA self-consistency for a network with the given degree sequence.

    The per-class map is R_k -> kick(Phi_tau(R_k; k K H), delta) with the
    global mean field H = sum_k k P(k) R_k / <k> solved self-consistently.
    H is held fixed across each free-flow interval, exactly as inside the map,
    and the pre-kick observables are read off after one such interval.
    """
    unique_k, counts = np.unique(deg_array, return_counts=True)
    Pk = counts.astype(float) / counts.sum()
    k_mean = (unique_k * Pk).sum()

    H = H_init if H_init is not None else (1.0/np.sqrt(N) if eps < 0 else 0.5)
    H = max(H, seed_floor(eps, N))      # see the note above find_ss_a2a
    Rk = {int(k): H for k in unique_k}

    for _ in range(2000):
        for k in unique_k:
            k = int(k)
            R = max(Rk[k], seed_floor(eps, N))
            for _ in range(500):
                R_flow = free_flow_tau(R, k * K * H, Delta, tau)
                R_kick = min(max(kick(R_flow, eps, delta), 0.0), 1.0)
                if _converged(R_kick, R):
                    R = R_kick
                    break
                R = R_kick
            Rk[k] = R
        H_new = sum(Pk[i] * unique_k[i] * Rk[int(unique_k[i])]
                    for i in range(len(unique_k))) / k_mean
        converged = abs(H_new - H) < 1e-8
        H = H_new
        if converged:
            break

    # Pre-kick observables: propagate each class over one free-flow interval
    # with the self-consistent field held fixed.
    Rk_obs = {int(k): free_flow_tau(Rk[int(k)], int(k) * K * H, Delta, tau)
              for k in unique_k}

    R_star = sum(Pk[i] * Rk[int(unique_k[i])] for i in range(len(unique_k)))
    R_obs  = sum(Pk[i] * Rk_obs[int(unique_k[i])] for i in range(len(unique_k)))
    sigma_star = sum(Pk[i] * kicked_fraction(Rk[int(unique_k[i])], delta)
                     for i in range(len(unique_k)))
    sigma_obs  = sum(Pk[i] * kicked_fraction(Rk_obs[int(unique_k[i])], delta)
                     for i in range(len(unique_k)))
    return {'R_star': R_star, 'sigma_star': sigma_star,
            'R_obs':  R_obs,  'sigma_obs':  sigma_obs,
            'H_star': H,
            'Rk_star': {int(k): Rk[int(k)] for k in unique_k},
            'Rk_obs':  Rk_obs}


def find_ss(K, eps, delta, network, deg_array=None, N=1000, warm=None):
    """Unified fixed-point solver dispatching on network type.

    warm : previous solution dict, used as an initial guess to speed up and
           stabilise sweeps over dense delta or K grids.
    """
    if network == 'a2a':
        return find_ss_a2a(K, eps, delta, N,
                           R_init=warm['R_star'] if warm else None)
    return find_ss_network(K, eps, delta, deg_array, N,
                           H_init=warm['H_star'] if warm else None)


# ------------------------------------------------------------------ #
# Simulation
#
# Both stroboscopic samples of R are recorded every cycle:
#   R_pre  : after the free flow, before the kick   (pre-kick convention)
#   R_post : immediately after the kick             (post-kick convention)
# The kicked fraction is measured where it is defined, on the pre-kick
# phase distribution.
# ------------------------------------------------------------------ #
def _run_kuramoto(rhs, theta, eps, delta, tmax, dt, tau):
    """Shared integration loop.

    Returns (R_pre, R_post, sigma, *tail) time averages, where tail holds the
    empirical fraction |sin(psi - theta)| > t for each t in TAIL_REF, measured
    on the pre-kick distribution.
    """
    n_sub = max(int(tau / dt), 1)
    h = tau / n_sub
    R_pre_list, R_post_list, frac_list = [], [], []
    tail_lists = [[] for _ in TAIL_REF]
    t = 0.0

    while t < tmax - 0.5*tau:
        for _ in range(n_sub):
            k1 = rhs(theta)
            k2 = rhs(theta + 0.5*h*k1)
            k3 = rhs(theta + 0.5*h*k2)
            k4 = rhs(theta + h*k3)
            theta = theta + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        theta = np.mod(theta, 2*np.pi)
        t += tau

        record = t > tmax * 0.5
        if record:
            R_pre_list.append(float(np.abs(np.mean(np.exp(1j*theta)))))

        psi  = np.angle(np.mean(np.exp(1j*theta)))
        sk   = np.sin(psi - theta)
        abs_sk = np.abs(sk)
        mask = abs_sk > delta
        if record:
            for j, tref in enumerate(TAIL_REF):
                tail_lists[j].append(float((abs_sk > tref).mean()))
        theta = theta + eps * np.sign(sk) * mask

        if record:
            R_post_list.append(float(np.abs(np.mean(np.exp(1j*theta)))))
            frac_list.append(float(mask.mean()))

    if not R_pre_list:
        return (0.0, 0.0, 0.0) + (0.0,)*len(TAIL_REF)
    return (float(np.mean(R_pre_list)),
            float(np.mean(R_post_list)),
            float(np.mean(frac_list))) + tuple(float(np.mean(v)) for v in tail_lists)


def sim_one_a2a(args):
    """All-to-all Kuramoto with periodic threshold kicks."""
    N, K, eps, delta, tmax, dt, tau, seed = args
    rng = np.random.default_rng(seed)
    omega = np.clip(rng.standard_cauchy(N) * Delta, -50*Delta, 50*Delta)
    theta = rng.uniform(0, 2*np.pi, N)

    def rhs(th):
        z = np.mean(np.exp(1j*th))
        return omega + K * np.abs(z) * np.sin(np.angle(z) - th)

    return _run_kuramoto(rhs, theta, eps, delta, tmax, dt, tau)


def sim_one(args):
    """
    Single simulation run for ER/BA networks.

    The network is generated inside the worker from the run seed, so each
    run uses a different network realisation.  Because the same seed list
    is reused for every (K, delta) condition, the comparison across
    conditions is paired: the same networks, frequencies and initial
    phases are used throughout, which removes most of the realisation
    noise from the delta-dependence.
    """
    network, N, mean_deg, K, eps, delta, tmax, dt, tau, seed = args
    rng = np.random.default_rng(seed)
    net_seed = int(rng.integers(1_000_000))

    G = build_graph(network, N, mean_deg, seed=net_seed)
    A = nx.to_scipy_sparse_array(G, format='csr', dtype=float)

    n = A.shape[0]
    omega = np.clip(rng.standard_cauchy(n) * Delta, -50*Delta, 50*Delta)
    theta = rng.uniform(0, 2*np.pi, n)

    def rhs(th):
        c = np.cos(th); s = np.sin(th)
        return omega + K * (c * A.dot(s) - s * A.dot(c))

    return _run_kuramoto(rhs, theta, eps, delta, tmax, dt, tau)


def sim_one_degree(args):
    """
    Single run for ER/BA, recording the degree-resolved local order parameter.

    At every recorded (pre-kick) instant the phases are referenced to the
    global mean phase and averaged within each degree class,
    R_k = |mean_{i in class k} exp(i(theta_i - psi))|, which is the quantity
    the annealed theory returns per class. Returned as running sums keyed by
    degree, so that runs with different network realisations can be pooled.
    """
    network, N, mean_deg, K, eps, delta, tmax, dt, tau, seed = args
    rng = np.random.default_rng(seed)
    net_seed = int(rng.integers(1_000_000))

    G = build_graph(network, N, mean_deg, seed=net_seed)
    A = nx.to_scipy_sparse_array(G, format='csr', dtype=float)
    deg = np.asarray(A.sum(axis=1)).ravel().astype(int)

    n = A.shape[0]
    omega = np.clip(rng.standard_cauchy(n) * Delta, -50*Delta, 50*Delta)
    theta = rng.uniform(0, 2*np.pi, n)

    def rhs(th):
        c = np.cos(th); s = np.sin(th)
        return omega + K * (c * A.dot(s) - s * A.dot(c))

    unique_k, inv = np.unique(deg, return_inverse=True)
    counts = np.bincount(inv, minlength=len(unique_k)).astype(float)
    Rk_sum  = np.zeros(len(unique_k))
    Rk2_sum = np.zeros(len(unique_k))
    n_sub = max(int(tau / dt), 1)
    h = tau / n_sub
    n_samp = 0
    t = 0.0

    while t < tmax - 0.5*tau:
        for _ in range(n_sub):
            k1 = rhs(theta)
            k2 = rhs(theta + 0.5*h*k1)
            k3 = rhs(theta + 0.5*h*k2)
            k4 = rhs(theta + h*k3)
            theta = theta + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        theta = np.mod(theta, 2*np.pi)
        t += tau

        psi = np.angle(np.mean(np.exp(1j*theta)))
        if t > tmax * 0.5:
            z = np.exp(1j*(theta - psi))
            zr = np.bincount(inv, weights=z.real, minlength=len(unique_k))
            zi = np.bincount(inv, weights=z.imag, minlength=len(unique_k))
            Rk = np.hypot(zr, zi) / counts
            Rk_sum  += Rk
            Rk2_sum += Rk**2      # for the finite-class-size correction below
            n_samp  += 1

        sk = np.sin(psi - theta)
        theta = theta + eps * np.sign(sk) * (np.abs(sk) > delta)

    if n_samp == 0:
        return {}
    return {int(k): (float(Rk_sum[j]/n_samp), float(Rk2_sum[j]/n_samp),
                     float(counts[j]))
            for j, k in enumerate(unique_k)}


def pool_degree_runs(run_dicts):
    """Pool per-run degree-resolved results, weighting each by its class size.

    A class of m nodes gives |mean of m unit vectors| a positive bias of order
    m^{-1/2} even for uniformly random phases, which matters for the sparsely
    populated high-degree classes of a scale-free network. The debiased value
    uses the standard correction R^2 -> (m <R^2> - 1)/(m - 1), and 'n_nodes'
    is kept so that under-populated classes can be dropped when plotting.
    """
    num, num2, den, nrun = {}, {}, {}, {}
    for rec in run_dicts:
        for k, (Rk, Rk2, cnt) in rec.items():
            num[k]  = num.get(k, 0.0)  + Rk*cnt
            num2[k] = num2.get(k, 0.0) + Rk2*cnt
            den[k]  = den.get(k, 0.0)  + cnt
            nrun[k] = nrun.get(k, 0) + 1
    out = {}
    for k in sorted(num):
        m = den[k]/nrun[k]                    # mean class size per realisation
        R2 = num2[k]/den[k]
        R2c = (m*R2 - 1.0)/(m - 1.0) if m > 1.0 else float('nan')
        out[k] = {'Rk':          num[k]/den[k],
                  'Rk_debiased': float(np.sqrt(R2c)) if R2c > 0 else 0.0,
                  'class_size':  float(m),
                  'n_nodes':     den[k]}
    return out


def make_args(network, N, mean_deg, K, eps, d, seed_base, n_runs):
    if network == 'a2a':
        return [(N, K, eps, d, tmax, dt, tau, seed_base + i*97)
                for i in range(n_runs)]
    return [(network, N, mean_deg, K, eps, d, tmax, dt, tau, seed_base + i*97)
            for i in range(n_runs)]


def summarize(results, n_runs):
    """Mean, sample std and standard error for R_pre, R_post, sigma and the tails."""
    arr = np.asarray(results, dtype=float)   # shape (n_runs, 3 + len(TAIL_REF))
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros(arr.shape[1])
    sem  = std / np.sqrt(len(arr))
    rec = {'R_pre':  float(mean[0]), 'R_pre_std':  float(std[0]), 'R_pre_sem':  float(sem[0]),
           'R_post': float(mean[1]), 'R_post_std': float(std[1]), 'R_post_sem': float(sem[1]),
           'sigma':  float(mean[2]), 'sigma_std':  float(std[2]), 'sigma_sem':  float(sem[2]),
           'n_runs': int(len(arr))}
    # Empirical tail fractions and the Poisson-kernel prediction at the same R,
    # so the closure can be checked without rerunning anything.
    rec['tail_emp'] = {float(t): float(mean[3+j]) for j, t in enumerate(TAIL_REF)}
    rec['tail_sem'] = {float(t): float(sem[3+j]) for j, t in enumerate(TAIL_REF)}
    rec['tail_poisson'] = {float(t): float(kicked_fraction(rec['R_pre'], t))
                           for t in TAIL_REF}
    return rec


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description='Generate cost-performance data for the sparse-kick paper.')
    parser.add_argument('--network', choices=['a2a', 'er', 'ba'], default='a2a',
                        help='Network type: all-to-all, Erdos-Renyi, or Barabasi-Albert')
    parser.add_argument('--N',        type=int,   default=1000, help='Number of oscillators')
    parser.add_argument('--mean_deg', type=int,   default=12,   help='Mean degree (ER/BA only)')
    parser.add_argument('--eps',      type=float, default=0.05,
                        help='Kick amplitude; positive enhances, negative suppresses')
    parser.add_argument('--n_runs',   type=int,   default=100,  help='Realisations per condition')
    parser.add_argument('--n_proc',   type=int,   default=None, help='Worker processes')
    parser.add_argument('--out_dir',  type=str,   default='.',  help='Output directory')
    parser.add_argument('--degree-resolved', action='store_true',
                        help='Run only the degree-resolved comparison (ER/BA only) '
                             'and write a separate, much smaller pkl')
    parser.add_argument('--dr-Kr',     type=float, default=2.0,
                        help='K/Kc for the degree-resolved run')
    parser.add_argument('--dr-deltas', type=str,   default='0.0,0.8',
                        help='Comma-separated thresholds for the degree-resolved run')
    args = parser.parse_args()

    N        = args.N
    network  = args.network
    mean_deg = args.mean_deg
    eps      = args.eps
    n_runs   = args.n_runs
    n_proc   = args.n_proc or min(cpu_count(), n_runs)

    # ---- Network setup ----
    if network == 'a2a':
        deg_arr = None
        Kc      = 2 * Delta
        print(f"Network: all-to-all, N={N}, Kc={Kc:.4f}", flush=True)
    else:
        print(f"Building pooled degree distribution "
              f"({n_theory_nets} samples)...", flush=True)
        deg_arr = avg_deg_array(network, N, mean_deg, n_theory_nets)
        Kc = 2*Delta * deg_arr.mean() / (deg_arr**2).mean()
        print(f"Network: {network.upper()}, N={N}, "
              f"<k>={deg_arr.mean():.2f}, <k^2>/<k>={(deg_arr**2).mean()/deg_arr.mean():.2f}, "
              f"Kc={Kc:.4f}", flush=True)

    # ---- Output filename ----
    if network == 'a2a':
        fname = f'a2a_costperf_eps{eps:.3f}_tau{tau:.2f}_N{N}.pkl'
    else:
        fname = f'{network}_costperf_eps{eps:.3f}_tau{tau:.2f}_N{N}_deg{mean_deg}.pkl'
    fname = os.path.join(args.out_dir, fname)

    # ---- Sweep parameters ----
    K_ratios_sim = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                    1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0]
    deltas_main  = [0.0, 0.5, 0.8, 1.0]   # 1.0 = no kick
    Kr_cp        = [0.7, 0.9, 1.2, 2.0]
    delta_cp     = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    delta_grid   = np.linspace(0, 1.0, 1000)   # theory dense grid (includes 1.0)
    Krs_th       = np.linspace(0.1, 3.2, 200)

    t0 = time.time()

    # ============================================================
    # Degree-resolved mode: one coupling, a few thresholds, nothing else
    # ============================================================
    if args.degree_resolved:
        if network == 'a2a':
            raise SystemExit('--degree-resolved requires --network er or ba')
        dr_deltas = [float(x) for x in args.dr_deltas.split(',')]
        Kr = args.dr_Kr
        K = Kr * Kc
        print(f"Degree-resolved run: Kr={Kr}, deltas={dr_deltas}, "
              f"{n_runs*len(dr_deltas)} simulations on {n_proc} workers", flush=True)
        sim_deg, theory_deg = {}, {}
        with Pool(n_proc) as pool:
            for d in dr_deltas:
                res = pool.map(sim_one_degree,
                               make_args(network, N, mean_deg, K, eps, d, 300, n_runs))
                sim_deg[(Kr, d)] = pool_degree_runs(res)
                ss = find_ss(K, eps, d, network, deg_arr, N)
                theory_deg[(Kr, d)] = {'Rk_obs':   ss['Rk_obs'],
                                       'Rk_star':  ss['Rk_star'],
                                       'R_obs':    ss['R_obs'],
                                       'sigma_obs': ss['sigma_obs']}
                print(f"  d={d}: {len(sim_deg[(Kr, d)])} degree classes, "
                      f"t={time.time()-t0:.0f}s", flush=True)
        out_dr = {
            'sim_deg': sim_deg, 'theory_deg': theory_deg,
            'Kr': Kr, 'deltas': dr_deltas,
            'eps': eps, 'tau': tau, 'dt': dt, 'tmax': tmax, 'Delta': Delta,
            'N': N, 'n_runs': n_runs, 'Kc': Kc, 'network': network,
            'mean_deg': mean_deg,
            'convention': 'pre_kick',
            'convention_note': (
                'Simulation: R_k is the modulus of the mean of exp(i(theta-psi)) '
                'over the nodes of degree k, sampled immediately before each kick '
                'and time-averaged over the second half of the run, then pooled '
                'across runs weighted by class size. Theory: Rk_obs is the '
                'per-class fixed point propagated over one free-flow interval.'),
        }
        fname_dr = os.path.join(
            args.out_dir,
            f'{network}_degres_eps{eps:.3f}_Kr{Kr:.1f}_N{N}_deg{mean_deg}.pkl')
        with open(fname_dr, 'wb') as f:
            pickle.dump(out_dr, f)
        print(f"\nDONE in {(time.time()-t0)/60:.1f} min, saved {fname_dr}")
        return

    n_sims = n_runs * (len(deltas_main)*len(K_ratios_sim) + len(Kr_cp)*len(delta_cp))
    print(f"Total simulation runs: {n_sims} on {n_proc} workers", flush=True)

    with Pool(n_proc) as pool:
        # ============================================================
        # (a) Transition curves: simulation
        # ============================================================
        sim_trans = {}
        print("\n=== (a) Transition curves ===", flush=True)
        for d in deltas_main:
            for Kr in K_ratios_sim:
                K = Kr * Kc
                res = pool.map(sim_one_a2a if network == 'a2a' else sim_one,
                               make_args(network, N, mean_deg, K, eps, d, 100, n_runs))
                rec = summarize(res, n_runs)
                sim_trans[(Kr, d)] = rec
                print(f"  Kr={Kr}, d={d}: R_pre={rec['R_pre']:.4f} "
                      f"(+/-{rec['R_pre_sem']:.4f}), R_post={rec['R_post']:.4f}, "
                      f"t={time.time()-t0:.0f}s", flush=True)

        # ============================================================
        # (b,c) Cost-performance: sweep delta at fixed K
        # ============================================================
        sim_cp = {}
        print("\n=== (b,c) Cost-performance ===", flush=True)
        for Kr in Kr_cp:
            K = Kr * Kc
            for d in delta_cp:
                res = pool.map(sim_one_a2a if network == 'a2a' else sim_one,
                               make_args(network, N, mean_deg, K, eps, d, 200, n_runs))
                rec = summarize(res, n_runs)
                sim_cp[(Kr, d)] = rec
                print(f"  Kr={Kr}, d={d}: R_pre={rec['R_pre']:.4f}, "
                      f"sigma={rec['sigma']:.4f} (+/-{rec['sigma_sem']:.4f}), "
                      f"t={time.time()-t0:.0f}s", flush=True)

    # ============================================================
    # Theory: dense delta grid at fixed K (warm-started along delta)
    # ============================================================
    theory = {}
    print("\n=== Theory (dense delta grid) ===", flush=True)
    for Kr in Kr_cp:
        K = Kr * Kc
        warm = None
        for d in delta_grid:
            warm = find_ss(K, eps, float(d), network, deg_arr, N, warm=warm)
            theory[(Kr, float(d))] = {k: warm[k] for k in
                                      ('R_obs', 'sigma_obs', 'R_star', 'sigma_star')}
        print(f"  Kr={Kr} done, t={time.time()-t0:.0f}s", flush=True)

    # ============================================================
    # Theory: transition curves (dense K grid, warm-started along K)
    # ============================================================
    theory_trans = {}
    print("\n=== Theory transition curves ===", flush=True)
    for d in deltas_main:
        warm = None
        for Kr in Krs_th:
            K = Kr * Kc
            warm = find_ss(K, eps, float(d), network, deg_arr, N, warm=warm)
            rec_th = {k: warm[k] for k in ('R_obs', 'sigma_obs',
                                           'R_star', 'sigma_star')}
            if 'Rk_obs' in warm:
                rec_th['Rk_obs'] = warm['Rk_obs']
            theory_trans[(float(Kr), float(d))] = rec_th
        print(f"  delta={d} done, t={time.time()-t0:.0f}s", flush=True)

    # ============================================================
    # Save
    # ============================================================
    out = {
        'sim_trans':    sim_trans,
        'sim_cp':       sim_cp,
        'theory':       theory,
        'theory_trans': theory_trans,
        'K_ratios_sim': K_ratios_sim,
        'Krs_th':       list(Krs_th),
        'deltas_main':  deltas_main,
        'Kr_cp':        Kr_cp,
        'delta_cp':     delta_cp,
        'delta_grid':   list(delta_grid),
        'eps':          eps,
        'tau':          tau,
        'dt':           dt,
        'tmax':         tmax,
        'Delta':        Delta,
        'N':            N,
        'n_runs':       n_runs,
        'Kc':           Kc,
        'network':      network,
        'mean_deg':     mean_deg if network != 'a2a' else None,
        'deg_stats':    None if deg_arr is None else {
            'mean_k':  float(deg_arr.mean()),
            'mean_k2': float((deg_arr**2).mean()),
            'k_max':   int(deg_arr.max()),
            'n_nets':  n_theory_nets},
        # Which keys to plot against which: pair R_obs with sigma_obs and the
        # simulated R_pre with the simulated sigma. Never mix _obs and _star.
        'tail_ref':     list(TAIL_REF),
        'convention':   'pre_kick',
        'convention_note': (
            'Theory: R_obs = Phi_tau(R*), sigma_obs = sigma_delta(Phi_tau(R*)). '
            'Simulation: R_pre sampled after the free flow and before the kick; '
            'sigma measured on the same pre-kick phase distribution. '
            'R_star/sigma_star and R_post are the post-kick counterparts, '
            'stored for reference only. Each record also carries tail_emp '
            '(measured fraction |sin phi| > t) against tail_poisson '
            '(sigma_delta at the measured R_pre) for t in tail_ref; the two '
            'agree in unkicked runs but differ systematically under kicks, '
            'which is a limit of the Poisson-kernel closure and not a '
            'finite-size effect.'),
    }
    with open(fname, 'wb') as f:
        pickle.dump(out, f)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min, saved {fname}")


if __name__ == '__main__':
    main()
