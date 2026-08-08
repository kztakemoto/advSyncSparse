"""
Data generation for the supporting figure:
Random-static annealed theory vs simulation.

The annealed kick map for random-static is the convex combination
    R_new = (1-p)*R + p * R_kick_full(R; eps)
which we evaluate as fixed points of the hybrid map (free flow + this kick).

All theory values are reported on the pre-kick (stroboscopic) sampling
convention, matching how the simulation records R.

For enhancement, single fixed point - simulation matches theory.
For suppression, the same map is bistable for K > Kc. Theory gives 3 FPs;
simulation explores the deterministic FPs subject to Kramers escape at finite N.
Per-run values are stored, not just their mean, because averaging over runs that
have settled on different branches returns a value on neither of them.

Run settings match the production runs used elsewhere in the paper
(tmax = 80, n_runs = 100, N = 1000).

Output: random_theory_vs_sim.pkl
"""
import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
from scipy.optimize import brentq
from multiprocessing import Pool

Delta = 0.5
tau = 0.3
dt = 0.01
tmax = 80
N_sim = 1000
n_runs = 100
n_proc = 8

# One root sequence for the whole script. Spawned children depend only on the
# run index, so every (K, eps, p) condition sees the same set of frequency and
# initial-phase realizations and the curves are paired across conditions.
SEED_ROOT = 20260801


def Sd_full(R):
    """S(R) at delta=0."""
    if R < 1e-12:
        return 1/np.pi
    return (1-R**2)/(np.pi*R) * np.arctanh(R)


def kick_full(R, eps):
    return R*np.cos(eps) + 2*np.sin(eps)*Sd_full(R)


def kick_random_static(R, eps, p):
    """Annealed kick map for random-static strategy."""
    return (1-p)*R + p*kick_full(R, eps)


def Phi_tau_a2a(R0, K):
    R0 = max(min(R0, 1), 0)
    if R0 < 1e-14:
        return 0.0
    beta = K - 2*Delta
    u0 = R0**2
    if abs(beta) < 1e-14:
        u = u0/(1 + K*u0*tau)
    else:
        E = np.exp(beta*tau)
        u = beta*u0*E/(beta + K*u0*(E-1))
    return np.sqrt(max(u, 0))


def all_fps(K, eps, p):
    """Find all fixed points of the random-static hybrid map."""
    map_fn = lambda R: kick_random_static(Phi_tau_a2a(R, K), eps, p)
    fps = []
    R_test = np.linspace(0.001, 0.999, 800)
    diffs = [map_fn(R) - R for R in R_test]
    for i in range(len(diffs)-1):
        if diffs[i]*diffs[i+1] < 0:
            try:
                root = brentq(lambda R: map_fn(R) - R, R_test[i], R_test[i+1])
                fps.append(root)
            except Exception:
                pass
    return fps


def sim_a2a_random_static(args):
    """All-to-all simulation with random-static kicks.

    R is recorded immediately before each kick, which is the pre-kick
    convention used throughout; the kicked subset is drawn once at t = 0.
    """
    K, eps, p, N, run_idx = args
    rng = np.random.default_rng(
        np.random.SeedSequence(SEED_ROOT).spawn(n_runs)[run_idx])
    omega = np.clip(rng.standard_cauchy(N)*Delta, -50*Delta, 50*Delta)
    theta = rng.uniform(0, 2*np.pi, N)
    n_kick = int(p*N)
    kicked = np.zeros(N, dtype=bool)
    kicked[rng.choice(N, n_kick, replace=False)] = True

    def rhs(th):
        z = np.mean(np.exp(1j*th))
        return omega + K*np.abs(z)*np.sin(np.angle(z) - th)

    R_list = []
    t = 0
    n_sub = max(int(tau/dt), 1)
    h = tau/n_sub
    while t < tmax - 0.5*tau:
        for _ in range(n_sub):
            k1 = rhs(theta); k2 = rhs(theta + 0.5*h*k1)
            k3 = rhs(theta + 0.5*h*k2); k4 = rhs(theta + h*k3)
            theta = theta + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        theta = np.mod(theta, 2*np.pi)
        t += tau
        if t > tmax*0.5:
            R_list.append(float(np.abs(np.mean(np.exp(1j*theta)))))
        psi = np.angle(np.mean(np.exp(1j*theta)))
        theta[kicked] = theta[kicked] + eps*np.sign(np.sin(psi - theta[kicked]))
    return float(np.mean(R_list))


def main():
    Kc = 2*Delta  # = 1.0
    ps = [0.1, 0.3, 0.5]

    results = {}

    # Theory: dense K grid
    print("Computing theory fixed points...", flush=True)
    for K_ratio in np.linspace(0.3, 3.5, 50):
        K_ratio = float(K_ratio)     # plain float, so the pkl keys stay hashable
        K = K_ratio*Kc
        for eps in [0.05, -0.05]:
            for p in ps:
                fps = all_fps(K, eps, p)
                # The map fixed points are post-kick; the simulation samples R
                # immediately before each kick, so report Phi_tau(R*) (the
                # pre-kick convention used throughout).
                fps_obs = [Phi_tau_a2a(R, K) for R in fps]
                results[('theory_fps', round(K_ratio, 3), eps, p)] = fps_obs
                results[('theory_fps_star', round(K_ratio, 3), eps, p)] = fps

    # Simulation
    print("Running simulations...", flush=True)
    K_sim = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    sim_args = []
    for K_ratio in K_sim:
        for eps in [0.05, -0.05]:
            for p in ps:
                for i in range(n_runs):
                    sim_args.append((K_ratio*Kc, eps, p, N_sim, i))

    t0 = time.time()
    with Pool(n_proc) as pool:
        sim_results = pool.map(sim_a2a_random_static, sim_args)
    print(f"  Sim done in {time.time()-t0:.0f}s ({len(sim_results)} runs)",
          flush=True)

    idx = 0
    for K_ratio in K_sim:
        for eps in [0.05, -0.05]:
            for p in ps:
                Rs = sim_results[idx:idx+n_runs]
                idx += n_runs
                Rs = np.asarray(Rs, dtype=float)
                # Per-run values are kept: for suppression the map is bistable
                # and the run-to-run mean lies between the two branches.
                results[('sim_runs', K_ratio, eps, p)] = Rs.tolist()
                results[('sim', K_ratio, eps, p)] = (float(Rs.mean()),
                                                     float(Rs.std()))

    out = {
        'results': results,
        'K_sim': K_sim,
        'p_values': ps,
        'Kc': Kc,
        'tau': tau,
        'dt': dt,
        'tmax': tmax,
        'Delta': Delta,
        'N_sim': N_sim,
        'n_runs': n_runs,
        'seed_root': SEED_ROOT,
        'convention': 'pre_kick',
        'convention_note': ('Theory: R_obs = Phi_tau(R*). Simulation: R sampled '
                            'at t = n*tau immediately before each kick, then '
                            'time-averaged over the second half of the run.'),
    }
    with open('random_theory_vs_sim.pkl', 'wb') as f:
        pickle.dump(out, f)
    print("DONE, saved random_theory_vs_sim.pkl")


if __name__ == '__main__':
    main()
