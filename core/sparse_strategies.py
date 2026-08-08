"""
Multi-strategy sparse adversarial kick simulator for Kuramoto on networks.

Strategies:
  'all': kick every node (full attack baseline, p=1)
  'none': no kick (baseline)
  'adaptive': Top-p% by current |sin(psi-theta_i)|, re-selected every kick
  'initial': Top-p% by |sin(psi-theta_i)| at t=tau (the first kick instant), fixed thereafter
  'random_dyn': random p% re-selected every kick
  'random_static': random p% chosen at t=0, fixed thereafter
  'hub': top-p% by degree, fixed
  'low_degree': bottom-p% by degree, fixed
"""
import numpy as np
from scipy import sparse


def simulate(A, K, eps, p, strategy, tau, dt, tmax, Delta=0.5, seed=0):
    """
    Run one realization of Kuramoto on network A with periodic kicks.
    Returns time-averaged R over the second half.
    """
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    # Lorentzian frequencies (clipped)
    omega = np.clip(rng.standard_cauchy(N) * Delta, -50*Delta, 50*Delta)
    theta = rng.uniform(0, 2*np.pi, N)
    
    # node count to kick
    n_kick = int(np.floor(p * N))
    
    # Static node selection: random_static, hub, low_degree, none, all
    if strategy == 'random_static':
        chosen_static = rng.choice(N, size=n_kick, replace=False)
    elif strategy == 'hub':
        deg = np.array(A.sum(axis=1)).ravel()
        # break ties randomly via small noise on degree
        deg_noised = deg + 1e-6*rng.random(N)
        chosen_static = np.argsort(-deg_noised)[:n_kick]  # top-n_kick
    elif strategy == 'low_degree':
        deg = np.array(A.sum(axis=1)).ravel()
        deg_noised = deg + 1e-6*rng.random(N)
        chosen_static = np.argsort(deg_noised)[:n_kick]
    else:
        chosen_static = None
    
    # for 'initial': chosen at t=tau, so we set chosen_initial=None now, fill in at first kick
    chosen_initial = None
    
    R_list = []
    t = 0.0
    n_kicks_done = 0
    n_substeps = max(int(tau/dt), 1)
    h = tau / n_substeps
    
    def rhs(th):
        c = np.cos(th); s = np.sin(th)
        coupling = K*(c*A.dot(s) - s*A.dot(c))
        return omega + coupling
    
    while t < tmax - 0.5*tau:
        # RK4 over interval tau
        for _ in range(n_substeps):
            k1 = rhs(theta); k2 = rhs(theta + 0.5*h*k1)
            k3 = rhs(theta + 0.5*h*k2); k4 = rhs(theta + h*k3)
            theta = theta + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        theta = np.mod(theta, 2*np.pi)
        t = t + tau
        
        # Measure R if past tmax/2
        if t > tmax * 0.5:
            R_list.append(float(np.abs(np.mean(np.exp(1j*theta)))))
        
        # Apply kick if strategy != 'none'
        if strategy == 'none' or eps == 0:
            n_kicks_done += 1
            continue
        
        # Decide which nodes to kick
        psi = np.angle(np.mean(np.exp(1j*theta)))
        sk = np.sin(psi - theta)
        
        if strategy == 'all':
            chosen = np.arange(N)
        elif strategy == 'adaptive':
            abs_sk = np.abs(sk)
            chosen = np.argsort(-abs_sk)[:n_kick]
        elif strategy == 'initial':
            if chosen_initial is None:
                # first kick: select by current |sin(psi-theta)|
                abs_sk = np.abs(sk)
                chosen_initial = np.argsort(-abs_sk)[:n_kick]
            chosen = chosen_initial
        elif strategy == 'random_dyn':
            chosen = rng.choice(N, size=n_kick, replace=False)
        elif strategy in ('random_static', 'hub', 'low_degree'):
            chosen = chosen_static
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Apply kick to chosen nodes
        theta[chosen] = theta[chosen] + eps * np.sign(sk[chosen])
        theta = np.mod(theta, 2*np.pi)
        n_kicks_done += 1
    
    return float(np.mean(R_list)) if R_list else 0.0


def build_ER(N, mean_deg, rng):
    p_edge = mean_deg/(N-1)
    iu = np.triu_indices(N, 1)
    mask = rng.random(len(iu[0])) < p_edge
    r = iu[0][mask]; c = iu[1][mask]
    A = sparse.coo_matrix((np.ones(len(r)), (r,c)), shape=(N,N))
    A = A + A.T
    return A.tocsr()


def build_BA(N, m, rng):
    er=[]; ec=[]; deg=np.zeros(N)
    for i in range(m+1):
        for j in range(i+1, m+1):
            er+=[i,j]; ec+=[j,i]; deg[i]+=1; deg[j]+=1
    for new in range(m+1, N):
        prob = deg[:new]/deg[:new].sum()
        tg = rng.choice(new, size=m, replace=False, p=prob)
        for t in tg: er+=[new,t]; ec+=[t,new]; deg[new]+=1; deg[t]+=1
    return sparse.coo_matrix((np.ones(len(er)),(er,ec)), shape=(N,N)).tocsr()
