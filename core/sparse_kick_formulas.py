"""
Closed-form quantities for threshold-based (sparse) adversarial kicks.
"""
import numpy as np


def S_delta(R, delta):
    """Kicked-region sin-weight. delta=0 gives S(R)=(1-R^2)arctanh(R)/(pi R)."""
    if R < 1e-14:
        return np.sqrt(1 - delta**2) / np.pi
    s = np.sqrt(1 - delta**2)
    return (1 - R**2) / (4 * np.pi * R) * np.log((1 + 2*R*s + R**2) /
                                                 (1 - 2*R*s + R**2))


def C_delta(R, delta):
    """No-kick region contribution. C_delta(R,0)=0, C_delta(R,1)=R."""
    if delta <= 0 or R < 1e-14:
        return 0.0
    a = np.arcsin(delta)
    s = np.sqrt(1 - delta**2)
    return (2 * R * a / np.pi
            + (1 + R**2) / (np.pi * R)
            * np.arctan2(2 * R**2 * delta * s, 1 - R**2 + 2 * R**2 * delta**2))


def kicked_fraction(R, delta):
    """Fraction of a class that is kicked (the per-class cost sigma_delta(R)).

    At R=0 (uniform phase distribution), the exact result is
    sigma_delta(0) = 1 - (2/pi)*arcsin(delta), NOT sqrt(1-delta^2).
    The sqrt form differs by O(delta) at small delta and reaches ~0.2
    error at delta=0.5.
    """
    if delta <= 0:
        return 1.0
    if delta >= 1:
        return 0.0
    if R < 1e-14:
        return 1.0 - (2.0/np.pi)*np.arcsin(delta)
    a = np.arcsin(delta)
    b = np.pi - a
    F = lambda phi: 0.5 + (1/np.pi)*np.arctan(((1+R)/(1-R))*np.tan(phi/2))
    return 2 * (F(b) - F(a))


def kick(R, eps, delta):
    """Corrected single-kick map."""
    C = C_delta(R, delta)
    return C + (R - C)*np.cos(eps) + 2*np.sin(eps)*S_delta(R, delta)


def free_flow_steady(kKH, Delta):
    """Steady state of the H-driven OA free flow
        dR/dt = -Delta R + (kKH/2)(1 - R^2),
    i.e. R* = (-Delta + sqrt(Delta^2 + (kKH)^2)) / (kKH).
    NOTE: this is the correct steady state of the companion paper's Eq.(16);
    the closed form quoted there as sqrt(1 - 2 Delta/(kKH)) corresponds instead to
    an Rk-driven flow dR/dt = -Delta R + (kKH/2) R (1 - R^2) and does NOT match Eq.(16)
    or direct simulation. Verified against simulation: the H-driven form is correct.
    """
    if kKH < 1e-14:
        return 0.0
    return (-Delta + np.sqrt(Delta**2 + kKH**2)) / kKH


def free_flow_tau(R0, kKH, Delta, tau):
    """Exact solution of the H-driven OA free flow
        dR/dt = -Delta R + (kKH/2)(1 - R^2)
    over time tau. The flow is a logistic between the two roots
        R_pm = (-Delta +/- sqrt(Delta^2 + (kKH)^2)) / (kKH),
    giving R(tau) = (R+ - R- E)/(1 - E) with E = [(R0-R+)/(R0-R-)] exp(-disc * tau),
    disc = sqrt(Delta^2 + (kKH)^2). Verified against fine numerical integration to ~1e-5.
    """
    R0 = min(max(R0, 0.0), 1.0)
    a = kKH
    if a < 1e-14:
        return R0*np.exp(-Delta*tau)
    disc = np.sqrt(Delta**2 + a**2)
    Rp = (-Delta + disc)/a
    Rm = (-Delta - disc)/a
    C = (R0 - Rp)/(R0 - Rm)
    E = C*np.exp(-disc*tau)
    R = (Rp - Rm*E)/(1 - E)
    return min(max(R, 0.0), 1.0)


# =====================================================================
# Measurement convention (pre-kick / stroboscopic sampling)
# =====================================================================
# The hybrid map is M = K_eps o Phi_tau, so its fixed point R* is the
# order parameter measured IMMEDIATELY AFTER a kick.  The direct
# simulations sample R stroboscopically at t = n*tau, that is,
# immediately BEFORE each kick, and they select the kicked subset from
# that same pre-kick phase distribution.
#
# The two sampling points differ at O(eps), and the difference shrinks
# monotonically with delta (3.7% at K=1.2Kc, delta=0 down to 0.5% at
# K=3Kc, delta=0.8 for eps=0.05).  Because the delta-dependence of that
# offset is comparable to the delta-dependence of R itself, theory and
# simulation must be reported on the same convention.
#
# We adopt the pre-kick convention throughout:
#     R_obs     = Phi_tau(R*)
#     sigma_obs = sigma_delta(Phi_tau(R*))
# Both the order parameter and the cost variable are therefore quoted
# at the kick instant, which is what an operator would actually observe
# and what the simulation records.

def phi_tau_a2a(R0, K, Delta, tau):
    """All-to-all free flow of dR/dt = -Delta R + (K/2) R (1-R^2) over time tau."""
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


def observables_a2a(R_star, K, Delta, tau, delta):
    """Pre-kick observables for the all-to-all hybrid map.

    Parameters
    ----------
    R_star : post-kick fixed point, i.e. the solution of R = K_eps(Phi_tau(R)).

    Returns
    -------
    (R_obs, sigma_obs) : order parameter and kicked fraction as sampled
                         immediately before a kick.
    """
    R_obs = phi_tau_a2a(R_star, K, Delta, tau)
    return R_obs, kicked_fraction(R_obs, delta)


def observables_network(Rk_star, H_star, K, Delta, tau, delta, deg_array):
    """Pre-kick observables for the annealed network hybrid map.

    Each degree class is propagated over one free-flow interval with the
    self-consistent mean field H_star held fixed, exactly as in the map
    itself, and the observables are read off at the end of that interval.

    Parameters
    ----------
    Rk_star   : dict {degree -> post-kick fixed-point R_k}
    H_star    : self-consistent degree-weighted mean field at the fixed point
    deg_array : degree of every node (used for the node average and P(k))

    Returns
    -------
    (R_obs, sigma_tot_obs, Rk_obs)
    """
    unique_k, counts = np.unique(deg_array, return_counts=True)
    Pk = counts.astype(float)/counts.sum()
    Rk_obs = {}
    for k in unique_k:
        k = int(k)
        Rk_obs[k] = free_flow_tau(Rk_star[k], k*K*H_star, Delta, tau)
    R_obs = float(sum(Rk_obs[int(k)] for k in deg_array)/len(deg_array))
    sigma_tot_obs = float(sum(Pk[i]*kicked_fraction(Rk_obs[int(unique_k[i])], delta)
                              for i in range(len(unique_k))))
    return R_obs, sigma_tot_obs, Rk_obs


if __name__ == "__main__":
    from scipy import integrate

    def rho(phi, R):
        return (1 - R**2) / (2*np.pi*(1 - 2*R*np.cos(phi) + R**2))

    def kick_exact(R, eps, delta):
        if delta >= 1:
            v, _ = integrate.quad(lambda p: rho(p, R)*np.cos(p), 0, np.pi)
            return 2*v
        a = np.arcsin(delta); b = np.pi - a
        v1, _ = integrate.quad(lambda p: rho(p, R)*np.cos(p), 0, a)
        v2, _ = integrate.quad(lambda p: rho(p, R)*np.cos(p-eps), a, b)
        v3, _ = integrate.quad(lambda p: rho(p, R)*np.cos(p), b, np.pi)
        return 2*(v1+v2+v3)

    print("Verifying corrected kick map vs direct integration:")
    maxerr = 0.0
    for delta in [0.0, 0.3, 0.5, 0.8]:
        for R in [0.1, 0.5, 0.9]:
            for eps in [0.05, -0.05]:
                e = abs(kick(R, eps, delta) - kick_exact(R, eps, delta))
                maxerr = max(maxerr, e)
    print(f"  max error = {maxerr:.2e}")
