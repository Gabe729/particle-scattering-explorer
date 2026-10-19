"""Validation suite for mie.py.

Two independent kinds of check:

1. Internal consistency and analytic limits. These need no external package and
   test the physics directly (optical theorem, energy conservation by quadrature,
   Maxwell boundary conditions at r = a, the Rayleigh limit, the far-field
   asymptotics of the near-field expansion).

2. Cross-check against `miepython`, a separately written implementation.
   miepython uses the m = n - ik convention, so we pass conj(m).

miepython is optional: without it Part 2 is skipped and fewer checks run.

Run:  python validate.py
"""

import numpy as np
import mie

TOL = 1e-6
results = []


def check(name, err, tol=TOL):
    ok = np.isfinite(err) and err < tol
    results.append((name, err, tol, ok))
    return ok


def relerr(a, b):
    """Elementwise relative error. Use only where every element is O(1)."""
    a, b = np.asarray(a), np.asarray(b)
    denom = np.maximum(np.abs(b), 1e-30)
    return float(np.max(np.abs(a - b) / denom))


def relerr_max(a, b):
    """Error normalised by the largest element of the reference.

    The right metric for a decaying coefficient array: the tail terms are ~1e-25
    and carry no information, so an elementwise relative error there is noise.
    """
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


CASES = [
    ("dielectric, small", 1.5 + 0.0j, 0.8),
    ("dielectric, resonant", 1.33 + 0.0j, 5.0),
    ("dielectric, large", 1.5 + 0.0j, 25.0),
    ("absorbing", 1.5 + 0.1j, 3.0),
    ("strongly absorbing", 1.7 + 0.5j, 6.0),
    ("metallic (gold-like)", 0.54 + 2.18j, 0.4),
    ("biological (RBC-like)", 1.042 + 0.0j, 29.3),
]

print("=" * 74)
print("PART 1  internal consistency and analytic limits")
print("=" * 74)

# ---- T1: m = 1 must reduce to a bare plane wave -------------------------
a, b = mie.mie_ab(1.0 + 0j, 4.0)
c, d = mie.mie_cd(1.0 + 0j, 4.0)
check("m=1: a_n, b_n vanish", float(np.max(np.abs(np.r_[a, b]))), 1e-12)
check("m=1: c_n, d_n equal 1", float(np.max(np.abs(np.r_[c, d] - 1.0))), 1e-12)
_, _, I = mie.near_field_slice(1.0 + 0j, 4.0, half_extent=2.0, n_grid=40)
check("m=1: |E|^2 == 1 everywhere", float(np.max(np.abs(I - 1.0))), 1e-6)

# Odd grid so a sample lands exactly on the origin: r = 0 is the one place where
# theta is ill-defined, and it must still return the plane wave. (The JS port got
# this wrong by deriving cos(theta) = z/r from a clamped r; nothing caught it until
# the browser self-test, because the even grid above never lands on r = 0.)
# The residual here is dominated by truncation at r -> 1^-, not by the origin, which
# is exact to 2e-16.
for ng in (41, 61):
    _, _, I0 = mie.near_field_slice(1.0 + 0j, 4.0, half_extent=2.0, n_grid=ng)
    check(f"m=1: |E|^2 == 1 including the origin (n_grid={ng})",
          float(np.max(np.abs(I0 - 1.0))), 5e-6)

# the residual above is truncation of the plane-wave VSH expansion, nothing else:
# it converges to machine precision once extra terms are retained
th_ = np.linspace(0.05, np.pi - 0.05, 40)
Er, Et, Ep = mie.fields_spherical(1.0 + 0j, 4.0, np.full_like(th_, 0.9), th_,
                                  np.full_like(th_, 0.7),
                                  nmax=mie.n_stop(4.0) + 9, part="internal")
I9 = np.abs(Er) ** 2 + np.abs(Et) ** 2 + np.abs(Ep) ** 2
check("m=1: VSH truncation converges", float(np.max(np.abs(I9 - 1.0))), 1e-12)

# ---- T2/T3/T4/T5: per-case invariants -----------------------------------
for name, m, x in CASES:
    eff = mie.efficiencies(m, x)
    # trapezoid error is O(h^2) and the phase function oscillates ~x times,
    # so the angular grid has to grow with x
    theta = np.linspace(0, np.pi, max(4001, int(700 * x) + 1))
    s1, s2 = mie.amplitudes(m, x, theta)

    # optical theorem: Qext = (4/x^2) Re S(0)
    s0 = mie.amplitudes(m, x, np.array([0.0]))[0][0]
    check(f"[{name}] optical theorem", abs((4 / x ** 2) * s0.real - eff["qext"]) / eff["qext"])

    # energy conservation: Qsca by angular quadrature
    w = np.abs(s1) ** 2 + np.abs(s2) ** 2
    qsca_q = np.trapezoid(w * np.sin(theta), theta) / x ** 2
    check(f"[{name}] Qsca by quadrature", abs(qsca_q - eff["qsca"]) / eff["qsca"], 1e-5)

    # asymmetry parameter by quadrature
    num = np.trapezoid(w * np.cos(theta) * np.sin(theta), theta)
    den = np.trapezoid(w * np.sin(theta), theta)
    g_q = num / den
    check(f"[{name}] g by quadrature", abs(g_q - eff["g"]) / max(abs(eff["g"]), 1e-3), 1e-4)

    # non-absorbing spheres must not absorb
    if m.imag == 0:
        check(f"[{name}] Qabs == 0", abs(eff["qabs"]) / eff["qsca"], 1e-9)

# ---- T5b: complex spherical Bessel functions against closed forms --------
# Independent of the Miller recurrence that generates them, so this validates
# both the recurrence and its normalisation.
for w in (0.3 + 0.2j, 2.0 + 0.0j, 8.0 - 3.0j, 30.0 + 1.0j):
    jn = mie.spherical_jn_complex(12, w)[:, 0]
    j0 = np.sin(w) / w
    j1 = np.sin(w) / w ** 2 - np.cos(w) / w
    j2 = (3.0 / w ** 3 - 1.0 / w) * np.sin(w) - 3.0 * np.cos(w) / w ** 2
    check(f"j_n(w={w}) closed form n=0,1,2",
          relerr(np.array([jn[0], jn[1], jn[2]]), np.array([j0, j1, j2])), 1e-10)
    n = np.arange(1, len(jn) - 1)
    resid = jn[:-2] + jn[2:] - (2 * n + 1) / w * jn[1:-1]
    check(f"j_n(w={w}) three-term recurrence", float(np.max(np.abs(resid))) / np.max(np.abs(jn)), 1e-10)

# ---- T6: Rayleigh limit --------------------------------------------------
x = 0.01
m = 1.5 + 0.0j
eff = mie.efficiencies(m, x)
ray = (8.0 / 3.0) * x ** 4 * abs((m ** 2 - 1) / (m ** 2 + 2)) ** 2
check("Rayleigh limit: Qsca", abs(eff["qsca"] - ray) / ray, 1e-4)

m = 1.5 + 0.1j
eff = mie.efficiencies(m, x)
ray_abs = 4 * x * ((m ** 2 - 1) / (m ** 2 + 2)).imag
check("Rayleigh limit: Qabs", abs(eff["qabs"] - ray_abs) / ray_abs, 1e-3)

# ---- T7: Maxwell boundary conditions at r = a ---------------------------
# Tangential E continuous; normal D continuous  =>  E_r,out = m^2 E_r,in.
# Residual is set by truncation of the Mie series *at the sphere surface*, where
# it converges more slowly than in the far field. Do NOT try to fix this by
# raising nmax: past roughly n_stop(x) + 10 the upward recurrence for the real
# Riccati-Bessel psi_n(rho) goes unstable (n >> rho) and the error explodes to
# O(1). Wiscombe's truncation is the sweet spot, and 1e-5 at the surface is it.
for name, m, x in CASES:
    if abs(m.imag) * x > 25:
        continue
    th = np.linspace(0.05, np.pi - 0.05, 37)
    ph = np.full_like(th, 0.7)
    eps = 1e-7
    Ei = mie.fields_spherical(m, x, np.full_like(th, 1 - eps), th, ph, part="internal")
    Eo = mie.fields_spherical(m, x, np.full_like(th, 1 + eps), th, ph, part="total")
    check(f"[{name}] BC: E_theta continuous", relerr_max(Ei[1], Eo[1]), 5e-5)
    check(f"[{name}] BC: E_phi continuous", relerr_max(Ei[2], Eo[2]), 5e-5)
    check(f"[{name}] BC: D_r continuous", relerr_max(m ** 2 * Ei[0], Eo[0]), 5e-4)

# ---- T8: near field -> far field asymptotics -----------------------------
# The residual is the O(1/kr) correction to the asymptotic form, so we assert
# both the magnitude at large kr and that it actually decays like 1/kr.
# Elementwise relative error is meaningless here: the pattern has deep nulls.
for name, m, x in CASES:
    th = np.linspace(0.05, np.pi - 0.05, 61)
    ph = np.full_like(th, np.pi / 4)
    s1, s2 = mie.amplitudes(m, x, th)
    rhs = np.abs(s2) ** 2 * np.cos(ph) ** 2 + np.abs(s1) ** 2 * np.sin(ph) ** 2

    # the neglected term in h_n(rho) ~ (1 + i n(n+1)/2rho), so the residual scales
    # as nmax^2 / kr; picking kr proportional to nmax^2 makes the test scale-free
    nmax = mie.n_stop(x)
    kr_far = 20.0 * nmax ** 2
    errs = {}
    for kr in (kr_far / 4, kr_far):
        _, Et, Ep = mie.fields_spherical(m, x, np.full_like(th, kr / x), th, ph, part="scattered")
        lhs = kr ** 2 * (np.abs(Et) ** 2 + np.abs(Ep) ** 2)
        errs[kr] = relerr_max(lhs, rhs)

    check(f"[{name}] near->far asymptotics", errs[kr_far], 3e-3)
    # quadrupling kr must shrink the residual ~4x if it is genuinely O(1/kr)
    ratio = errs[kr_far / 4] / max(errs[kr_far], 1e-15)
    check(f"[{name}] near->far residual is O(1/kr)", 1.0 / ratio, 1.0 / 3.0)

print()
print("=" * 74)
print("PART 2  cross-check against miepython (independent implementation)")
print("=" * 74)

try:
    import miepython as mp
except ImportError:
    mp = None
    print("  miepython not installed: skipping the optional cross-check "
          "(pip install miepython to run it)")

try:
    if mp is None:
        raise StopIteration

    # Convention mapping, established empirically and anchored on the optical theorem:
    #   * efficiencies_mx expects m = n - ik.
    #   * an_bn / cn_dn are insensitive to the sign of Im(m) and return the same
    #     coefficients as this module does.
    #   * S1_S2 conjugates internally, so compare against conj(S).
    #   * S1_S2(norm='bohren') is 2x the textbook amplitude despite the name;
    #     norm='wiscombe' is the one that satisfies Qext = (4/x^2) Re S(0).
    # Everything the explorer displays (|E|^2, dC/dOmega, the efficiencies) is
    # invariant under a global conjugation, so only magnitudes are load-bearing.

    for name, m, x in CASES:
        mm = np.conj(m)

        qext, qsca, qback, g = mp.efficiencies_mx(mm, x)
        eff = mie.efficiencies(m, x)
        check(f"[{name}] miepython Qext", abs(qext - eff["qext"]) / eff["qext"], 1e-6)
        check(f"[{name}] miepython Qsca", abs(qsca - eff["qsca"]) / eff["qsca"], 1e-6)
        check(f"[{name}] miepython Qback", abs(qback - eff["qback"]) / max(eff["qback"], 1e-12), 1e-5)
        check(f"[{name}] miepython g", abs(g - eff["g"]) / max(abs(eff["g"]), 1e-3), 1e-5)

        an, bn = mp.an_bn(mm, x)
        a, b = mie.mie_ab(m, x)
        k = min(len(an), len(a))
        check(f"[{name}] miepython a_n", relerr_max(a[:k], an[:k]), 1e-8)
        check(f"[{name}] miepython b_n", relerr_max(b[:k], bn[:k]), 1e-8)

        # miepython's cn_dn agrees to ~3e-5, not to machine precision. Ours is the
        # more accurate of the two: it reproduces c_n = d_n = 1 at m = 1 to 1e-12
        # and satisfies the Maxwell boundary conditions above. Tolerance reflects
        # miepython's accuracy, not ours.
        cn, dn = mp.cn_dn(mm, x, len(a))
        c, d = mie.mie_cd(m, x)
        k = min(len(cn), len(c))
        check(f"[{name}] miepython c_n", relerr_max(c[:k], cn[:k]), 1e-4)
        check(f"[{name}] miepython d_n", relerr_max(d[:k], dn[:k]), 1e-4)

        th = np.linspace(0.02, np.pi - 0.02, 25)
        S1, S2 = mp.S1_S2(mm, x, np.cos(th), norm="wiscombe")
        s1, s2 = mie.amplitudes(m, x, th)
        check(f"[{name}] miepython S1", relerr_max(np.conj(s1), S1), 1e-8)
        check(f"[{name}] miepython S2", relerr_max(np.conj(s2), S2), 1e-8)

    # near field: unit vacuum wavelength, n_env = 1 so x = pi*d/lambda0
    for name, m, x in CASES:
        if abs(m.imag) * x > 25:
            continue
        lam, n_env = 1.0, 1.0
        d_sph = x / np.pi
        th = np.linspace(0.1, np.pi - 0.1, 17)
        ph = np.full_like(th, 0.6)
        for rfac in (0.5, 0.9, 1.5, 4.0):
            r = rfac * d_sph / 2
            out = mp.e_near(lam, d_sph, np.conj(m), n_env, r, th, ph)
            Er, Et, Ep = mie.fields_spherical(m, x, np.full_like(th, rfac), th, ph)
            got = np.abs(Er) ** 2 + np.abs(Et) ** 2 + np.abs(Ep) ** 2
            ref = np.sum(np.abs(np.asarray(out)) ** 2, axis=0)
            check(f"[{name}] miepython |E|^2 at r/a={rfac}", relerr(got, ref), 1e-5)

except StopIteration:
    pass
except Exception as exc:  # pragma: no cover
    # miepython is installed but its API did not behave as expected (e.g. a major
    # version change). Fail loudly rather than silently reporting fewer checks.
    print(f"  miepython cross-check raised {exc!r}")
    check("miepython cross-check ran to completion", float("inf"))

print()
print("=" * 74)
npass = sum(1 for *_, ok in results if ok)
for name, err, tol, ok in results:
    if not ok:
        print(f"  FAIL  {name:52s} err={err:.3e} tol={tol:.1e}")
print(f"{npass}/{len(results)} checks passed")
print("=" * 74)
raise SystemExit(0 if npass == len(results) else 1)
