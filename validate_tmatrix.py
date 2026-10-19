"""Validation suite for tmatrix.py (EBCM, axisymmetric particles).

There is no independent T-matrix package that installs without a Fortran compiler
(pytmatrix needs one), so validation leans on analytic limits and internal consistency
between *independent routes to the same number*. Four families:

1. The sphere limit. At eps = 1 the EBCM must reproduce Lorenz-Mie exactly, for the
   cross sections, the amplitude matrix, and the orientation-averaged asymmetry
   parameter. This exercises the entire chain (quadrature, Wigner d functions,
   Bessel tables, Q/RgQ assembly, matrix inversion, amplitude assembly) against a
   fully independent implementation, mie.py, which is itself validated 174 ways.

2. Shape-sensitive analytic limits. A small spheroid must reproduce the Rayleigh
   ellipsoid result built from depolarisation factors. This is the test that would
   catch a wrong particle profile or a mis-assembled Q matrix for a *non-sphere*,
   which the sphere limit cannot see.

3. Independent routes to Csca. The exact T-matrix trace formula and the numerical
   angular quadrature of the phase function must agree. They share no code path
   beyond the T-matrix itself.

4. Energy conservation and unitarity. For a lossless particle S = I + 2T must be
   unitary. The residual is set by multipole truncation, so we assert it *decays*
   with nmax rather than pinning an absolute value.

Run:  python validate_tmatrix.py
"""

import numpy as np

import mie
import tmatrix as tm

np.seterr(all="ignore")   # numpy 2.0 complex matmul emits spurious FPE flags

results = []


def check(name, err, tol):
    ok = np.isfinite(err) and err < tol
    results.append((name, err, tol, ok))
    return ok


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def relmax(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def depolarisation_polar(eps):
    """L along the symmetry axis of a spheroid with axis ratio eps = a_eq / c_polar."""
    if abs(eps - 1.0) < 1e-12:
        return 1.0 / 3.0
    if eps > 1.0:                                   # oblate
        f = np.sqrt(eps * eps - 1.0)
        return (1.0 + f * f) / (f * f) * (1.0 - np.arctan(f) / f)
    e = np.sqrt(1.0 - eps * eps)                    # prolate
    return (1.0 - e * e) / (e * e) * (-1.0 + np.log((1.0 + e) / (1.0 - e)) / (2.0 * e))


print("=" * 78)
print("PART 1  the sphere limit: eps = 1 must reproduce Lorenz-Mie exactly")
print("=" * 78)

SPHERE_CASES = [
    ("dielectric", 1.5 + 0.0j, 3.0),
    ("absorbing", 1.5 + 0.05j, 4.0),
    ("low contrast", 1.042 + 0.0j, 5.0),
    ("larger", 1.33 + 0.0j, 8.0),
]

for name, m, x in SPHERE_CASES:
    rev, k = 1.0, x
    blocks, nmax = tm.tmatrix(rev, k, m, tm.SPHEROID, eps=1.0)
    cs = tm.cross_sections_random(blocks, k, rev)
    e = mie.efficiencies(m, x)
    # Qext is LINEAR in T (a trace), Qsca is QUADRATIC (a sum of |T|^2), so Qext
    # converges more slowly in nmax. Mishchenko's default nmax = x + 4.05 x^(1/3) is
    # also smaller than Wiscombe's x + 4 x^(1/3) + 2 used by mie.py. Hence the split
    # tolerances here, and the explicit convergence check below.
    check(f"[{name}] Qext == Mie", rel(cs["qext"], e["qext"]), 1e-9)
    check(f"[{name}] Qsca == Mie", rel(cs["qsca"], e["qsca"]), 1e-10)

    b3, _ = tm.tmatrix(rev, k, m, tm.SPHEROID, eps=1.0, nmax=nmax + 3, ngauss=(nmax + 3) * 2)
    cs3 = tm.cross_sections_random(b3, k, rev)
    check(f"[{name}] Qext == Mie at nmax+3", rel(cs3["qext"], e["qext"]), 1e-12)

    Tf = tm.full_T(blocks, nmax)
    th = np.linspace(0.15, np.pi - 0.15, 13)

    # axial incidence, scattering plane through z: the meridional basis coincides
    # with the Bohren & Huffman (parallel, perpendicular) basis
    S = tm.amplitudes_vec(Tf, nmax, k, 1.0, np.cos(th), np.zeros(th.size))
    s1, s2 = mie.amplitudes(m, x, th)
    check(f"[{name}] |S_tt| == |S2|/k", relmax(np.abs(S[:, 0, 0]), np.abs(s2) / k), 1e-7)
    check(f"[{name}] |S_pp| == |S1|/k", relmax(np.abs(S[:, 1, 1]), np.abs(s1) / k), 1e-7)
    check(f"[{name}] S off-diagonal == 0",
          float(np.max(np.abs(S[:, 0, 1])) + np.max(np.abs(S[:, 1, 0]))) / np.max(np.abs(s1) / k), 1e-12)

    # rotational invariance: a sphere cannot know where its "axis" is
    th0 = 1.0
    S = tm.amplitudes_vec(Tf, nmax, k, np.cos(th0), np.cos(th), np.zeros(th.size))
    s1o, s2o = mie.amplitudes(m, x, np.abs(th - th0))
    check(f"[{name}] rotational invariance |S_tt|", relmax(np.abs(S[:, 0, 0]), np.abs(s2o) / k), 1e-7)
    check(f"[{name}] rotational invariance |S_pp|", relmax(np.abs(S[:, 1, 1]), np.abs(s1o) / k), 1e-7)

    # orientation average of a sphere is the sphere
    _, _, csca_q, g = tm.phase_function_random(blocks, nmax, k, n_beta=8, n_psi=16, n_theta=90)
    check(f"[{name}] orientation-averaged g == Mie g", rel(g, e["g"]), 1e-9)
    check(f"[{name}] Csca quadrature == exact T", rel(csca_q, cs["csca"]), 1e-9)

print()
print("=" * 78)
print("PART 2  Rayleigh ellipsoid limit: catches a wrong profile that eps=1 cannot")
print("=" * 78)

for eps in (1.0, 1.5, 2.0, 3.0, 0.6, 0.35):
    rev, x, m = 1.0, 0.03, 1.5 + 0.0j
    k = x
    blocks, _ = tm.tmatrix(rev, k, m, tm.SPHEROID, eps=eps, nmax=4, ngauss=16)
    cs = tm.cross_sections_random(blocks, k, rev)

    epsr = m ** 2
    V = 4.0 / 3.0 * np.pi * rev ** 3
    Lc = depolarisation_polar(eps)
    La = 0.5 * (1.0 - Lc)
    a_eq = V * (epsr - 1) / (1 + La * (epsr - 1))
    a_po = V * (epsr - 1) / (1 + Lc * (epsr - 1))
    # <|alpha . e|^2> over random rotations = (1/3) sum_j |alpha_j|^2
    csca_ray = (k ** 4 / (6 * np.pi)) * (2 * abs(a_eq) ** 2 + abs(a_po) ** 2) / 3.0

    # the Rayleigh formula itself carries an O(x^2) error, hence the 1e-3 tolerance
    check(f"Rayleigh spheroid eps={eps}", rel(cs["csca"], csca_ray), 1e-3)

print()
print("=" * 78)
print("PART 3  independent routes to Csca, and energy conservation, for NON-spheres")
print("=" * 78)

SHAPES = [
    ("oblate 2.0", tm.SPHEROID, 2.0, 2),
    ("prolate 0.5", tm.SPHEROID, 0.5, 2),
    ("cylinder D/L=1", tm.CYLINDER, 1.0, 2),
    ("cylinder D/L=0.5", tm.CYLINDER, 0.5, 2),
    ("chebyshev T2(0.1)", tm.CHEBYSHEV, 0.1, 2),
]

for name, shape, eps, cn in SHAPES:
    rev, k = 1.0, 3.0
    m = 1.4 + 0.0j
    blocks, nmax = tm.tmatrix(rev, k, m, shape, eps=eps, nmax=14, ngauss=56, cheb_n=cn)
    cs = tm.cross_sections_random(blocks, k, rev)
    _, _, csca_q, _ = tm.phase_function_random(blocks, nmax, k, n_beta=12, n_psi=24, n_theta=90)

    check(f"[{name}] Csca: exact T vs angular quadrature", rel(csca_q, cs["csca"]), 1e-8)
    check(f"[{name}] Qabs == 0 (lossless)", abs(cs["qabs"]) / cs["qsca"], 1e-6)

    # ngauss independence: the surface integrals must be converged
    b2, _ = tm.tmatrix(rev, k, m, shape, eps=eps, nmax=14, ngauss=112, cheb_n=cn)
    cs2 = tm.cross_sections_random(b2, k, rev)
    check(f"[{name}] Qsca independent of ngauss", rel(cs2["qsca"], cs["qsca"]), 1e-6)

    # absorbing variant must have Qabs > 0 and Qext = Qsca + Qabs by construction
    ba, _ = tm.tmatrix(rev, k, 1.4 + 0.05j, shape, eps=eps, nmax=14, ngauss=56, cheb_n=cn)
    csa = tm.cross_sections_random(ba, k, rev)
    check(f"[{name}] absorbing: 0 < albedo < 1", 0.0 if 0 < csa["albedo"] < 1 else 1.0, 0.5)

print()
print("=" * 78)
print("PART 4  lossless unitarity: S = I + 2T must be unitary as nmax -> infinity")
print("=" * 78)


def unitarity(shape, eps, nmax, ngauss, x=3.0, m=1.4 + 0j, cn=2):
    blocks, _ = tm.tmatrix(1.0, x, m, shape, eps=eps, nmax=nmax, ngauss=ngauss, cheb_n=cn)
    worst = 0.0
    for T in blocks.values():
        U = np.eye(T.shape[0]) + 2 * T
        worst = max(worst, float(np.max(np.abs(U.conj().T @ U - np.eye(T.shape[0])))))
    return worst


check("sphere: unitarity exact", unitarity(tm.SPHEROID, 1.0, 10, 40), 1e-12)

for name, shape, eps in [("oblate 2.0", tm.SPHEROID, 2.0), ("prolate 0.5", tm.SPHEROID, 0.5)]:
    u_lo = unitarity(shape, eps, 10, 40)
    u_hi = unitarity(shape, eps, 16, 64)
    check(f"[{name}] unitarity converged at high nmax", u_hi, 1e-8)
    # the residual is truncation, so it must fall by orders of magnitude
    check(f"[{name}] unitarity residual decays with nmax", u_hi / u_lo, 1e-2)

# a corner (the cylinder) converges much more slowly: assert decay, not a bound
u_lo = unitarity(tm.CYLINDER, 1.0, 8, 32)
u_hi = unitarity(tm.CYLINDER, 1.0, 20, 80)
check("cylinder: unitarity residual decays (slowly: it has a corner)", u_hi / u_lo, 0.5)

print()
print("=" * 78)
print("PART 5  the convergence driver, and its honest failure mode")
print("=" * 78)

blocks, nmax, ngauss = tm.tmatrix_converged(1.0, 5.0, 1.3 + 0j, tm.SPHEROID, eps=1.0)
cs = tm.cross_sections_random(blocks, 5.0, 1.0)
e = mie.efficiencies(1.3 + 0j, 5.0)
check("tmatrix_converged reproduces Mie", rel(cs["qsca"], e["qsca"]), 1e-8)

try:
    tm.tmatrix_converged(1.0, 5.0, 1.3 + 0j, tm.SPHEROID, eps=1.0, nmax_cap=5)
    check("converged() raises at nmax_cap", 1.0, 0.5)
except RuntimeError:
    check("converged() raises at nmax_cap", 0.0, 0.5)

print()
print("=" * 78)
print("PART 6  biconcave disc: correct where EBCM converges, and its hard boundary")
print("=" * 78)

# The Evans-Fung disc is star-shaped and axisymmetric, so EBCM applies in principle.
# In practice the deep central concavity restricts the null-field method's
# double-precision convergence domain to the near-Rayleigh regime (x_ev <~ 0.5):
# beyond it the albedo *grows* with the multipole cutoff (the series diverges), so
# adding terms makes the answer worse. These tests assert (a) the geometry is exact,
# (b) the physics is correct where EBCM does converge, and (c) the divergence sets in
# where we say it does. The RBC itself (x_ev ~ 10-30) is far beyond reach; the true
# biconcave number needs DDA or extended precision. See rbc_shape_study.py STEP 0.

# (a) geometry: equal volume to the sphere, star-shaped (r(theta) single-valued)
rev = 1.0
a0, a1, a2 = tm._EF
R0 = rev * ((4.0 / 3.0 * np.pi) / tm._RBC_VDIM) ** (1.0 / 3.0)
rho = np.linspace(1e-6, R0 * (1 - 1e-9), 20000)
u = rho / R0
zt = R0 * np.sqrt(1 - u * u) * (a0 + a1 * u * u + a2 * u ** 4)
theta_of_rho = np.arctan2(rho, zt)
check("disc: theta(rho) monotonic (star-shaped)",
      0.0 if np.all(np.diff(theta_of_rho) > 0) else 1.0, 0.5)
Vdisc = np.trapezoid(4 * np.pi * rho * zt, rho)
check("disc: volume equals equal-volume sphere", rel(Vdisc, 4.0 / 3.0 * np.pi * rev ** 3), 1e-5)

# (b) deep Rayleigh limit: g -> 0 for ANY shape (dipole radiation). This is a strong,
# shape-independent proof the implementation computes the right physics, and it needs
# the disc's characteristically high ngauss to pass.
b, nmax = tm.tmatrix(0.03, 1.0, 1.5 + 0j, tm.BICONCAVE, eps=1.0, nmax=6, ngauss=6 * 200)
cs = tm.cross_sections_random(b, 1.0, 0.03)
_, _, _, g = tm.phase_function_random(b, nmax, 1.0, 8, 20, 60)
check("disc: Rayleigh g -> 0 (implementation correct)", abs(g), 2e-3)
check("disc: Rayleigh energy conservation", abs(cs["albedo"] - 1.0), 1e-4)
# and it is stable there under a large change in ngauss
b2, _ = tm.tmatrix(0.03, 1.0, 1.5 + 0j, tm.BICONCAVE, eps=1.0, nmax=6, ngauss=6 * 400)
check("disc: Rayleigh stable under ngauss",
      rel(tm.cross_sections_random(b2, 1.0, 0.03)["qsca"], cs["qsca"]), 1e-6)


def disc_albedo_err(xev, m, nmax, ndgs=40):
    b_, _ = tm.tmatrix(xev, 1.0, m, tm.BICONCAVE, eps=1.0, nmax=nmax, ngauss=nmax * ndgs)
    return abs(tm.cross_sections_random(b_, 1.0, xev)["albedo"] - 1.0)


# (c) the divergence signature: for x_ev = 1 the albedo gets WORSE as nmax grows,
# the hallmark of EBCM breakdown (a convergent shape like a spheroid gets better).
err_lo = disc_albedo_err(1.0, 1.5 + 0j, 6)
err_hi = disc_albedo_err(1.0, 1.5 + 0j, 12)
check("disc: x_ev=1 diverges with nmax (albedo grows, not shrinks)",
      0.0 if err_hi > 3.0 * err_lo else 1.0, 0.5)

# (d) at RBC size the disc is hopeless in double precision, at RBC contrast, at any
# resolution. Asserted so a future change that appears to converge here is scrutinised.
best = min(disc_albedo_err(3.0, 1.042 + 0j, nmax, ndgs)
           for nmax in (8, 10, 12) for ndgs in (30, 60))
check("disc: RBC-regime (x_ev=3) beyond double-precision EBCM",
      0.0 if best > 1e-1 else 1.0, 0.5)

print()
print("=" * 78)
npass = sum(1 for *_, ok in results if ok)
for name, err, tol, ok in results:
    if not ok:
        print(f"  FAIL  {name:56s} err={err:.3e} tol={tol:.1e}")
print(f"{npass}/{len(results)} checks passed")
print("=" * 78)
raise SystemExit(0 if npass == len(results) else 1)
