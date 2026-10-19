"""T-matrix (extended boundary condition method) for axisymmetric particles.

Waterman's EBCM as implemented by M. I. Mishchenko. This module is a direct port of
the public-domain Fortran `ampld.lp.f` (NASA GISS), taken from the copy vendored in
`pytmatrix`, where the non-profit-use restriction was dropped with Mishchenko's
permission.

    Acknowledgement, as requested by the authors:
    M. I. Mishchenko and L. D. Travis, "T-matrix computations of light scattering by
    large spheroidal particles", Opt. Commun. 109, 16-21 (1994).
    M. I. Mishchenko, L. D. Travis, D. W. Mackowski, "T-matrix computations of light
    scattering by nonspherical particles: a review", JQSRT 55, 535-575 (1996).

Conventions (identical to mie.py, so the two are directly comparable)
---------------------------------------------------------------------
* exp(-i omega t); m = n + ik relative to the surrounding medium, k >= 0.
* The particle's symmetry axis is the z axis of the particle frame.
* Size is set by `rev`, the equal-volume-sphere radius, so a spheroid and its
  volume-equivalent sphere share the same size parameter x = k * rev.
* `eps` is the axis ratio (equatorial semi-axis)/(polar semi-axis) for spheroids:
  eps > 1 oblate, eps < 1 prolate, eps = 1 a sphere. At eps = 1 this module must
  reproduce Lorenz-Mie exactly; that is the sharpest test in validate_tmatrix.py.
* For cylinders `eps` is diameter/length. For Chebyshev particles it is the
  deformation parameter and `cheb_n` the degree.

The T-matrix is block diagonal in the azimuthal index m and is *independent of the
particle's orientation*: once T is known, changing the incidence direction is a
cheap operation on it. That is what makes an incidence-angle slider free.
"""

import numpy as np

from mie import spherical_jn_complex

__all__ = [
    "SPHEROID", "CYLINDER", "CHEBYSHEV", "BICONCAVE",
    "gauss_nodes", "profile", "vig", "vigampl",
    "tmatrix", "tmatrix_converged", "tmatrix_best_lossless", "cross_sections_random",
    "full_T", "amplitude_matrix", "amplitudes_vec", "phase_function_random",
]

SPHEROID = "spheroid"
CYLINDER = "cylinder"
CHEBYSHEV = "chebyshev"
BICONCAVE = "biconcave"

# Evans & Fung (1972) dimensionless red-blood-cell profile. With u = rho / R0 in
# [0, 1] the half-thickness of the disc is
#     z_top(u) = R0 * sqrt(1 - u^2) * (A0 + A1 u^2 + A2 u^4).
# At u = 0 this gives the central dimple (z_top = A0 R0), and the surface rises to a
# rim before returning to z = 0 at the equator. The shape is axisymmetric about z, so
# EBCM applies; it is star-shaped with respect to its centroid (verified numerically
# in validate_tmatrix.py), so r(theta) is single-valued.
_EF = (0.1035805, 1.001279, -0.561381)


def _rbc_ztop(u):
    a0, a1, a2 = _EF
    return np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0)) * (a0 + a1 * u * u + a2 * u ** 4)


# dimensionless volume for R0 = 1:  V = int_0^1 4 pi u z_top(u) du
_u = np.linspace(0.0, 1.0, 200001)
_RBC_VDIM = float(np.trapezoid(4.0 * np.pi * _u * _rbc_ztop(_u), _u))
del _u


# --------------------------------------------------------------------------
# quadrature and particle profile
# --------------------------------------------------------------------------

def gauss_nodes(ngauss, shape=SPHEROID, eps=1.0):
    """Nodes/weights in cos(theta) on [-1,1], increasing and antisymmetric.

    A cylinder's profile has a corner where the flat cap meets the side. Mishchenko
    splits the quadrature there (CONST, NP=-2); integrating across a kink with one
    smooth rule converges far more slowly.
    """
    ng = 2 * ngauss
    if shape != CYLINDER:
        return np.polynomial.legendre.leggauss(ng)

    ng1 = int(ngauss / 2.0)
    ng2 = ngauss - ng1
    xx = -np.cos(np.arctan(eps))                       # cos(theta) at the corner
    x = np.zeros(ng)
    w = np.zeros(ng)
    x1, w1 = np.polynomial.legendre.leggauss(ng1)
    x2, w2 = np.polynomial.legendre.leggauss(ng2)
    x[:ng1] = 0.5 * (xx + 1.0) * x1 + 0.5 * (xx - 1.0)
    w[:ng1] = 0.5 * (xx + 1.0) * w1
    x[ng1:ngauss] = -0.5 * xx * x2 + 0.5 * xx
    w[ng1:ngauss] = -0.5 * xx * w2
    x[ngauss:] = -x[ngauss - 1::-1]
    w[ngauss:] = w[ngauss - 1::-1]
    return x, w


def profile(x, rev, shape, eps, cheb_n=2):
    """(r2, drr) with r2 = r(theta)^2 and drr = (dr/dtheta)/r, on the nodes x=cos(theta)."""
    if shape == SPHEROID:
        a = rev * eps ** (1.0 / 3.0)                   # equatorial semi-axis
        ee = eps * eps
        cc = x * x
        ss = 1.0 - cc
        rr = 1.0 / (ss + ee * cc)
        return a * a * rr, rr * x * np.sqrt(ss) * (ee - 1.0)

    if shape == CYLINDER:
        h = rev * (2.0 / (3.0 * eps * eps)) ** (1.0 / 3.0)   # half-length
        a = h * eps                                          # radius
        n = len(x) // 2
        co = -x[:n]                                          # strictly positive
        si = np.sqrt(1.0 - co * co)
        on_cap = (si / co) <= (a / h)
        rad = np.where(on_cap, h / co, a / si)
        rthet = np.where(on_cap, h * si / (co * co), -a * co / (si * si))
        r2h = rad * rad
        drh = -rthet / rad
        return np.concatenate([r2h, r2h[::-1]]), np.concatenate([drh, -drh[::-1]])

    if shape == CHEBYSHEV:
        dnp = float(cheb_n)
        dn = dnp * dnp
        ep = eps * eps
        a = 1.0 + 1.5 * ep * (4.0 * dn - 2.0) / (4.0 * dn - 1.0)
        if cheb_n % 2 == 0:
            a -= 3.0 * eps * (1.0 + 0.25 * ep) / (dn - 1.0) + 0.25 * ep * eps / (9.0 * dn - 1.0)
        r0 = rev * a ** (-1.0 / 3.0)
        xi = np.arccos(x) * dnp
        ri = r0 * (1.0 + eps * np.cos(xi))
        return ri * ri, -r0 * eps * dnp * np.sin(xi) / ri

    if shape == BICONCAVE:
        a0, a1, a2 = _EF
        R0 = rev * ((4.0 / 3.0 * np.pi) / _RBC_VDIM) ** (1.0 / 3.0)   # equal-volume scaling

        def ztop(rho):
            u = rho / R0
            return R0 * _rbc_ztop(u)

        def ztop_p(rho):                                             # d z_top / d rho
            u = rho / R0
            s = np.sqrt(np.clip(1.0 - u * u, 1e-300, 1.0))
            P = a0 + a1 * u * u + a2 * u ** 4
            return -u / s * P + 2.0 * u * s * (a1 + 2.0 * a2 * u * u)

        # solve z_top(rho) = t * rho  for rho, per node, by vectorised bisection.
        # theta(rho) is monotonic (star-shaped), so the root is unique.
        t = np.abs(x) / np.sqrt(1.0 - x * x)
        lo = np.full_like(x, 1e-12 * R0)
        hi = np.full_like(x, R0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f = ztop(mid) - t * mid
            pos = f > 0.0
            lo = np.where(pos, mid, lo)
            hi = np.where(pos, hi, mid)
        rho = 0.5 * (lo + hi)
        z = ztop(rho)
        zp = ztop_p(rho)
        r2 = rho * rho + z * z
        drr_top = (rho + z * zp) / (z - rho * zp)
        return r2, np.sign(x) * drr_top

    raise ValueError(f"unknown shape {shape!r}")


# --------------------------------------------------------------------------
# Wigner d functions
# --------------------------------------------------------------------------

def vig(x, nmax, m):
    """d^n_{0m}(theta) and d/dtheta of it. x = cos(theta), array. Index n = 0..nmax."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    dv1 = np.zeros((x.size, nmax + 1))
    dv2 = np.zeros((x.size, nmax + 1))
    qs = np.sqrt(1.0 - x * x)
    qs1 = 1.0 / qs

    if m == 0:
        d1 = np.ones_like(x)
        d2 = x.copy()
        for n in range(1, nmax + 1):
            qn, qn1, qn2 = float(n), float(n + 1), float(2 * n + 1)
            d3 = (qn2 * x * d2 - qn * d1) / qn1
            dv1[:, n] = d2
            dv2[:, n] = qs1 * (qn1 * qn / qn2) * (-d1 + d3)
            d1, d2 = d2, d3
        return dv1, dv2

    qmm = float(m * m)
    a = np.ones_like(x)
    for i in range(1, m + 1):
        a = a * np.sqrt(float(2 * i - 1) / float(2 * i)) * qs
    d1 = np.zeros_like(x)
    d2 = a
    for n in range(m, nmax + 1):
        qn, qn1, qn2 = float(n), float(n + 1), float(2 * n + 1)
        qnm = np.sqrt(qn * qn - qmm)
        qnm1 = np.sqrt(qn1 * qn1 - qmm)
        d3 = (qn2 * x * d2 - qnm * d1) / qnm1
        dv1[:, n] = d2
        dv2[:, n] = qs1 * (-qn1 * qnm * d1 + qn * qnm1 * d3) / qn2
        d1, d2 = d2, d3
    return dv1, dv2


def vigampl(x, nmax, m):
    """d^n_{0m}/sin(theta) and its derivative, vectorised over x. Handles x = +-1."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    dv1 = np.zeros((x.size, nmax + 1))
    dv2 = np.zeros((x.size, nmax + 1))

    pole = np.abs(1.0 - np.abs(x)) <= 1e-10
    reg = ~pole

    if reg.any():
        a1, a2 = vig(x[reg], nmax, m)
        dv1[reg] = a1 / np.sqrt(1.0 - x[reg] ** 2)[:, None]
        dv2[reg] = a2

    if pole.any() and m == 1:
        n = np.arange(nmax + 1, dtype=float)
        dn = 0.5 * np.sqrt(n * (n + 1.0))
        for i in np.nonzero(pole)[0]:
            if x[i] < 0:
                sgn = np.array([(-1) ** (int(nn) + 1) for nn in range(nmax + 1)], dtype=float)
                dv1[i] = dn * sgn
                dv2[i] = -dn * sgn
            else:
                dv1[i] = dn
                dv2[i] = dn
    return dv1, dv2


# --------------------------------------------------------------------------
# Bessel tables on the quadrature nodes
# --------------------------------------------------------------------------

def _bessel_tables(kr, m_rel, nmax):
    """j_n(kr), y_n(kr), j_n(m kr) and the (1/z) d/dz [z f(z)] derivatives."""
    ng = kr.size
    jr = spherical_jn_complex(nmax, kr.astype(complex)).T.real          # (ng, nmax+1)

    y = np.zeros((ng, nmax + 1))
    y[:, 0] = -np.cos(kr) / kr
    y[:, 1] = -np.cos(kr) / kr ** 2 - np.sin(kr) / kr
    for n in range(1, nmax):
        y[:, n + 1] = (2 * n + 1) / kr * y[:, n] - y[:, n - 1]

    w = m_rel * kr
    jc = spherical_jn_complex(nmax, w).T                                # (ng, nmax+1)

    n = np.arange(nmax + 1)
    dj = np.zeros((ng, nmax + 1))
    dy = np.zeros((ng, nmax + 1))
    djc = np.zeros((ng, nmax + 1), dtype=complex)
    dj[:, 1:] = jr[:, :-1] - n[1:] * jr[:, 1:] / kr[:, None]
    dy[:, 1:] = y[:, :-1] - n[1:] * y[:, 1:] / kr[:, None]
    djc[:, 1:] = jc[:, :-1] - n[1:] * jc[:, 1:] / w[:, None]
    return jr, y, jc, dj, dy, djc


# --------------------------------------------------------------------------
# the T-matrix
# --------------------------------------------------------------------------

def _tmatrix_block(m, k, m_rel, x, w, r2, drr, jr, y, jc, dj, dy, djc, nmax):
    """T^m: a (2 NM, 2 NM) complex block, NM = nmax - max(m,1) + 1.

    Each surface-integral term factorises into (function of n1) x (function of n2),
    so the double sum over n collapses into a matrix product over quadrature nodes.
    """
    nmin = max(m, 1)
    ns = np.arange(nmin, nmax + 1)
    nm = ns.size

    sinth = np.sqrt(1.0 - x * x)
    r = np.sqrt(r2)
    rr = w * r2                                    # W_i r_i^2
    ds = rr * m / sinth                            # zero when m = 0
    dss = (m * m) / (sinth * sinth)
    ddr = 1.0 / (k * r)                            # 1/(k r)
    invmkr = 1.0 / (m_rel * k * r)                 # 1/(m k r), complex

    d1, d2 = vig(x, nmax, m)
    d1 = d1[:, nmin:nmax + 1]
    d2 = d2[:, nmin:nmax + 1]

    j1 = jr[:, nmin:nmax + 1]
    dj1 = dj[:, nmin:nmax + 1]
    h1 = j1 + 1j * y[:, nmin:nmax + 1]
    dh1 = dj1 + 1j * dy[:, nmin:nmax + 1]
    jc2 = jc[:, nmin:nmax + 1]
    djc2 = djc[:, nmin:nmax + 1]

    an = ns * (ns + 1.0)

    def acc(wgt, f1, f2):
        """sum_i wgt_i f1[i,n1] f2[i,n2] -> (nm, nm)

        numpy 2.0's complex matmul raises spurious divide-by-zero / overflow FPE
        flags on entirely finite inputs (a SIMD artefact: `np.zeros((16,8),complex).T
        @ finite` reproduces it under np.seterr('raise')). Suppress here rather than
        globally, and guard the block's output for real non-finite values below.
        """
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return (f1 * wgt[:, None]).T @ f2

    out = {}
    for tag, hh, dhh in (("Q", h1, dh1), ("Rg", j1, dj1)):
        a11 = acc(ds, d1 * hh, d2 * jc2) + acc(ds, d2 * hh, d1 * jc2)

        a12 = acc(rr * dss, d1 * dhh, d1 * jc2) + acc(rr, d2 * dhh, d2 * jc2)
        a12 = a12 + an[:, None] * acc(rr * drr * ddr, d1 * hh, d2 * jc2)

        a21 = acc(rr * dss, d1 * hh, d1 * djc2) + acc(rr, d2 * hh, d2 * djc2)
        a21 = a21 + an[None, :] * acc(rr * drr, d2 * hh, d1 * jc2 * invmkr[:, None])

        a22 = acc(ds, d1 * dhh, d2 * djc2) + acc(ds, d2 * dhh, d1 * djc2)
        a22 = a22 + an[:, None] * acc(ds * drr * ddr, d1 * hh, d1 * djc2)
        a22 = a22 + an[None, :] * acc(ds * drr, d1 * dhh, d1 * jc2 * invmkr[:, None])

        out[tag] = (a11, a12, a21, a22)

    dd = np.sqrt((2.0 * ns + 1.0) / (ns * (ns + 1.0)))
    ann = 0.5 * dd[:, None] * dd[None, :]

    def assemble(a11, a12, a21, a22):
        a11, a12, a21, a22 = (v * ann for v in (a11, a12, a21, a22))
        ta11, ta22 = -a11, -a22
        ta12, ta21 = -1j * a12, 1j * a21
        tpi = k * k * m_rel
        tppi = k * k
        M = np.empty((2 * nm, 2 * nm), dtype=complex)
        M[:nm, :nm] = tpi * ta21 + tppi * ta12
        M[:nm, nm:] = tpi * ta11 + tppi * ta22
        M[nm:, :nm] = tpi * ta22 + tppi * ta11
        M[nm:, nm:] = tpi * ta12 + tppi * ta21
        return M

    Q = assemble(*out["Q"])
    RgQ = assemble(*out["Rg"])
    if not (np.isfinite(Q).all() and np.isfinite(RgQ).all()):
        raise FloatingPointError(f"non-finite Q/RgQ at m={m} (nmax={nmax})")
    # T = -RgQ Q^{-1}  <=>  Q^T T^T = -RgQ^T
    T = -np.linalg.solve(Q.T, RgQ.T).T
    if not np.isfinite(T).all():
        raise FloatingPointError(f"non-finite T at m={m} (nmax={nmax}); Q is singular")
    return T


def tmatrix(rev, k, m_rel, shape=SPHEROID, eps=1.0, nmax=None, ngauss=None,
            ndgs=2, cheb_n=2):
    """All T^m blocks. Returns (blocks, nmax); blocks[m] has shape (2 NM, 2 NM)."""
    m_rel = complex(m_rel)
    xev = k * rev
    if nmax is None:
        nmax = max(4, int(xev + 4.05 * xev ** (1.0 / 3.0)))
    if ngauss is None:
        ngauss = nmax * ndgs

    x, w = gauss_nodes(ngauss, shape, eps)
    r2, drr = profile(x, rev, shape, eps, cheb_n)
    kr = k * np.sqrt(r2)
    tabs = _bessel_tables(kr, m_rel, nmax)

    blocks = {m: _tmatrix_block(m, k, m_rel, x, w, r2, drr, *tabs, nmax)
              for m in range(nmax + 1)}
    return blocks, nmax


def tmatrix_converged(rev, k, m_rel, shape=SPHEROID, eps=1.0, ddelt=1e-3,
                      ndgs=2, cheb_n=2, nmax_cap=100, verbose=False):
    """Mishchenko's CALCTMAT procedure: grow nmax, then grow ngauss.

    Convergence is judged on the (2n+1)-weighted m=0 diagonal, as the Fortran does.
    Raises RuntimeError at nmax_cap: the honest signal that EBCM has run out of
    double precision for this (size, aspect ratio, index), not a bug to paper over.
    """
    m_rel = complex(m_rel)
    xev = k * rev
    inm1 = max(4, int(xev + 4.05 * xev ** (1.0 / 3.0)))

    def m0_probe(nmax, ngauss):
        x, w = gauss_nodes(ngauss, shape, eps)
        r2, drr = profile(x, rev, shape, eps, cheb_n)
        kr = k * np.sqrt(r2)
        tabs = _bessel_tables(kr, m_rel, nmax)
        T = _tmatrix_block(0, k, m_rel, x, w, r2, drr, *tabs, nmax)
        n = np.arange(1, nmax + 1)
        dn1 = 2.0 * n + 1.0
        d1 = np.diag(T)[:nmax]
        d2 = np.diag(T)[nmax:]
        qext = float(np.sum((d1.real + d2.real) * dn1))
        qsca = float(np.sum((np.abs(d1) ** 2 + np.abs(d2) ** 2) * dn1))
        return qext, qsca

    qext1 = qsca1 = 0.0
    converged = False
    for nmax in range(inm1, nmax_cap + 1):
        qext, qsca = m0_probe(nmax, nmax * ndgs)
        dext = abs((qext1 - qext) / qext) if qext else 1.0
        dsca = abs((qsca1 - qsca) / qsca) if qsca else 1.0
        qext1, qsca1 = qext, qsca
        if verbose:
            print(f"  nmax={nmax:3d}  dext={dext:.2e}  dsca={dsca:.2e}")
        if dsca <= ddelt and dext <= ddelt:
            converged = True
            break
    if not converged:
        raise RuntimeError(
            f"EBCM did not converge for x_ev={xev:.2f}, eps={eps}, m={m_rel} "
            f"within nmax<={nmax_cap}: the double-precision boundary, not a bug.")

    ngauss = nmax * ndgs
    step = max(1, nmax // 4)
    for ng in range(nmax * ndgs + step, 4 * nmax + 1, step):
        qext, qsca = m0_probe(nmax, ng)
        dext = abs((qext1 - qext) / qext) if qext else 1.0
        dsca = abs((qsca1 - qsca) / qsca) if qsca else 1.0
        qext1, qsca1 = qext, qsca
        if dsca <= ddelt and dext <= ddelt:
            break
        ngauss = ng

    blocks, nmax = tmatrix(rev, k, m_rel, shape, eps, nmax=nmax, ngauss=ngauss, cheb_n=cheb_n)
    return blocks, nmax, ngauss


def tmatrix_best_lossless(rev, k, m_rel, shape=SPHEROID, eps=1.0, cheb_n=2,
                          ndgs=2, span=(-4, 13), verbose=False):
    """Pick nmax by minimising |albedo - 1|, for a lossless (real m) particle.

    Why not just grow nmax until it converges? Because for high-aspect oblate
    spheroids at large size parameters, EBCM's Q matrix becomes numerically singular
    in double precision (cond(Q) ~ 1e21 by nmax=56 for an RBC at 800 nm), and the
    series *diverges* rather than converging: adding terms makes the answer worse.
    Equilibrating Q does not help; the loss is intrinsic, and the standard cure is
    extended precision (Mishchenko ships a quadruple-precision variant for this).

    There is therefore an optimal nmax where truncation error meets amplified
    roundoff. For a lossless particle the single-scattering albedo must be exactly 1,
    which gives a free, physical, computable meter for locating it, and a residual
    |albedo - 1| that is an honest accuracy estimate for everything derived from T.

    Returns (blocks, nmax, albedo_err).
    """
    if abs(complex(m_rel).imag) > 0:
        raise ValueError("albedo criterion only applies to a lossless (real m) particle")

    xev = k * rev
    n0 = max(4, int(xev + 4.05 * xev ** (1.0 / 3.0)))
    best = None
    for nmax in range(max(4, n0 + span[0]), n0 + span[1]):
        try:
            blocks, _ = tmatrix(rev, k, m_rel, shape, eps, nmax=nmax,
                                ngauss=nmax * ndgs, cheb_n=cheb_n)
            err = abs(cross_sections_random(blocks, k, rev)["albedo"] - 1.0)
        except (FloatingPointError, np.linalg.LinAlgError):
            break
        if verbose:
            print(f"  nmax={nmax:3d}  |albedo-1| = {err:.3e}")
        if best is None or err < best[2]:
            best = (blocks, nmax, err)
        if best is not None and err > 100.0 * best[2] and err > 1e-3:
            break                                   # diverging; stop wasting time
    if best is None:
        raise RuntimeError("no usable nmax found")
    return best


def cross_sections_random(blocks, k, rev):
    """Orientation-averaged cross sections, exact from T (no angular quadrature)."""
    qext = 0.0
    qsca = 0.0
    for m, T in blocks.items():
        fac = 1.0 if m == 0 else 2.0
        qext += fac * np.trace(T).real
        qsca += fac * float(np.sum(np.abs(T) ** 2))
    cext = -(2.0 * np.pi / k ** 2) * qext
    csca = (2.0 * np.pi / k ** 2) * qsca
    geo = np.pi * rev ** 2
    return dict(cext=cext, csca=csca, cabs=cext - csca,
                qext=cext / geo, qsca=csca / geo, qabs=(cext - csca) / geo,
                albedo=csca / cext if cext else 0.0)


# --------------------------------------------------------------------------
# amplitude matrix (particle frame)
# --------------------------------------------------------------------------

def full_T(blocks, nmax):
    """Re-index blocks as T11/T12/T21/T22 with shape (nmax+1, nmax+1, nmax+1) [m, n, n']."""
    T = [np.zeros((nmax + 1, nmax + 1, nmax + 1), dtype=complex) for _ in range(4)]
    for m, B in blocks.items():
        nmin = max(m, 1)
        nm = nmax - nmin + 1
        sl = slice(nmin, nmax + 1)
        T[0][m][sl, sl] = B[:nm, :nm]
        T[1][m][sl, sl] = B[:nm, nm:]
        T[2][m][sl, sl] = B[nm:, :nm]
        T[3][m][sl, sl] = B[nm:, nm:]
    return tuple(T)


def _cal(nmax):
    n = np.arange(nmax + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rn = np.sqrt((2 * n + 1.0) / (n * (n + 1.0)))
    rn[0] = 0.0
    p = (np.arange(nmax + 1)[None, :] - np.arange(nmax + 1)[:, None] - 1) % 4
    return (1j ** p) * rn[:, None] * rn[None, :]


def amplitudes_vec(Tfull, nmax, k, ct0, ct, dphi):
    """Amplitude matrix for one incidence direction and many scattering directions.

    ct0 scalar = cos(theta_inc); ct, dphi arrays over scattering directions, all in
    the particle frame. Returns S with shape (ndir, 2, 2).

    The basis is the *meridional* (theta_hat, phi_hat) frame, NOT the scattering
    plane. It coincides with the Bohren & Huffman (S2, S1) basis only when the
    scattering plane contains the symmetry axis, i.e. dphi = 0 or pi. Comparisons
    against Mie must use dphi = 0.

    The double sum over (n, n') factorises: contract the incidence index once per m,
    leaving an O(ndir * nmax) contraction per m instead of O(ndir * nmax^2).
    """
    T11, T12, T21, T22 = Tfull
    cal = _cal(nmax)
    ct = np.atleast_1d(np.asarray(ct, dtype=float))
    dphi = np.atleast_1d(np.asarray(dphi, dtype=float))

    DV1, DV2 = vigampl(ct, nmax, 0)      # allocate; recomputed per m below
    vv = np.zeros(ct.size, dtype=complex)
    vh = np.zeros_like(vv)
    hv = np.zeros_like(vv)
    hh = np.zeros_like(vv)

    # see the note in _tmatrix_block.acc: numpy 2.0 complex matmul flags are spurious
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for m in range(nmax + 1):
            DV1, DV2 = vigampl(ct, nmax, m)                 # (ndir, nmax+1)
            dv01, dv02 = vigampl(np.array([ct0]), nmax, m)
            dv01, dv02 = dv01[0], dv02[0]

            M11 = cal * T11[m]
            M12 = cal * T12[m]
            M21 = cal * T21[m]
            M22 = cal * T22[m]

            if m == 0:
                b = dv02
                vv += DV2 @ (M22 @ b)
                hh += DV2 @ (M11 @ b)
                continue

            a = m * dv01
            b = dv02
            U = m * DV1
            V = DV2
            qa1 = M11 @ a + M12 @ b
            qa2 = M21 @ a + M22 @ b
            qb1 = M11 @ b + M12 @ a
            qb2 = M21 @ b + M22 @ a

            fc = 2.0 * np.cos(m * dphi)
            fs = 2.0 * np.sin(m * dphi)
            vv += fc * (U @ qa1 + V @ qa2)
            vh += fs * (U @ qb1 + V @ qb2)
            hv -= fs * (V @ qa1 + U @ qa2)
            hh += fc * (V @ qb1 + U @ qb2)

    S = np.empty((ct.size, 2, 2), dtype=complex)
    S[:, 0, 0] = vv / k
    S[:, 0, 1] = vh / k
    S[:, 1, 0] = hv / k
    S[:, 1, 1] = hh / k
    return S


def amplitude_matrix(Tfull, nmax, k, ct0, ct, dphi):
    """Scalar convenience wrapper: 2x2 amplitude matrix for one direction pair."""
    return amplitudes_vec(Tfull, nmax, k, ct0, np.array([ct]), np.array([dphi]))[0]


def phase_function_random(blocks, nmax, k, n_beta=32, n_psi=32, n_theta=181):
    """Orientation-averaged phase function Z11(Theta), with Csca and g by quadrature.

    Averages over the incidence direction in the particle frame (equivalent, for an
    axisymmetric particle, to averaging over orientations) and over the azimuth psi
    around the incident direction. Z11 = 0.5 Tr(S S^dagger) is invariant under
    rotations of the polarisation basis, so no basis bookkeeping is needed.

    Scattering-angle integrals use Gauss-Legendre nodes in cos(Theta), so the Csca
    returned here is an *independent* route to the exact value from
    cross_sections_random. Their agreement is a real test, not a tautology.

    Returns (theta, Z11, csca, g); theta = arccos of the Gauss nodes, descending mu.
    """
    Tfull = full_T(blocks, nmax)

    cb, wb = np.polynomial.legendre.leggauss(n_beta)      # cos(theta_inc)
    mu, wmu = np.polynomial.legendre.leggauss(n_theta)    # cos(Theta_scatter)
    psi = 2.0 * np.pi * np.arange(n_psi) / n_psi

    CT, PS = np.meshgrid(mu, psi, indexing="ij")
    ST = np.sqrt(np.clip(1.0 - CT ** 2, 0.0, 1.0))
    cosPS, sinPS = np.cos(PS), np.sin(PS)

    Z11 = np.zeros(n_theta)
    for b in range(n_beta):
        ct0 = cb[b]
        st0 = np.sqrt(max(1.0 - ct0 * ct0, 0.0))
        nz = ct0 * CT - st0 * ST * cosPS
        nx = st0 * CT + ct0 * ST * cosPS
        ny = ST * sinPS
        ct_out = np.clip(nz, -1.0, 1.0).ravel()
        dphi = np.arctan2(ny, nx).ravel()

        S = amplitudes_vec(Tfull, nmax, k, ct0, ct_out, dphi)
        z = 0.5 * np.sum(np.abs(S) ** 2, axis=(1, 2)).reshape(n_theta, n_psi)
        Z11 += 0.5 * wb[b] * z.mean(axis=1)

    denom = float(np.sum(wmu * Z11))
    csca = 2.0 * np.pi * denom
    g = float(np.sum(wmu * Z11 * mu)) / denom
    return np.arccos(mu), Z11, csca, g
