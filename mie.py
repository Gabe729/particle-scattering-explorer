"""Lorenz-Mie scattering from a homogeneous sphere.

Reference implementation, Bohren & Huffman (1983) conventions throughout.

Conventions
-----------
* Time dependence exp(-i omega t).
* m = n + i k, relative to the surrounding medium, with k >= 0 for absorption.
* Size parameter x = k_med * a = 2 pi n_med a / lambda_vac.
* Incident plane wave: unit amplitude, propagating along +z, polarised along x.
* theta measured from +z (theta = 0 is forward scattering); phi from the x axis.

This module is the numerical ground truth. The browser JavaScript in
`explorer_template.html` is a port of it and is cross-checked against it
(see `validate.py` and `crosscheck.py`).
"""

import numpy as np

__all__ = [
    "n_stop",
    "mie_ab",
    "mie_cd",
    "efficiencies",
    "amplitudes",
    "differential_cross_section",
    "fields_spherical",
    "near_field_slice",
]


# --------------------------------------------------------------------------
# series truncation
# --------------------------------------------------------------------------

def n_stop(x):
    """Wiscombe truncation: number of terms retained in the Mie series."""
    return max(int(np.floor(x + 4.0 * x ** (1.0 / 3.0) + 2.0)) + 1, 4)


# --------------------------------------------------------------------------
# Bessel machinery
# --------------------------------------------------------------------------

def _logderiv(z, nmx):
    """D_n(z) = psi'_n(z) / psi_n(z) by downward recurrence (stable)."""
    D = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        nz = n / z
        D[n - 1] = nz - 1.0 / (D[n] + nz)
    return D


def _riccati_real(x, nmax):
    """psi_n(x) and chi_n(x) for real x, upward recurrence. Returns n = 0..nmax."""
    psi = np.zeros(nmax + 1)
    chi = np.zeros(nmax + 1)
    psi_m1, chi_m1 = np.cos(x), -np.sin(x)          # n = -1
    psi[0], chi[0] = np.sin(x), np.cos(x)           # n = 0
    for n in range(1, nmax + 1):
        p2 = psi_m1 if n == 1 else psi[n - 2]
        c2 = chi_m1 if n == 1 else chi[n - 2]
        psi[n] = (2 * n - 1) / x * psi[n - 1] - p2
        chi[n] = (2 * n - 1) / x * chi[n - 1] - c2
    return psi, chi


def spherical_jn_complex(nmax, w):
    """j_n(w) for complex w, n = 0..nmax, via Miller's downward recurrence.

    `w` may be a scalar or an ndarray. Returns shape (nmax+1,) + w.shape.
    Renormalisation is per-point, so the recurrence stays in range.
    """
    w = np.atleast_1d(np.asarray(w, dtype=complex))
    nstart = nmax + int(np.max(np.abs(w))) + 25
    j = np.zeros((nstart + 2,) + w.shape, dtype=complex)
    # Seed at 1.0, not at a tiny value: the recurrence grows downward, so a tiny
    # seed leaves j[0] denormal and |j[0]|^2 underflows in the normalisation.
    j[nstart] = 1.0
    for n in range(nstart, 0, -1):
        j[n - 1] = (2 * n + 1) / w * j[n] - j[n + 1]
        big = np.abs(j[n - 1]) > 1e200
        if big.any():
            # linear homogeneous recurrence: rescaling a point's whole column is exact
            f = np.where(big, 1e-200, 1.0)
            j[n - 1:] *= f
    # rescale so |j[0]| ~ 1 before dividing (keeps the port bit-comparable in JS,
    # where complex division is naive and would underflow here)
    jj = j[: nmax + 1] / np.abs(j[0])
    j0 = np.sin(w) / w
    return jj * (j0 / jj[0])


# --------------------------------------------------------------------------
# Mie coefficients
# --------------------------------------------------------------------------

def mie_ab(m, x, nmax=None):
    """Scattering coefficients a_n, b_n (B&H 4.88). Returns arrays for n = 1..nmax."""
    if nmax is None:
        nmax = n_stop(x)
    m = complex(m)
    z = m * x
    nmx = int(max(nmax, abs(z))) + 16
    D = _logderiv(z, nmx)
    psi, chi = _riccati_real(x, nmax)
    xi = psi - 1j * chi

    a = np.zeros(nmax, dtype=complex)
    b = np.zeros(nmax, dtype=complex)
    for n in range(1, nmax + 1):
        ta = D[n] / m + n / x
        tb = D[n] * m + n / x
        a[n - 1] = (ta * psi[n] - psi[n - 1]) / (ta * xi[n] - xi[n - 1])
        b[n - 1] = (tb * psi[n] - psi[n - 1]) / (tb * xi[n] - xi[n - 1])
    return a, b


def mie_cd(m, x, nmax=None):
    """Internal-field coefficients c_n, d_n (B&H 4.53), simplified by the Wronskian.

    psi_n(x) xi'_n(x) - xi_n(x) psi'_n(x) = i, hence both numerators collapse to i*m.
    """
    if nmax is None:
        nmax = n_stop(x)
    m = complex(m)
    psi, chi = _riccati_real(x, nmax)
    xi = psi - 1j * chi

    jmx = spherical_jn_complex(nmax, m * x)[:, 0]
    psi_mx = (m * x) * jmx                                   # psi_n(mx)

    c = np.zeros(nmax, dtype=complex)
    d = np.zeros(nmax, dtype=complex)
    for n in range(1, nmax + 1):
        xi_p = xi[n - 1] - (n / x) * xi[n]                   # xi'_n(x)
        psi_mx_p = psi_mx[n - 1] - (n / (m * x)) * psi_mx[n]  # psi'_n(mx)
        c[n - 1] = 1j * m / (psi_mx[n] * xi_p - m * xi[n] * psi_mx_p)
        d[n - 1] = 1j * m / (m * psi_mx[n] * xi_p - xi[n] * psi_mx_p)
    return c, d


# --------------------------------------------------------------------------
# integral quantities
# --------------------------------------------------------------------------

def efficiencies(m, x):
    """Qext, Qsca, Qabs, Qback, g. Dimensionless, normalised by pi a^2."""
    a, b = mie_ab(m, x)
    n = np.arange(1, len(a) + 1)
    tn = 2 * n + 1

    qext = (2.0 / x ** 2) * np.sum(tn * (a + b).real)
    qsca = (2.0 / x ** 2) * np.sum(tn * (np.abs(a) ** 2 + np.abs(b) ** 2))
    qabs = qext - qsca
    qback = (1.0 / x ** 2) * np.abs(np.sum(tn * (-1) ** n * (a - b))) ** 2

    # asymmetry parameter (B&H 4.62)
    nn = n[:-1]
    t1 = np.sum(nn * (nn + 2) / (nn + 1) *
                (a[:-1] * np.conj(a[1:]) + b[:-1] * np.conj(b[1:])).real)
    t2 = np.sum(tn / (n * (n + 1)) * (a * np.conj(b)).real)
    g = (4.0 / x ** 2) * (t1 + t2) / qsca if qsca > 0 else 0.0

    return dict(qext=qext, qsca=qsca, qabs=qabs, qback=qback, g=g)


def _pi_tau(nmax, mu):
    """Angular functions pi_n(mu), tau_n(mu). Returns arrays (nmax+1,) + mu.shape."""
    mu = np.asarray(mu, dtype=float)
    pi = np.zeros((nmax + 1,) + mu.shape)
    tau = np.zeros((nmax + 1,) + mu.shape)
    pi[1] = 1.0
    tau[1] = mu
    for n in range(2, nmax + 1):
        pi[n] = ((2 * n - 1) / (n - 1)) * mu * pi[n - 1] - (n / (n - 1)) * pi[n - 2]
        tau[n] = n * mu * pi[n] - (n + 1) * pi[n - 1]
    return pi, tau


def amplitudes(m, x, theta):
    """Scattering amplitude functions S1(theta), S2(theta) (B&H 4.74)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    a, b = mie_ab(m, x)
    nmax = len(a)
    pi, tau = _pi_tau(nmax, np.cos(theta))

    n = np.arange(1, nmax + 1)
    w = (2 * n + 1) / (n * (n + 1))
    s1 = np.tensordot(w * a, pi[1:], axes=(0, 0)) + np.tensordot(w * b, tau[1:], axes=(0, 0))
    s2 = np.tensordot(w * a, tau[1:], axes=(0, 0)) + np.tensordot(w * b, pi[1:], axes=(0, 0))
    return s1, s2


def differential_cross_section(m, x, theta, phi, polarisation="x"):
    """dC_sca/dOmega in units of 1/k^2, for unit incident irradiance.

    polarisation = 'x'  -> incident linear polarisation along x
    polarisation = 'un' -> unpolarised incident light
    """
    s1, s2 = amplitudes(m, x, theta)
    if polarisation == "un":
        return 0.5 * (np.abs(s1) ** 2 + np.abs(s2) ** 2)
    c, s = np.cos(phi), np.sin(phi)
    return np.abs(s2) ** 2 * c ** 2 + np.abs(s1) ** 2 * s ** 2


# --------------------------------------------------------------------------
# near field
# --------------------------------------------------------------------------

# Beyond this optical depth the interior field is numerically zero anyway
# (skin depth << radius); computing it would overflow the Bessel recurrences.
_INTERNAL_CUTOFF = 30.0


def fields_spherical(m, x, r, theta, phi, nmax=None, part="total"):
    """Electric field components (E_r, E_theta, E_phi) at spherical coordinates.

    `r` is in units of the sphere radius a, so r < 1 is inside the particle.
    Field is normalised to unit incident amplitude.

    part = 'total'      incident + scattered outside, internal field inside
           'scattered'  scattered field only (zero inside)
           'internal'   internal field only (zero outside)
           'incident'   incident plane wave only
    """
    m = complex(m)
    if nmax is None:
        nmax = n_stop(x)

    r = np.atleast_1d(np.asarray(r, dtype=float))
    theta = np.broadcast_to(np.asarray(theta, dtype=float), r.shape)
    phi = np.broadcast_to(np.asarray(phi, dtype=float), r.shape)

    ct, st = np.cos(theta), np.sin(theta)
    cph, sph = np.cos(phi), np.sin(phi)
    pi_n, tau_n = _pi_tau(nmax, ct)

    n = np.arange(1, nmax + 1)
    En = (1j ** n) * (2 * n + 1) / (n * (n + 1))

    Er = np.zeros(r.shape, dtype=complex)
    Et = np.zeros(r.shape, dtype=complex)
    Ep = np.zeros(r.shape, dtype=complex)

    if part == "incident":
        eikz = np.exp(1j * x * r * ct)
        return eikz * st * cph, eikz * ct * cph, -eikz * sph

    out = r >= 1.0
    ins = ~out

    # ---------------- outside: scattered (+ incident if total) ----------------
    if out.any() and part in ("total", "scattered"):
        a, b = mie_ab(m, x, nmax)
        rho = x * r[out]
        psi = np.zeros((nmax + 1, rho.size))
        chi = np.zeros((nmax + 1, rho.size))
        psi[0], chi[0] = np.sin(rho), np.cos(rho)
        psi_m1, chi_m1 = np.cos(rho), -np.sin(rho)
        for k in range(1, nmax + 1):
            p2 = psi_m1 if k == 1 else psi[k - 2]
            c2 = chi_m1 if k == 1 else chi[k - 2]
            psi[k] = (2 * k - 1) / rho * psi[k - 1] - p2
            chi[k] = (2 * k - 1) / rho * chi[k - 1] - c2
        xi = psi - 1j * chi                       # xi_n(rho) = rho * h_n^(1)(rho)

        er = np.zeros(rho.size, dtype=complex)
        et = np.zeros(rho.size, dtype=complex)
        ep = np.zeros(rho.size, dtype=complex)
        cf, sf, st_o = cph[out], sph[out], st[out]
        for k in range(1, nmax + 1):
            hn = xi[k] / rho                                    # h_n(rho)
            drh = (xi[k - 1] - (k / rho) * xi[k]) / rho         # [rho h_n]' / rho
            pk, tk = pi_n[k][out], tau_n[k][out]
            er += En[k - 1] * 1j * a[k - 1] * cf * k * (k + 1) * st_o * pk * hn / rho
            et += En[k - 1] * (1j * a[k - 1] * cf * tk * drh - b[k - 1] * cf * pk * hn)
            ep += En[k - 1] * (-1j * a[k - 1] * sf * pk * drh + b[k - 1] * sf * tk * hn)

        if part == "total":
            eikz = np.exp(1j * x * r[out] * ct[out])
            er += eikz * st[out] * cph[out]
            et += eikz * ct[out] * cph[out]
            ep += -eikz * sph[out]

        Er[out], Et[out], Ep[out] = er, et, ep

    # ---------------- inside: internal field ----------------
    if ins.any() and part in ("total", "internal"):
        if abs(m.imag) * x <= _INTERNAL_CUTOFF:
            c, d = mie_cd(m, x, nmax)
            w = m * x * np.maximum(r[ins], 1e-6)
            jn = spherical_jn_complex(nmax, w)                  # (nmax+1, npts)
            psi_w = w[None, :] * jn                             # psi_n(w)

            er = np.zeros(w.size, dtype=complex)
            et = np.zeros(w.size, dtype=complex)
            ep = np.zeros(w.size, dtype=complex)
            cf, sf, st_i = cph[ins], sph[ins], st[ins]
            for k in range(1, nmax + 1):
                dwr = (psi_w[k - 1] - (k / w) * psi_w[k]) / w   # [w j_n(w)]' / w
                pk, tk = pi_n[k][ins], tau_n[k][ins]
                er += En[k - 1] * (-1j * d[k - 1]) * cf * k * (k + 1) * st_i * pk * jn[k] / w
                et += En[k - 1] * (c[k - 1] * cf * pk * jn[k] - 1j * d[k - 1] * cf * tk * dwr)
                ep += En[k - 1] * (-c[k - 1] * sf * tk * jn[k] + 1j * d[k - 1] * sf * pk * dwr)

            Er[ins], Et[ins], Ep[ins] = er, et, ep

    return Er, Et, Ep


def near_field_slice(m, x, half_extent=3.0, n_grid=180, plane="E"):
    """Total |E|^2 on a plane through the sphere centre, relative to the incident wave.

    plane 'E' -> the x-z plane (contains the polarisation vector)
    plane 'H' -> the y-z plane (perpendicular to the polarisation vector)

    Coordinates are in units of the sphere radius a. Returns (u, z, I).
    """
    u = np.linspace(-half_extent, half_extent, n_grid)
    zz = np.linspace(-half_extent, half_extent, n_grid)
    U, Z = np.meshgrid(u, zz, indexing="xy")

    R = np.maximum(np.sqrt(U ** 2 + Z ** 2), 1e-4)
    theta = np.arctan2(np.abs(U), Z)
    if plane == "E":
        phi = np.where(U >= 0, 0.0, np.pi)
    else:
        phi = np.where(U >= 0, 0.5 * np.pi, -0.5 * np.pi)

    Er, Et, Ep = fields_spherical(m, x, R.ravel(), theta.ravel(), phi.ravel())
    I = (np.abs(Er) ** 2 + np.abs(Et) ** 2 + np.abs(Ep) ** 2).reshape(U.shape)
    return u, zz, I
