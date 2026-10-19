"""Generate reference.json: Python ground truth that the browser JS checks itself against.

The explorer embeds this and runs MIE.selfTest() and TM.selfTest() on load, so the shipped artefact
carries its own proof that the JavaScript port reproduces the validated Python.

Run:  python crosscheck.py   (explorer_build.py runs it for you)
"""

import json
import pathlib

import numpy as np
import mie
import tmatrix as tm

np.seterr(all="ignore")

OUT = pathlib.Path(__file__).resolve().parent / "reference.json"

CASES = [
    ("dielectric", 1.5, 0.0, 0.8),
    ("resonant", 1.33, 0.0, 5.0),
    ("absorbing", 1.5, 0.1, 3.0),
    ("metallic", 0.54, 2.18, 0.4),
    ("rbc-like", 1.042, 0.0, 29.3),
]

# near-field cases kept small: the JSON is embedded in the HTML
NEARFIELD = [
    ("dielectric", 1.5, 0.0, 3.0, 2.5, 31, "E"),
    ("absorbing", 1.5, 0.1, 4.0, 2.5, 31, "H"),
    ("metallic", 0.54, 2.18, 0.4, 2.5, 31, "E"),
]

# T-matrix cases: small nmax so the browser self-test stays under ~100 ms.
# Deliberately spans every shape and both lossless and absorbing particles.
TMATRIX = [
    ("tm-sphere", 1.5, 0.0, tm.SPHEROID, 1.0, 8, 16, 2),
    ("tm-oblate", 1.4, 0.0, tm.SPHEROID, 2.0, 10, 20, 2),
    ("tm-prolate-absorbing", 1.4, 0.05, tm.SPHEROID, 0.5, 10, 20, 2),
    ("tm-cylinder", 1.4, 0.0, tm.CYLINDER, 1.0, 10, 20, 2),
    ("tm-chebyshev", 1.4, 0.0, tm.CHEBYSHEV, 0.1, 10, 20, 2),
    # biconcave disc, sub-wavelength so it converges (the regime the JS port must match)
    ("tm-biconcave", 1.5, 0.0, tm.BICONCAVE, 1.0, 6, 240, 2),
]
TM_QUAD = dict(n_beta=8, n_psi=16, n_theta=30)

ref = {"cases": [], "nearfield": [], "tmatrix": []}

theta = np.linspace(0.0, np.pi, 25)
for name, mr, mi, x in CASES:
    m = complex(mr, mi)
    eff = mie.efficiencies(m, x)
    s1, s2 = mie.amplitudes(m, x, theta)
    ref["cases"].append(dict(
        name=name, mr=mr, mi=mi, x=x,
        qext=eff["qext"], qsca=eff["qsca"], qabs=eff["qabs"],
        qback=eff["qback"], g=eff["g"],
        theta=theta.tolist(),
        s1abs=np.abs(s1).tolist(),
        s2abs=np.abs(s2).tolist(),
    ))

for name, mr, mi, x, half, n, plane in NEARFIELD:
    _, _, I = mie.near_field_slice(complex(mr, mi), x, half_extent=half, n_grid=n, plane=plane)
    # python I is indexed [iz, iu]; the JS buffer is I[iu*N + iz]
    ref["nearfield"].append(dict(
        name=name, mr=mr, mi=mi, x=x, halfExtent=half, N=n, plane=plane,
        I=I.T.ravel().tolist(),
    ))

REV, K = 1.0, 3.0
ct_probe = np.array([0.9, 0.5, 0.0, -0.4, -0.85])
dphi_probe = np.array([0.0, 0.6, 1.3, 2.4, 3.0])
CT0 = 0.3

for name, mr, mi, shape, eps, nmax, ngauss, cheb_n in TMATRIX:
    m = complex(mr, mi)
    # the biconcave disc only converges sub-wavelength, so anchor it at small x_ev
    k = 0.5 if shape == tm.BICONCAVE else K
    blocks, _ = tm.tmatrix(REV, k, m, shape, eps, nmax=nmax, ngauss=ngauss, cheb_n=cheb_n)
    cs = tm.cross_sections_random(blocks, k, REV)
    _, _, csca_q, g = tm.phase_function_random(blocks, nmax, k, **TM_QUAD)
    Tf = tm.full_T(blocks, nmax)
    S = tm.amplitudes_vec(Tf, nmax, k, CT0, ct_probe, dphi_probe)
    ref["tmatrix"].append(dict(
        name=name, rev=REV, k=k, mr=mr, mi=mi, shape=shape, eps=eps,
        nmax=nmax, ngauss=ngauss, chebN=cheb_n,
        qext=cs["qext"], qsca=cs["qsca"], g=g, csca_quad=csca_q,
        nBeta=TM_QUAD["n_beta"], nPsi=TM_QUAD["n_psi"], nTheta=TM_QUAD["n_theta"],
        ct0=CT0, ct=ct_probe.tolist(), dphi=dphi_probe.tolist(),
        # all four elements of the 2x2 amplitude matrix, so the browser check covers
        # the cross-polarised terms as well as the co-polarised ones
        absS=np.abs(S[:, 0, 0]).tolist(),
        absSvh=np.abs(S[:, 0, 1]).tolist(),
        absShv=np.abs(S[:, 1, 0]).tolist(),
        absShh=np.abs(S[:, 1, 1]).tolist(),
    ))

with open(OUT, "w") as f:
    json.dump(ref, f)

n_nums = sum(len(c["theta"]) * 3 + 5 for c in ref["cases"]) + sum(len(nf["I"]) for nf in ref["nearfield"])
print(f"wrote reference.json  mie={len(ref['cases'])} nearfield={len(ref['nearfield'])} "
      f"tmatrix={len(ref['tmatrix'])} numbers~{n_nums}")
