# Particle Scattering Explorer

Exact single-particle light scattering in the browser, with the numerics validated against analytic limits and an independent implementation.

- **Spheres:** exact Lorenz–Mie theory. Far field, near field inside and around the sphere, phase function and efficiency spectrum, all live at slider speed.
- **Axisymmetric particles:** Waterman's extended boundary condition method (EBCM, the "T-matrix" method), ported from M. I. Mishchenko's public-domain Fortran. Spheroids (oblate and prolate), finite cylinders, Chebyshev particles, and the Evans–Fung biconcave red-blood-cell profile. Orientation-averaged cross sections, asymmetry parameter $g$ and phase function, compared against the volume-equivalent sphere.

The interactive page is a single self-contained HTML file. It needs no server and no network connection, and on every load it re-checks its JavaScript solvers against numbers produced by the validated Python reference.

**Live demo:** https://gabe729.github.io/particle-scattering-explorer/

## Quick start

Open `docs/index.html` in a browser. That is the whole app.

To run the Python reference or the validation suites, or to rebuild the page:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python validate.py                # Lorenz–Mie core: 174/174 with miepython, 76/76 without
python validate_tmatrix.py        # EBCM core: 81/81 (about 6 s)
python explorer_build.py          # validates, then writes dist/scattering_explorer.html
python explorer_build.py --docs   # also refreshes docs/index.html for GitHub Pages
python rbc_shape_study.py         # worked example: sphere vs spheroid for a red cell (about 1 min)
```

The solvers need only NumPy. `plotly` is needed only to build the page: the build inlines the `plotly.min.js` that ships inside the Python package. Alternatives: `--plotly PATH` inlines a `plotly.min.js` you supply, and `--cdn` references plotly.js 3.6.0 from jsDelivr instead of inlining it (about 0.16 MB rather than 5 MB, but the page then needs a network connection). `miepython` is optional; if it is installed, `validate.py` cross-checks against it.

The build refuses to produce a page if either validation suite fails.

## Using the page

Pick a preset or a shape, then move the sliders: equal-volume radius $r_\text{ev}$, vacuum wavelength, particle $n$ and $k$, medium index, and for non-spheres the axis ratio and the incidence angle.

- **Far field:** a 3D radiation lobe, radius proportional to $dC_\text{sca}/d\Omega$ (dB or linear).
- **Near field** (spheres only): $|E|^2$ on the E-plane or H-plane slice through the centre.
- **Phase function:** E- and H-plane for spheres; for non-spheres, the orientation-averaged $Z_{11}$ overlaid on the volume-equivalent Mie sphere.
- **Efficiency spectrum:** $Q_\text{ext}$, $Q_\text{sca}$, $Q_\text{abs}$ against wavelength. For non-spheres the sphere curve is drawn at once and an 11-point T-matrix spectrum is computed on request.
- **Shape sweep** (non-spheres): $g$ and the reduced-scattering ratio $C_\text{sca}(1-g)$ relative to the volume-equivalent sphere, against axis ratio.

The T-matrix does not depend on the particle's orientation, so moving the incidence-angle slider reuses the cached matrix and only redraws.

## Conventions

- Time dependence $e^{-i\omega t}$.
- Relative refractive index $m = n + ik$, with $k \ge 0$ for absorption, relative to the surrounding medium.
- Size parameter $x = 2\pi n_\text{med} a / \lambda_\text{vac}$. For non-spheres $x_\text{ev}$ uses the equal-volume-sphere radius, so a particle and its volume-equivalent sphere share the same size parameter.
- Incident plane wave of unit amplitude travelling along $+z$, polarised along $x$; $\theta$ is measured from $+z$ ($\theta = 0$ is forward), $\varphi$ from the $x$ axis.
- Mie amplitudes $S_1$, $S_2$ and efficiencies follow Bohren & Huffman (1983).
- Non-spheres: the symmetry axis is $z$ in the particle frame. The axis ratio $\varepsilon$ is equatorial/polar semi-axis for spheroids ($\varepsilon > 1$ oblate, $< 1$ prolate), diameter/length for cylinders, and the deformation parameter for Chebyshev particles.
- The EBCM amplitude matrix is expressed in the meridional $(\hat\theta, \hat\varphi)$ basis, as in Mishchenko's code. It coincides with Bohren & Huffman's $(S_2, S_1)$ only when the azimuth difference is $0$ or $\pi$.
- Units in the page: lengths in nm, cross sections in µm².

## Files

| File | Role |
|---|---|
| `mie.py` | Lorenz–Mie reference: coefficients $a_n, b_n, c_n, d_n$, efficiencies, amplitude functions, internal and external near field. |
| `tmatrix.py` | EBCM reference, a direct port of Mishchenko's `ampld.lp.f`: spheroids, cylinders, Chebyshev particles, the Evans–Fung disc; orientation-averaged cross sections and phase function; convergence helpers. |
| `mie.js`, `tmatrix.js` | Browser ports of the two Python modules, kept structurally parallel (line by line) so each pair can be compared by eye. |
| `validate.py` | 174 checks on the Mie core (76 without `miepython`). |
| `validate_tmatrix.py` | 81 checks on the EBCM core. |
| `crosscheck.py` | Writes `reference.json`: Python results that the page checks its JavaScript against on load. |
| `reference.json` | Generated by `crosscheck.py` (the build regenerates it). |
| `explorer_template.html` | Page layout, controls, plots and the in-browser self-test. |
| `explorer_build.py` | Runs both suites, regenerates `reference.json`, and inlines Plotly and both engines into one HTML file. |
| `rbc_shape_study.py` | Worked example: a red-blood-cell-shaped oblate spheroid against its volume-equivalent sphere, with explicit convergence tests. |
| `docs/index.html` | The built page, served by GitHub Pages (`docs/.nojekyll` turns off Jekyll processing). |

## Validation

**Lorenz–Mie (`validate.py`, 174 checks)** over seven cases from the Rayleigh regime to $x = 29$, lossless, absorbing and metallic:

- the optical theorem; $Q_\text{sca}$ and $g$ by direct angular quadrature;
- Rayleigh limits; complex spherical Bessel functions against closed forms;
- the Maxwell boundary conditions at $r = a$;
- the $m \to 1$ limit (no scatterer, a bare plane wave);
- near-to-far-field asymptotics, asserting an $O(1/kr)$ residual;
- optionally, coefficient-by-coefficient and field-by-field agreement with `miepython`.

**EBCM (`validate_tmatrix.py`, 81 checks):**

- the sphere limit: at axis ratio 1 the T-matrix reproduces Lorenz–Mie for the cross sections, the full amplitude matrix and the orientation-averaged $g$, which exercises the whole chain against an independent implementation;
- the Rayleigh-ellipsoid limit from depolarisation factors, which catches errors in a non-spherical profile that the sphere limit cannot see;
- two independent routes to $C_\text{sca}$: the T-matrix trace formula and numerical angular quadrature of the phase function;
- energy conservation, and unitarity of $S = I + 2T$ for lossless particles, asserted to *decay* with the multipole cutoff rather than pinned at an absolute value;
- the biconcave disc: exact geometry (volume, surface round trip, analytic $dr/d\theta$), the correct Rayleigh limit, and its divergence boundary (see below).

**In the browser:** on every load the page runs both JavaScript ports on 5 Mie, 3 near-field and 6 T-matrix cases (all five shapes, lossless and absorbing; all four amplitude-matrix elements) and compares them with the embedded Python results. Current worst-case agreement is $1.3\times10^{-10}$ for Mie and $1.2\times10^{-9}$ for the T-matrix (tolerances $10^{-9}$ and $10^{-8}$), in about 50 ms. The badge at the top of the page shows the result.

## Accuracy, limits and gotchas

These are the things most worth knowing before relying on the numbers or editing the solvers.

1. **The albedo is a free accuracy meter, but it is necessary, not sufficient.** For a lossless particle $\varpi = C_\text{sca}/C_\text{ext}$ must be exactly 1, so $|\varpi - 1|$ bounds the error in anything derived from the T-matrix. The page shows it and refuses to print derived numbers when it exceeds $10^{-3}$. But near the edge of convergence it can look fine while $g$ is still drifting with the multipole cutoff (for example $|\varpi - 1| = 1.8\times10^{-4}$ while $g$ moves by $5\times10^{-4}$ per multipole). For quantitative work, also require a plateau in the quantity you care about, as `rbc_shape_study.py` does. Absorbing particles have no such meter, and the readout says so.
2. **Double-precision EBCM has a hard convergence domain, and outside it the series diverges.** For high-aspect particles at large size parameter the $Q$ matrix becomes numerically singular (condition numbers around $10^{21}$), and adding multipoles makes the answer worse. Row/column equilibration does not help. The usual remedies are extended precision (Mishchenko distributes a quadruple-precision variant) or a different method.
3. **The biconcave disc converges only for $x_\text{ev} \lesssim 1$.** The Evans–Fung profile is axisymmetric and star-shaped, so EBCM applies in principle, and the geometry is exact. But its deep central concavity shrinks the double-precision convergence domain to the near-Rayleigh regime: beyond it the albedo error *grows* with $n_\text{max}$, to $10^3$ and beyond. A real red cell in visible or near-infrared light ($x_\text{ev} \approx 10$–$30$) is far outside it. The page's "disc demo" preset shows the converging case; the "true biconcave DISC" preset is refused with an explanation. For the true shape use the discrete dipole approximation (for example ADDA) or extended-precision EBCM.
4. **The interactive T-matrix is capped at $n_\text{max} = 34$** (roughly $x_\text{ev} \lesssim 22$) to keep the page responsive. Larger particles show a message instead of numbers. The Python reference has no such cap.
5. **Elongated particles need more multipoles than $x_\text{ev}$ suggests.** The default cutoff (Mishchenko's $x + 4.05x^{1/3}$) is set by the equal-volume size parameter, but a long cylinder's circumscribing sphere is much larger. For lossless particles the page retries with a larger cutoff and more quadrature points when the albedo meter is poor, and keeps the best result. Absorbing elongated particles get no such retry, because there is no meter to judge it by.
6. **Cylinders converge more slowly than spheroids.** The profile has a corner, and (as in Mishchenko's code) the surface quadrature is split there. The unitarity residual falls only slowly with $n_\text{max}$ for a cylinder, but geometrically for a spheroid. That is physics, not a bug.
7. **Mie truncation is a ceiling, not a floor.** Wiscombe's $n_\text{stop} = x + 4x^{1/3} + 2$ is used deliberately. Adding many more terms makes the *near field* worse, because the upward recurrence for the real Riccati–Bessel function becomes unstable once $n \gg \rho$. Near-field accuracy at the sphere surface is therefore about $10^{-5}$.
8. **$Q_\text{ext}$ converges more slowly in $n_\text{max}$ than $Q_\text{sca}$**: the first is linear in $T$ (a trace), the second quadratic.
9. **No material dispersion.** $n$ and $k$ are held constant across the spectrum panel and in the shape study. The panel caption says so.
10. **Near field for spheres only.** EBCM fields inside the circumscribing sphere are unreliable (the Rayleigh hypothesis), so for non-spheres the near-field panel is replaced by the shape sweep.
11. **Axisymmetric shapes only.** Aggregates and arbitrary shapes need DDA or another general method.
12. **Presets are illustrative.** Biological and metallic presets use rounded literature values, not measurements of any specific sample. Not for dosimetry or any safety-critical use.

Notes for anyone editing the solvers:

- **Seed Miller's downward recurrence at 1.0, not at a tiny value.** The recurrence grows as $n$ decreases, so a tiny seed leaves $j_0$ denormal. NumPy's complex division survives this; a hand-written JavaScript division that forms $|j_0|^2$ underflows to zero, and every internal-field value becomes NaN. The validated Python hid this bug; only the browser self-test found it.
- **At the origin, take $\theta$ from `atan2(|u|, z)`** on the raw coordinates, never from $\cos\theta = z/r$ after clamping $r$. An even-sized grid never samples $r = 0$, so only an odd grid exposes the bug.
- **`miepython` conventions** (version 3): it uses $m = n - ik$; `S1_S2` conjugates internally; `norm='wiscombe'` is the normalisation that satisfies the optical theorem, while `norm='bohren'` is twice the textbook amplitude. Its `cn_dn` agrees with ours to about $3\times10^{-5}$; ours reproduces $c_n = d_n = 1$ at $m = 1$ to $10^{-12}$.
- **NumPy 2 complex `matmul`** can raise spurious divide-by-zero or overflow floating-point flags on finite inputs. `tmatrix.py` suppresses them locally and checks the block output for genuine non-finite values.
- **Relative error is the wrong metric** at the deep nulls of a phase function, for $Q_\text{abs}$ of a transparent particle (legitimately 0, arriving as roundoff), and for the tail of a decaying coefficient array. The tests normalise by the pattern peak, by $Q_\text{ext}$, or by the largest element.
- **Plotly does not decode named HTML entities** in trace names or axis titles; use Unicode characters.

## Worked example: red-cell shape and reduced scattering

`rbc_shape_study.py` compares a red-blood-cell-shaped oblate spheroid ($\varepsilon = a/c = 2.77$, volume 90 fL, $r_\text{ev} = 2.78$ µm, $m = 1.042$) with its volume-equivalent sphere, both randomly oriented. At fixed number density the reduced scattering coefficient $\mu_s' = N C_\text{sca}(1-g)$ depends on shape only through $C_\text{sca}(1-g)$. With the convergence criteria above (albedo plus a plateau in $g$) it finds:

| $\lambda$ (nm) | $x_\text{ev}$ | $g$, sphere | $g$, spheroid | $\mu_s'$ ratio, spheroid / sphere |
|---|---|---|---|---|
| 1000 | 23.5 | 0.99133 | 0.98935 | 1.106 |
| 1300 | 18.1 | 0.98769 | 0.98500 | 1.103 |
| 1550 | 15.1 | 0.98411 | 0.98084 | 1.093 |
| 2200 | 10.7 | 0.97279 | 0.96797 | 1.070 |

The sphere is more forward-scattering, so the sphere model gives a reduced scattering coefficient 7–11 % lower than the spheroid. At 800 nm ($x_\text{ev} = 29$) double-precision EBCM does not converge for this shape, and no number is reported. An oblate spheroid is only a proxy for the biconcave disc; the indices are illustrative, and random orientation is assumed.

## Credits

- **T-matrix code:** M. I. Mishchenko (NASA GISS), `ampld.lp.f`, public domain, ported from the copy distributed with `pytmatrix`. Please cite M. I. Mishchenko and L. D. Travis, *Opt. Commun.* **109**, 16–21 (1994), and M. I. Mishchenko, L. D. Travis and D. W. Mackowski, *JQSRT* **55**, 535–575 (1996). The method is P. C. Waterman's EBCM.
- **Mie conventions:** C. F. Bohren and D. R. Huffman, *Absorption and Scattering of Light by Small Particles* (Wiley, 1983); truncation after W. J. Wiscombe, *Appl. Opt.* **19**, 1505 (1980).
- **Red-cell profile:** E. Evans and Y.-C. Fung, *Microvasc. Res.* **4**, 335–347 (1972).
- **Plotting:** [plotly.js](https://github.com/plotly/plotly.js) (MIT License), inlined in the built page.
- **Optional cross-check:** [miepython](https://github.com/scottprahl/miepython) (MIT License).

Full attributions are in [`NOTICE`](NOTICE).

## Licence

MIT; see [`LICENSE`](LICENSE). Provided as is, without warranty and without support. Issues and pull requests may not receive a response.
