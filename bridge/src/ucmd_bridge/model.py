"""The model of a system package as an OpenMM System (E3.7).

The same terms and conventions as the engine's `force-field` provider (spec/system-package.md
§5): harmonic bonds and angles, periodic / Ryckaert–Bellemans / OPLS-Fourier dihedrals, periodic
and harmonic impropers, Lennard-Jones with the package's combination rule (plain cutoff or
potential shift, optional tail correction counted as the engine counts it), Coulomb with PME,
scaled 1-4 pairs without cutoff.
PME uses the engine's α and grid (OpenMM's tolerance rules, computed here explicitly so that the
GPU platform cannot choose a different grid).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import openmm as mm

from .package import Topology


class ModelError(ValueError):
    """The provider configuration asks for something this backend does not implement."""


@dataclass(frozen=True)
class NonbondedSettings:
    cutoff_nm: float | None
    lj_cutoff_mode: str = "truncate"
    dispersion_correction: bool = False
    coulomb: str | None = None
    ewald_tolerance: float = 1e-5
    ewald_alpha_per_nm: float | None = None
    pme_grid: tuple[int, int, int] | None = None
    pme_order: int = 5

    @classmethod
    def from_provider(cls, p: dict) -> NonbondedSettings:
        kind = p.get("kind")
        if kind not in ("force-field", "nonbonded"):
            raise ModelError(f'the OpenMM backend runs kind = "force-field", not {kind!r}')
        known = {
            "kind", "name", "cutoff_nm", "lj_cutoff_mode", "dispersion_correction", "coulomb",
            "ewald_tolerance", "ewald_alpha_per_nm", "pme_grid", "pme_order", "neighbor_skin_nm",
        }  # fmt: skip
        unknown = set(p) - known
        if unknown:
            raise ModelError(f"provider keys the OpenMM backend does not know: {sorted(unknown)}")
        s = cls(
            cutoff_nm=p.get("cutoff_nm"),
            lj_cutoff_mode=p.get("lj_cutoff_mode", "truncate"),
            dispersion_correction=bool(p.get("dispersion_correction", False)),
            coulomb=p.get("coulomb"),
            ewald_tolerance=float(p.get("ewald_tolerance", 1e-5)),
            ewald_alpha_per_nm=p.get("ewald_alpha_per_nm"),
            pme_grid=tuple(p["pme_grid"]) if "pme_grid" in p else None,
            pme_order=int(p.get("pme_order", 5)),
        )
        if s.pme_order != 5:
            raise ModelError("OpenMM's PME has B-spline order 5; pme_order must be 5")
        if s.coulomb == "ewald":
            raise ModelError(
                'coulomb = "ewald" is not implemented in the OpenMM backend; use "pme"'
            )
        if s.dispersion_correction and s.lj_cutoff_mode != "truncate":
            raise ModelError("the tail correction is implemented with lj_cutoff_mode = truncate")
        return s


def legal_size(n: int) -> int:
    """Smallest product of 2, 3, 5 and 7 that is ≥ n (OpenMM's and the engine's rule)."""
    m = max(n, 1)
    while True:
        r = m
        for f in (2, 3, 5, 7):
            while r % f == 0:
                r //= f
        if r == 1:
            return m
        m += 1


def pme_parameters(cell_nm, s: NonbondedSettings) -> tuple[float, int, int, int]:
    """α (nm⁻¹) and grid as the engine chooses them: α = √(−ln 2δ)/r_c and
    n_d = legal(⌈2α·h_d / (3δ^{1/5})⌉) with h_d the diagonal of the reduced cell."""
    assert s.cutoff_nm is not None
    alpha = s.ewald_alpha_per_nm or math.sqrt(-math.log(2.0 * s.ewald_tolerance)) / s.cutoff_nm
    if s.pme_grid is not None:
        return (alpha, *s.pme_grid)
    scale = 2.0 * alpha / (3.0 * s.ewald_tolerance**0.2)
    grid = [legal_size(math.ceil(scale * cell_nm[d][d])) for d in range(3)]
    return (alpha, *grid)


def _mix(top: Topology, i: int, j: int) -> tuple[float, float]:
    """σ_ij, ε_ij of atoms i, j: the pair override if there is one, else the rule."""
    if top.pairs is not None:
        a, b = top.pairs.type_index[i], top.pairs.type_index[j]
        return float(top.pairs.sigma_nm[a, b]), float(top.pairs.epsilon_kjmol[a, b])
    si, sj = top.sigma_nm[i], top.sigma_nm[j]
    sigma = math.sqrt(si * sj) if top.combination_rule == "geometric" else 0.5 * (si + sj)
    return sigma, math.sqrt(top.epsilon_kjmol[i] * top.epsilon_kjmol[j])


def tail_weights(top: Topology) -> tuple[list[int], list[float], float]:
    """Per atom its LJ class and that class's diagonal weight, and the off-diagonal weight, which
    make OpenMM's long-range correction equal the engine's tail (2026-11-21).

    The engine's tail is 2π/V Σ_ab n_a n_b ∫_rc^∞ r² u_ab(r) dr over the LJ types, with N²/2
    pairs (Allen & Tildesley). OpenMM (CustomNonbondedForceImpl::calcLongRangeCorrection) counts
    each pair of classes once, a class with itself n(n+1)/2 times, and scales the mean over
    N(N+1)/2 pairs of all N atoms by N². Both agree for one type with LJ on every atom (argon)
    and differ by O(1/N) otherwise: for SPC/Fw water, 1.3·10⁻³ of the tail (0.17 kJ/mol for 500
    molecules, 6·10⁻⁶ of the conformance energy). Weights w_aa = n_a (N+1) / (N (n_a+1)) and
    w_ab = (N+1)/N on the integrand of the class pair turn OpenMM's sum into the engine's.
    Classes are distinct (σ, ε); the engine's formula does not change when types with equal
    parameters are merged. With pair overrides (spec 0.1.2) the classes are the atom types: a
    type's pairs may differ from another's with the same σ and ε, and OpenMM groups the particles
    by their per-particle parameters, which are then the types."""
    if top.pairs is not None:
        keys = [(int(t), 0.0) for t in top.pairs.type_index]
    else:
        keys = list(zip(top.sigma_nm.tolist(), top.epsilon_kjmol.tolist(), strict=True))
    index: dict[tuple[float, float], int] = {}
    cls = [index.setdefault(k, len(index)) for k in keys]
    counts = [0] * len(index)
    for c in cls:
        counts[c] += 1
    n = len(cls)
    diagonal = [counts[c] * (n + 1) / (n * (counts[c] + 1)) for c in cls]
    return cls, diagonal, (n + 1) / n


def _lj_force(top: Topology, shift: str, tail: bool) -> mm.CustomNonbondedForce:
    """LJ between non-excluded pairs: per-particle mixing, or per-type tables with overrides.

    Without overrides the expression is the one used before spec 0.1.2, so existing runs do not
    change. With them, σ_ij and ε_ij come from Discrete2DFunction tables (OpenMM's usual NBFIX
    route), which keeps the cutoff, the shift and the long-range correction. With the tail
    correction, OpenMM's integral beyond the cutoff carries the class-pair weights of
    `tail_weights`; inside the cutoff the factor is exactly 1.
    """
    energy = f"4*eps*((sig/r)^12-(sig/r)^6){shift}"
    if tail:
        energy = f"({energy})*(1 + step(r - rc)*(w - 1)); w = select(delta(cls1 - cls2), wd1, wo)"
    if top.pairs is None:
        mix = (
            "sig=sqrt(sig1*sig2); eps=sqrt(eps1*eps2)"
            if top.combination_rule == "geometric"
            else "sig=0.5*(sig1+sig2); eps=sqrt(eps1*eps2)"
        )
        lj = mm.CustomNonbondedForce(f"{energy}; {mix}")
        lj.addPerParticleParameter("sig")
        lj.addPerParticleParameter("eps")
        values = [
            [float(sg), float(ep)] for sg, ep in zip(top.sigma_nm, top.epsilon_kjmol, strict=True)
        ]
    else:
        n = len(top.pairs.sigma_nm)
        lj = mm.CustomNonbondedForce(f"{energy}; sig=sigt(t1, t2); eps=epst(t1, t2)")
        # Discrete2DFunction values run with the first index fastest; the tables are symmetric.
        sig = top.pairs.sigma_nm.T.ravel().tolist()
        eps = top.pairs.epsilon_kjmol.T.ravel().tolist()
        lj.addTabulatedFunction("sigt", mm.Discrete2DFunction(n, n, sig))
        lj.addTabulatedFunction("epst", mm.Discrete2DFunction(n, n, eps))
        lj.addPerParticleParameter("t")
        values = [[float(t)] for t in top.pairs.type_index]
    if tail:
        lj.addPerParticleParameter("cls")
        lj.addPerParticleParameter("wd")
        cls, diagonal, off_diagonal = tail_weights(top)
        lj.addGlobalParameter("wo", off_diagonal)
        values = [[*v, float(c), wd] for v, c, wd in zip(values, cls, diagonal, strict=True)]
    for v in values:
        lj.addParticle(v)
    return lj


def build_system(top: Topology, s: NonbondedSettings) -> tuple[mm.System, dict]:
    """The OpenMM System and a description of the electrostatics, for provenance."""
    periodic = top.cell_nm is not None
    if periodic and s.cutoff_nm is None:
        raise ModelError("a periodic package needs cutoff_nm")
    if periodic and s.coulomb not in (None, "pme"):
        raise ModelError(f"coulomb = {s.coulomb!r} for a periodic package; use pme")
    system = mm.System()
    for mass in top.masses_amu:
        system.addParticle(float(mass))
    if periodic:
        system.setDefaultPeriodicBoxVectors(*(mm.Vec3(*row) for row in top.cell_nm))
    for i, j, length in top.constraints:
        system.addConstraint(i, j, length)

    bonds = mm.HarmonicBondForce()
    for i, j, r0, k in top.bonds:
        bonds.addBond(i, j, r0, k)
    angles = mm.HarmonicAngleForce()
    for i, j, k, theta0, kk in top.angles:
        angles.addAngle(i, j, k, theta0, kk)
    dihedrals = mm.PeriodicTorsionForce()
    for i, j, k, m, n, phase, kk in top.periodic:
        dihedrals.addTorsion(i, j, k, m, n, phase, kk)
    rb = mm.RBTorsionForce()
    for i, j, k, m, c in top.rb:
        rb.addTorsion(i, j, k, m, *c)
    impropers = mm.PeriodicTorsionForce()
    for i, j, k, m, n, phase, kk in top.improper_periodic:
        impropers.addTorsion(i, j, k, m, n, phase, kk)
    harmonic = mm.CustomTorsionForce(
        "0.5*k*d^2; d = theta - xi0 - 2*pi*floor((theta - xi0 + pi)/(2*pi)); pi = 3.141592653589793"
    )
    harmonic.addPerTorsionParameter("xi0")
    harmonic.addPerTorsionParameter("k")
    for i, j, k, m, xi0, kk in top.improper_harmonic:
        harmonic.addTorsion(i, j, k, m, [xi0, kk])

    coulomb = mm.NonbondedForce()
    for q in top.charges_e:
        coulomb.addParticle(float(q), 1.0, 0.0)
    for i, j in top.excluded:
        coulomb.addException(i, j, 0.0, 1.0, 0.0)
    for i, j in top.pairs14:
        qq = top.scale_14_coulomb * top.charges_e[i] * top.charges_e[j]
        coulomb.addException(i, j, float(qq), 1.0, 0.0)
    shift = " - 4*eps*((sig/rc)^12-(sig/rc)^6)" if s.lj_cutoff_mode == "potential-shift" else ""
    if s.dispersion_correction and not periodic:
        raise ModelError("the tail correction needs a periodic package")
    lj = _lj_force(top, shift, s.dispersion_correction)
    for i, j in (*top.excluded, *top.pairs14):
        lj.addExclusion(i, j)
    lj.setUseLongRangeCorrection(s.dispersion_correction)
    info: dict = {"coulomb": "direct"}
    if periodic:
        alpha, nx, ny, nz = pme_parameters(top.cell_nm, s)
        coulomb.setNonbondedMethod(mm.NonbondedForce.PME)
        coulomb.setCutoffDistance(s.cutoff_nm)
        coulomb.setPMEParameters(alpha, nx, ny, nz)
        coulomb.setUseDispersionCorrection(False)
        lj.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        lj.setCutoffDistance(s.cutoff_nm)
        lj.addGlobalParameter("rc", s.cutoff_nm)
        info = {"coulomb": "pme", "alpha_per_nm": alpha, "pme_grid": [nx, ny, nz]}
    else:
        coulomb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
        lj.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
        if s.lj_cutoff_mode != "truncate":
            raise ModelError("potential-shift needs a cutoff")
    lj14 = mm.CustomBondForce("4*eps*((sig/r)^12-(sig/r)^6)")
    lj14.addPerBondParameter("sig")
    lj14.addPerBondParameter("eps")
    for i, j in top.pairs14:
        sigma, epsilon = _mix(top, i, j)
        lj14.addBond(i, j, [sigma, top.scale_14_lj * epsilon])
    lj14.setUsesPeriodicBoundaryConditions(periodic)
    for force in (bonds, angles, dihedrals, rb, impropers, harmonic, coulomb, lj, lj14):
        system.addForce(force)
    return system, info
