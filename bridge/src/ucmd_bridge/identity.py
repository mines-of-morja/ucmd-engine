"""Molecule identity across position reads: outputs in canonical order (architecture invariant 4).

**Why.**
- In the 20 ns pilot of 2026-11-20 (OpenCL, mixed precision, CustomIntegrator, OpenMM 8.3.1),
  `Context.getState` returned identical molecules with their identities exchanged. That
  happened every 84 ps, for about half of the 594 molecules, of every type.
- The dynamics are unaffected:
  - identical molecules are interchangeable;
  - box images move with the molecules, so sums over a species stay continuous.
- What is wrong is the labels. Everything that follows one molecule over time (self-diffusion,
  residence times) is corrupted.
- OpenMM reorders identical molecules internally every 250 steps and maps them back through
  `atomIndex`. Where that mapping fails on this path is not yet known.

**What this module does.**
- `Tracker` matches every molecule to the nearest molecule of the same type at the previous
  read, in unwrapped coordinates. It keeps the permutation from OpenMM's labels to the
  canonical ones.
- The distance is the atom-wise RMSD (atom k to atom k), not the distance between centroids.
  A neighbour of the same type would have to sit in the same place *and* in the same
  orientation to be confused with the molecule. With 2 ps between frames, centroids alone
  were ambiguous for EC (0.27 against 0.34 nm, pilot of 2026-11-20).
- A match is accepted only when it is unambiguous. For every molecule, the nearest candidate
  must lie within `ratio` times the distance to the second nearest, and the matches must form a
  permutation. Otherwise `IdentityError`: a wrong label is worse than a stopped run
  (invariant 7).
- **Strict mode (the backend)** relies on a physical bound:
  - between reads 0.25 ps apart, no molecule moves more than about 0.1 nm (atom-wise RMSD);
  - two distinct identical molecules are never closer than about 0.3 nm.
  - So only molecules that moved more than `max_move_nm` (0.2 nm) are suspects. Only suspects
    are matched, among themselves (a swap makes both partners suspects).
  - A relabelling must be unambiguous; a suspect that finds itself nearest keeps its label even
    when a neighbour is about as close (no label changes, nothing to decide).
  - After relabelling, a molecule may still have moved more than `max_move_nm` if its match was
    unambiguous: fast but real motion, e.g. in the violent first picoseconds of a too dense
    start (A4, 2026-11-20). Only moves beyond `MAX_PHYSICAL_MOVE_FACTOR × max_move_nm`
    (0.6 nm per read) raise IdentityError. An error there would repeat at every restart from
    the same checkpoint, so it is kept for the impossible.
  - The cost scales with the number of suspects, not with the square of the system.
- The backend reads every 250 steps at most, OpenMM's reorder interval, so there is at most one
  reorder between reads.

**Units:** positions in nm.
"""

from __future__ import annotations

import numpy as np

DEFAULT_RATIO = 0.5  # nearest / second-nearest distance above which a match is refused
DEFAULT_MAX_MOVE_NM = 0.2  # strict mode: molecules that moved more between two reads are matched
MAX_PHYSICAL_MOVE_FACTOR = 3.0  # strict mode: a move beyond 3 × max_move_nm per read is an error


class IdentityError(RuntimeError):
    """The molecules cannot be matched unambiguously between two reads."""


def molecule_groups(molecule_index: np.ndarray, molecule_type: list[str]) -> dict[str, np.ndarray]:
    """Molecule ids per molecule type. Molecules must be contiguous runs of atoms, numbered from 0
    in canonical order, and molecule_type[m] names the type of molecule m."""
    groups: dict[str, list[int]] = {}
    for m, name in enumerate(molecule_type):
        groups.setdefault(name, []).append(m)
    return {k: np.array(v, dtype=np.int64) for k, v in groups.items()}


class Tracker:
    """Keeps canonical molecule labels across position reads.

    `perm[a]` is the index in OpenMM's (raw) order of canonical atom a, so the canonical
    positions are `raw[perm]`. The permutation only ever exchanges whole molecules of the same
    type.
    """

    def __init__(self, molecule_index: np.ndarray, molecule_type: list[str], *,
                 ratio: float = DEFAULT_RATIO, max_move_nm: float = DEFAULT_MAX_MOVE_NM,
                 slots: np.ndarray | None = None) -> None:  # fmt: skip
        mol = np.asarray(molecule_index, dtype=np.int64)
        if np.any(np.diff(mol) < 0) or (len(mol) and mol[0] != 0):
            raise ValueError("molecules must be contiguous runs of atoms numbered from 0")
        self.n_atoms = len(mol)
        self.starts = np.flatnonzero(np.r_[True, mol[1:] != mol[:-1]])
        self.sizes = np.diff(np.r_[self.starts, self.n_atoms])
        if len(self.starts) != len(molecule_type):
            raise ValueError("one molecule type per molecule is needed")
        self.groups = molecule_groups(mol, molecule_type)
        for name, ids in self.groups.items():
            if len(set(self.sizes[ids].tolist())) != 1:
                raise ValueError(f"molecules of type {name} differ in size")
        self.ratio = ratio
        self.max_move_nm = max_move_nm
        self.slot = np.arange(len(self.starts))  # canonical molecule m sits in raw slot slot[m]
        if slots is not None:
            self.set_slots(np.asarray(slots, dtype=np.int64))
        self.previous: np.ndarray | None = None  # canonical positions at the last read
        self.exchanges = 0  # molecules relabelled so far
        self.worst_ratio = 0.0  # largest nearest / second-nearest distance accepted so far
        self.ambiguous = 0  # repair mode: assignments above the ratio limit
        self.events = 0  # repair mode: reads with a relabelling event

    # -- the permutation -------------------------------------------------------------------
    @property
    def perm(self) -> np.ndarray:
        """Atom permutation: canonical positions = raw[perm]."""
        return np.concatenate([np.arange(self.starts[s], self.starts[s] + self.sizes[s])
                               for s in self.slot])  # fmt: skip

    def set_slots(self, slots: np.ndarray) -> None:
        """Restore the molecule permutation (e.g. from a checkpoint): slots[m] is the raw slot of
        canonical molecule m. Only exchanges within a type are allowed."""
        if sorted(slots.tolist()) != list(range(len(self.starts))):
            raise ValueError("not a permutation of the molecules")
        for name, ids in self.groups.items():
            if set(slots[ids].tolist()) != set(ids.tolist()):
                raise ValueError(f"the permutation moves a molecule of type {name} to another type")
        self.slot = slots.copy()

    def perm_of(self, slot: np.ndarray) -> np.ndarray:
        return np.concatenate([np.arange(self.starts[s], self.starts[s] + self.sizes[s])
                               for s in slot])  # fmt: skip

    # -- reads -----------------------------------------------------------------------------
    def _molecules(self, x: np.ndarray, ids: np.ndarray, slots: np.ndarray) -> np.ndarray:
        """(len(ids), atoms·3) coordinates of the molecules in `slots` (all of one type)."""
        n = int(self.sizes[ids[0]])
        index = self.starts[slots][:, None] + np.arange(n)[None, :]
        return x[index].reshape(len(slots), 3 * n)

    @staticmethod
    def _rmsd(a: np.ndarray, b: np.ndarray, n_atoms: int) -> np.ndarray:
        """(len(a), len(b)) atom-wise RMSD between rows of a and rows of b (nm)."""
        a0 = a - a.mean(axis=0)  # a common shift keeps the expansion well conditioned
        b0 = b - a.mean(axis=0)
        d2 = (a0**2).sum(1)[:, None] + (b0**2).sum(1)[None, :] - 2.0 * a0 @ b0.T
        return np.sqrt(np.maximum(d2, 0.0) / n_atoms)

    def _own(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Per canonical molecule: atom-wise RMSD between two canonical frames (nm)."""
        d2 = ((a - b) ** 2).sum(axis=1)
        return np.sqrt(np.add.reduceat(d2, self.starts) / self.sizes)

    def update(self, raw: np.ndarray, *, jump_nm: float | None = None) -> np.ndarray:
        """Match the molecules of a new read (raw OpenMM order, nm) to the last read and return
        the positions in canonical order.

        - **Strict** (`jump_nm` None, the backend): every read is matched; any ambiguous match
          raises IdentityError.
        - **Repair** (`jump_nm` given, old files): labels are kept unless some molecule moved
          more than `jump_nm` since the last read, a relabelling event. Then every type is
          assigned anew by the assignment of least total distance (`min_cost_assignment`):
          with coarse frames a greedy choice can take a molecule's only continuation for a
          neighbour (pilot, frame 967). Ambiguous assignments are counted in
          `self.ambiguous` instead of raising. Afterwards no molecule may still move more than
          `jump_nm`, else IdentityError.
        """
        raw = np.asarray(raw, dtype=np.float64)
        if self.previous is None:
            self.previous = raw[self.perm]
            return self.previous.copy()
        if jump_nm is None:
            return self._update_strict(raw)
        if self._own(self.previous, raw[self.perm]).max() <= jump_nm:
            self.previous = raw[self.perm]
            return self.previous.copy()
        self.events += 1
        new_slot = self.slot.copy()
        for ids in self.groups.values():
            slots = self.slot[ids]
            prev = self._molecules(self.previous, ids, ids)  # canonical order
            cur = self._molecules(raw, ids, slots)
            d = self._rmsd(prev, cur, int(self.sizes[ids[0]]))
            match, ratio = _ratios(d, min_cost_assignment(d))
            bad = ratio > self.ratio
            if np.any(bad):
                self.ambiguous += int(bad.sum())
            if np.any(~bad):
                self.worst_ratio = max(self.worst_ratio, float(ratio[~bad].max()))
            changed = match != np.arange(len(ids))
            self.exchanges += int(changed.sum())
            new_slot[ids] = slots[match]
        self.slot = new_slot
        canonical = raw[self.perm]
        moved = self._own(self.previous, canonical)
        if moved.max() > jump_nm:
            raise IdentityError(
                f"molecule {int(np.argmax(moved))} still moves {moved.max():.2f} nm after "
                f"relabelling (limit {jump_nm} nm)"
            )
        self.previous = canonical
        return canonical.copy()

    def _update_strict(self, raw: np.ndarray) -> np.ndarray:
        own = self._own(self.previous, raw[self.perm])
        suspects = np.flatnonzero(own > self.max_move_nm)
        if len(suspects):
            new_slot = self.slot.copy()
            mtype = np.empty(len(self.starts), dtype=object)
            for name, ids in self.groups.items():
                mtype[ids] = name
            for name in sorted(set(mtype[suspects].tolist())):
                ids = suspects[mtype[suspects] == name]
                slots = self.slot[ids]
                prev = self._molecules(self.previous, ids, ids)
                cur = self._molecules(raw, ids, slots)
                match, ratio = _assign(self._rmsd(prev, cur, int(self.sizes[ids[0]])))
                # Ambiguity matters only where the matching would change a label. Fast molecules
                # that each find themselves nearest keep their labels even if a neighbour is
                # about as close (the violent first picoseconds of a dense NPT start, EC
                # 2026-11-20); a relabelling must be unambiguous.
                changed = match != np.arange(len(ids))
                doubtful = changed & (ratio > self.ratio)
                if np.any(doubtful):
                    k = int(np.flatnonzero(doubtful)[np.argmax(ratio[doubtful])])
                    raise IdentityError(
                        f"{name}: molecule {int(ids[k])} is ambiguous (nearest / second-nearest "
                        f"atom RMSD {ratio[k]:.2f} > {self.ratio}); reads are too far apart to "
                        f"keep molecule identities"
                    )
                if np.any(changed):
                    self.worst_ratio = max(self.worst_ratio, float(ratio[changed].max()))
                self.exchanges += int((match != np.arange(len(ids))).sum())
                new_slot[ids] = slots[match]
            self.slot = new_slot
        canonical = raw[self.perm]
        if len(suspects):
            moved = self._own(self.previous, canonical)
            limit = MAX_PHYSICAL_MOVE_FACTOR * self.max_move_nm
            if moved.max() > limit:
                raise IdentityError(
                    f"molecule {int(np.argmax(moved))} moved {moved.max():.2f} nm between two "
                    f"reads (limit {limit:.2f} nm); reads are too far apart to keep molecule "
                    f"identities"
                )
        self.previous = canonical
        return canonical.copy()


def _assign(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign rows to columns of a distance matrix, closest pairs first (greedy).

    Returns the column of each row and a per-row ambiguity ratio: the assigned distance over
    the smallest distance to any other column. Below 1 the assigned column is the row's
    nearest; near 1 another column is about as close.
    """
    n = d.shape[0]
    match = np.full(n, -1)
    taken = np.zeros(n, dtype=bool)
    for flat in np.argsort(d, axis=None, kind="stable"):
        i, j = divmod(int(flat), n)
        if match[i] < 0 and not taken[j]:
            match[i] = j
            taken[j] = True
            if taken.all():
                break
    return _ratios(d, match)


def _ratios(d: np.ndarray, match: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(match, per-row assigned distance over the smallest distance to any other column)."""
    n = d.shape[0]
    if n == 1:
        return match, np.zeros(1)
    rows = np.arange(n)
    other = d.copy()
    other[rows, match] = np.inf
    return match, d[rows, match] / np.maximum(other.min(axis=1), 1e-300)


def min_cost_assignment(cost: np.ndarray) -> np.ndarray:
    """The column of each row in the assignment of least total cost (square matrix).

    Hungarian algorithm with potentials (Kuhn–Munkres, O(n³)), the inner minimisation
    vectorised over columns.
    """
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=np.int64)  # p[j]: row (1-based) assigned to column j
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = np.flatnonzero(~used)
            cur = cost[i0 - 1, free - 1] - u[i0] - v[free]
            better = cur < minv[free]
            minv[free[better]] = cur[better]
            way[free[better]] = j0
            j1 = int(free[np.argmin(minv[free])])
            delta = minv[j1]
            u[p[used]] += delta
            v[used] -= delta
            minv[free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
    match = np.empty(n, dtype=np.int64)
    match[p[1:] - 1] = np.arange(n)
    return match
