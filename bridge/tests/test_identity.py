"""Molecule identity across reads (identity.py): the swaps seen in the 2026-11-20 pilot are undone
exactly, and anything ambiguous is an error."""

from __future__ import annotations

import numpy as np
import pytest

from ucmd_bridge.identity import IdentityError, Tracker

# three types as in the electrolyte: an ion (1 atom), an anion (7 atoms), a solvent (10 atoms)
TYPES = ["Li"] * 12 + ["PF6"] * 12 + ["EC"] * 30
SIZES = {"Li": 1, "PF6": 7, "EC": 10}


def _system(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """molecule_index per atom, and positions: molecules on a lattice 0.6 nm apart (as in a
    liquid, where identical molecules are ≥ 0.3 nm apart), atoms within 0.15 nm of the centre."""
    mol = np.concatenate([np.full(SIZES[t], m) for m, t in enumerate(TYPES)])
    grid = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3) * 0.6
    centres = grid[rng.permutation(len(grid))[: len(TYPES)]]
    x = centres[mol] + rng.uniform(-0.15, 0.15, (len(mol), 3))
    return mol, x


def _swap(rng: np.random.Generator, mol: np.ndarray, n: int) -> np.ndarray:
    """An atom permutation that exchanges whole molecules of the same type (raw = true[p])."""
    starts = np.flatnonzero(np.r_[True, mol[1:] != mol[:-1]])
    slots = np.arange(len(TYPES))
    for name in SIZES:
        ids = np.array([m for m, t in enumerate(TYPES) if t == name])
        pick = rng.choice(ids, size=min(n, len(ids)), replace=False)
        slots[pick] = rng.permutation(pick)
    return np.concatenate([np.arange(starts[s], starts[s] + SIZES[TYPES[s]]) for s in slots])


def test_swaps_within_types_are_undone_exactly() -> None:
    rng = np.random.default_rng(20261120)
    mol, x = _system(rng)
    tracker = Tracker(mol, TYPES)
    hidden = np.arange(len(mol))  # the backend's raw order: raw = true[hidden]
    swaps = 0
    base = x.copy()
    for frame in range(200):
        # molecules rattle around their sites (repulsion keeps identical ones ≥ 0.3 nm apart);
        # 0.02 nm per read and axis, as at 0.25 ps in a liquid, plus a slow common drift
        x = base + rng.normal(0.0, 0.02, (len(TYPES), 3))[mol] + 0.003 * frame
        if frame % 17 == 5:
            p = _swap(rng, mol, 5)
            hidden = hidden[p]
            swaps += 1
        raw = x[hidden]
        # whole box images travel with a swapped molecule: add them to the raw coordinates
        assert np.array_equal(tracker.update(raw), x)
    assert swaps > 10 and tracker.exchanges > 0


def test_box_images_travel_with_the_molecules() -> None:
    rng = np.random.default_rng(1)
    mol, x = _system(rng)
    shift = np.zeros_like(x)
    shift[mol == 3] += np.array([4.3, 0.0, -8.6])  # molecule 3 is two images away (unwrapped)
    x = x + shift
    tracker = Tracker(mol, TYPES)
    tracker.update(x)
    p = _swap(rng, mol, 12)
    x = x + rng.normal(0.0, 0.01, x.shape)
    assert np.array_equal(tracker.update(x[p]), x)


def test_reads_too_far_apart_are_an_error() -> None:
    rng = np.random.default_rng(2)
    mol, x = _system(rng)
    tracker = Tracker(mol, TYPES)
    tracker.update(x)
    with pytest.raises(IdentityError, match="too far apart"):
        tracker.update(x + rng.normal(0.0, 0.25, (len(TYPES), 3))[mol])


def _two_li(rng: np.random.Generator, gap_nm: float) -> tuple:
    mol, x = _system(rng)
    a, b = np.flatnonzero(mol == 0), np.flatnonzero(mol == 1)
    x[b] = x[a] + [gap_nm, 0.0, 0.0]  # two Li close together (the lattice keeps others away)
    return mol, x, a, b


def test_a_swap_of_close_neighbours_is_undone() -> None:
    mol, x, a, b = _two_li(np.random.default_rng(3), 0.35)
    tracker = Tracker(mol, TYPES)
    tracker.update(x)
    y = x + 0.01
    y[a], y[b] = y[b].copy(), y[a].copy()  # labels exchanged, partners 0.35 nm apart
    assert np.array_equal(tracker.update(y), x + 0.01)
    assert tracker.exchanges == 2


def test_an_ambiguous_move_between_reads_is_an_error() -> None:
    mol, x, a, b = _two_li(np.random.default_rng(3), 0.42)
    tracker = Tracker(mol, TYPES)
    tracker.update(x)
    y = x.copy()
    y[a] = x[a] + [0.62, 0.0, 0.0]  # one Li at 0.20 and one at 0.62 nm along the pair axis:
    y[b] = x[a] + [0.20, 0.0, 0.0]  # b's old place (0.42) is 0.22 and 0.20 nm from both
    with pytest.raises(IdentityError, match="ambiguous"):
        tracker.update(y)


def test_the_permutation_survives_a_checkpoint() -> None:
    rng = np.random.default_rng(4)
    mol, x = _system(rng)
    p = _swap(rng, mol, 6)
    first = Tracker(mol, TYPES)
    first.update(x)
    canonical = first.update(x[p])
    second = Tracker(mol, TYPES, slots=first.slot)  # as after a restart
    assert np.array_equal(second.update(x[p]), canonical)


def test_a_wrong_stored_permutation_is_refused() -> None:
    rng = np.random.default_rng(5)
    mol, _ = _system(rng)
    slots = np.arange(len(TYPES))
    slots[[11, 12]] = slots[[12, 11]]  # the last Li and the first PF6: different types
    with pytest.raises(ValueError, match="another type"):
        Tracker(mol, TYPES, slots=slots)
    with pytest.raises(ValueError, match="not a permutation"):
        Tracker(mol, TYPES, slots=np.zeros(len(TYPES), dtype=np.int64))
    with pytest.raises(ValueError, match="differ in size"):
        Tracker(mol, ["Li"] * 13 + ["PF6"] * 11 + ["EC"] * 30)


def test_repair_mode_relabels_only_at_events() -> None:
    """Old files: frames as coarse as 2 ps, swaps in bursts (every 42nd frame in the pilot)."""
    rng = np.random.default_rng(6)
    mol, x = _system(rng)
    base = x.copy()
    tracker = Tracker(mol, TYPES)
    hidden = np.arange(len(mol))
    for frame in range(120):
        x = base + rng.normal(0.0, 0.04, (len(TYPES), 3))[mol]  # coarse frames: larger moves
        if frame % 42 == 5:
            hidden = hidden[_swap(rng, mol, 8)]
        assert np.array_equal(tracker.update(x[hidden], jump_nm=0.5), x)
    assert tracker.events == 3 and tracker.ambiguous == 0


def test_repair_mode_refuses_a_jump_without_partner() -> None:
    rng = np.random.default_rng(7)
    mol, x = _system(rng)
    tracker = Tracker(mol, TYPES)
    tracker.update(x, jump_nm=0.5)
    y = x.copy()
    y[mol == 30] += 2.0  # one EC jumps 2 nm: no relabelling explains it
    with pytest.raises(IdentityError, match="still moves"):
        tracker.update(y, jump_nm=0.5)


def test_repair_mode_counts_ambiguous_assignments() -> None:
    rng = np.random.default_rng(8)
    mol, x = _system(rng)
    a, b = np.flatnonzero(mol == 0), np.flatnonzero(mol == 1)  # two Li 0.02 nm apart
    x[b] = x[a] + [0.02, 0.0, 0.0]
    tracker = Tracker(mol, TYPES)
    tracker.update(x, jump_nm=0.5)
    y = x.copy()
    p = np.arange(len(mol))
    e1, e2 = np.flatnonzero(mol == 30), np.flatnonzero(mol == 31)
    p[e1], p[e2] = e2, e1  # far swap of two EC: an event
    y[a] += [0.011, 0.0, 0.0]
    y[b] = x[a] + [0.012, 0.0, 0.0]
    tracker.update(y[p], jump_nm=0.5)
    assert tracker.events == 1 and tracker.ambiguous >= 1


def test_min_cost_assignment_is_optimal() -> None:
    from itertools import permutations

    from ucmd_bridge.identity import min_cost_assignment

    rng = np.random.default_rng(9)
    for n in (1, 2, 3, 5, 7):
        for _ in range(20):
            cost = rng.uniform(0.0, 1.0, (n, n))
            if n > 2:
                cost[0] = cost[1]  # ties, as two molecules near one continuation
            match = min_cost_assignment(cost)
            assert sorted(match.tolist()) == list(range(n))
            best = min(sum(cost[i, q[i]] for i in range(n)) for q in permutations(range(n)))
            assert cost[np.arange(n), match].sum() == pytest.approx(best, rel=1e-12, abs=0)


def test_repair_mode_finds_the_only_consistent_assignment() -> None:
    """Pilot frame 967: two molecules have the same nearest continuation; the greedy choice
    leaves one of them without any. The least-cost assignment does not."""
    from ucmd_bridge.identity import _assign, min_cost_assignment

    d = np.array([[0.335, 0.869], [0.326, 0.45]])
    assert _assign(d)[0].tolist() == [1, 0]  # greedy: row 0 gets 0.869 > jump
    assert min_cost_assignment(d).tolist() == [0, 1]
