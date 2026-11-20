"""The repair tool end to end: an H5MD file with a hidden relabelling event is restored exactly,
the source stays unchanged, and a wrong package or an existing destination is refused."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from ucmd_bridge.package import load
from ucmd_bridge.repair import repair

ENGINE = Path(__file__).resolve().parents[2]
SPEC = Path(os.environ.get("UCMD_SPEC_DIR", ENGINE.parent / "spec"))
NACL = SPEC / "fixtures/valid/nacl-crystal"
ARGON = SPEC / "fixtures/valid/argon-864"
pytestmark = pytest.mark.skipif(not NACL.is_dir(), reason="needs the spec fixtures")


def _write(path: Path, sha: str, frames: np.ndarray) -> None:
    with h5py.File(path, "w") as fh:
        fh.create_group("parameters/ucmd").attrs["system_sha256"] = sha
        g = fh.create_group("particles/all/position")
        g["value"] = frames.astype(np.float32)
        g["step"] = np.arange(len(frames), dtype=np.int64) * 2000


def _trajectory(rng: np.random.Generator, x0: np.ndarray, mol_type: list[str]) -> tuple:
    """True frames and the frames as a faulty backend writes them: two events, each swapping
    whole sets of identical ions."""
    true = np.array([x0 + rng.normal(0.0, 0.01, x0.shape) for _ in range(12)])
    hidden = np.arange(len(x0))
    written = []
    for k, x in enumerate(true):
        if k in (4, 9):
            for name in ("Na+", "Cl-"):
                ids = np.flatnonzero(np.array(mol_type) == name)
                pick = rng.choice(ids, 10, replace=False)
                hidden[pick] = hidden[rng.permutation(pick)]
        written.append(x[hidden])
    return true, np.array(written)


def test_a_relabelled_file_is_restored(tmp_path: Path) -> None:
    top = load(NACL)
    rng = np.random.default_rng(11)
    true, written = _trajectory(rng, top.positions_nm, list(top.molecule_type))
    src, dst = tmp_path / "trajectory.h5", tmp_path / "repaired.h5"
    _write(src, top.system_sha256, written)
    before = src.read_bytes()
    record = repair(src, dst, NACL, jump_nm=0.1)
    assert src.read_bytes() == before
    with h5py.File(dst, "r") as fh:
        fixed = fh["particles/all/position/value"][()]
        stored = json.loads(fh["parameters/ucmd"].attrs["identity_repair"])
    assert np.array_equal(fixed, true.astype(np.float32))
    assert record == stored
    assert record["events"] == 2 and record["event_steps"] == [8000, 18000]
    assert record["ambiguous_assignments"] == 0


def test_a_wrong_package_or_an_existing_destination_is_refused(tmp_path: Path) -> None:
    top = load(NACL)
    src, dst = tmp_path / "trajectory.h5", tmp_path / "repaired.h5"
    _write(src, top.system_sha256, np.array([top.positions_nm] * 2))
    with pytest.raises(ValueError, match="not the trajectory's system"):
        repair(src, dst, ARGON)
    assert not dst.exists()
    dst.write_bytes(b"")
    with pytest.raises(FileExistsError):
        repair(src, dst, NACL)
