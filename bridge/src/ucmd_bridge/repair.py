"""Repair the molecule labels of an H5MD trajectory written before the identity fix.

    python -m ucmd_bridge.repair IN.h5 OUT.h5 --package SYSTEM.ucsys

- Writes a corrected copy; IN is never changed.
- Frames are matched with `identity.Tracker` in repair mode:
  - labels are kept unless some molecule moved more than `--jump-nm` since the last frame
    (atom-wise RMSD), a relabelling event;
  - at an event, every type is assigned anew, closest pairs first;
  - after the event no molecule may still move more than `--jump-nm`, else the repair stops
    and the incomplete copy is removed.
- Old files have coarse frames (2 ps in the pilot). For EC and EMC a neighbour can then be
  about as close as the molecule's own new position, so a few assignments at events can be
  ambiguous. They are counted, not hidden.
- `--jump-nm` must lie above every real move between frames. In the pilot no molecule moved
  more than 0.6 nm in 2 ps outside the events.
- `--skip-ps`: frames before this time are copied unchanged and tracking starts at the first
  frame after it. For runs whose first picoseconds are violent (a too dense start under NPT
  moved molecules by up to 1.1 nm in the first 2 ps, A4 2026-11-20); the analysis skips them
  anyway. The labels then follow the first tracked frame. Identical molecules are
  interchangeable, so a relabelling before it changes nothing in the analysed window.
- The package must be the one the run used: its system hash is compared with the trajectory's
  provenance.
- The copy records what was done in `parameters/ucmd/identity_repair`: tool and version, the
  source file's name, molecules relabelled, frames with relabelling, the worst ratio accepted.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

from . import __version__
from .identity import DEFAULT_RATIO, IdentityError, Tracker
from .package import load

DEFAULT_JUMP_NM = 0.7  # nm: above every move of a molecule between the pilot's 2 ps frames


def repair(src: Path, dst: Path, package: Path, *, ratio: float = DEFAULT_RATIO,
           jump_nm: float = DEFAULT_JUMP_NM, skip_ps: float = 0.0) -> dict:  # fmt: skip
    top = load(package)
    with h5py.File(src, "r") as fh:
        sha = fh["parameters/ucmd"].attrs.get("system_sha256")
    if sha != top.system_sha256:
        raise ValueError(f"{package} is not the trajectory's system (hash {sha})")
    if dst.exists():
        raise FileExistsError(dst)
    shutil.copyfile(src, dst)
    try:
        tracker = Tracker(top.molecule_index, list(top.molecule_type), ratio=ratio)
        frames = []
        with h5py.File(dst, "r+") as fh:
            pos = fh["particles/all/position/value"]
            steps = fh["particles/all/position/step"][()]
            times = fh["particles/all/position/time"][()]
            first = int(np.searchsorted(times, skip_ps - 1e-9))
            for k in range(first, pos.shape[0]):
                events = tracker.events
                fixed = tracker.update(pos[k].astype(np.float64), jump_nm=jump_nm)
                if tracker.events != events:
                    frames.append(int(steps[k]))
                if not np.array_equal(tracker.slot, np.arange(len(tracker.slot))):
                    pos[k] = fixed.astype(pos.dtype)
            record = {
                "tool": f"ucmd_bridge.repair {__version__}",
                "source": src.name,
                "molecules_relabelled": tracker.exchanges,
                "events": tracker.events,
                "event_steps": frames,
                "ambiguous_assignments": tracker.ambiguous,
                "ratio_limit": ratio,
                "worst_unambiguous_ratio": tracker.worst_ratio,
                "jump_nm": jump_nm,
                "skip_ps": skip_ps,
                "first_tracked_step": int(steps[first]) if first < len(steps) else None,
                "frames": int(pos.shape[0]),
            }
            fh["parameters/ucmd"].attrs["identity_repair"] = json.dumps(record)
    except BaseException:
        dst.unlink(missing_ok=True)
        raise
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ucmd_bridge.repair", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("source", type=Path)
    ap.add_argument("destination", type=Path)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--ratio", type=float, default=DEFAULT_RATIO)
    ap.add_argument("--jump-nm", type=float, default=DEFAULT_JUMP_NM)
    ap.add_argument("--skip-ps", type=float, default=0.0)
    args = ap.parse_args(argv)
    try:
        record = repair(args.source, args.destination, args.package, ratio=args.ratio,
                        jump_nm=args.jump_nm, skip_ps=args.skip_ps)  # fmt: skip
    except (IdentityError, ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
