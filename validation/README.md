# Physics validation cases

One folder per case (for example `argon/`), each containing:

- `REPORT.md`: what was compared, the numbers, pass/fail against the acceptance
  criterion in PLAN.md, the exact commands, and the versions used (engine git hash,
  OpenMM version, …);
- the inputs and scripts needed to reproduce the case.

Large outputs such as trajectories are not committed; the report says how to
regenerate them. Cases run explicitly (`cargo test --workspace -- --ignored` or the
case's own script), not in normal CI.
