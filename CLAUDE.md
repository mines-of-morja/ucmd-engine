# ucmd-engine — agent notes

The shared rules are in `../CLAUDE.md`, in the UnitcellMD umbrella repo one level up in the
workspace. The plan is `../PLAN.md` (track E §5, crate layout §2.7); the E1 design is
`../docs/design/E1.md`.

## Commands

```sh
. "$HOME/.cargo/env"                                  # only if cargo is not on PATH
cargo build --workspace
UCMD_REQUIRE_SPEC=1 cargo test --workspace            # fails if ../spec fixtures are missing
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
scripts/check-architecture.sh                         # crate dependency rules
cargo build --workspace --no-default-features         # HDF5-free build must stay clean
validation/argon/run.sh                               # E1.12 physics validation vs OpenMM (~5 min)
```

When chaining checks in a shell, use `set -o pipefail`: a check piped into `head` otherwise
hides its exit status.

## Repo rules

- `scripts/check-architecture.sh` enforces the crate dependency rules. A new crate needs an
  entry there, or the check fails.
- Library crates: no `unwrap`/`expect` outside tests (enforced by clippy). Every public item is
  documented, including its units (enforced by `missing_docs`). Integration-test files start
  with `#![allow(clippy::unwrap_used, clippy::expect_used)]`.
- Range checks on floats must reject NaN (TOML allows `nan`). Use positive forms (`x > 0.0`) or
  the helpers in `ucmd-io::error`, not `!(x > 0.0)`.
- Every provider passes `ucmd_force::testkit` (finite-difference forces and strain virial,
  momentum, determinism) in its own tests.
- New spec behaviour needs a matching fixture in `../spec/fixtures`. The contract test
  (`crates/ucmd-io/tests/contract.rs`) must keep agreeing with the Python validator.
- Physics validation evidence goes in `validation/<case>/REPORT.md`. Outputs stay out of git.
