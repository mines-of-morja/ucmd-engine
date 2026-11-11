# ucmd-engine — agent notes

The shared rules are in `../CLAUDE.md`, in the UnitcellMD umbrella repo one level up in the
workspace. The plan is `../PLAN.md`: track E (§5) and the crate layout (§2.7).

## Commands

```sh
. "$HOME/.cargo/env"                     # only if cargo is not on PATH
cargo build --workspace
cargo test --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
scripts/check-architecture.sh            # crate dependency rules
cargo test --workspace -- --ignored      # slow physics validation (from E1 on)
```

## Repo rules

- `scripts/check-architecture.sh` enforces the crate dependency rules. A new crate needs an
  entry there, or the check fails.
- Library crates: no `unwrap`/`expect` outside tests (enforced by clippy). Every public item
  is documented, including its units (enforced by `missing_docs`).
- Physics validation evidence goes in `validation/<case>/REPORT.md`.
