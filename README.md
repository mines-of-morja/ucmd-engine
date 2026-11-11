# ucmd-engine

The molecular-dynamics engine of **UnitcellMD**. It propagates a system package with
interchangeable force providers: the integrator never knows where forces come from.
An in-process classical force field, an ML potential or a quantum-chemistry program
behind a socket all implement the same `ForceProvider` contract.

**Status:** Phase 0 skeleton. Crate layout and checks only; no physics yet.
The plan, the shared spec and the agent rules live in the UnitcellMD umbrella repository.

## Crates

| Crate | Role |
|---|---|
| `ucmd-core` | units, vectors, periodic cell, immutable system, state |
| `ucmd-force` | `ForceProvider` contract, result guard, composite provider, provider test kit |
| `ucmd-providers-classical` | harmonic, Lennard-Jones, OPLS-AA terms, Ewald/PME, neighbour lists |
| `ucmd-providers-ipi` | out-of-process providers over the i-PI socket protocol |
| `ucmd-md` | integrators as operator splittings, thermostats, barostats, constraints, run loop |
| `ucmd-io` | system-package reader, run config, H5MD trajectories, checkpoints |
| `ucmd-cli` | the `ucmd` binary; the only crate that knows the concrete providers |

`ucmd-md` depends only on `ucmd-core` and `ucmd-force`.
`scripts/check-architecture.sh` enforces all dependency rules.

## Build and test

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
scripts/check-architecture.sh
```

Requires a stable Rust toolchain (rustup); edition 2024, MSRV 1.85.

## Units

nm, ps, amu, kJ/mol, e and K, everywhere inside the engine and in its files.

## License

MIT; see [LICENSE](LICENSE).
