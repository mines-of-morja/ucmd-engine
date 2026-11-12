# ucmd-engine

The molecular-dynamics engine of **UnitcellMD**. It propagates a system package with
interchangeable force providers: the integrator never knows where forces come from. An in-process
classical force field, an ML potential or a quantum-chemistry program behind a socket all
implement the same `ForceProvider` contract.

The plan, the shared spec and the agent rules live in the umbrella repository
[UnitcellMD](https://github.com/mines-of-morja/UnitcellMD).

## Status

**Phase E1 (minimal engine), branch `e1`, provisional until the author's review.**

- velocity Verlet with the CSVR thermostat;
- periodic triclinic cells;
- Lennard-Jones with a neighbour list;
- H5MD trajectories;
- exact checkpoints and restarts;
- validated against OpenMM on argon: `validation/argon/REPORT.md`.

Molecular force fields (bonded terms, PME, constraints) follow in E2.

## Usage

```sh
cargo build --release
ucmd=target/release/ucmd
$ucmd check path/to/system.ucsys            # validate a package (same codes as ucmd-spec)
$ucmd check run.toml                         # validate a run configuration and its package
$ucmd singlepoint run.toml [--forces f.npy]  # energy, terms, virial of the initial state (JSON)
$ucmd run run.toml                           # run; outputs in [output] of run.toml
$ucmd run run.toml --restart out/checkpoint.ucchk   # continue bitwise identically
$ucmd export run.toml --checkpoint out/checkpoint.ucchk --out equilibrated.ucsys
$ucmd rerun run.toml --trajectory out/trajectory.h5 --out energies.tsv
```

`run.toml` and the system package are described in the spec (`spec/run-config.md`,
`spec/system-package.md`). Example: `spec/fixtures/runs/argon-nvt.toml`.

## Crates

| Crate | Role |
|---|---|
| `ucmd-core` | units (CODATA 2018), periodic cell, immutable system, state |
| `ucmd-force` | `ForceProvider` contract, result guard, composite provider, observers, provider test kit |
| `ucmd-providers-classical` | harmonic, Lennard-Jones, neighbour lists (bonded OPLS-AA terms and PME in E2) |
| `ucmd-providers-ipi` | out-of-process providers over the i-PI socket protocol (E3) |
| `ucmd-md` | velocity Verlet as operators, CSVR, kinetics, Maxwell–Boltzmann velocities, run loop |
| `ucmd-io` | package and run-config readers, NPY, thermo/XYZ/H5MD writers, H5MD reader, checkpoints |
| `ucmd-cli` | the `ucmd` binary; the only crate that knows the concrete providers |

`ucmd-md` depends only on `ucmd-core` and `ucmd-force`. `scripts/check-architecture.sh`
enforces all dependency rules.

## Build and test

```sh
cargo build --workspace
UCMD_REQUIRE_SPEC=1 cargo test --workspace    # contract tests need ../spec (workspace layout)
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
scripts/check-architecture.sh
cargo build --workspace --no-default-features # without HDF5: XYZ trajectories only
validation/argon/run.sh                       # E1.12 against OpenMM (needs tools/oracle)
```

- Requires a stable Rust toolchain (rustup); edition 2024, MSRV 1.85.
- The default `h5md` feature builds a bundled static HDF5, which needs cmake and a C compiler
  (one-time build of about a minute).

## Units

nm, ps, amu (g/mol), kJ/mol, e and K, everywhere inside the engine and in its files. The
extended-XYZ debug output uses Å for viewers.

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
