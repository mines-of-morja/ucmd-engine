# ucmd-engine

The molecular-dynamics engine of **UnitcellMD**. It propagates a system package with
interchangeable force providers: the integrator never knows where forces come from. An in-process
classical force field, an ML potential or a quantum-chemistry program behind a socket all
implement the same `ForceProvider` contract.

The plan, the shared spec and the agent rules live in the umbrella repository
[UnitcellMD](https://github.com/mines-of-morja/UnitcellMD).

## Status

**E1 (minimal engine) is on `main`. E2 core is on branch `e2`, awaiting the author's review.**

- velocity Verlet with thermostats as operators: CSVR, Nosé–Hoover chains, Langevin BAOAB;
- periodic triclinic cells;
- the `force-field` provider for parameterized packages:
  - bonds, angles, proper and improper dihedrals;
  - Lennard-Jones with exclusions and scaled 1-4 pairs;
  - Coulomb direct, Ewald or smooth PME;
- a FIRE minimizer (`ucmd minimize`);
- H5MD trajectories; exact checkpoints and restarts;
- threads that change only the speed: results are bitwise identical on any machine and for
  any thread count (umbrella ADR 0010);
- validated against OpenMM:
  - argon (`validation/argon/REPORT.md`);
  - the 1 M LiPF₆ in EC:EMC electrolyte box (`validation/m2a/REPORT.md`).

Constraints, the barostat and speed work follow in the rest of E2.

## Usage

```sh
cargo build --release
ucmd=target/release/ucmd
$ucmd check path/to/system.ucsys            # validate a package (same codes as ucmd-spec)
$ucmd check run.toml                         # validate a run configuration and its package
$ucmd singlepoint run.toml [--forces f.npy]  # energy, terms, virial of the initial state (JSON)
$ucmd minimize run.toml --out minimized.ucsys  # FIRE until max |F| < 100 kJ/mol/nm (--max-force)
$ucmd run run.toml                           # run; outputs in [output] of run.toml
$ucmd run run.toml --restart out/checkpoint.ucchk   # continue bitwise identically
$ucmd export run.toml --checkpoint out/checkpoint.ucchk --out equilibrated.ucsys
$ucmd rerun run.toml --trajectory out/trajectory.h5 --out energies.tsv
```

`run.toml` and the system package are described in the spec (`spec/run-config.md`,
`spec/system-package.md`). Examples: `examples/argon-quickstart/` and `examples/electrolyte/`
(a builder-made LiPF₆ box: minimize, then NVT).

The number of threads comes from the machine profile (`threads`, default all logical cores;
`UCMD_MACHINE_PROFILE` points to another profile).

## Crates

| Crate | Role |
|---|---|
| `ucmd-core` | units (CODATA 2018), periodic cell, immutable system, state |
| `ucmd-force` | `ForceProvider` contract, result guard, composite provider, observers, provider test kit |
| `ucmd-providers-classical` | harmonic, Lennard-Jones, neighbour lists, bonded terms, nonbonded with Ewald and PME, the `force-field` provider |
| `ucmd-providers-ipi` | out-of-process providers over the i-PI socket protocol (E3) |
| `ucmd-md` | velocity Verlet as operators, CSVR, Nosé–Hoover chains, Langevin BAOAB, FIRE, kinetics, Maxwell–Boltzmann velocities, run loop |
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
validation/m2a/run.sh                         # M2a electrolyte chain (needs the builder, ~30 min)
```

- Requires a stable Rust toolchain (rustup); edition 2024, MSRV 1.85.
- The default `h5md` feature builds a bundled static HDF5, which needs cmake and a C compiler
  (one-time build of about a minute).

## Units

nm, ps, amu (g/mol), kJ/mol, e and K, everywhere inside the engine and in its files. The
extended-XYZ debug output uses Å for viewers.

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
