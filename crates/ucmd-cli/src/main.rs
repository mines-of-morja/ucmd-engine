//! `ucmd`: the command-line entry point and composition root of the UnitcellMD engine.
//!
//! This is the only crate that knows the concrete force providers. It wires them
//! by name from `run.toml` (PLAN.md §2.7). The subcommands `run`, `check`,
//! `singlepoint` and `rerun` arrive in task E1.11.

fn main() {
    println!(
        "ucmd {} (skeleton build: no subcommands yet, see PLAN.md task E1.11)",
        env!("CARGO_PKG_VERSION")
    );
}
