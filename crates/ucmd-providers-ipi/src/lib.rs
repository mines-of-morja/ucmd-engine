//! Out-of-process force providers over the i-PI socket protocol.
//!
//! Planned for track E3 (task E3.1): the engine acts as i-PI server, so any i-PI
//! client (for example an ASE calculator run by `ucmd-bridge`, on this machine or
//! on another node over TCP) becomes a force provider. Positions and forces travel
//! in atomic units and are converted at this boundary.
