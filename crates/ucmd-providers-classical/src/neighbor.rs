//! Neighbour search: a cell list on fractional coordinates builds a half Verlet list with skin
//! (PLAN.md E1.8). Positions are unwrapped, so the rebuild test needs no minimum image.
//!
//! The rows are built in parallel and each is sorted, so the list does not depend on the number
//! of threads (ADR 0010).

use rayon::prelude::*;
use ucmd_core::{Cell, Vec3};

/// A half neighbour list (i < j) of all pairs within `cutoff + skin` at build time, stored in
/// compressed form: the neighbours of atom i are `neighbors[offsets[i]..offsets[i + 1]]`.
#[derive(Debug, Clone, Default)]
pub struct NeighborList {
    cutoff_nm: f64,
    skin_nm: f64,
    offsets: Vec<usize>,
    neighbors: Vec<u32>,
    positions_at_build: Vec<Vec3>,
    cell_at_build: Option<Cell>,
    builds: u64,
}

impl NeighborList {
    /// A list for pairs within `cutoff_nm`, rebuilt when an atom moved more than `skin_nm / 2`.
    pub fn new(cutoff_nm: f64, skin_nm: f64) -> Self {
        Self {
            cutoff_nm,
            skin_nm,
            ..Self::default()
        }
    }

    /// Number of builds so far (for tests and statistics).
    pub fn builds(&self) -> u64 {
        self.builds
    }

    /// Neighbours of atom `i` with j > i.
    pub fn neighbors(&self, i: usize) -> &[u32] {
        &self.neighbors[self.offsets[i]..self.offsets[i + 1]]
    }

    /// Number of pairs in the rows before atom `i` (`i` may be the atom count: all pairs).
    pub fn pairs_before(&self, i: usize) -> usize {
        self.offsets[i]
    }

    /// Rebuild if needed; returns true if it rebuilt.
    pub fn update(&mut self, positions: &[Vec3], cell: Option<&Cell>) -> bool {
        let stale = self.positions_at_build.len() != positions.len()
            || self.cell_at_build.as_ref() != cell
            || positions
                .iter()
                .zip(&self.positions_at_build)
                .any(|(r, r0)| (*r - *r0).length_squared() > 0.25 * self.skin_nm * self.skin_nm);
        if stale {
            self.build(positions, cell);
        }
        stale
    }

    fn build(&mut self, positions: &[Vec3], cell: Option<&Cell>) {
        let n = positions.len();
        let range = self.cutoff_nm + self.skin_nm;
        let range2 = range * range;
        let grid = cell.and_then(|c| CellGrid::new(c, range, positions));
        let lists: Vec<Vec<u32>> = match (&grid, cell) {
            (Some(grid), Some(cell)) => (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut list = Vec::new();
                    for neighbor_cell in grid.neighbor_cells(grid.cell_of[i]) {
                        for &j in &grid.members[neighbor_cell] {
                            if j as usize > i
                                && cell
                                    .minimum_image(positions[i] - positions[j as usize])
                                    .length_squared()
                                    < range2
                            {
                                list.push(j);
                            }
                        }
                    }
                    list.sort_unstable(); // the same order for any cell traversal
                    list
                })
                .collect(),
            // small cells (fewer than 3 grid cells along a direction) or no cell: all pairs
            _ => (0..n)
                .into_par_iter()
                .map(|i| {
                    (i + 1..n)
                        .filter(|&j| {
                            let mut d = positions[i] - positions[j];
                            if let Some(cell) = cell {
                                d = cell.minimum_image(d);
                            }
                            d.length_squared() < range2
                        })
                        .map(|j| j as u32)
                        .collect()
                })
                .collect(),
        };
        self.offsets.clear();
        self.neighbors.clear();
        self.offsets.push(0);
        for list in lists {
            self.neighbors.extend_from_slice(&list);
            self.offsets.push(self.neighbors.len());
        }
        self.positions_at_build = positions.to_vec();
        self.cell_at_build = cell.cloned();
        self.builds += 1;
    }
}

/// Binning of wrapped fractional coordinates into a grid whose cells are at least `range` wide
/// perpendicular to their faces, so all neighbours lie in the 27 surrounding cells.
struct CellGrid {
    dims: [usize; 3],
    cell_of: Vec<usize>,
    members: Vec<Vec<u32>>,
}

impl CellGrid {
    fn new(cell: &Cell, range: f64, positions: &[Vec3]) -> Option<Self> {
        let heights = cell.heights();
        let mut dims = [1usize; 3];
        for k in 0..3 {
            if cell.periodic()[k] {
                dims[k] = (heights[k] / range).floor() as usize;
            }
            if dims[k] < 3 {
                return None;
            }
        }
        let n_cells = dims[0] * dims[1] * dims[2];
        let mut members = vec![Vec::new(); n_cells];
        let mut cell_of = Vec::with_capacity(positions.len());
        for (i, &r) in positions.iter().enumerate() {
            let (wrapped, _) = cell.wrap(r);
            let s = cell.fractional(wrapped);
            let idx: [usize; 3] =
                std::array::from_fn(|k| ((s[k] * dims[k] as f64) as usize).min(dims[k] - 1));
            let flat = (idx[0] * dims[1] + idx[1]) * dims[2] + idx[2];
            members[flat].push(i as u32);
            cell_of.push(flat);
        }
        Some(Self {
            dims,
            cell_of,
            members,
        })
    }

    fn neighbor_cells(&self, flat: usize) -> impl Iterator<Item = usize> + '_ {
        let [nx, ny, nz] = self.dims;
        let (x, y, z) = (flat / (ny * nz), (flat / nz) % ny, flat % nz);
        (0..27).map(move |k| {
            let (dx, dy, dz) = (k / 9, (k / 3) % 3, k % 3);
            let wrap = |c: usize, d: usize, n: usize| (c + n + d - 1) % n;
            (wrap(x, dx, nx) * ny + wrap(y, dy, ny)) * nz + wrap(z, dz, nz)
        })
    }
}
