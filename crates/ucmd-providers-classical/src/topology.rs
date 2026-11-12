//! Pair topology from the bond graph: exclusions and 1-4 pairs (spec/system-package.md §5.2).
//!
//! Within a molecule, let d be the shortest path length in the bond graph. Pairs with d = 1 or
//! 2 are excluded, pairs with d = 3 are 1-4 pairs, all other pairs interact normally. Rings
//! count by their shortest path. The lists are built once per molecule type and expanded to
//! global (canonical) atom indices.

use std::collections::VecDeque;
use ucmd_core::{MoleculeType, System};

/// Exclusions and 1-4 pairs of a whole system, in global atom indices.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PairTopology {
    /// CSR offsets into `skip`: atom i skips `skip[skip_start[i]..skip_start[i + 1]]`.
    skip_start: Vec<u32>,
    /// Sorted partners with bond distance 1, 2 or 3 (both directions).
    skip: Vec<u32>,
    /// Pairs (i < j) with bond distance 1 or 2.
    pub excluded: Vec<[u32; 2]>,
    /// Pairs (i < j) with bond distance 3.
    pub pairs14: Vec<[u32; 2]>,
}

/// Bond distances up to 3 within one molecule type: (local i, local j, d) with i < j.
fn local_pairs(mtype: &MoleculeType) -> Vec<(u32, u32, u8)> {
    let n = mtype.atoms.len();
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); n];
    for bond in &mtype.bonds {
        let [a, b] = bond.atoms;
        neighbours[a].push(b);
        neighbours[b].push(a);
    }
    let mut out = Vec::new();
    let mut depth = vec![u8::MAX; n];
    let mut queue = VecDeque::new();
    for start in 0..n {
        depth.iter_mut().for_each(|d| *d = u8::MAX);
        depth[start] = 0;
        queue.clear();
        queue.push_back(start);
        while let Some(a) = queue.pop_front() {
            if depth[a] == 3 {
                continue;
            }
            for &b in &neighbours[a] {
                if depth[b] == u8::MAX {
                    depth[b] = depth[a] + 1;
                    queue.push_back(b);
                }
            }
        }
        for (j, &d) in depth.iter().enumerate().skip(start + 1) {
            if (1..=3).contains(&d) {
                out.push((start as u32, j as u32, d));
            }
        }
    }
    out
}

impl PairTopology {
    /// Build the topology of `system` (molecule types × instances).
    pub fn new(system: &System) -> Self {
        let n = system.n_atoms();
        let per_type: Vec<Vec<(u32, u32, u8)>> =
            system.molecule_types.iter().map(local_pairs).collect();
        let mut excluded = Vec::new();
        let mut pairs14 = Vec::new();
        let mut partners: Vec<Vec<u32>> = vec![Vec::new(); n];
        for instance in &system.instances {
            let offset = instance.first_atom as u32;
            for &(i, j, d) in &per_type[instance.molecule_type] {
                let (gi, gj) = (offset + i, offset + j);
                if d <= 2 {
                    excluded.push([gi, gj]);
                } else {
                    pairs14.push([gi, gj]);
                }
                partners[gi as usize].push(gj);
                partners[gj as usize].push(gi);
            }
        }
        let mut skip_start = Vec::with_capacity(n + 1);
        let mut skip = Vec::new();
        skip_start.push(0);
        for mut list in partners {
            list.sort_unstable();
            skip.extend_from_slice(&list);
            skip_start.push(skip.len() as u32);
        }
        Self {
            skip_start,
            skip,
            excluded,
            pairs14,
        }
    }

    /// True if the normal pair loop must skip (i, j): same molecule, bond distance ≤ 3.
    pub fn skips(&self, i: usize, j: usize) -> bool {
        let (start, end) = (self.skip_start[i] as usize, self.skip_start[i + 1] as usize);
        self.skip[start..end].binary_search(&(j as u32)).is_ok()
    }

    /// Partners of atom `i` that the normal pair loop skips (sorted).
    pub fn skipped(&self, i: usize) -> &[u32] {
        &self.skip[self.skip_start[i] as usize..self.skip_start[i + 1] as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{chain_molecule, ring_molecule, system_of};

    #[test]
    fn chain_distances() {
        // 0-1-2-3-4: excluded (0,1)(0,2)(1,2)(1,3)(2,3)(2,4)(3,4); 1-4 (0,3)(1,4)
        let system = system_of(vec![chain_molecule(5)], &[(0, 1)], [false; 3]);
        let t = PairTopology::new(&system);
        let mut ex = t.excluded.clone();
        ex.sort();
        assert_eq!(
            ex,
            vec![[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4]]
        );
        assert_eq!(t.pairs14, vec![[0, 3], [1, 4]]);
        assert!(t.skips(0, 3) && t.skips(3, 0) && !t.skips(0, 4));
    }

    #[test]
    fn five_ring_counts_by_shortest_path() {
        // In a 5-ring every pair is at distance 1 or 2: no 1-4 pairs.
        let system = system_of(vec![ring_molecule(5)], &[(0, 1)], [false; 3]);
        let t = PairTopology::new(&system);
        assert_eq!(t.excluded.len(), 10);
        assert!(t.pairs14.is_empty());
        // In a 6-ring the para pairs are 1-4 pairs.
        let system = system_of(vec![ring_molecule(6)], &[(0, 1)], [false; 3]);
        let t = PairTopology::new(&system);
        assert_eq!(t.pairs14, vec![[0, 3], [1, 4], [2, 5]]);
    }

    #[test]
    fn instances_are_offset_and_independent() {
        let system = system_of(vec![chain_molecule(4)], &[(0, 3)], [false; 3]);
        let t = PairTopology::new(&system);
        assert_eq!(t.pairs14, vec![[0, 3], [4, 7], [8, 11]]);
        assert!(
            !t.skips(3, 4),
            "atoms of different molecules always interact"
        );
        assert_eq!(t.skipped(5), &[4, 6, 7]);
    }
}
