//! The periodic cell in reduced lower-triangular form (spec/units.md §3).

use crate::{Mat3, Vec3};

/// Relative slack allowed on the reduced-form inequalities (floating-point noise only); the same
/// value as the reference validator.
pub const REDUCED_FORM_TOLERANCE: f64 = 1e-9;

/// Why a cell was rejected.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CellError {
    /// a_y, a_z or b_z is not zero.
    #[error("cell is not lower-triangular (a_y = a_z = b_z = 0 required)")]
    NotLowerTriangular,
    /// a_x, b_y or c_z is not positive.
    #[error("cell diagonal (a_x, b_y, c_z) must be positive")]
    NonPositiveDiagonal,
    /// A tilt factor exceeds half the corresponding edge.
    #[error("cell is not reduced (|b_x| <= a_x/2, |c_x| <= a_x/2, |c_y| <= b_y/2)")]
    NotReduced,
    /// No direction is periodic; such systems have no cell.
    #[error("a cell needs at least one periodic direction")]
    NoPeriodicDirection,
}

/// A periodic cell with lattice vectors a, b, c (nm), stored in reduced lower-triangular form.
///
/// `h` has the lattice vectors as **columns**, so Cartesian r = h·s for fractional s.
#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    h_inv: Mat3,
    periodic: [bool; 3],
}

impl Cell {
    /// Build a cell from its rows a, b, c (nm) and the periodic flags of the a, b, c directions.
    pub fn new(rows: [[f64; 3]; 3], periodic: [bool; 3]) -> Result<Self, CellError> {
        let [a, b, c] = rows.map(Vec3::from_array);
        if a.y != 0.0 || a.z != 0.0 || b.z != 0.0 {
            return Err(CellError::NotLowerTriangular);
        }
        if a.x <= 0.0 || b.y <= 0.0 || c.z <= 0.0 {
            return Err(CellError::NonPositiveDiagonal);
        }
        let slack = REDUCED_FORM_TOLERANCE * a.x.max(b.y).max(c.z);
        if b.x.abs() > a.x / 2.0 + slack
            || c.x.abs() > a.x / 2.0 + slack
            || c.y.abs() > b.y / 2.0 + slack
        {
            return Err(CellError::NotReduced);
        }
        if !periodic.iter().any(|&p| p) {
            return Err(CellError::NoPeriodicDirection);
        }
        let h_inv = Mat3::from_cols(a, b, c).inverse();
        Ok(Self {
            a,
            b,
            c,
            h_inv,
            periodic,
        })
    }

    /// Lattice vectors a, b, c in nm.
    pub fn rows(&self) -> [Vec3; 3] {
        [self.a, self.b, self.c]
    }

    /// The matrix h with the lattice vectors as columns (nm).
    pub fn h(&self) -> Mat3 {
        Mat3::from_cols(self.a, self.b, self.c)
    }

    /// Periodic flags of the a, b, c directions.
    pub fn periodic(&self) -> [bool; 3] {
        self.periodic
    }

    /// Cell volume in nm³.
    pub fn volume(&self) -> f64 {
        self.a.x * self.b.y * self.c.z
    }

    /// Perpendicular distances between opposite faces for the a, b, c directions (nm).
    pub fn heights(&self) -> [f64; 3] {
        let v = self.volume();
        [
            v / self.b.cross(self.c).length(),
            v / self.c.cross(self.a).length(),
            v / self.a.cross(self.b).length(),
        ]
    }

    /// Half the smallest perpendicular height over the periodic directions (nm): the largest
    /// distance for which [`Cell::minimum_image`] is exact.
    pub fn max_minimum_image_distance(&self) -> f64 {
        let heights = self.heights();
        (0..3)
            .filter(|&k| self.periodic[k])
            .map(|k| heights[k])
            .fold(f64::INFINITY, f64::min)
            / 2.0
    }

    /// Fractional coordinates s = h⁻¹·r.
    pub fn fractional(&self, r: Vec3) -> Vec3 {
        self.h_inv * r
    }

    /// The periodic image of the displacement `d` (nm) with the smallest length.
    ///
    /// Reduces along c, then b, then a. For a reduced cell this is exact whenever the true
    /// minimum image is shorter than [`Cell::max_minimum_image_distance`]: each step fixes one
    /// lattice coefficient, because the perpendicular height in that direction is at most the
    /// corresponding diagonal element.
    pub fn minimum_image(&self, d: Vec3) -> Vec3 {
        let mut d = d;
        if self.periodic[2] {
            d -= self.c * (d.z / self.c.z).round();
        }
        if self.periodic[1] {
            d -= self.b * (d.y / self.b.y).round();
        }
        if self.periodic[0] {
            d -= self.a * (d.x / self.a.x).round();
        }
        d
    }

    /// The image of `r` inside the primary cell (fractional coordinates in [0, 1) along periodic
    /// directions), and the integer lattice shift n such that `r = wrapped + h·n`.
    pub fn wrap(&self, r: Vec3) -> (Vec3, [i64; 3]) {
        let s = self.fractional(r);
        let mut shift = [0i64; 3];
        for k in 0..3 {
            if self.periodic[k] {
                shift[k] = s[k].floor() as i64;
            }
        }
        let n = Vec3::new(shift[0] as f64, shift[1] as f64, shift[2] as f64);
        (r - self.h() * n, shift)
    }

    /// The same cell with every lattice vector multiplied by `factor` (isotropic scaling).
    pub fn scaled(&self, factor: f64) -> Result<Self, CellError> {
        Self::new(self.rows().map(|v| (v * factor).to_array()), self.periodic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn cubic(l: f64) -> Cell {
        Cell::new([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]], [true; 3]).unwrap()
    }

    #[test]
    fn rejects_cells_outside_the_reduced_form() {
        let p = [true; 3];
        assert_eq!(
            Cell::new([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], p),
            Err(CellError::NotLowerTriangular)
        );
        assert_eq!(
            Cell::new([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], p),
            Err(CellError::NonPositiveDiagonal)
        );
        assert_eq!(
            Cell::new([[1.0, 0.0, 0.0], [0.6, 1.0, 0.0], [0.0, 0.0, 1.0]], p),
            Err(CellError::NotReduced)
        );
        assert_eq!(
            Cell::new(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [false; 3]
            ),
            Err(CellError::NoPeriodicDirection)
        );
    }

    #[test]
    fn volume_and_heights_of_a_cube() {
        let cell = cubic(2.0);
        assert_eq!(cell.volume(), 8.0);
        assert_eq!(cell.heights(), [2.0, 2.0, 2.0]);
        assert_eq!(cell.max_minimum_image_distance(), 1.0);
    }

    #[test]
    fn minimum_image_of_a_cube() {
        let cell = cubic(2.0);
        let d = cell.minimum_image(Vec3::new(1.9, -1.2, 0.3));
        assert!((d - Vec3::new(-0.1, 0.8, 0.3)).length() < 1e-15);
    }

    /// A random reduced triclinic cell with edges 1..3 nm and tilts within the reduced form.
    fn reduced_cell() -> impl Strategy<Value = Cell> {
        (
            1.0..3.0f64,
            1.0..3.0f64,
            1.0..3.0f64,
            -0.5..0.5f64,
            -0.5..0.5f64,
            -0.5..0.5f64,
        )
            .prop_map(|(ax, by, cz, tbx, tcx, tcy)| {
                Cell::new(
                    [
                        [ax, 0.0, 0.0],
                        [tbx * ax, by, 0.0],
                        [tcx * ax, tcy * by, cz],
                    ],
                    [true; 3],
                )
                .unwrap()
            })
    }

    proptest! {
        /// E1.1: whenever the shortest image is shorter than half the smallest height, the
        /// sequential reduction finds exactly that image (checked against a 5x5x5 image search).
        #[test]
        fn minimum_image_matches_brute_force_search(
            cell in reduced_cell(),
            x in -10.0..10.0f64, y in -10.0..10.0f64, z in -10.0..10.0f64,
        ) {
            let d = Vec3::new(x, y, z);
            let reduced = cell.minimum_image(d);
            let [a, b, c] = cell.rows();
            let mut best = reduced;
            for i in -2..=2 {
                for j in -2..=2 {
                    for k in -2..=2 {
                        let image = reduced + a * f64::from(i) + b * f64::from(j) + c * f64::from(k);
                        if image.length() < best.length() {
                            best = image;
                        }
                    }
                }
            }
            if best.length() < cell.max_minimum_image_distance() {
                prop_assert!((reduced - best).length() < 1e-9, "reduced {reduced:?} best {best:?}");
            }
            // the reduction never changes the vector by anything but a lattice vector
            let n = cell.fractional(d - reduced);
            prop_assert!((n - n.round()).abs().max_element() < 1e-9);
        }

        /// E1.1: wrapping lands in [0, 1) fractional and unwrapping restores r.
        #[test]
        fn wrap_unwrap_roundtrip(cell in reduced_cell(), x in -20.0..20.0f64, y in -20.0..20.0f64, z in -20.0..20.0f64) {
            let r = Vec3::new(x, y, z);
            let (wrapped, shift) = cell.wrap(r);
            let s = cell.fractional(wrapped);
            prop_assert!(s.min_element() > -1e-12 && s.max_element() < 1.0 + 1e-12);
            let back = wrapped + cell.h() * Vec3::new(shift[0] as f64, shift[1] as f64, shift[2] as f64);
            prop_assert!((back - r).length() < 1e-12 * (1.0 + r.length()));
        }
    }
}
