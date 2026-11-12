//! Chemical element symbols (for parsing `"element"` only; masses always come from the package,
//! CLAUDE.md invariant 5).

const SYMBOLS: [&str; 118] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
    "Fl", "Mc", "Lv", "Ts", "Og",
];

/// Atomic number of a chemical symbol, e.g. `"Ar"` → 18.
pub fn atomic_number(symbol: &str) -> Option<u8> {
    SYMBOLS
        .iter()
        .position(|s| *s == symbol)
        .and_then(|i| u8::try_from(i + 1).ok())
}

/// Chemical symbol of an atomic number (1…118).
pub fn symbol(atomic_number: u8) -> Option<&'static str> {
    SYMBOLS
        .get(usize::from(atomic_number).checked_sub(1)?)
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_complete_and_consistent() {
        assert_eq!(atomic_number("H"), Some(1));
        assert_eq!(atomic_number("Ar"), Some(18));
        assert_eq!(atomic_number("Ag"), Some(47));
        assert_eq!(atomic_number("Og"), Some(118));
        assert_eq!(atomic_number("Xx"), None);
        for z in 1..=118u8 {
            assert_eq!(symbol(z).and_then(atomic_number), Some(z));
        }
        assert_eq!(symbol(0), None);
    }
}
