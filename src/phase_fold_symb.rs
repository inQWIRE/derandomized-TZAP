//! Deterministic phase folding over exact symbolic parities.
//!
//! This is the symbolic variant of Alg. 1 (§ 2, § 5.2, and the ablation in
//! § 7.3 of the tzap paper): the abstract state still maps each qubit to a
//! Boolean function of Hadamard-introduced variables, but that function is
//! kept as an exact formula rather than as a random fingerprint. Everything
//! else — the per-gate transfer functions, the table of seen parities, and
//! the merging rule (`PhaseGroups`) — is shared with
//! [`PhaseFoldRand`](crate::phase_fold_rand::PhaseFoldRand) unchanged.
//!
//! The result never depends on a random draw: the same input always yields
//! the same output, and two rotations are merged exactly when their parities
//! are equal as Boolean functions of the tracked variables. The price is that
//! the pass is no longer linear time. Each XOR, hash, and equality check costs
//! time proportional to the weight of the formulas involved, and the CNOT
//! network between Hadamards mixes fresh variables into ever-heavier parities
//! (the paper measures 6×–25× slowdowns on its benchmarks, and quadratic
//! growth on the hwb-like family of § C).
//!
//! Formulas are stored in algebraic normal form, split into a constant bit,
//! a sorted list of linear variables, and a sorted list of nonlinear
//! monomials. CNOT-only circuits never populate the nonlinear list; CCX
//! multiplies its controls' formulas, reducing `x·x = x`.

use std::cmp::Ordering;

use crate::circuit::{Circuit, Gate};
use crate::pass::Pass;
use crate::phase_fold_rand::{PhaseGroups, rotation_of};

/// A Hadamard-introduced (or otherwise opaque) Boolean variable.
type Var = u32;

/// A CCX whose controls' formulas would multiply out to more than this many
/// monomials makes its target a fresh opaque variable instead. This mirrors
/// `PhaseFoldRand`'s degree cutoff: it is sound (later equalities are still
/// exact), it only forgets how the target relates to its old value, and it
/// keeps pathological Toffoli cascades from growing formulas exponentially.
const MAX_PRODUCT_TERMS: usize = 1 << 16;

/// The non-constant part of a Boolean formula in algebraic normal form: an
/// XOR of `linear` variables and `nonlinear` monomials. This is the key of
/// the phase table; the constant term is carried separately as the
/// complement flag, so `p` and `¬p` share one group exactly as they do in
/// `PhaseFoldRand`.
///
/// Invariants (the normal form that makes derived equality and hashing
/// agree with equality of Boolean functions):
/// - `linear` is strictly increasing;
/// - each monomial in `nonlinear` is strictly increasing with length ≥ 2;
/// - `nonlinear` is strictly increasing in lexicographic order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct Parity {
    linear: Vec<Var>,
    nonlinear: Vec<Box<[Var]>>,
}

impl Parity {
    fn var(v: Var) -> Self {
        Parity {
            linear: vec![v],
            nonlinear: Vec::new(),
        }
    }

    fn is_zero(&self) -> bool {
        self.linear.is_empty() && self.nonlinear.is_empty()
    }

    fn num_terms(&self) -> usize {
        self.linear.len() + self.nonlinear.len()
    }

    fn xor_assign(&mut self, other: &Parity) {
        if !other.linear.is_empty() {
            self.linear = symmetric_difference(&self.linear, &other.linear);
        }
        if !other.nonlinear.is_empty() {
            self.nonlinear = symmetric_difference(&self.nonlinear, &other.nonlinear);
        }
    }
}

/// XOR of two strictly increasing sequences viewed as sets.
fn symmetric_difference<T: Ord + Clone>(left: &[T], right: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            Ordering::Less => {
                out.push(left[i].clone());
                i += 1;
            }
            Ordering::Greater => {
                out.push(right[j].clone());
                j += 1;
            }
            Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&left[i..]);
    out.extend_from_slice(&right[j..]);
    out
}

/// Union of two strictly increasing variable lists (`x·x = x`).
fn monomial_product(left: &[Var], right: &[Var]) -> Vec<Var> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            Ordering::Less => {
                out.push(left[i]);
                i += 1;
            }
            Ordering::Greater => {
                out.push(right[j]);
                j += 1;
            }
            Ordering::Equal => {
                out.push(left[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&left[i..]);
    out.extend_from_slice(&right[j..]);
    out
}

/// A qubit's full computational-basis value: `negated ⊕ parity`.
#[derive(Clone, Debug, Default)]
struct Formula {
    parity: Parity,
    negated: bool,
}

impl Formula {
    fn var(v: Var) -> Self {
        Formula {
            parity: Parity::var(v),
            negated: false,
        }
    }

    fn xor_assign(&mut self, other: &Formula) {
        self.parity.xor_assign(&other.parity);
        self.negated ^= other.negated;
    }

    /// Every monomial, the constant `1` included as the empty monomial.
    fn monomials(&self) -> impl Iterator<Item = &[Var]> {
        let constant: &[Var] = &[];
        self.negated
            .then_some(constant)
            .into_iter()
            .chain(self.parity.linear.iter().map(std::slice::from_ref))
            .chain(self.parity.nonlinear.iter().map(|m| &m[..]))
    }

    fn num_monomials(&self) -> usize {
        self.parity.num_terms() + usize::from(self.negated)
    }

    /// The product `self · other`, or `None` if it would exceed
    /// [`MAX_PRODUCT_TERMS`] monomials before cancellation.
    fn product(&self, other: &Formula) -> Option<Formula> {
        let bound = self.num_monomials().checked_mul(other.num_monomials())?;
        if bound > MAX_PRODUCT_TERMS {
            return None;
        }
        let mut terms: Vec<Vec<Var>> = Vec::with_capacity(bound);
        for left in self.monomials() {
            for right in other.monomials() {
                terms.push(monomial_product(left, right));
            }
        }
        // Over GF(2), a monomial appearing an even number of times cancels.
        terms.sort_unstable();
        let mut product = Formula::default();
        let mut i = 0;
        while i < terms.len() {
            let mut j = i + 1;
            while j < terms.len() && terms[j] == terms[i] {
                j += 1;
            }
            if (j - i) % 2 == 1 {
                let term = std::mem::take(&mut terms[i]);
                match term.len() {
                    0 => product.negated = true,
                    1 => product.parity.linear.push(term[0]),
                    _ => product.parity.nonlinear.push(term.into_boxed_slice()),
                }
            }
            i = j;
        }
        // Sorting `Vec<Var>` lexicographically keeps each of the linear and
        // nonlinear sublists in the order the normal form requires.
        Some(product)
    }
}

/// Merges T/Rz gates that act on the same Boolean function, tracking each
/// qubit's value as an exact symbolic formula. Deterministic, and folds the
/// same rotations as [`PhaseFoldRand`](crate::phase_fold_rand::PhaseFoldRand)
/// does with high probability — but in superlinear time on circuits whose
/// parities grow heavy. See the module documentation.
pub struct PhaseFoldSymb;

impl Pass for PhaseFoldSymb {
    fn name(&self) -> &str {
        "Phase folding (symbolic)"
    }
    fn run(&self, circuit: &Circuit) -> Circuit {
        phase_fold_symb(circuit)
    }
}

/// Supplies fresh variables in a fixed order, so runs are reproducible.
struct Fresh(Var);

impl Fresh {
    fn var(&mut self) -> Formula {
        let v = self.0;
        self.0 = self
            .0
            .checked_add(1)
            .expect("symbolic phase folding ran out of variable identifiers");
        Formula::var(v)
    }
}

/// Free-function form of [`PhaseFoldSymb`], for use outside a [`Pass`] pipeline.
pub fn phase_fold_symb(circuit: &Circuit) -> Circuit {
    let mut fresh = Fresh(0);
    let mut qubits: Vec<Formula> = (0..circuit.num_qubits).map(|_| fresh.var()).collect();
    let mut groups: PhaseGroups<Parity> = PhaseGroups::new(circuit.gates.len());

    for (idx, gate) in circuit.gates.iter().enumerate() {
        if let Some((q, rotation)) = rotation_of(gate) {
            let value = &qubits[q as usize];
            let parity = (!value.parity.is_zero()).then_some((&value.parity, value.negated));
            groups.record(parity, q, rotation, idx);
            continue;
        }
        match gate {
            Gate::t(_) | Gate::tdg(_) | Gate::s(_) | Gate::sdg(_) | Gate::z(_) | Gate::rz(..) => {
                unreachable!("phase gates are recorded above")
            }
            Gate::h(q) => {
                qubits[*q as usize] = fresh.var();
            }
            Gate::x(q) => {
                qubits[*q as usize].negated ^= true;
            }
            Gate::cnot { control, target } => {
                // Take the control out and put it back, rather than borrowing
                // two elements of `qubits` at once.
                let (control, target) = (*control as usize, *target as usize);
                let source = std::mem::take(&mut qubits[control]);
                qubits[target].xor_assign(&source);
                qubits[control] = source;
            }
            Gate::cz { .. } | Gate::ccz { .. } => {
                // Diagonal: phase changes only, tracked values unchanged.
            }
            Gate::ccx {
                control1,
                control2,
                target,
            } => {
                let product =
                    qubits[*control1 as usize].product(&qubits[*control2 as usize]);
                match product {
                    Some(product) => qubits[*target as usize].xor_assign(&product),
                    // Too large to multiply out: forget the target's history
                    // and treat its new value as opaque (sound; see
                    // `MAX_PRODUCT_TERMS`).
                    None => qubits[*target as usize] = fresh.var(),
                }
            }
            Gate::measure { .. } => {
                // Computational-basis measurement preserves the basis value.
            }
            Gate::reset(q) => {
                qubits[*q as usize] = Formula::default();
            }
        }
    }

    groups.rebuild(circuit)
}

/// The hwb-like family used to isolate the cost of heavy parities (our
/// reconstruction of the § C family from its description in § 7.3).
///
/// Qubit 0 is an accumulator; qubits `1..=width` are workers. Stage `i`
/// puts a fresh Hadamard variable on worker `w = 1 + i % width`, rotates it,
/// and XORs it into the accumulator, so the accumulator's parity gains one
/// variable per stage and its weight grows linearly with circuit size. Every
/// fourth stage also rotates the accumulator, passes it through a CNOT pair
/// (which temporarily changes and then restores its parity), and applies the
/// inverse rotation — a fold both analyses must find.
#[cfg(test)]
pub(crate) fn hwb_like(stages: usize, width: usize) -> Circuit {
    assert!(width >= 2);
    let mut c = Circuit::new(width + 1);
    for i in 0..stages {
        let w = (1 + i % width) as crate::circuit::Qubit;
        c.apply(Gate::h(w));
        c.apply(Gate::t(w));
        c.apply(Gate::cnot {
            control: w,
            target: 0,
        });
        if i % 4 == 3 {
            let v = (1 + (i + 1) % width) as crate::circuit::Qubit;
            c.apply(Gate::t(0));
            c.apply(Gate::cnot {
                control: v,
                target: 0,
            });
            c.apply(Gate::cnot {
                control: v,
                target: 0,
            });
            c.apply(Gate::tdg(0));
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::cancel::CancelGates;
    use crate::circuit::Qubit;
    use crate::decompose::DecomposeToffoli;
    use crate::pass::count_t;
    use crate::phase_fold_rand::phase_fold_rand;
    use crate::unitary::circuits_equiv;

    fn formula(negated: bool, linear: &[Var], nonlinear: &[&[Var]]) -> Formula {
        Formula {
            parity: Parity {
                linear: linear.to_vec(),
                nonlinear: nonlinear.iter().map(|m| m.to_vec().into_boxed_slice()).collect(),
            },
            negated,
        }
    }

    fn assert_same_formula(left: &Formula, right: &Formula) {
        assert_eq!(left.parity, right.parity);
        assert_eq!(left.negated, right.negated);
    }

    #[test]
    fn xor_is_symmetric_difference_in_normal_form() {
        let mut a = formula(true, &[1, 3, 5], &[&[1, 2]]);
        let b = formula(true, &[2, 3], &[&[1, 2], &[2, 4]]);
        a.xor_assign(&b);
        assert_same_formula(&a, &formula(false, &[1, 2, 5], &[&[2, 4]]));
    }

    #[test]
    fn product_is_boolean_and_cancels_pairs() {
        // x·x = x.
        let x = formula(false, &[7], &[]);
        assert_same_formula(&x.product(&x).unwrap(), &x);
        // (a ⊕ b)(a ⊕ c) = a ⊕ ab ⊕ ac ⊕ bc.
        let ab = formula(false, &[0, 1], &[]);
        let ac = formula(false, &[0, 2], &[]);
        assert_same_formula(
            &ab.product(&ac).unwrap(),
            &formula(false, &[0], &[&[0, 1], &[0, 2], &[1, 2]]),
        );
        // (1 ⊕ a)(1 ⊕ a) = 1 ⊕ a, and (1 ⊕ a)·a = 0.
        let not_a = formula(true, &[0], &[]);
        let a = formula(false, &[0], &[]);
        assert_same_formula(&not_a.product(&not_a).unwrap(), &not_a);
        assert!(not_a.product(&a).unwrap().parity.is_zero());
        assert!(!not_a.product(&a).unwrap().negated);
    }

    #[test]
    fn oversized_products_are_refused() {
        let wide = Formula {
            parity: Parity {
                linear: (0..300).collect(),
                nonlinear: Vec::new(),
            },
            negated: false,
        };
        assert!(wide.product(&wide).is_none());
    }

    #[test]
    fn folds_basic_patterns() {
        // t; cx; t on the same parity across a CNOT pair → s.
        let mut c = Circuit::new(2);
        c.apply(Gate::h(0));
        c.apply(Gate::h(1));
        c.apply(Gate::t(1));
        c.apply(Gate::cnot { control: 0, target: 1 });
        c.apply(Gate::cnot { control: 0, target: 1 });
        c.apply(Gate::t(1));
        let out = phase_fold_symb(&c);
        assert!(circuits_equiv(&c, &out, 1e-10));
        assert_eq!(count_t(&out), 0);
        assert!(out.gates.contains(&Gate::s(1)));

        // X complements: t; x; t; x → identity on the rotations.
        let mut c = Circuit::new(1);
        c.apply(Gate::h(0));
        c.apply(Gate::t(0));
        c.apply(Gate::x(0));
        c.apply(Gate::t(0));
        c.apply(Gate::x(0));
        let out = phase_fold_symb(&c);
        assert!(circuits_equiv(&c, &out, 1e-10));
        assert_eq!(count_t(&out), 0);

        // H is a boundary.
        let mut c = Circuit::new(1);
        c.apply(Gate::t(0));
        c.apply(Gate::h(0));
        c.apply(Gate::t(0));
        let out = phase_fold_symb(&c);
        assert_eq!(count_t(&out), 2);

        // A rotation on a reset (known |0⟩) qubit disappears.
        let mut c = Circuit::new(1);
        c.apply(Gate::reset(0));
        c.apply(Gate::t(0));
        assert_eq!(phase_fold_symb(&c).gates, vec![Gate::reset(0)]);
    }

    #[test]
    fn nonlinear_ccx_pair_restores_parity() {
        let mut circuit = Circuit::new(3);
        circuit.apply(Gate::t(2));
        circuit.apply(Gate::ccx {
            control1: 0,
            control2: 1,
            target: 2,
        });
        circuit.apply(Gate::ccx {
            control1: 1,
            control2: 0,
            target: 2,
        });
        circuit.apply(Gate::tdg(2));
        let folded = phase_fold_symb(&circuit);
        assert_eq!(count_t(&folded), 0);
        assert!(circuits_equiv(&circuit, &folded, 1e-10));
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    fn random_circuit(rng: &mut Rng, num_qubits: usize, num_gates: usize, ccx: bool) -> Circuit {
        let mut c = Circuit::new(num_qubits);
        for _ in 0..num_gates {
            let q = rng.below(num_qubits) as Qubit;
            let other = (q + 1 + rng.below(num_qubits - 1) as Qubit) % num_qubits as Qubit;
            match rng.below(if ccx { 14 } else { 13 }) {
                0..=2 => c.apply(Gate::cnot {
                    control: q,
                    target: other,
                }),
                3 => c.apply(Gate::h(q)),
                4 => c.apply(Gate::x(q)),
                5 | 6 => c.apply(Gate::t(q)),
                7 => c.apply(Gate::tdg(q)),
                8 => c.apply(Gate::s(q)),
                9 => c.apply(Gate::z(q)),
                10 => c.apply(Gate::rz(0.1 * rng.below(40) as f64, q)),
                11 => c.apply(Gate::cz {
                    control: q,
                    target: other,
                }),
                12 => c.apply(Gate::sdg(q)),
                _ => {
                    let third = (0..num_qubits as Qubit)
                        .find(|&r| r != q && r != other)
                        .expect("ccx needs three qubits");
                    c.apply(Gate::ccx {
                        control1: q,
                        control2: other,
                        target: third,
                    })
                }
            }
        }
        c
    }

    /// Unitary-checked against the input, and — on CNOT-dihedral circuits,
    /// where both analyses are exact up to fingerprint collisions — the same
    /// T-count as the randomized pass.
    #[test]
    fn matches_randomized_pass_on_random_circuits() {
        let mut rng = Rng(0x5eed_5eed_5eed_5eed);
        for case in 0..400 {
            let num_qubits = 2 + rng.below(4);
            let num_gates = 10 + rng.below(120);
            let circuit = random_circuit(&mut rng, num_qubits, num_gates, false);
            let symb = phase_fold_symb(&circuit);
            let rand = phase_fold_rand(&circuit);
            assert!(
                circuits_equiv(&circuit, &symb, 1e-9),
                "case {case}: symbolic output not equivalent\n{circuit}"
            );
            assert_eq!(
                symb.gates, rand.gates,
                "case {case}: symbolic and randomized passes disagree\n{circuit}"
            );
        }
    }

    #[test]
    fn sound_on_random_circuits_with_toffolis() {
        let mut rng = Rng(0x7011_0f11_7011_0f11);
        for case in 0..300 {
            let num_qubits = 3 + rng.below(3);
            let num_gates = 10 + rng.below(80);
            let circuit = random_circuit(&mut rng, num_qubits, num_gates, true);
            let symb = phase_fold_symb(&circuit);
            assert!(
                circuits_equiv(&circuit, &symb, 1e-9),
                "case {case}: symbolic output not equivalent\n{circuit}"
            );
            assert!(count_t(&symb) <= count_t(&circuit));
        }
    }

    #[test]
    fn is_deterministic() {
        let mut rng = Rng(42);
        let circuit = random_circuit(&mut rng, 5, 2_000, true);
        let first = phase_fold_symb(&circuit);
        for _ in 0..3 {
            assert_eq!(phase_fold_symb(&circuit).gates, first.gates);
        }
    }

    fn load(name: &str) -> Circuit {
        let path = format!("{}/benchmarks/feynman/{name}.qasm", env!("CARGO_MANIFEST_DIR"));
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        crate::qasm::parse(&src).unwrap_or_else(|e| panic!("parse {name}: {e}"))
    }

    /// The paper reports T-count reductions matching tzap exactly.
    #[test]
    fn matches_randomized_t_count_on_benchmarks() {
        for name in [
            "tof_3",
            "tof_5",
            "barenco_tof_4",
            "mod5_4",
            "vbe_adder_3",
            "grover_5",
            "adder_8",
            "gf2^8_mult",
            "hwb6",
        ] {
            let input = CancelGates.run(&DecomposeToffoli.run(&load(name)));
            let symb = phase_fold_symb(&input);
            let rand = phase_fold_rand(&input);
            assert_eq!(count_t(&symb), count_t(&rand), "{name}: T-counts differ");
            assert_eq!(symb.gates.len(), rand.gates.len(), "{name}: gate counts differ");
        }
    }

    #[test]
    fn hwb_like_family_folds_like_randomized_pass() {
        for stages in [1, 4, 17, 64] {
            let circuit = hwb_like(stages, 3);
            let symb = phase_fold_symb(&circuit);
            assert!(circuits_equiv(&circuit, &symb, 1e-9), "{stages} stages");
            assert_eq!(symb.gates, phase_fold_rand(&circuit).gates, "{stages} stages");
            assert!(count_t(&symb) < count_t(&circuit) || stages < 4);
        }
    }

    /// Fig. 6: symbolic runtime grows quadratically on the hwb-like family,
    /// randomized runtime linearly. Run with
    /// `cargo test --release phase_fold_symb::tests::hwb_like_scaling -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn hwb_like_scaling() {
        println!("\n{:>8} {:>12} {:>12} {:>9}", "gates", "symb (s)", "rand (s)", "speedup");
        // Sizes stop at 32k stages: every accumulator rotation stores its
        // (linear-weight) parity in the phase table, so memory is quadratic too.
        for stages in [2_000, 4_000, 8_000, 16_000, 32_000] {
            let circuit = hwb_like(stages, 12);
            let start = Instant::now();
            let symb = phase_fold_symb(&circuit);
            let symb_time = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let rand = phase_fold_rand(&circuit);
            let rand_time = start.elapsed().as_secs_f64();
            assert_eq!(count_t(&symb), count_t(&rand));
            println!(
                "{:>8} {:>12.4} {:>12.4} {:>8.0}×",
                circuit.gates.len(),
                symb_time,
                rand_time,
                symb_time / rand_time
            );
        }
    }
}
