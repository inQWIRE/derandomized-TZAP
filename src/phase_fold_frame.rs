//! Exact, deterministic phase folding with fixed-width parity vectors.
//!
//! # Idea
//!
//! Without CCX or reset, the qubits' current values always span exactly `n`
//! dimensions: CNOT is invertible, and a Hadamard replaces one direction by a
//! new one. A function that leaves that span can never be produced again, so
//! a phase-table key outside it is dead and may be dropped without losing a
//! fold. That makes `n` coordinates enough, forever.
//!
//! Each qubit `q` keeps
//!
//! - `R[q]`: its value in a fixed `n`-bit coordinate system, and
//! - `D[q]`: a dual vector with `D[q]·R[j] = [q = j]`.
//!
//! | gate      | update                                                        |
//! |-----------|---------------------------------------------------------------|
//! | `cx c→t`  | `R[t] ^= R[c]`, `D[c] ^= D[t]`                                |
//! | `x q`     | flip `negated[q]`                                             |
//! | `h q`     | `R[q]` unchanged — the new variable takes over its coordinates; every key `k` with `D[q]·k = 1` is now dead |
//! | rotation  | key `R[q]`, complemented if `negated[q]`                      |
//!
//! # Soundness and exactness
//!
//! Let each coordinate vector denote a Boolean function through a linear map
//! `φ`. After `h q`, the new map agrees with the old one on
//! `span{R[j] : j ≠ q}` and sends `R[q]` to the fresh variable; `R` stays a
//! basis. A key keeps its meaning exactly when it lies in that span, i.e.
//! when `D[q]·k = 0`; the others denote functions no longer in the live span,
//! which no future parity can equal. Evicting exactly those keys therefore
//! merges precisely the rotations the exact symbolic analysis merges.
//!
//! # Lazy eviction
//!
//! Scanning the table at every Hadamard would cost `O(keys)` per `h`.
//! Instead every renaming appends its dual vector to a [`History`], and each
//! group remembers how much of the history it has been checked against. A
//! table hit tests the key only against the renamings since then; a dead key
//! starts a new group. Each (key, renaming) pair is tested at most once, and
//! only for keys that are hit again.
//!
//! The history is kept twice. Row-wise, testing a key against `g` renamings
//! costs `g · n/64` words. Bit-sliced — column `j` is the bitset of renamings
//! whose dual has bit `j` set — the same test XORs the key's columns over
//! the range, `weight(key) · g/64` words, stopping at the first nonzero
//! word. Each check takes whichever is cheaper. Long-lived sparse keys
//! revisited after many unrelated Hadamards (multiplier controls, idle
//! ancillas) are the case the sliced form exists for.
//!
//! A renaming whose dual was already recorded since the last time a group
//! was stamped is not recorded again: every group's pending range already
//! contains the earlier copy, so the test would be redundant. (Skipping a
//! repeat *across* a stamp would be unsound — the newly stamped group never
//! saw the earlier copy.)
//!
//! # Beyond linear circuits
//!
//! Anything that makes a qubit's new value opaque is a *renaming* of its
//! coordinates, handled exactly like `h`: a CCX target (unless a control is a
//! known constant, when the CCX is resolved exactly) and a reset. A reset
//! qubit is additionally marked constant, so rotations on it are dropped and
//! CNOTs from it touch only the target's complement flag. `R` remains a
//! basis throughout; a constant's coordinates simply denote the zero
//! function, which can cost folds but never correctness.

use std::f64::consts::PI;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::circuit::{Circuit, Gate, Qubit};
use crate::pass::Pass;
use crate::phase_fold_rand::{angle_is_zero, classify_quarter_pi, emit_rotation};

/// Exact deterministic phase folding over `n`-bit coordinate vectors with
/// dual-vector eviction. See the module documentation.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseFoldFrame;

impl Pass for PhaseFoldFrame {
    fn name(&self) -> &str {
        "Phase folding (frame)"
    }
    fn run(&self, circuit: &Circuit) -> Circuit {
        phase_fold_frame(circuit)
    }
}

/// The dual vector of every recorded renaming, stored row-wise (one dual per
/// renaming) and bit-sliced (one bitset over renamings per coordinate).
struct History {
    words: usize,
    /// Renaming `r`'s dual at `rows[r * words..(r + 1) * words]`.
    rows: Vec<u64>,
    /// Bit `r` of `columns[j]` is bit `j` of renaming `r`'s dual. Columns
    /// grow on demand; missing words are zero.
    columns: Vec<Vec<u64>>,
    len: usize,
}

impl History {
    fn new(num_qubits: usize, words: usize) -> Self {
        History {
            words,
            rows: Vec::new(),
            columns: vec![Vec::new(); num_qubits],
            len: 0,
        }
    }

    fn push(&mut self, dual: &[u64]) {
        self.rows.extend_from_slice(dual);
        let (word, bit) = (self.len / 64, self.len % 64);
        for (wi, &w) in dual.iter().enumerate() {
            let mut bits = w;
            while bits != 0 {
                let j = wi * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let column = &mut self.columns[j];
                if column.len() <= word {
                    column.resize(word + 1, 0);
                }
                column[word] |= 1 << bit;
            }
        }
        self.len += 1;
    }

    /// Whether `key · dual = 0` for every renaming from `since` on, using
    /// whichever representation is cheaper for this key and range.
    fn orthogonal_since(&self, key: &[u64], since: usize) -> bool {
        if since >= self.len {
            return true;
        }
        let gap = self.len - since;
        let weight: usize = key.iter().map(|w| w.count_ones() as usize).sum();
        let sliced_words = (self.len - 1) / 64 - since / 64 + 1;
        if gap * self.words <= weight * sliced_words {
            self.orthogonal_rows(key, since)
        } else {
            self.orthogonal_sliced(key, since)
        }
    }

    fn orthogonal_rows(&self, key: &[u64], since: usize) -> bool {
        self.rows[since * self.words..]
            .chunks_exact(self.words)
            .all(|dual| {
                let overlap = dual
                    .iter()
                    .zip(key)
                    .fold(0u64, |acc, (d, k)| acc ^ (d & k));
                overlap.count_ones() % 2 == 0
            })
    }

    fn orthogonal_sliced(&self, key: &[u64], since: usize) -> bool {
        if since >= self.len {
            return true;
        }
        let (first, last) = (since / 64, (self.len - 1) / 64);
        for word in first..=last {
            // Bit r of `acc` is key · dual_r for the 64 renamings in this word.
            let mut acc = 0u64;
            for (wi, &kw) in key.iter().enumerate() {
                let mut bits = kw;
                while bits != 0 {
                    let j = wi * 64 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    acc ^= self.columns[j].get(word).copied().unwrap_or(0);
                }
            }
            if word == first {
                acc &= !0u64 << (since % 64);
            }
            // Bits at or beyond `len` were never set.
            if acc != 0 {
                return false;
            }
        }
        true
    }
}

/// Per-qubit coordinates and duals, each stored as `words` machine words per
/// qubit in one flat array.
struct Frame {
    words: usize,
    rows: Vec<u64>,
    duals: Vec<u64>,
    negated: Vec<bool>,
    /// The qubit holds a known computational-basis value (`negated`); its
    /// coordinates denote the zero function.
    constant: Vec<bool>,
    /// The dual of every recorded renaming.
    history: History,
    /// Duals recorded since the last group was stamped; a repeat of one of
    /// them need not be recorded.
    since_stamp: FxHashSet<Box<[u64]>>,
}

impl Frame {
    fn new(num_qubits: usize) -> Self {
        let words = num_qubits.div_ceil(64).max(1);
        let mut frame = Frame {
            words,
            rows: vec![0; num_qubits * words],
            duals: vec![0; num_qubits * words],
            negated: vec![false; num_qubits],
            constant: vec![false; num_qubits],
            history: History::new(num_qubits, words),
            since_stamp: FxHashSet::default(),
        };
        for q in 0..num_qubits {
            frame.rows[q * words + q / 64] |= 1 << (q % 64);
            frame.duals[q * words + q / 64] |= 1 << (q % 64);
        }
        frame
    }

    fn row(&self, q: usize) -> &[u64] {
        &self.rows[q * self.words..(q + 1) * self.words]
    }

    /// Whether `key`'s meaning survived every renaming from `since` on.
    fn still_valid(&self, key: &[u64], since: usize) -> bool {
        self.history.orthogonal_since(key, since)
    }

    /// Stamp a group as validated against the whole history so far, and
    /// return the stamp.
    fn stamp(&mut self) -> usize {
        if !self.since_stamp.is_empty() {
            self.since_stamp.clear();
        }
        self.history.len
    }

    /// Qubit `q`'s coordinates now denote a new function; keys depending on
    /// the old one are invalidated lazily through the history.
    fn rename(&mut self, q: usize) {
        let w = self.words;
        let dual = &self.duals[q * w..(q + 1) * w];
        if !self.since_stamp.contains(dual) {
            self.history.push(dual);
            self.since_stamp.insert(Box::from(dual));
        }
        self.negated[q] = false;
    }

    fn hadamard(&mut self, q: usize) {
        self.rename(q);
        self.constant[q] = false;
    }

    fn reset(&mut self, q: usize) {
        self.rename(q);
        self.constant[q] = true;
    }

    fn cnot(&mut self, control: usize, target: usize) {
        if control == target {
            // Malformed `cx q,q`: treated as the identity.
            return;
        }
        if self.constant[control] {
            // Only the control's constant reaches the target.
            self.negated[target] ^= self.negated[control];
            return;
        }
        let w = self.words;
        let (c, t) = (control * w, target * w);
        for i in 0..w {
            let row = self.rows[c + i];
            self.rows[t + i] ^= row;
            let dual = self.duals[t + i];
            self.duals[c + i] ^= dual;
        }
        self.negated[target] ^= self.negated[control];
        self.constant[target] = false;
    }

    fn ccx(&mut self, control1: usize, control2: usize, target: usize) {
        match (self.constant[control1], self.constant[control2]) {
            (true, true) => {
                if self.negated[control1] && self.negated[control2] {
                    self.negated[target] ^= true;
                }
            }
            // A control fixed at |1⟩ makes the CCX a CNOT from the other one;
            // fixed at |0⟩, it is the identity.
            (true, false) => {
                if self.negated[control1] {
                    self.cnot(control2, target);
                }
            }
            (false, true) => {
                if self.negated[control2] {
                    self.cnot(control1, target);
                }
            }
            // target ⊕ c1·c2 is not a parity: treat it as a new variable.
            (false, false) => {
                self.rename(target);
                self.constant[target] = false;
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Rotation {
    /// `k · π/4` (mod 2π).
    Int(u8),
    /// An `rz` angle that is not a multiple of π/4.
    Float(f64),
}

fn rotation_of(gate: &Gate) -> Option<(Qubit, Rotation)> {
    Some(match gate {
        Gate::t(q) => (*q, Rotation::Int(1)),
        Gate::tdg(q) => (*q, Rotation::Int(7)),
        Gate::s(q) => (*q, Rotation::Int(2)),
        Gate::sdg(q) => (*q, Rotation::Int(6)),
        Gate::z(q) => (*q, Rotation::Int(4)),
        Gate::rz(theta, q) => match classify_quarter_pi(*theta) {
            Some(k) => (*q, Rotation::Int(k)),
            None => (*q, Rotation::Float(*theta)),
        },
        _ => return None,
    })
}

/// Accumulated phase for one parity group, as in `PhaseFoldRand`, plus the
/// number of renamings its key has been validated against.
struct LivePhase {
    int_part: u8,
    float_part: f64,
    qubit: Qubit,
    current_idx: usize,
    /// `current_idx` sits on the complement of the group's parity.
    current_sign: bool,
    checked: usize,
}

/// Free-function form of [`PhaseFoldFrame`], for use outside a [`Pass`] pipeline.
pub fn phase_fold_frame(circuit: &Circuit) -> Circuit {
    let mut frame = Frame::new(circuit.num_qubits);
    let mut live: Vec<LivePhase> = Vec::new();
    let mut groups: FxHashMap<Box<[u64]>, usize> = FxHashMap::default();
    let mut skip = vec![false; circuit.gates.len()];

    for (idx, gate) in circuit.gates.iter().enumerate() {
        if let Some((q, rotation)) = rotation_of(gate) {
            let qi = q as usize;
            if frame.constant[qi] {
                // Identity on |0⟩, global phase on |1⟩.
                skip[idx] = true;
                continue;
            }
            let is_complement = frame.negated[qi];
            let key = frame.row(qi);
            let hit = groups
                .get(key)
                .copied()
                .filter(|&gi| frame.still_valid(key, live[gi].checked));
            if let Some(gi) = hit {
                let group = &mut live[gi];
                skip[group.current_idx] = true;
                // RZ(θ) on ¬p ≡ RZ(−θ) on p up to global phase.
                match rotation {
                    Rotation::Int(k) if is_complement => {
                        group.int_part = group.int_part.wrapping_sub(k) & 7;
                    }
                    Rotation::Int(k) => group.int_part = (group.int_part + k) & 7,
                    Rotation::Float(theta) if is_complement => group.float_part -= theta,
                    Rotation::Float(theta) => group.float_part += theta,
                }
                group.current_idx = idx;
                group.qubit = q;
                group.current_sign = is_complement;
                group.checked = frame.stamp();
            } else {
                // New parity, or the old group's key is dead: that group
                // stays in `live` for emission, and this key now maps here.
                let (int_part, float_part) = match rotation {
                    Rotation::Int(k) if is_complement => (8u8.wrapping_sub(k) & 7, 0.0),
                    Rotation::Int(k) => (k, 0.0),
                    Rotation::Float(theta) if is_complement => (0, -theta),
                    Rotation::Float(theta) => (0, theta),
                };
                groups.insert(Box::from(key), live.len());
                live.push(LivePhase {
                    int_part,
                    float_part,
                    qubit: q,
                    current_idx: idx,
                    current_sign: is_complement,
                    checked: frame.stamp(),
                });
            }
            continue;
        }
        match gate {
            Gate::t(_) | Gate::tdg(_) | Gate::s(_) | Gate::sdg(_) | Gate::z(_) | Gate::rz(..) => {
                unreachable!("phase gates are recorded above")
            }
            Gate::h(q) => frame.hadamard(*q as usize),
            Gate::x(q) => frame.negated[*q as usize] ^= true,
            Gate::cnot { control, target } => frame.cnot(*control as usize, *target as usize),
            // Diagonal: they change phases, not computational-basis values.
            Gate::cz { .. } | Gate::ccz { .. } => {}
            Gate::ccx {
                control1,
                control2,
                target,
            } => frame.ccx(*control1 as usize, *control2 as usize, *target as usize),
            // Measurement preserves the computational-basis value.
            Gate::measure { .. } => {}
            Gate::reset(q) => frame.reset(*q as usize),
        }
    }

    // Emit each group at its last site, as PhaseFoldRand does.
    let mut emit_at: Vec<Option<(Qubit, u8, f64)>> = vec![None; circuit.gates.len()];
    for lp in &live {
        let (int_part, float_part) = if lp.current_sign {
            (8u8.wrapping_sub(lp.int_part) & 7, -lp.float_part)
        } else {
            (lp.int_part, lp.float_part)
        };
        if float_part == 0.0 {
            if int_part == 0 {
                skip[lp.current_idx] = true;
            } else {
                emit_at[lp.current_idx] = Some((lp.qubit, int_part, 0.0));
            }
        } else {
            let total = (int_part as f64) * (PI / 4.0) + float_part;
            if angle_is_zero(total) {
                skip[lp.current_idx] = true;
            } else {
                emit_at[lp.current_idx] = Some((lp.qubit, 0, total));
            }
        }
    }

    let mut output = Circuit::with_cbits(circuit.num_qubits, circuit.num_cbits);
    for (idx, gate) in circuit.gates.iter().enumerate() {
        if skip[idx] {
            continue;
        }
        if let Some((qubit, int_part, float_part)) = emit_at[idx] {
            emit_rotation(&mut output, qubit, int_part, float_part);
        } else {
            output.apply(gate.clone());
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass::count_t;
    use crate::phase_fold_rand::phase_fold_rand;
    use crate::unitary::circuits_equiv;

    fn circuit(num_qubits: usize, gates: &[Gate]) -> Circuit {
        let mut c = Circuit::new(num_qubits);
        for g in gates {
            c.apply(g.clone());
        }
        c
    }

    fn cx(control: Qubit, target: Qubit) -> Gate {
        Gate::cnot { control, target }
    }

    fn check(c: &Circuit) -> Circuit {
        let out = phase_fold_frame(c);
        assert!(circuits_equiv(c, &out, 1e-9), "changed semantics\n{c}");
        out
    }

    #[test]
    fn basic_folds() {
        assert_eq!(check(&circuit(1, &[Gate::t(0), Gate::t(0)])).gates, vec![Gate::s(0)]);
        let c = circuit(2, &[Gate::t(1), cx(0, 1), cx(0, 1), Gate::t(1)]);
        assert_eq!(count_t(&check(&c)), 0);
        let c = circuit(1, &[Gate::t(0), Gate::x(0), Gate::t(0)]);
        assert_eq!(count_t(&check(&c)), 0);
        let c = circuit(1, &[Gate::t(0), Gate::h(0), Gate::t(0)]);
        assert_eq!(count_t(&check(&c)), 2);
    }

    /// Exact where generation counters are not: the control keeps its
    /// identity across a CNOT into a Hadamard-conjugated target.
    #[test]
    fn keeps_control_across_cnot_into_fresh_target() {
        let c = circuit(2, &[Gate::t(0), Gate::h(1), cx(0, 1), Gate::t(0)]);
        assert_eq!(check(&c).gates, vec![Gate::h(1), cx(0, 1), Gate::s(0)]);
    }

    /// H followed by a 3-CNOT SWAP, in both orientations: both values are
    /// tracked through the SWAP, so every T pairs up.
    #[test]
    fn folds_through_swap_after_hadamard() {
        for (a, b) in [(0, 1), (1, 0)] {
            let c = circuit(
                2,
                &[
                    Gate::h(0),
                    Gate::t(0),
                    Gate::t(1),
                    cx(a, b),
                    cx(b, a),
                    cx(a, b),
                    Gate::t(0),
                    Gate::t(1),
                ],
            );
            let out = check(&c);
            assert_eq!(count_t(&out), 0, "swap starting cx {a}→{b}");
        }
    }

    #[test]
    fn genuinely_different_parities_do_not_fold() {
        let cases = [
            circuit(
                2,
                &[cx(0, 1), Gate::h(0), cx(1, 0), Gate::h(1), Gate::t(1), Gate::t(0)],
            ),
            circuit(
                2,
                &[
                    cx(1, 0),
                    cx(0, 1),
                    Gate::h(0),
                    Gate::tdg(0),
                    cx(0, 1),
                    cx(1, 0),
                    Gate::t(0),
                ],
            ),
            circuit(2, &[Gate::h(0), Gate::t(0), cx(1, 0), Gate::h(0), Gate::t(0)]),
        ];
        for c in &cases {
            assert_eq!(count_t(&check(c)), count_t(c), "\n{c}");
        }
    }

    #[test]
    fn constants_after_reset() {
        let c = circuit(
            2,
            &[
                Gate::h(1),
                Gate::t(1),
                Gate::reset(0),
                Gate::t(0),
                cx(0, 1),
                Gate::t(1),
            ],
        );
        assert_eq!(
            phase_fold_frame(&c).gates,
            vec![Gate::h(1), Gate::reset(0), cx(0, 1), Gate::s(1)]
        );
    }

    #[test]
    fn ccx_with_a_constant_control_is_a_cnot() {
        let ccx = Gate::ccx {
            control1: 0,
            control2: 1,
            target: 2,
        };
        let c = circuit(
            3,
            &[
                Gate::reset(0),
                Gate::x(0),
                Gate::t(2),
                ccx.clone(),
                ccx,
                Gate::t(2),
            ],
        );
        assert_eq!(count_t(&phase_fold_frame(&c)), 0);
    }

    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    fn random_circuit(rng: &mut Rng, num_qubits: usize, num_gates: usize, ccx: bool) -> Circuit {
        let mut c = Circuit::new(num_qubits);
        for _ in 0..num_gates {
            let q = rng.below(num_qubits) as Qubit;
            let other = (q + 1 + rng.below(num_qubits - 1) as Qubit) % num_qubits as Qubit;
            let third = (0..num_qubits as Qubit).find(|&r| r != q && r != other);
            match rng.below(15) {
                0..=3 => c.apply(cx(q, other)),
                4 | 5 => c.apply(Gate::h(q)),
                6 if ccx && third.is_some() => c.apply(Gate::ccx {
                    control1: q,
                    control2: other,
                    target: third.unwrap(),
                }),
                7 => c.apply(Gate::x(q)),
                8 => c.apply(Gate::cz {
                    control: q,
                    target: other,
                }),
                9 => c.apply(Gate::rz(0.1 * rng.below(40) as f64, q)),
                10 => c.apply(Gate::s(q)),
                11 => c.apply(Gate::tdg(q)),
                _ => c.apply(Gate::t(q)),
            }
        }
        c
    }

    #[test]
    fn sound_on_random_circuits_with_toffolis() {
        let mut rng = Rng(0xf4a3_e5ee_d000_0001);
        for case in 0..2_000 {
            let n = 3 + rng.below(3);
            let len = 5 + rng.below(60);
            let c = random_circuit(&mut rng, n, len, true);
            let out = phase_fold_frame(&c);
            assert!(circuits_equiv(&c, &out, 1e-9), "case {case}\n{c}");
            assert!(count_t(&out) <= count_t(&c));
        }
    }

    /// On CNOT-dihedral circuits both analyses are exact (the randomized
    /// one up to fingerprint collisions) and share their bookkeeping, so
    /// they must produce the same circuit gate for gate.
    #[test]
    fn matches_randomized_pass_gate_for_gate() {
        let mut rng = Rng(0x5a5a_1234);
        for case in 0..3_000 {
            let n = 2 + rng.below(7);
            let len = 5 + rng.below(120);
            let c = random_circuit(&mut rng, n, len, false);
            assert_eq!(
                phase_fold_frame(&c).gates,
                phase_fold_rand(&c).gates,
                "case {case}\n{c}"
            );
        }
    }

    #[test]
    fn matches_randomized_pass_on_benchmarks() {
        // Files with a malformed `cx q,q` (cycle_17_3, mod_adder_1048576)
        // are left out: the passes interpret that gate differently.
        for name in [
            "tof_3",
            "tof_10",
            "barenco_tof_4",
            "mod5_4",
            "vbe_adder_3",
            "grover_5",
            "adder_8",
            "gf2^8_mult",
            "hwb6",
            "qcla_mod_7",
            "ham15-med",
        ] {
            let path = format!("{}/benchmarks/feynman/{name}.qasm", env!("CARGO_MANIFEST_DIR"));
            let src = std::fs::read_to_string(&path).unwrap();
            let c = crate::qasm::parse(&src).unwrap();
            assert_eq!(
                phase_fold_frame(&c).gates,
                phase_fold_rand(&c).gates,
                "{name}"
            );
        }
    }

    #[test]
    fn is_deterministic_and_handles_wide_circuits() {
        let mut rng = Rng(11);
        let c = random_circuit(&mut rng, 130, 20_000, true);
        let first = phase_fold_frame(&c);
        assert_eq!(phase_fold_frame(&c).gates, first.gates);
        assert!(count_t(&first) <= count_t(&c));
    }

    /// A repeated dual may be skipped only if no group was stamped since its
    /// earlier copy: here the first `t` is stamped between the two `h`s and
    /// must be checked against the second.
    #[test]
    fn repeated_hadamards_with_a_stamp_between_are_both_recorded() {
        let c = circuit(1, &[Gate::h(0), Gate::t(0), Gate::h(0), Gate::t(0)]);
        assert_eq!(count_t(&check(&c)), 2);
        // Without a stamp between, the repeat adds nothing.
        let c = circuit(1, &[Gate::t(0), Gate::h(0), Gate::h(0), Gate::t(0)]);
        assert_eq!(count_t(&check(&c)), 2);
    }

    /// Row-wise and bit-sliced checks agree on every key and range.
    #[test]
    fn history_representations_agree() {
        let mut rng = Rng(0xb175_11ce);
        for n in [3usize, 64, 65, 130] {
            let words = n.div_ceil(64);
            let mut history = History::new(n, words);
            let random_vector = |rng: &mut Rng| -> Vec<u64> {
                let mut v = vec![0u64; words];
                // Sparse and dense vectors both.
                let density = 1 + rng.below(4);
                for j in 0..n {
                    if rng.below(8) < density {
                        v[j / 64] |= 1 << (j % 64);
                    }
                }
                v
            };
            for _ in 0..300 {
                let dual = random_vector(&mut rng);
                history.push(&dual);
                for _ in 0..5 {
                    let key = random_vector(&mut rng);
                    let since = rng.below(history.len + 1);
                    let rows = history.orthogonal_rows(&key, since);
                    assert_eq!(rows, history.orthogonal_sliced(&key, since), "n={n}");
                    assert_eq!(rows, history.orthogonal_since(&key, since), "n={n}");
                }
            }
        }
    }

    /// `n` idle qubits carry a T before and after a long run of
    /// `h w; t w` on one worker. Each idle key is revisited after `rounds`
    /// unrelated renamings — the pattern the bit-sliced history is for.
    fn hadamard_storm(idle: usize, rounds: usize) -> Circuit {
        let w = idle as Qubit;
        let mut c = Circuit::new(idle + 1);
        for q in 0..idle {
            c.apply(Gate::t(q as Qubit));
        }
        for _ in 0..rounds {
            c.apply(Gate::h(w));
            c.apply(Gate::t(w));
        }
        for q in 0..idle {
            c.apply(Gate::t(q as Qubit));
        }
        c
    }

    #[test]
    fn hadamard_storm_folds_idle_pairs_only() {
        let small = hadamard_storm(2, 3);
        check(&small);
        let (idle, rounds) = (64, 20_000);
        let out = phase_fold_frame(&hadamard_storm(idle, rounds));
        assert_eq!(count_t(&out), rounds);
        assert_eq!(
            out.gates.iter().filter(|g| matches!(g, Gate::s(_))).count(),
            idle
        );
        assert_eq!(out.gates, phase_fold_rand(&hadamard_storm(idle, rounds)).gates);
    }

    /// Run with
    /// `cargo test --release phase_fold_frame::tests::hadamard_storm_scaling -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn hadamard_storm_scaling() {
        println!("\n{:>6} {:>9} {:>9} | {:>9} {:>9}", "idle", "rounds", "gates", "Rand ms", "Frame ms");
        for (idle, rounds) in [(64, 100_000), (64, 1_000_000), (512, 100_000), (512, 1_000_000)] {
            let c = hadamard_storm(idle, rounds);
            let t = std::time::Instant::now();
            let rand = phase_fold_rand(&c);
            let tr = t.elapsed();
            let t = std::time::Instant::now();
            let frame = phase_fold_frame(&c);
            let tf = t.elapsed();
            assert_eq!(count_t(&rand), count_t(&frame));
            println!(
                "{:>6} {:>9} {:>9} | {:>9.2} {:>9.2}",
                idle,
                rounds,
                c.gates.len(),
                tr.as_secs_f64() * 1e3,
                tf.as_secs_f64() * 1e3
            );
        }
    }
}
