//! Deterministic phase folding over generation-stamped parity vectors.
//!
//! [`PhaseFoldRand`](crate::phase_fold_rand::PhaseFoldRand) gives every
//! Hadamard a fresh random 128-bit tag. This pass gets the same
//! reset-on-Hadamard behavior without randomness, and with a representation
//! whose size never grows: each qubit's value is a *generation* counter plus
//! an `n`-bit vector over the qubits.
//!
//! # Abstract domain
//!
//! Slot `j` of a vector whose generation is `k` names the variable
//! `(j, k)`: "the value qubit `j` held when it entered generation `k`". A
//! qubit's value is `negated ⊕ XOR of its slots' variables`. Qubit `j`'s
//! generation starts at 0 and only ever increases; each increase introduces
//! the variable `(j, new generation)` in qubit `j`'s own slot.
//!
//! Two vectors are XORed only when their generations are equal. When a CNOT
//! meets operands of different generations, one side is *forgotten*: its
//! current value is renamed to a variable in a newer generation. Which side
//! is forgotten is the [`ResetPolicy`].
//!
//! # Soundness
//!
//! Three invariants make equal keys imply equal Boolean functions:
//!
//! 1. **Each variable is introduced once.** `(j, k)` is created only when
//!    qubit `j` moves to generation `k` from a lower one, and generations
//!    never decrease.
//! 2. **Generations never mix.** Every vector operation combines vectors of
//!    one generation, so a slot in a generation-`k` vector always names a
//!    generation-`k` variable that has already been introduced.
//! 3. **Each variable stands for one fixed function.** A variable introduced
//!    by `h` is the new path variable; one introduced by forgetting stands for
//!    the qubit's concrete value at that moment. Every transfer function keeps
//!    "concretization of the abstract value = the qubit's actual value".
//!
//! Distinct variables may happen to denote the same function; that only
//! costs folds, never correctness.
//!
//! # Cost
//!
//! Every operation is `O(n / 64)` machine words — one word pair for
//! circuits of up to 64 qubits — independent of how many Hadamards have
//! occurred or how heavily the CNOT network has mixed them. Phase-table keys
//! are the qubit's row itself (generation word followed by the bit vector),
//! looked up without allocating.

use std::f64::consts::PI;

use rustc_hash::FxHashMap;

use crate::circuit::{Circuit, Gate, Qubit};
use crate::pass::Pass;
use crate::phase_fold_rand::{angle_is_zero, classify_quarter_pi, emit_rotation};

/// Which operand a CNOT forgets when its operands' generations differ.
///
/// When the *control* is ahead, both policies forget the target (rename its
/// value to a variable in the control's generation) and then XOR the control
/// in, so the relation to the control survives. They differ only when the
/// *target* is ahead.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResetPolicy {
    /// Target ahead: give the target a fresh variable in its next generation
    /// and leave the control untouched. Keeps controls' identities across
    /// Toffoli-style gadgets, where the target is Hadamard-conjugated.
    #[default]
    ForgetTarget,
    /// Target ahead: forget the control (lift it to the target's generation),
    /// then XOR as usual. Keeps the target's relation to the control but
    /// loses the control's history.
    ForgetLower,
}

/// Merges T/Rz gates acting on equal parities, tracked deterministically as
/// generation-stamped bit vectors. See the module documentation.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseFoldDet {
    pub policy: ResetPolicy,
}

impl Pass for PhaseFoldDet {
    fn name(&self) -> &str {
        match self.policy {
            ResetPolicy::ForgetTarget => "Phase folding (deterministic)",
            ResetPolicy::ForgetLower => "Phase folding (deterministic, forget lower)",
        }
    }
    fn run(&self, circuit: &Circuit) -> Circuit {
        phase_fold_det(circuit, self.policy)
    }
}

/// Per-qubit abstract values, stored as one flat array of rows. Row `q` is
/// `[generation, bits_0, bits_1, …]`, so a row is directly usable as a phase
/// table key.
struct State {
    stride: usize,
    rows: Vec<u64>,
    negated: Vec<bool>,
}

impl State {
    fn new(num_qubits: usize) -> Self {
        let stride = 1 + num_qubits.div_ceil(64).max(1);
        let mut state = State {
            stride,
            rows: vec![0; num_qubits * stride],
            negated: vec![false; num_qubits],
        };
        for q in 0..num_qubits {
            state.fresh(q, 0);
        }
        state
    }

    fn row(&self, q: usize) -> &[u64] {
        &self.rows[q * self.stride..(q + 1) * self.stride]
    }

    fn generation(&self, q: usize) -> u64 {
        self.rows[q * self.stride]
    }

    /// The qubit holds a known computational-basis constant (`negated`).
    fn is_constant(&self, q: usize) -> bool {
        self.rows[q * self.stride + 1..(q + 1) * self.stride]
            .iter()
            .all(|&w| w == 0)
    }

    /// Introduce the variable `(q, generation)` as qubit `q`'s whole value.
    /// Callers only ever pass a generation above `q`'s current one (or 0 at
    /// start-up), which is what keeps each variable introduced once.
    fn fresh(&mut self, q: usize, generation: u64) {
        let base = q * self.stride;
        self.rows[base] = generation;
        self.rows[base + 1..base + self.stride].fill(0);
        self.rows[base + 1 + q / 64] |= 1 << (q % 64);
        self.negated[q] = false;
    }

    /// Move a constant to a later generation. Its value has no variables,
    /// so nothing is lost.
    fn lift_constant(&mut self, q: usize, generation: u64) {
        debug_assert!(self.is_constant(q) && generation >= self.generation(q));
        self.rows[q * self.stride] = generation;
    }

    /// `target ^= control`, for operands of the same generation (or a
    /// constant control).
    fn xor_into(&mut self, control: usize, target: usize) {
        let (c, t) = (control * self.stride, target * self.stride);
        for w in 1..self.stride {
            let bits = self.rows[c + w];
            self.rows[t + w] ^= bits;
        }
        self.negated[target] ^= self.negated[control];
    }

    fn hadamard(&mut self, q: usize) {
        let next = self.generation(q) + 1;
        self.fresh(q, next);
    }

    fn cnot(&mut self, control: usize, target: usize, policy: ResetPolicy) {
        if control == target {
            return;
        }
        let (gc, gt) = (self.generation(control), self.generation(target));
        if gc == gt || self.is_constant(control) {
            // Same generation: plain XOR. A constant control contributes
            // only its constant, whatever its generation.
            self.xor_into(control, target);
        } else if gc > gt {
            // Control ahead: bring the target up to the control's
            // generation, then XOR. A constant target loses nothing.
            if self.is_constant(target) {
                self.lift_constant(target, gc);
            } else {
                self.fresh(target, gc);
            }
            self.xor_into(control, target);
        } else {
            // Target ahead. Generations never decrease, so the control
            // cannot be brought down; one side must be forgotten.
            match policy {
                ResetPolicy::ForgetTarget => self.fresh(target, gt + 1),
                ResetPolicy::ForgetLower => {
                    self.fresh(control, gt);
                    self.xor_into(control, target);
                }
            }
        }
    }

    fn ccx(&mut self, control1: usize, control2: usize, target: usize, policy: ResetPolicy) {
        let (c1, c2) = (self.is_constant(control1), self.is_constant(control2));
        match (c1, c2) {
            (true, true) => {
                if self.negated[control1] && self.negated[control2] {
                    self.negated[target] ^= true;
                }
            }
            // A control fixed at |1⟩ makes the CCX a CNOT from the other one;
            // fixed at |0⟩, it is the identity.
            (true, false) => {
                if self.negated[control1] {
                    self.cnot(control2, target, policy);
                }
            }
            (false, true) => {
                if self.negated[control2] {
                    self.cnot(control1, target, policy);
                }
            }
            // target ⊕ c1·c2 is not a parity: the target becomes opaque.
            (false, false) => {
                let next = self.generation(target) + 1;
                self.fresh(target, next);
            }
        }
    }

    fn reset(&mut self, q: usize) {
        let base = q * self.stride;
        self.rows[base + 1..base + self.stride].fill(0);
        self.negated[q] = false;
    }
}

/// The rotation a phase gate applies, split as [`LivePhase`] accumulates it.
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

/// Accumulated phase for one parity group; same bookkeeping as
/// `PhaseFoldRand`, so the two passes differ only in how parities are
/// represented.
struct LivePhase {
    int_part: u8,
    float_part: f64,
    qubit: Qubit,
    current_idx: usize,
    /// `current_idx` sits on the complement of the group's parity.
    current_sign: bool,
}

/// Free-function form of [`PhaseFoldDet`], for use outside a [`Pass`] pipeline.
pub fn phase_fold_det(circuit: &Circuit, policy: ResetPolicy) -> Circuit {
    let mut state = State::new(circuit.num_qubits);
    let mut live: Vec<LivePhase> = Vec::new();
    let mut groups: FxHashMap<Box<[u64]>, usize> = FxHashMap::default();
    let mut skip = vec![false; circuit.gates.len()];

    for (idx, gate) in circuit.gates.iter().enumerate() {
        if let Some((q, rotation)) = rotation_of(gate) {
            let qi = q as usize;
            if state.is_constant(qi) {
                // Identity on |0⟩, global phase on |1⟩.
                skip[idx] = true;
                continue;
            }
            let is_complement = state.negated[qi];
            let key = state.row(qi);
            if let Some(&gi) = groups.get(key) {
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
            } else {
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
                });
            }
            continue;
        }
        match gate {
            Gate::t(_) | Gate::tdg(_) | Gate::s(_) | Gate::sdg(_) | Gate::z(_) | Gate::rz(..) => {
                unreachable!("phase gates are recorded above")
            }
            Gate::h(q) => state.hadamard(*q as usize),
            Gate::x(q) => state.negated[*q as usize] ^= true,
            Gate::cnot { control, target } => {
                state.cnot(*control as usize, *target as usize, policy)
            }
            // Diagonal: they change phases, not computational-basis values.
            Gate::cz { .. } | Gate::ccz { .. } => {}
            Gate::ccx {
                control1,
                control2,
                target,
            } => state.ccx(
                *control1 as usize,
                *control2 as usize,
                *target as usize,
                policy,
            ),
            // Measurement preserves the computational-basis value.
            Gate::measure { .. } => {}
            Gate::reset(q) => state.reset(*q as usize),
        }
    }

    // Emit each surviving group at its last site, as PhaseFoldRand does.
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
    use std::time::{Duration, Instant};

    use super::*;
    use crate::pass::count_t;
    use crate::phase_fold_rand::phase_fold_rand;
    use crate::unitary::circuits_equiv;

    const POLICIES: [ResetPolicy; 2] = [ResetPolicy::ForgetTarget, ResetPolicy::ForgetLower];

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

    fn check(c: &Circuit, policy: ResetPolicy) -> Circuit {
        let out = phase_fold_det(c, policy);
        assert!(circuits_equiv(c, &out, 1e-9), "{policy:?} changed semantics\n{c}");
        out
    }

    #[test]
    fn basic_folds() {
        for policy in POLICIES {
            // T·T → S.
            let out = check(&circuit(1, &[Gate::t(0), Gate::t(0)]), policy);
            assert_eq!(out.gates, vec![Gate::s(0)]);
            // Across a CNOT pair.
            let c = circuit(2, &[Gate::t(1), cx(0, 1), cx(0, 1), Gate::t(1)]);
            assert_eq!(count_t(&check(&c, policy)), 0);
            // Across X: T·X·T = X.
            let c = circuit(1, &[Gate::t(0), Gate::x(0), Gate::t(0)]);
            assert_eq!(count_t(&check(&c, policy)), 0);
            // H is a boundary.
            let c = circuit(1, &[Gate::t(0), Gate::h(0), Gate::t(0)]);
            assert_eq!(count_t(&check(&c, policy)), 2);
            // Equal generations mix freely: both qubits' Hadamards happen first.
            let c = circuit(
                2,
                &[Gate::h(0), Gate::h(1), Gate::t(1), cx(0, 1), cx(0, 1), Gate::t(1)],
            );
            assert_eq!(count_t(&check(&c, policy)), 0);
        }
    }

    /// The counterexamples that break single-count schemes without resets
    /// must not be folded.
    #[test]
    fn generation_mixing_counterexamples_do_not_fold() {
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
            circuit(
                2,
                &[Gate::h(0), Gate::t(0), cx(1, 0), Gate::h(0), Gate::t(0)],
            ),
        ];
        for c in &cases {
            for policy in POLICIES {
                assert_eq!(count_t(&check(c, policy)), count_t(c), "{policy:?}\n{c}");
            }
        }
    }

    /// Where the policies differ: a CNOT into a Hadamard-conjugated target.
    #[test]
    fn forget_target_keeps_the_control() {
        let c = circuit(2, &[Gate::t(0), Gate::h(1), cx(0, 1), Gate::t(0)]);
        assert_eq!(
            check(&c, ResetPolicy::ForgetTarget).gates,
            vec![Gate::h(1), cx(0, 1), Gate::s(0)]
        );
        assert_eq!(count_t(&check(&c, ResetPolicy::ForgetLower)), 2);
    }

    #[test]
    fn constants_are_tracked_exactly() {
        let mut c = Circuit::new(2);
        c.apply(Gate::reset(0));
        c.apply(Gate::t(0));
        c.apply(Gate::h(1));
        c.apply(cx(0, 1)); // constant control: no reset of either side
        c.apply(Gate::x(0));
        c.apply(Gate::t(0)); // known |1⟩: global phase only
        let out = phase_fold_det(&c, ResetPolicy::ForgetTarget);
        assert_eq!(
            out.gates,
            vec![Gate::reset(0), Gate::h(1), cx(0, 1), Gate::x(0)]
        );
    }

    #[test]
    fn ccx_with_a_constant_control() {
        for policy in POLICIES {
            // Control 0 is |1⟩ after X, so the CCX is a CNOT from qubit 1;
            // applying it twice restores qubit 2's parity.
            let ccx = Gate::ccx {
                control1: 0,
                control2: 1,
                target: 2,
            };
            let mut c = Circuit::new(3);
            c.apply(Gate::reset(0));
            c.apply(Gate::x(0));
            c.apply(Gate::t(2));
            c.apply(ccx.clone());
            c.apply(ccx);
            c.apply(Gate::t(2));
            let out = phase_fold_det(&c, policy);
            assert_eq!(count_t(&out), 0, "{policy:?}");
        }
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

    fn random_circuit(rng: &mut Rng, num_qubits: usize, num_gates: usize, h_and_ccx: bool) -> Circuit {
        let mut c = Circuit::new(num_qubits);
        for _ in 0..num_gates {
            let q = rng.below(num_qubits) as Qubit;
            let other = (q + 1 + rng.below(num_qubits - 1) as Qubit) % num_qubits as Qubit;
            let third = (0..num_qubits as Qubit).find(|&r| r != q && r != other);
            match rng.below(14) {
                0..=3 => c.apply(cx(q, other)),
                4 | 5 if h_and_ccx => c.apply(Gate::h(q)),
                6 if h_and_ccx && third.is_some() => c.apply(Gate::ccx {
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
    fn sound_on_random_circuits() {
        let mut rng = Rng(0xde7e_de7e_de7e_de7e);
        for case in 0..2_000 {
            let n = 2 + rng.below(4);
            let len = 5 + rng.below(60);
            let c = random_circuit(&mut rng, n, len, true);
            for policy in POLICIES {
                let out = phase_fold_det(&c, policy);
                assert!(
                    circuits_equiv(&c, &out, 1e-9),
                    "case {case}, {policy:?}\n{c}"
                );
                assert!(count_t(&out) <= count_t(&c));
            }
        }
    }

    /// With no Hadamards every qubit stays in generation 0, the analysis is
    /// exact, and it must fold exactly what the randomized pass does.
    #[test]
    fn matches_randomized_pass_without_hadamards() {
        let mut rng = Rng(0x0dd_ba11);
        for case in 0..2_000 {
            let n = 2 + rng.below(5);
            let len = 5 + rng.below(80);
            let c = random_circuit(&mut rng, n, len, false);
            for policy in POLICIES {
                assert_eq!(
                    phase_fold_det(&c, policy).gates,
                    phase_fold_rand(&c).gates,
                    "case {case}, {policy:?}\n{c}"
                );
            }
        }
    }

    #[test]
    fn is_deterministic_and_handles_wide_circuits() {
        // 130 qubits spans three bit-vector words.
        let mut rng = Rng(7);
        let c = random_circuit(&mut rng, 130, 20_000, true);
        let first = phase_fold_det(&c, ResetPolicy::ForgetTarget);
        assert_eq!(phase_fold_det(&c, ResetPolicy::ForgetTarget).gates, first.gates);
        assert!(count_t(&first) <= count_t(&c));
    }

    fn benchmark_paths() -> Vec<std::path::PathBuf> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
        let mut paths: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            // Skip loose files such as benchmarks/README.md.
            .filter(|path| path.is_dir())
            .flat_map(|dir| std::fs::read_dir(dir).unwrap())
            .map(|entry| entry.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "qasm"))
            .collect();
        paths.sort();
        paths
    }

    fn timed(f: impl FnOnce() -> Circuit) -> (Circuit, Duration) {
        let start = Instant::now();
        let out = f();
        (out, start.elapsed())
    }

    /// Each phase-folding pass alone on every benchmark: T-count and time.
    /// Run with
    /// `cargo test --release phase_fold_det::tests::compare_with_phase_fold_rand -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn compare_with_phase_fold_rand() {
        let passes: [(&str, fn(&Circuit) -> Circuit); 4] = [
            ("Rand", phase_fold_rand),
            ("Det", |c| phase_fold_det(c, ResetPolicy::ForgetTarget)),
            ("DetLower", |c| phase_fold_det(c, ResetPolicy::ForgetLower)),
            ("Frame", crate::phase_fold_frame::phase_fold_frame),
        ];
        print!("\n{:<38} {:>9}", "benchmark", "T in");
        for (name, _) in &passes {
            print!(" {name:>9}");
        }
        for (name, _) in &passes {
            print!(" {:>9}", format!("{name} ms"));
        }
        println!();
        // Signed: rz groups can sum to a π/4 multiple and emit a T.
        let mut removed = [0i64; 4];
        let mut time = [Duration::ZERO; 4];
        for path in benchmark_paths() {
            let src = std::fs::read_to_string(&path).unwrap();
            let c = crate::qasm::parse(&src).unwrap();
            let t_in = count_t(&c);
            let results: Vec<(usize, Duration)> = passes
                .iter()
                .map(|(_, pass)| {
                    let (out, t) = timed(|| pass(&c));
                    (count_t(&out), t)
                })
                .collect();
            let name = path
                .strip_prefix(concat!(env!("CARGO_MANIFEST_DIR"), "/benchmarks/"))
                .unwrap_or(&path)
                .with_extension("");
            print!("{:<38} {:>9}", name.display(), t_in);
            for (i, (t, _)) in results.iter().enumerate() {
                removed[i] += t_in as i64 - *t as i64;
                print!(" {t:>9}");
            }
            for (i, (_, d)) in results.iter().enumerate() {
                time[i] += *d;
                print!(" {:>9.2}", d.as_secs_f64() * 1e3);
            }
            println!();
        }
        println!();
        for (i, (name, _)) in passes.iter().enumerate() {
            println!(
                "{name:<9} T removed {:>9} ({:>5.1}% of Rand) · total {:>9.1} ms",
                removed[i],
                100.0 * removed[i] as f64 / removed[0] as f64,
                time[i].as_secs_f64() * 1e3
            );
        }
    }

    /// Algorithm 2 (Appendix C of the tzap paper): hwb-like circuits whose
    /// largest parity grows linearly with gate count.
    ///
    /// Stage `i` uses worker `w = q[1 + i mod (n − 1)]`:
    /// `h w; cx w,q0; rz(π/4) q0; cx q0,w; rz(−π/4) w` — 5 gates per stage.
    /// Qubit 0 accumulates one fresh Hadamard variable per stage (its parity
    /// has weight `i + 2` at stage `i`), and the second CNOT copies the
    /// accumulator's *previous* value into the worker. So each stage's
    /// `rz(−π/4)` acts on exactly the parity of the previous stage's
    /// `rz(π/4)`, and an exact analysis cancels every rotation but the first
    /// and the last — but only by comparing parities of weight up to `s + 1`.
    pub(crate) fn hwb_like(num_qubits: usize, stages: usize) -> Circuit {
        assert!(num_qubits >= 2, "Algorithm 2 needs n ≥ 2");
        let quarter = PI / 4.0;
        let mut c = Circuit::new(num_qubits);
        for i in 0..stages {
            let w = (1 + i % (num_qubits - 1)) as Qubit;
            c.apply(Gate::h(w));
            c.apply(cx(w, 0));
            c.apply(Gate::rz(quarter, 0));
            c.apply(cx(0, w));
            c.apply(Gate::rz(-quarter, w));
        }
        c
    }

    fn rotations(c: &Circuit) -> usize {
        c.gates
            .iter()
            .filter(|g| rotation_of(g).is_some())
            .count()
    }

    /// Exact analyses keep 2 rotations. The generation-stamped analysis
    /// loses one fold each time the round-robin wraps back to a worker whose
    /// generation is behind the accumulator's (under either policy only the
    /// control-ahead case arises): `2 + 2·⌊(s − 1)/(n − 1)⌋` rotations.
    #[test]
    fn hwb_like_family_characterization() {
        for n in [2, 3, 5, 13] {
            for s in [1, 2, 5, 20, 100] {
                let c = hwb_like(n, s);
                assert_eq!(c.gates.len(), 5 * s);
                if n <= 5 && s <= 5 {
                    let out = phase_fold_det(&c, ResetPolicy::ForgetTarget);
                    assert!(circuits_equiv(&c, &out, 1e-9), "n={n} s={s}");
                }
                assert_eq!(rotations(&phase_fold_rand(&c)), 2, "Rand n={n} s={s}");
                assert_eq!(
                    rotations(&crate::phase_fold_global_expr::phase_fold_global_expr(&c)),
                    2,
                    "GlobalExpr n={n} s={s}"
                );
                assert_eq!(
                    rotations(&crate::phase_fold_frame::phase_fold_frame(&c)),
                    2,
                    "Frame n={n} s={s}"
                );
                let expected = 2 + 2 * ((s - 1) / (n - 1));
                for policy in POLICIES {
                    assert_eq!(
                        rotations(&phase_fold_det(&c, policy)),
                        expected,
                        "{policy:?} n={n} s={s}"
                    );
                }
            }
        }
    }

    /// Fig. 6 on our passes: runtime and surviving rotations on the hwb-like
    /// family. Run with
    /// `cargo test --release phase_fold_det::tests::hwb_like_scaling -- --ignored --nocapture`.
    /// Set `TZAP_HWB_QUBITS` to change the width (default 12), and
    /// `TZAP_HWB_DUMP=<dir>` to also write each circuit as
    /// `<dir>/hwb_like_n<n>_s<stages>.qasm` for use with the CLI.
    #[test]
    #[ignore]
    fn hwb_like_scaling() {
        let n: usize = std::env::var("TZAP_HWB_QUBITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(12);
        // GlobalExpr stores one weight-i key per stage, so its memory is
        // quadratic too; stop it at 8k stages (~256 MB of keys).
        const EXPR_MAX_STAGES: usize = 8_000;
        println!(
            "\nhwb-like, n = {n}\n{:>8} {:>9} | {:>10} {:>5} | {:>10} {:>5} | {:>10} {:>5} | {:>10} {:>5} | {:>10} {:>5}",
            "stages", "gates", "Rand ms", "rot", "Det ms", "rot", "Lower ms", "rot", "Frame ms", "rot", "Expr ms", "rot"
        );
        for s in [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 48_000] {
            let c = hwb_like(n, s);
            if let Ok(dir) = std::env::var("TZAP_HWB_DUMP") {
                std::fs::create_dir_all(&dir)
                    .unwrap_or_else(|e| panic!("create {dir}: {e}"));
                let path = std::path::Path::new(&dir).join(format!("hwb_like_n{n}_s{s}.qasm"));
                std::fs::write(&path, c.to_qasm())
                    .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
            }
            let ms = |d: Duration| d.as_secs_f64() * 1e3;
            let (rand, tr) = timed(|| phase_fold_rand(&c));
            let (det, td) = timed(|| phase_fold_det(&c, ResetPolicy::ForgetTarget));
            let (lower, tl) = timed(|| phase_fold_det(&c, ResetPolicy::ForgetLower));
            let (frame, tf) = timed(|| crate::phase_fold_frame::phase_fold_frame(&c));
            let expr = (s <= EXPR_MAX_STAGES)
                .then(|| timed(|| crate::phase_fold_global_expr::phase_fold_global_expr(&c)));
            let (expr_ms, expr_rot) = match &expr {
                Some((out, t)) => (format!("{:.2}", ms(*t)), rotations(out).to_string()),
                None => ("skipped".to_owned(), "-".to_owned()),
            };
            println!(
                "{:>8} {:>9} | {:>10.2} {:>5} | {:>10.2} {:>5} | {:>10.2} {:>5} | {:>10.2} {:>5} | {:>10} {:>5}",
                s,
                c.gates.len(),
                ms(tr),
                rotations(&rand),
                ms(td),
                rotations(&det),
                ms(tl),
                rotations(&lower),
                ms(tf),
                rotations(&frame),
                expr_ms,
                expr_rot
            );
        }
    }
}
