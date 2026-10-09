use std::collections::BTreeMap;
use std::f64::consts::PI;

use crate::circuit::{Circuit, Gate, Qubit};
use crate::pass::Pass;

/// A parity expression: XOR of a set of variables, optionally complemented.
/// Two expressions are equivalent iff they have the same variable set and
/// the same `negated` flag.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ParityExpr {
    /// Sorted variable indices whose XOR forms this expression.
    vars: Vec<usize>,
    /// Whether the expression is bitwise-complemented (from X gates).
    negated: bool,
}

impl ParityExpr {
    fn new(var: usize) -> Self {
        ParityExpr {
            vars: vec![var],
            negated: false,
        }
    }

    fn fresh(var: usize) -> Self {
        Self::new(var)
    }

    /// XOR two parity expressions (symmetric difference of variable sets).
    fn xor(&self, other: &ParityExpr) -> ParityExpr {
        let mut result = Vec::new();
        let (a, b) = (&self.vars, &other.vars);
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            if a[i] < b[j] {
                result.push(a[i]);
                i += 1;
            } else if a[i] > b[j] {
                result.push(b[j]);
                j += 1;
            } else {
                // Same variable in both — cancels out in XOR
                i += 1;
                j += 1;
            }
        }
        result.extend_from_slice(&a[i..]);
        result.extend_from_slice(&b[j..]);
        ParityExpr {
            vars: result,
            negated: self.negated ^ other.negated,
        }
    }

    /// Complement (NOT) — flips the negated flag.
    fn complement(&self) -> ParityExpr {
        ParityExpr {
            vars: self.vars.clone(),
            negated: !self.negated,
        }
    }
}

struct Group {
    /// Accumulated phase in canonical frame (negated=false).
    /// Contributions from sites on the complement parity are subtracted.
    angle: f64,
    last_idx: usize,
    last_qubit: Qubit,
    /// True when `last_idx` sits on the complement of the canonical parity.
    /// On emission, the rotation is negated to compensate.
    last_sign: bool,
    indices: Vec<usize>,
}

/// Merges T/Rz gates that act on the same linear parity of qubits, tracked
/// via exact symbolic parity expressions rather than randomized sampling.
pub struct PhaseFoldGlobalExpr;

impl Pass for PhaseFoldGlobalExpr {
    fn name(&self) -> &str {
        "Phase folding (expr)"
    }
    fn run(&self, circuit: &Circuit) -> Circuit {
        phase_fold_global_expr(circuit)
    }
}

/// Free-function form of [`PhaseFoldGlobalExpr`], for use outside a [`Pass`] pipeline.
pub fn phase_fold_global_expr(circuit: &Circuit) -> Circuit {
    let n = circuit.num_qubits;
    let mut next_var = 0usize;
    let mut fresh = || {
        let v = next_var;
        next_var += 1;
        ParityExpr::fresh(v)
    };
    let mut qubits: Vec<ParityExpr> = (0..n).map(|_| fresh()).collect();

    let mut groups: BTreeMap<ParityExpr, Group> = BTreeMap::new();
    let mut skip = vec![false; circuit.gates.len()];
    let mut emit_at: Vec<Option<(Qubit, f64)>> = vec![None; circuit.gates.len()];

    for (idx, gate) in circuit.gates.iter().enumerate() {
        match gate {
            Gate::t(q) => record_phase(&qubits, *q, PI / 4.0, idx, &mut groups),
            Gate::tdg(q) => record_phase(&qubits, *q, -PI / 4.0, idx, &mut groups),
            Gate::s(q) => record_phase(&qubits, *q, PI / 2.0, idx, &mut groups),
            Gate::sdg(q) => record_phase(&qubits, *q, -PI / 2.0, idx, &mut groups),
            Gate::z(q) => record_phase(&qubits, *q, PI, idx, &mut groups),
            Gate::rz(theta, q) => record_phase(&qubits, *q, *theta, idx, &mut groups),
            Gate::h(q) => {
                qubits[*q as usize] = fresh();
            }
            Gate::x(q) => {
                qubits[*q as usize] = qubits[*q as usize].complement();
            }
            Gate::cnot { control, target } => {
                let ctrl = qubits[*control as usize].clone();
                qubits[*target as usize] = qubits[*target as usize].xor(&ctrl);
            }
            // CZ and CCZ are diagonal and do not change tracked parities. Keep the
            // gate in the output without introducing an artificial analysis boundary.
            Gate::cz { .. } | Gate::ccz { .. } => {}
            Gate::ccx {
                control1: _,
                control2: _,
                target,
            } => {
                // AND-based parity — opaque, just refresh the target.
                qubits[*target as usize] = fresh();
            }
            Gate::measure { qubit, .. } | Gate::reset(qubit) => {
                qubits[*qubit as usize] = fresh();
            }
        }
    }

    // Finalize skip/emit from groups.
    for g in groups.values() {
        for &i in &g.indices {
            skip[i] = true;
        }
        let emit_angle = if g.last_sign { -g.angle } else { g.angle };
        if !angle_is_zero(emit_angle) {
            skip[g.last_idx] = false;
            emit_at[g.last_idx] = Some((g.last_qubit, emit_angle));
        }
    }

    // Reconstruct circuit.
    let mut output = Circuit::with_cbits(n, circuit.num_cbits);
    for (idx, gate) in circuit.gates.iter().enumerate() {
        if skip[idx] {
            continue;
        }
        if let Some((qubit, angle)) = emit_at[idx] {
            emit_rotation(&mut output, qubit, angle);
        } else {
            output.apply(gate.clone());
        }
    }

    output
}

fn record_phase(
    qubits: &[ParityExpr],
    q: Qubit,
    angle: f64,
    idx: usize,
    groups: &mut BTreeMap<ParityExpr, Group>,
) {
    let parity = &qubits[q as usize];
    let is_complement = parity.negated;
    // Canonicalize: keys always have negated=false. Contributions from the
    // complement side are folded in with a sign flip — matches the hash
    // version's complementary-parity lookup so rotations bracketed by X
    // (or any CNOT·X·CNOT pattern) merge across the negation.
    let key = ParityExpr {
        vars: parity.vars.clone(),
        negated: false,
    };
    let signed_angle = if is_complement { -angle } else { angle };

    if let Some(g) = groups.get_mut(&key) {
        g.angle += signed_angle;
        g.last_idx = idx;
        g.last_qubit = q;
        g.last_sign = is_complement;
        g.indices.push(idx);
        return;
    }

    groups.insert(
        key,
        Group {
            angle: signed_angle,
            last_idx: idx,
            last_qubit: q,
            last_sign: is_complement,
            indices: vec![idx],
        },
    );
}

fn angle_is_zero(angle: f64) -> bool {
    let n = angle.rem_euclid(2.0 * PI);
    n < 1e-6 || (2.0 * PI - n) < 1e-6
}

fn emit_rotation(output: &mut Circuit, q: Qubit, angle: f64) {
    let n = angle.rem_euclid(2.0 * PI);
    if angle_is_zero(n) {
        return;
    }
    let quarter = PI / 4.0;
    let k = (n / quarter).round();
    // Use 1e-6 tolerance to handle floating-point drift from summing many π/4 multiples.
    if (n - k * quarter).abs() < 1e-6 {
        let k = k as u32 % 8;
        match k {
            0 => {}
            1 => output.apply(Gate::t(q)),
            2 => output.apply(Gate::s(q)),
            3 => {
                output.apply(Gate::s(q));
                output.apply(Gate::t(q));
            }
            4 => output.apply(Gate::z(q)),
            5 => {
                output.apply(Gate::z(q));
                output.apply(Gate::t(q));
            }
            6 => output.apply(Gate::sdg(q)),
            7 => output.apply(Gate::tdg(q)),
            _ => unreachable!(),
        }
    } else {
        output.apply(Gate::rz(n, q));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unitary::circuits_equiv;

    const TOL: f64 = 1e-10;

    fn count_phase_gates(c: &Circuit) -> usize {
        c.gates
            .iter()
            .filter(|g| {
                matches!(
                    g,
                    Gate::t(_)
                        | Gate::tdg(_)
                        | Gate::s(_)
                        | Gate::sdg(_)
                        | Gate::z(_)
                        | Gate::rz(..)
                )
            })
            .count()
    }

    fn count_t_gates(c: &Circuit) -> usize {
        c.gates
            .iter()
            .filter(|g| matches!(g, Gate::t(_) | Gate::tdg(_)))
            .count()
    }

    fn run(c: &Circuit) -> Circuit {
        phase_fold_global_expr(c)
    }

    // --- Expr equivalence tests ---

    #[test]
    fn expr_fresh_not_equal() {
        let a = ParityExpr::fresh(0);
        let b = ParityExpr::fresh(1);
        assert_ne!(a, b);
    }

    #[test]
    fn expr_same_var_equal() {
        let a = ParityExpr::fresh(0);
        let b = ParityExpr::fresh(0);
        assert_eq!(a, b);
    }

    #[test]
    fn expr_xor_self_is_empty() {
        let a = ParityExpr::fresh(0);
        let r = a.xor(&a);
        assert!(r.vars.is_empty());
        assert!(!r.negated);
    }

    #[test]
    fn expr_xor_commutative() {
        let a = ParityExpr::fresh(0);
        let b = ParityExpr::fresh(1);
        let ab = a.xor(&b);
        let ba = b.xor(&a);
        assert_eq!(ab, ba);
    }

    #[test]
    fn expr_complement_not_equal() {
        let a = ParityExpr::fresh(0);
        let b = a.complement();
        assert_ne!(a, b);
    }

    #[test]
    fn expr_double_complement_equal() {
        let a = ParityExpr::fresh(0);
        let b = a.complement().complement();
        assert_eq!(a, b);
    }

    #[test]
    fn expr_xor_associative() {
        let a = ParityExpr::fresh(0);
        let b = ParityExpr::fresh(1);
        let c = ParityExpr::fresh(2);
        let ab_c = a.xor(&b).xor(&c);
        let a_bc = a.xor(&b.xor(&c));
        assert_eq!(ab_c, a_bc);
    }

    // --- Circuit-level tests ---

    #[test]
    fn two_t_merge_to_s() {
        let mut c = Circuit::new(1);
        c.apply(Gate::t(0));
        c.apply(Gate::t(0));
        let out = run(&c);
        assert_eq!(count_phase_gates(&out), 1);
        assert_eq!(count_t_gates(&out), 0);
        assert!(circuits_equiv(&c, &out, TOL));
    }

    #[test]
    fn phases_fold_through_ccz_on_every_operand() {
        for q in 0..3 {
            let mut c = Circuit::new(3);
            c.apply(Gate::t(q));
            c.apply(Gate::ccz {
                control1: 0,
                control2: 1,
                target: 2,
            });
            c.apply(Gate::t(q));

            let out = run(&c);

            assert!(matches!(out.gates[0], Gate::ccz { .. }), "operand {q}");
            assert!(matches!(out.gates[1], Gate::s(p) if p == q), "operand {q}");
            assert!(circuits_equiv(&c, &out, TOL), "operand {q}");
        }
    }

    #[test]
    fn t_tdg_cancel() {
        let mut c = Circuit::new(1);
        c.apply(Gate::t(0));
        c.apply(Gate::tdg(0));
        let out = run(&c);
        assert_eq!(count_phase_gates(&out), 0);
        assert!(circuits_equiv(&c, &out, TOL));
    }

    #[test]
    fn s_s_merge_to_z() {
        let mut c = Circuit::new(1);
        c.apply(Gate::s(0));
        c.apply(Gate::s(0));
        let out = run(&c);
        assert_eq!(count_phase_gates(&out), 1);
        assert!(circuits_equiv(&c, &out, TOL));
    }

    #[test]
    fn four_t_merge_to_z() {
        let mut c = Circuit::new(1);
        for _ in 0..4 {
            c.apply(Gate::t(0));
        }
        let out = run(&c);
        assert_eq!(count_phase_gates(&out), 1);
        assert_eq!(count_t_gates(&out), 0);
        assert!(circuits_equiv(&c, &out, TOL));
    }

    #[test]
    fn eight_t_cancel() {
        let mut c = Circuit::new(1);
        for _ in 0..8 {
            c.apply(Gate::t(0));
        }
        let out = run(&c);
        assert_eq!(count_phase_gates(&out), 0);
        assert!(circuits_equiv(&c, &out, TOL));
    }

    #[test]
    fn h_blocks_merge() {
        let mut c = Circuit::new(1);
        c.apply(Gate::t(0));
        c.apply(Gate::h(0));
        c.apply(Gate::t(0));
        let out = run(&c);
        assert_eq!(count_t_gates(&out), 2);
        assert!(circuits_equiv(&c, &out, TOL));
    }

    #[test]
    fn cnot_enables_cross_qubit_merge() {
        let mut c = Circuit::new(2);
        c.apply(Gate::t(0));
        c.apply(Gate::cnot {
            control: 0,
            target: 1,
        });
        c.apply(Gate::t(1));
        // After CNOT, q1 parity = q0 XOR q1. Not the same as q0.
        let out = run(&c);
        assert_eq!(count_t_gates(&out), 2);
        assert!(circuits_equiv(&c, &out, TOL));
    }

    #[test]
    fn cnot_same_parity_merge() {
        // q0 = v0, q1 = v1
        // CNOT 0->1: q1 = v0 ^ v1
        // CNOT 0->1 again: q1 = v0 ^ v1 ^ v0 = v1
        // So T on q0, CNOT, CNOT, T on q1 should NOT merge (v0 != v1).
        // But: q0 = v0 and q1 after double CNOT = v1, so no merge.
        let mut c = Circuit::new(2);
        c.apply(Gate::t(0));
        c.apply(Gate::cnot {
            control: 0,
            target: 1,
        });
        c.apply(Gate::cnot {
            control: 0,
            target: 1,
        });
        c.apply(Gate::t(1));
        let out = run(&c);
        // v0 and v1 are different, no merge
        assert_eq!(count_t_gates(&out), 2);
        assert!(circuits_equiv(&c, &out, TOL));
    }

    #[test]
    fn x_folds_t_t_to_identity() {
        // T · X · T = X (since X · T · X = Tdg), so the two T contributions
        // cancel in the canonical frame and only X survives.
        let mut c = Circuit::new(1);
        c.apply(Gate::t(0));
        c.apply(Gate::x(0));
        c.apply(Gate::t(0));
        let out = run(&c);
        assert_eq!(count_t_gates(&out), 0);
        assert!(circuits_equiv(&c, &out, TOL));
    }

    #[test]
    fn x_folds_t_tdg_to_s() {
        // T · X · Tdg = X · S (in canonical frame the two contributions add
        // to π/2, then negate because the second site is on ¬canonical).
        let mut c = Circuit::new(1);
        c.apply(Gate::t(0));
        c.apply(Gate::x(0));
        c.apply(Gate::tdg(0));
        let out = run(&c);
        assert_eq!(count_t_gates(&out), 0);
        assert!(circuits_equiv(&c, &out, TOL));
    }

    #[test]
    fn double_x_allows_merge() {
        let mut c = Circuit::new(1);
        c.apply(Gate::t(0));
        c.apply(Gate::x(0));
        c.apply(Gate::x(0));
        c.apply(Gate::t(0));
        let out = run(&c);
        assert_eq!(count_t_gates(&out), 0); // merged to S
        assert!(circuits_equiv(&c, &out, TOL));
    }

    #[test]
    fn rz_merge() {
        let mut c = Circuit::new(1);
        c.apply(Gate::rz(0.5, 0));
        c.apply(Gate::rz(0.3, 0));
        let out = run(&c);
        assert_eq!(count_phase_gates(&out), 1);
        assert!(circuits_equiv(&c, &out, TOL));
    }

    #[test]
    fn z_z_cancel() {
        let mut c = Circuit::new(1);
        c.apply(Gate::z(0));
        c.apply(Gate::z(0));
        let out = run(&c);
        assert_eq!(count_phase_gates(&out), 0);
        assert!(circuits_equiv(&c, &out, TOL));
    }

    #[test]
    fn ccx_refreshes_target() {
        let mut c = Circuit::new(3);
        c.apply(Gate::t(2));
        c.apply(Gate::ccx {
            control1: 0,
            control2: 1,
            target: 2,
        });
        c.apply(Gate::t(2));
        let out = run(&c);
        // CCX refreshes target parity, so no merge
        assert_eq!(count_t_gates(&out), 2);
    }

    #[test]
    fn phase_fold_preserves_semantics_3q() {
        let mut c = Circuit::new(3);
        c.apply(Gate::t(0));
        c.apply(Gate::cnot {
            control: 0,
            target: 1,
        });
        c.apply(Gate::t(1));
        c.apply(Gate::cnot {
            control: 1,
            target: 2,
        });
        c.apply(Gate::s(2));
        c.apply(Gate::t(0));
        let out = run(&c);
        assert!(circuits_equiv(&c, &out, TOL));
    }

    #[test]
    fn agrees_with_hash_version_simple() {
        use crate::phase_fold_rand::phase_fold_rand;
        let mut c = Circuit::new(2);
        c.apply(Gate::t(0));
        c.apply(Gate::t(0));
        c.apply(Gate::cnot {
            control: 0,
            target: 1,
        });
        c.apply(Gate::t(1));
        c.apply(Gate::tdg(1));
        c.apply(Gate::s(0));
        let hash_out = phase_fold_rand(&c);
        let expr_out = run(&c);
        assert_eq!(hash_out.gates.len(), expr_out.gates.len());
        assert!(circuits_equiv(&hash_out, &expr_out, TOL));
    }

    // --- measurement / reset barrier tests ---

    #[test]
    fn measure_blocks_t_t_merge_expr() {
        let mut c = Circuit::with_cbits(1, 1);
        c.apply(Gate::t(0));
        c.apply(Gate::measure { qubit: 0, cbit: 0 });
        c.apply(Gate::t(0));
        let opt = run(&c);
        assert_eq!(count_t_gates(&opt), 2);
        assert_eq!(opt.gates.len(), 3);
        assert!(matches!(&opt.gates[1], Gate::measure { qubit: 0, cbit: 0 }));
        assert!(opt.has_measurement());
    }

    #[test]
    fn reset_blocks_t_tdg_cancel_expr() {
        let mut c = Circuit::new(1);
        c.apply(Gate::t(0));
        c.apply(Gate::reset(0));
        c.apply(Gate::tdg(0));
        let opt = run(&c);
        assert_eq!(opt.gates.len(), 3);
        assert!(matches!(&opt.gates[1], Gate::reset(0)));
    }

    #[test]
    fn measure_on_other_qubit_does_not_block_expr() {
        let mut c = Circuit::with_cbits(2, 1);
        c.apply(Gate::t(0));
        c.apply(Gate::measure { qubit: 1, cbit: 0 });
        c.apply(Gate::t(0));
        let opt = run(&c);
        assert!(opt.gates.iter().any(|g| matches!(g, Gate::s(0))));
        assert!(
            opt.gates
                .iter()
                .any(|g| matches!(g, Gate::measure { qubit: 1, cbit: 0 }))
        );
        // No T survives.
        assert_eq!(count_t_gates(&opt), 0);
    }

    #[test]
    fn measure_preserved_with_no_rotations_expr() {
        let mut c = Circuit::with_cbits(2, 2);
        c.apply(Gate::h(0));
        c.apply(Gate::cnot {
            control: 0,
            target: 1,
        });
        c.apply(Gate::measure { qubit: 0, cbit: 0 });
        c.apply(Gate::measure { qubit: 1, cbit: 1 });
        let opt = run(&c);
        assert_eq!(opt.gates.len(), 4);
        assert_eq!(opt.num_cbits, 2);
        assert!(opt.has_measurement());
    }

    #[test]
    fn agrees_with_hash_version_3q() {
        use crate::phase_fold_rand::phase_fold_rand;
        let mut c = Circuit::new(3);
        c.apply(Gate::h(0));
        c.apply(Gate::cnot {
            control: 0,
            target: 1,
        });
        c.apply(Gate::t(0));
        c.apply(Gate::t(1));
        c.apply(Gate::cnot {
            control: 1,
            target: 2,
        });
        c.apply(Gate::s(2));
        c.apply(Gate::x(0));
        c.apply(Gate::t(0));
        c.apply(Gate::x(0));
        c.apply(Gate::t(0));
        c.apply(Gate::h(1));
        c.apply(Gate::t(1));
        let hash_out = phase_fold_rand(&c);
        let expr_out = run(&c);
        // Both passes canonicalize parity over X (negation), so they should
        // produce equal-sized outputs that are semantically equivalent.
        assert_eq!(hash_out.gates.len(), expr_out.gates.len());
        assert!(circuits_equiv(&c, &hash_out, TOL));
        assert!(circuits_equiv(&c, &expr_out, TOL));
    }
}
