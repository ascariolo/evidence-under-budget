"""Tests of the frozen seed-count function. ALL numeric inputs are SYNTHETIC test values,
chosen only to exercise the code paths. None is, or is derived from, pilot or experimental data."""

from __future__ import annotations

import ast, inspect, math, os, sys
from dataclasses import FrozenInstanceError, replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.elasticity.seed_count as S
from experiments.elasticity import protocol as EP

BUD = ("tight", "medium", "loose")


def fixed(delta=0.05, dD=0.2, EQ=1.5):
    return S.FixedQuantities(delta={b: delta for b in BUD}, delta_D=dD, E_Q=EQ)


def pilot(pi=0.5, M2=0.04, rr=0.5, rl=0.3, sS=0.05):
    return S.PilotQuantities(pi={b: pi for b in BUD}, M2={b: M2 for b in BUD},
                             rho_red={b: rr for b in BUD}, rho_lvl={b: rl for b in BUD}, sigma_S=sS)


def stops(fn):
    try:
        fn()
    except S.SeedCountStop:
        return True
    return False


# --- protocol state -------------------------------------------------------- #
def test_regime_r2_and_before_after_distinction():
    R = EP.SEED_COUNT_RULE
    assert R["regime"] == "R2" and R["role_of_pilot"] == "PARAMETER ESTIMATION ONLY"
    assert "frozen" in R["before_pilot"] and "substituted" in R["after_pilot"]
    assert set(R["edge_case_rules"]) == set("abcdefghi") | {"general"}
    assert "arbitrary numerical value" in EP.GENERAL_EDGE_INVARIANT


def test_t8_sesoi_is_probability_difference_not_slope():
    assert "NOT a slope of 0.03" in EP.SEED_COUNT_RULE["formulas"]["T8"]
    assert S.T8_PROB_SESOI == 0.03


def test_populations_fixed():
    assert "0-59" in EP.SEED_COUNT_RULE["populations"]["B"]
    assert "independent of n" in EP.SEED_COUNT_RULE["populations"]["B"]


# --- determinism and floor ------------------------------------------------- #
def test_map_is_deterministic():
    assert S.n_final(fixed(), pilot()) == S.n_final(fixed(), pilot())


def test_floor_60_never_reduced():
    out = S.n_final(fixed(delta=0.5), pilot(pi=0.9, M2=0.5, sS=0.001))   # synthetic: very easy to detect
    assert out["n_required"] < 60 and out["n_final"] == 60


def test_above_300_stops():
    assert stops(lambda: S.n_final(fixed(delta=0.001), pilot()))


def test_friedman_power_monotone():
    p = [S.friedman_power(n, 0.1, 0.3) for n in (10, 50, 150, 300)]
    assert p == sorted(p)


# --- edge-case rules -------------------------------------------------------- #
def test_rule_a_degenerate_variance_stops_and_term_not_dropped():
    assert stops(lambda: S.sigma2(fixed(delta=0.3), pilot(pi=0.5, M2=0.04), "tight"))   # 0.02 - 0.09 <= 0


def test_rule_b_undefined_pearson_stops():
    assert stops(lambda: S._pearson([1.0, 1.0, 1.0], [0.1, 0.2, 0.3], "synthetic"))


def test_rule_c_no_clamping():
    lo = -1.0 / (S.K - 1)
    assert stops(lambda: S._check_rho_lvl(lo - 1e-6, "tight"))         # materially below
    assert S._check_rho_lvl(lo - 1e-11, "tight") == lo - 1e-11         # within tolerance: used UNCHANGED
    assert stops(lambda: S._check_rho_lvl(1.0, "tight"))               # 1 - rho <= 0
    assert stops(lambda: S._check_rho_lvl(1.0 + 1e-11, "tight"))       # within tol but invalid
    assert S._check_rho_lvl(0.999999, "tight") == 0.999999             # no n_t = 1 shortcut, no clamp


def test_rule_d_search_bounds():
    src = inspect.getsource(S.n_friedman)
    assert "range(1, N_MAX + 1)" in src and S.N_MAX == 300 and S.N_MIN == 60


def test_rule_e_f_constants():
    assert S.NONZERO_TOL == 1e-12
    assert "ddof=1" in inspect.getsource(S.pilot_quantities)


def test_rule_i_unproven_ilp_stops():
    recs = [{"seed": s, "beta_target": b, "redundancy_level": g, "budget_level": bud,
             "objective_score": 1.0, "ilp_proven_optimal": True}
            for s in range(3) for b in EP.BETA_LEVELS for g in EP.REDUNDANCY_LEVELS for bud in BUD]
    assert S.fixed_opt_b(recs, range(3))["tight"]["delta"] == 0.03
    recs[0]["ilp_proven_optimal"] = False
    assert stops(lambda: S.fixed_opt_b(recs, range(3)))


def test_general_invariant_degenerate_inputs_stop():
    assert stops(lambda: S.n_t8(fixed(dD=0.0), pilot()))
    assert stops(lambda: S.n_t8(fixed(EQ=0.0), pilot()))
    assert stops(lambda: S.n_t8(fixed(), pilot(sS=0.0)))
    assert stops(lambda: S.n_final(fixed(delta=0.0), pilot()))
    assert stops(lambda: S.n_final(fixed(), pilot(rr=float("nan"))))


# --- locked percentile convention (synthetic values) ------------------------ #
def test_percentile_type7_matches_hand_computation():
    x = [float(i) ** 1.5 for i in range(60)]                 # synthetic, non-uniform spacing
    xs = sorted(x)
    assert math.isclose(S.percentile_type7(x, 1, 10), xs[5] + 0.9 * (xs[6] - xs[5]), rel_tol=0, abs_tol=1e-12)
    assert math.isclose(S.percentile_type7(x, 9, 10), xs[53] + 0.1 * (xs[54] - xs[53]), rel_tol=0, abs_tol=1e-12)
    assert S.percentile_type7([3.0, 1.0, 2.0], 1, 2) == 2.0       # integer h -> exact order statistic
    assert S.percentile_type7(list(reversed(x)), 1, 10) == S.percentile_type7(x, 1, 10)   # order-invariant


def test_percentile_convention_is_explicit_in_protocol():
    P = EP.PERCENTILE_CONVENTION
    assert P["N"] == 60 and "h = (N-1)*p" in P["definition"] and "0-based" in P["definition"]
    assert "5.9" in P["P10"] and "53.1" in P["P90"] and "NO library default" in P["name"]
    src = inspect.getsource(S)
    assert "np.percentile" not in src and "np.quantile" not in src


def test_delta_d_requires_exactly_60_baseline_seeds():
    D = {(s, b): 0.1 + 0.01 * s for s in range(60) for b in EP.BETA_LEVELS}     # synthetic D
    assert math.isclose(S.fixed_delta_D(D, range(60)), 0.472, abs_tol=1e-12)
    assert stops(lambda: S.fixed_delta_D(D, range(59)))


# --- rule H (synthetic dispositions) ----------------------------------------- #
def _grid(seeds):
    return [(s, b, g, bud, m) for s in seeds for b in EP.BETA_LEVELS for g in EP.REDUNDANCY_LEVELS
            for bud in BUD for m in ("greedy_objective", "greedy_token_aware")]


def test_rule_h_rejection_alone_does_not_stop():
    acc = [s for s in EP.VARIANCE_PILOT_SEEDS if s != 2007]
    out = S.assert_pilot_sufficient(acc, [2007], _grid(acc))
    assert out["n_P"] == 39 and out["rejected_reported"] == [2007]


def test_rule_h_declared_conditions_stop():
    allp = list(EP.VARIANCE_PILOT_SEEDS)
    assert stops(lambda: S.assert_pilot_sufficient(allp[:-1], [], _grid(allp[:-1])))                # H1
    assert stops(lambda: S.assert_pilot_sufficient(allp, [], _grid(allp)[:-1]))                      # H2
    assert stops(lambda: S.assert_pilot_sufficient(allp[:1], allp[1:], _grid(allp[:1])))             # H3


def test_rule_h_imposes_no_arbitrary_minimum():
    two = allp2 = list(EP.VARIANCE_PILOT_SEEDS)[:2]
    assert S.assert_pilot_sufficient(two, list(EP.VARIANCE_PILOT_SEEDS)[2:], _grid(two))["n_P"] == 2
    assert [c["id"] for c in EP.RULE_H_CONDITIONS] == ["H1", "H2", "H3", "H4"]


# --- separation: pilot quantities cannot alter protocol definitions ---------- #
def test_fixed_quantities_are_immutable():
    f = fixed()
    try:
        f.delta_D = 9.0
    except FrozenInstanceError:
        return
    raise AssertionError("FixedQuantities is mutable")


def test_fixed_layer_never_references_pilot_names():
    tree = ast.parse(inspect.getsource(S))
    pilot_names = {"PilotQuantities", "pilot", "pilot_quantities", "VARIANCE_PILOT_SEEDS", "sigma_S", "rho_lvl", "rho_red"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("fixed_"):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | \
                    {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)} | \
                    {a.arg for a in node.args.args}
            assert not names & pilot_names, (node.name, names & pilot_names)


def test_pilot_layer_never_references_sesoi_or_formula_constants():
    tree = ast.parse(inspect.getsource(S))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pilot_quantities")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    for forbidden in ("delta", "SESOI_FRACTION_OF_OPTIMUM", "T8_PROB_SESOI", "ALPHA", "POWER", "POLY_LINEAR",
                      "POLY_QUADRATIC", "ARE", "N_MIN", "N_MAX", "FixedQuantities"):
        assert forbidden not in names, forbidden


def test_changing_pilot_inputs_cannot_change_fixed_quantities():
    f = fixed()
    before = (dict(f.delta), f.delta_D, f.E_Q)
    for p in (pilot(), pilot(pi=0.9, rl=-0.1, sS=0.2)):
        try:
            S.n_final(f, p)
        except S.SeedCountStop:
            pass
    assert (dict(f.delta), f.delta_D, f.E_Q) == before
    assert inspect.signature(S.n_final).parameters.keys() == {"fixed", "pilot"}


def test_protocol_formula_text_has_no_pilot_seed_reference():
    blob = str({k: v for k, v in EP.SEED_COUNT_RULE.items() if k in ("formulas", "fixed_quantities_from_B", "constants")})
    assert "2000" not in blob and "2039" not in blob


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                fails += 1; print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{fails} failure(s)"); raise SystemExit(1 if fails else 0)
