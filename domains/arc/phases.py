"""
ARC-specific wake phases for the Universal Learning Loop.

These phases implement domain-specific search strategies for ARC grids:
object decomposition, cross-reference, local rules, procedural learning,
synthesis, conditional search, color fix, and input-prediction correction.

Each phase is a standalone function: (WakeContext, ARCEnv) -> Optional[str].
They are returned by ARCEnv.domain_wake_phases() and run after the
domain-agnostic exhaustive enumeration phase.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Optional, TYPE_CHECKING

from core.types import Primitive, Program, ScoredProgram

if TYPE_CHECKING:
    from core.learner import WakeContext
    from .environment import ARCEnv

logger = logging.getLogger(__name__)


# =============================================================================
# Phase: Object Decomposition
# =============================================================================

def phase_object_decomposition(ctx: "WakeContext", env: "ARCEnv") -> Optional[str]:
    """Try per-object transforms via connected components."""
    if ctx.solved:
        return None
    t = time.time()
    result = env.try_object_decomposition(ctx.task, ctx.all_prims) if ctx.budget_ok() else None
    if result is not None:
        name, fn = result
        sp = ctx.evaluate(Program(root=name))
        ctx.n_evals += 1
        ctx.update_pareto(sp)
        ctx.update_best(sp)
        ctx.enum_candidates.append(sp)
    logger.debug(f"  [wake] Object decomp: {time.time()-t:.2f}s")
    return "object decomposition" if ctx.solved else None


# =============================================================================
# Phase: For-Each Object
# =============================================================================

def phase_for_each_object(ctx: "WakeContext", env: "ARCEnv") -> Optional[str]:
    """Apply top-K enumeration candidates per-object."""
    if ctx.solved or not ctx.enum_candidates or not ctx.budget_ok():
        return None
    t = time.time()
    result = env.try_for_each_object(ctx.task, ctx.enum_candidates, top_k=10)
    if result is not None:
        name, fn = result
        sp = ctx.evaluate(Program(root=name))
        ctx.n_evals += 1
        ctx.update_pareto(sp)
        ctx.update_best(sp)
        ctx.enum_candidates.append(sp)
    logger.debug(f"  [wake] For-each-object: {time.time()-t:.2f}s")
    return "for-each-object" if ctx.solved else None


# =============================================================================
# Phase: Cross-Reference
# =============================================================================

def phase_cross_reference(ctx: "WakeContext", env: "ARCEnv") -> Optional[str]:
    """Cross-reference: one grid part informs another."""
    if ctx.solved:
        return None
    t = time.time()
    result = env.try_cross_reference(ctx.task, ctx.all_prims)
    if result is not None:
        name, fn = result
        sp = ctx.evaluate(Program(root=name))
        ctx.n_evals += 1
        ctx.update_pareto(sp)
        ctx.update_best(sp)
        ctx.enum_candidates.append(sp)
    logger.debug(f"  [wake] Cross-reference: {time.time()-t:.2f}s")
    return "cross-reference" if ctx.solved else None


# =============================================================================
# Phase: Local Rules
# =============================================================================

def phase_local_rules(ctx: "WakeContext", env: "ARCEnv") -> Optional[str]:
    """Learn cellular automaton rules from training examples."""
    if ctx.solved:
        return None
    if not hasattr(env, 'try_local_rules'):
        return None
    t = time.time()
    result = env.try_local_rules(ctx.task)
    if result is not None:
        name, fn = result
        sp = ctx.evaluate(Program(root=name))
        ctx.n_evals += 1
        ctx.update_pareto(sp)
        ctx.update_best(sp)
        ctx.enum_candidates.append(sp)
    logger.debug(f"  [wake] Local rules: {time.time()-t:.2f}s")
    return "local rules" if ctx.solved else None


# =============================================================================
# Phase: Procedural
# =============================================================================

def phase_procedural(ctx: "WakeContext", env: "ARCEnv") -> Optional[str]:
    """Learn per-object action rules from pixel diffs."""
    if ctx.solved:
        return None
    if not hasattr(env, 'try_procedural'):
        return None
    t = time.time()
    result = env.try_procedural(ctx.task)
    if result is not None:
        name, fn = result
        sp = ctx.evaluate(Program(root=name))
        ctx.n_evals += 1
        ctx.update_pareto(sp)
        ctx.update_best(sp)
        ctx.enum_candidates.append(sp)
    logger.debug(f"  [wake] Procedural: {time.time()-t:.2f}s")
    return "procedural" if ctx.solved else None


# =============================================================================
# Phase: Synthesis
# =============================================================================

def phase_synthesis(ctx: "WakeContext", env: "ARCEnv") -> Optional[str]:
    """Auto-synthesize primitives from input→output diffs."""
    if ctx.solved:
        return None
    if not hasattr(env, 'synthesize_from_diff'):
        return None
    t = time.time()
    results = env.synthesize_from_diff(ctx.task)
    for name, fn in results:
        sp = ctx.evaluate(Program(root=name))
        ctx.n_evals += 1
        ctx.update_pareto(sp)
        ctx.update_best(sp)
        ctx.enum_candidates.append(sp)
        if ctx.solved:
            break
    logger.debug(f"  [wake] Synthesis: {time.time()-t:.2f}s")
    return "synthesis" if ctx.solved else None


# =============================================================================
# Phase: Conditional Search
# =============================================================================

def phase_conditional_search(ctx: "WakeContext", env: "ARCEnv") -> Optional[str]:
    """Try if(predicate, A, B) programs."""
    predicates = ctx.grammar.get_predicates()
    if not predicates or not ctx.enum_candidates or not ctx.budget_ok():
        return None
    if ctx.solved:
        return None
    t = time.time()
    result, n_evals = _try_conditional_search(
        ctx, env, predicates, ctx.enum_candidates, ctx.all_prims, ctx.task,
        top_k=10)
    ctx.n_evals += n_evals
    if result is not None:
        ctx.update_pareto(result)
        ctx.update_best(result)
        ctx.enum_candidates.append(result)
    logger.debug(f"  [wake] Conditional: {time.time()-t:.2f}s, {n_evals} evals")
    return "conditional" if ctx.solved else None


def _try_conditional_search(
    ctx: "WakeContext",
    env: "ARCEnv",
    predicates: list[tuple[str, Any]],
    candidates: list[ScoredProgram],
    primitives: list[Primitive],
    task,
    top_k: int = 15,
) -> tuple[Optional[ScoredProgram], int]:
    """Search for if(pred, A, B) programs.

    For each predicate that non-trivially splits the training examples,
    score top-K primitives on each group, then try best 5x5 combinations.
    """
    n_evals = 0
    solve_thresh = ctx.cfg.solve_threshold
    t_start = time.time()

    # Build candidate pool from top depth-1 programs
    depth1 = sorted(
        [sp for sp in candidates if sp.program.depth == 1],
        key=lambda s: s.prediction_error)
    prim_map = {p.name: p for p in primitives}
    seen = set()
    top_prims = []
    for sp in depth1:
        if sp.program.root not in seen and sp.program.root in prim_map:
            top_prims.append(prim_map[sp.program.root])
            seen.add(sp.program.root)
        if len(top_prims) >= top_k:
            break

    if len(top_prims) < 2:
        return None, 0

    best_result: Optional[ScoredProgram] = None

    # Pre-cache all prim outputs ONCE (the expensive part)
    prim_cache: dict[str, tuple] = {}  # name -> (prim, [(out, exp), ...])
    for prim in top_prims:
        if time.time() - t_start > 0.3:
            break
        outputs = []
        ok = True
        for inp, exp in task.train_examples:
            try:
                out = prim.fn(inp)
                # Validate: must be a 2D grid
                if not isinstance(out, list) or not out or not isinstance(out[0], list):
                    ok = False
                    break
                outputs.append((out, exp))
            except Exception:
                ok = False
                break
        if ok:
            prim_cache[prim.name] = (prim, outputs)

    if len(prim_cache) < 2:
        return None, 0

    for pred_name, pred_fn in predicates:
        if time.time() - t_start > 0.5:
            break

        true_idx, false_idx = [], []
        for idx, (inp, _) in enumerate(task.train_examples):
            try:
                (true_idx if pred_fn(inp) else false_idx).append(idx)
            except Exception:
                false_idx.append(idx)

        if not true_idx or not false_idx:
            continue

        true_scores = []
        false_scores = []
        for name, (prim, outputs) in prim_cache.items():
            te = sum(ctx.drive.prediction_error(outputs[i][0], outputs[i][1])
                     for i in true_idx) / len(true_idx)
            fe = sum(ctx.drive.prediction_error(outputs[i][0], outputs[i][1])
                     for i in false_idx) / len(false_idx)
            true_scores.append((te, prim))
            false_scores.append((fe, prim))

        true_scores.sort(key=lambda x: x[0])
        false_scores.sort(key=lambda x: x[0])

        for then_p in [p for _, p in true_scores[:5]]:
            for else_p in [p for _, p in false_scores[:5]]:
                if then_p.name == else_p.name:
                    continue

                cond_name = f"if_{pred_name}_{then_p.name}_else_{else_p.name}"
                total_error = 0.0
                max_err = 0.0
                solve_score = 0
                for inp, expected in task.train_examples:
                    try:
                        pred = then_p.fn(inp) if pred_fn(inp) else else_p.fn(inp)
                    except Exception:
                        pred = inp
                    err = ctx.drive.prediction_error(pred, expected)
                    total_error += err
                    max_err = max(max_err, err)
                    if err <= solve_thresh:
                        solve_score += 1
                n_ex = len(task.train_examples)
                avg_error = total_error / n_ex if n_ex else 1.0
                if avg_error <= solve_thresh:
                    # Register only if it actually solves
                    def _make(pf, tf, ef):
                        def fn(g):
                            try:
                                return tf(g) if pf(g) else ef(g)
                            except Exception:
                                return g
                        return fn
                    cond_fn = _make(pred_fn, then_p.fn, else_p.fn)
                    cond_prim = Primitive(
                        name=cond_name, arity=0, fn=cond_fn, domain="arc")
                    env.register_primitive(cond_prim)
                    sp = ctx.evaluate(Program(root=cond_name))
                    n_evals += 1
                    return sp, n_evals
                complexity = 0.003  # 3 nodes
                energy = avg_error + ctx.cfg.energy_beta * complexity
                sp = ScoredProgram(
                    program=Program(root=cond_name),
                    energy=energy,
                    prediction_error=avg_error,
                    complexity_cost=complexity,
                    max_example_error=max_err,
                    example_solve_score=(solve_score / n_ex) ** 2 if n_ex else 0,
                )
                n_evals += 1
                if best_result is None or sp.energy < best_result.energy:
                    best_result = sp

    return best_result, n_evals


# =============================================================================
# Phase: Color Fix
# =============================================================================

def phase_color_fix(ctx: "WakeContext", env: "ARCEnv") -> Optional[str]:
    """Learn color remapping from near-miss programs."""
    if ctx.solved or not ctx.enum_candidates:
        return None
    t = time.time()
    result = _try_color_fix(ctx, env, ctx.enum_candidates, ctx.task)
    if result is not None:
        ctx.n_evals += 1
        ctx.update_pareto(result)
        ctx.update_best(result)
    logger.debug(f"  [wake] Phase 2 color fix: {time.time()-t:.2f}s")
    return "color fix" if ctx.solved else None


def _try_color_fix(
    ctx: "WakeContext",
    env: "ARCEnv",
    candidates: list[ScoredProgram],
    task,
    threshold: float = 0.30,
) -> Optional[ScoredProgram]:
    """Try to fix near-miss programs by learning a color correction."""
    solve_thresh = ctx.cfg.solve_threshold
    near_misses = [
        sp for sp in candidates
        if solve_thresh < sp.prediction_error <= threshold
    ]
    if not near_misses:
        return None

    near_misses.sort(key=lambda s: s.prediction_error)
    near_misses = near_misses[:20]

    best_fix: Optional[ScoredProgram] = None

    # Try correction(identity)
    identity_outputs = [inp for inp, _ in task.train_examples]
    identity_expected = [exp for _, exp in task.train_examples]
    correction = env.infer_output_correction(
        identity_outputs, identity_expected)
    if correction is not None:
        sp = ctx.evaluate(correction)
        if sp.prediction_error <= solve_thresh:
            return sp
        if best_fix is None or sp.energy < best_fix.energy:
            best_fix = sp

    for nm in near_misses:
        outputs = []
        expected = []
        ok = True
        for inp, exp in task.train_examples:
            try:
                out = env.execute(nm.program, inp)
                outputs.append(out)
                expected.append(exp)
            except Exception:
                ok = False
                break
        if not ok:
            continue

        correction = env.infer_output_correction(outputs, expected)
        if correction is None:
            continue

        base = nm.program
        if base.root == "identity" and not base.children:
            fixed_prog = correction
        else:
            fixed_prog = Program(
                root=correction.root,
                children=[copy.deepcopy(base)],
                params=correction.params,
            )
        sp = ctx.evaluate(fixed_prog)

        if sp.prediction_error >= nm.prediction_error:
            continue

        if sp.prediction_error <= solve_thresh:
            return sp
        if best_fix is None or sp.energy < best_fix.energy:
            best_fix = sp

    return best_fix


# =============================================================================
# Phase: Input-Prediction Correction
# =============================================================================

def phase_input_pred_correction(ctx: "WakeContext", env: "ARCEnv") -> Optional[str]:
    """Learn pixel-level correction rules on near-miss predictions.

    Tries multiple key strategies:
    1. (input_pixel, pred_pixel) — original
    2. (pred_pixel, n_nonzero_4neighbors_in_input) — neighborhood context

    Each is LOOCV-validated to avoid overfitting.
    """
    if ctx.solved or not ctx.enum_candidates:
        return None
    t = time.time()
    solve_thresh = ctx.cfg.solve_threshold
    task = ctx.task

    # Key strategy 1: (input_pixel, pred_pixel)
    def key_original(inp, pred, r, c):
        return (inp[r][c], pred[r][c])

    def apply_original(grid, pred, rule):
        return [[rule.get((grid[r][c], pred[r][c]), pred[r][c])
                 for c in range(len(pred[0]))] for r in range(len(pred))]

    # Key strategy 2: (pred_pixel, n_nonzero_4neighbors_in_input)
    def key_nbr_count(inp, pred, r, c):
        h, w = len(inp), len(inp[0])
        n_nz = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                   if 0 <= r + dr < h and 0 <= c + dc < w
                   and inp[r + dr][c + dc] != 0)
        return (pred[r][c], n_nz)

    def apply_nbr_count(grid, pred, rule):
        h, w = len(grid), len(grid[0])
        result = []
        for r in range(h):
            row = []
            for c in range(w):
                n_nz = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                           if 0 <= r + dr < h and 0 <= c + dc < w
                           and grid[r + dr][c + dc] != 0)
                row.append(rule.get((pred[r][c], n_nz), pred[r][c]))
            result.append(row)
        return result

    key_strategies = [
        ("input_pred_correct", key_original, apply_original),
        ("nbr_pred_correct", key_nbr_count, apply_nbr_count),
    ]

    # Try ALL same-dims candidates (correction only works when dims match)
    candidates = sorted(ctx.enum_candidates, key=lambda s: s.prediction_error)[:500]

    for nm in candidates:
        if nm.prediction_error <= solve_thresh:
            continue

        # Execute program on all training inputs to get predictions
        preds = []
        ok = True
        for inp, expected in task.train_examples:
            try:
                pred = env.execute(nm.program, inp)
                if (not isinstance(pred, list) or not pred or
                        len(pred) != len(expected) or len(pred[0]) != len(expected[0]) or
                        len(inp) != len(expected) or len(inp[0]) != len(expected[0])):
                    ok = False
                    break
                preds.append((pred, inp, expected))
            except Exception:
                ok = False
                break
        if not ok or len(preds) < 2:
            continue

        for strat_name, key_fn, apply_fn in key_strategies:
            # Learn rule
            rule: dict[tuple, int] = {}
            consistent = True
            for pred, inp, expected in preds:
                h, w = len(pred), len(pred[0])
                for r in range(h):
                    for c in range(w):
                        key = key_fn(inp, pred, r, c)
                        val = expected[r][c]
                        if key in rule:
                            if rule[key] != val:
                                consistent = False
                                break
                        else:
                            rule[key] = val
                    if not consistent:
                        break
                if not consistent:
                    break
            if not consistent:
                continue

            # Check non-trivial
            trivial = True
            for pred, inp, expected in preds:
                h, w = len(pred), len(pred[0])
                for r in range(h):
                    for c in range(w):
                        key = key_fn(inp, pred, r, c)
                        if rule.get(key, pred[r][c]) != pred[r][c]:
                            trivial = False
                            break
                    if not trivial:
                        break
                if not trivial:
                    break
            if trivial:
                continue

            # LOOCV
            loocv_pass = True
            for hold in range(len(preds)):
                sub = [x for i, x in enumerate(preds) if i != hold]
                sub_rule: dict[tuple, int] = {}
                sub_ok = True
                for pred, inp, expected in sub:
                    h, w = len(pred), len(pred[0])
                    for r in range(h):
                        for c in range(w):
                            key = key_fn(inp, pred, r, c)
                            if key in sub_rule and sub_rule[key] != expected[r][c]:
                                sub_ok = False
                                break
                            sub_rule[key] = expected[r][c]
                        if not sub_ok:
                            break
                    if not sub_ok:
                        break
                if not sub_ok:
                    loocv_pass = False
                    break
                hp, hi, he = preds[hold]
                h, w = len(hp), len(hp[0])
                for r in range(h):
                    for c in range(w):
                        key = key_fn(hi, hp, r, c)
                        if sub_rule.get(key, hp[r][c]) != he[r][c]:
                            loocv_pass = False
                            break
                    if not loocv_pass:
                        break
                if not loocv_pass:
                    break

            if not loocv_pass:
                continue

            # Build the corrected program
            base_prog = nm.program
            final_rule = dict(rule)

            def _make_correction(bp=base_prog, fr=final_rule,
                                 _env=env, afn=apply_fn):
                def fn(grid):
                    pred = _env.execute(bp, grid)
                    if not isinstance(pred, list) or not pred:
                        return grid
                    return afn(grid, pred, fr)
                return fn

            corr_fn = _make_correction()
            name = f"{strat_name}({repr(base_prog)[:30]})"
            prim = Primitive(name=name, arity=0, fn=corr_fn, domain="arc")
            env.register_primitive(prim)
            sp = ctx.evaluate(Program(root=name))
            ctx.n_evals += 1
            ctx.update_pareto(sp)
            ctx.update_best(sp)
            ctx.enum_candidates.append(sp)
            if sp.prediction_error <= solve_thresh:
                logger.debug(f"  [wake] Input-pred correction ({strat_name}): "
                             f"{time.time()-t:.2f}s SOLVED")
                return "input-pred correction"

    logger.debug(f"  [wake] Input-pred correction: {time.time()-t:.2f}s")
    return "input-pred correction" if ctx.solved else None


# =============================================================================
# Phase list constructor
# =============================================================================

def arc_wake_phases(env: "ARCEnv") -> list:
    """Return all ARC-specific wake phases bound to the given environment.

    Each returned callable has signature: (WakeContext) -> Optional[str]
    """
    return [
        lambda ctx, _env=env: phase_object_decomposition(ctx, _env),
        lambda ctx, _env=env: phase_for_each_object(ctx, _env),
        lambda ctx, _env=env: phase_cross_reference(ctx, _env),
        lambda ctx, _env=env: phase_local_rules(ctx, _env),
        lambda ctx, _env=env: phase_procedural(ctx, _env),
        lambda ctx, _env=env: phase_synthesis(ctx, _env),
        lambda ctx, _env=env: phase_conditional_search(ctx, _env),
        lambda ctx, _env=env: phase_color_fix(ctx, _env),
        lambda ctx, _env=env: phase_input_pred_correction(ctx, _env),
    ]
