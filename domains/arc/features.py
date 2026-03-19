"""Generic pixel feature pool and search for local (cellular automaton) rules.

Replaces 12 hardcoded _learn_*/_apply_* rule types with a single generic
auto-discovery mechanism. The ONE piece of manual input is the feature pool:
"what can a pixel observe about its neighborhood and position?"

The search tries feature combinations of increasing complexity:
  Singles (center, f) → Pairs (center, f1, f2) → Triples → Raw 3×3

Each combination builds a lookup table from training data. If the table is
consistent (every key maps to exactly one output) AND passes LOOCV, it
becomes a valid local rule.

Coverage of old hardcoded rules:
  compact (rule 1):     pair (n_nz_4, majority_4)
  count (rule 2):       single (n_nz_8)
  raw3x3 (rule 3):     step 4 (special)
  pos_mod2 (rule 4a):   pair (r_mod2, c_mod2)
  pos_mod3 (rule 4b):   pair (r_mod3, c_mod3)
  ncolors (rule 5):     single (n_distinct_4)
  nbr_set (rule 6):     single (nbr_set_4)
  8nbr_diag (rule 7):   pair (n_nz_8, has_diag_nz)
  min_nz_nbr (rule 8):  single (min_nz_8)
  lr_context (rule 9):  pair (left, right)
  rowcol_nz (rule 10):  pair (row_nz, col_nz)
  ndist_rowcol (rule 11): pair (row_ndist, col_ndist)
  rowcol_maj (rule 12): pair (row_maj, col_maj)
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Pixel feature context: lazily caches per-grid row/col aggregates
# ---------------------------------------------------------------------------

class PixelFeatureContext:
    """Precomputed per-grid context for pixel features.

    Row/col aggregates (nz counts, distinct colors, majority color)
    are lazily computed at most once per grid.
    """

    __slots__ = ('grid', 'h', 'w',
                 '_row_nz', '_col_nz',
                 '_row_ndist', '_col_ndist',
                 '_row_maj', '_col_maj')

    def __init__(self, grid):
        self.grid = grid
        self.h = len(grid)
        self.w = len(grid[0]) if grid else 0
        self._row_nz = None
        self._col_nz = None
        self._row_ndist = None
        self._col_ndist = None
        self._row_maj = None
        self._col_maj = None

    @property
    def row_nz(self):
        if self._row_nz is None:
            self._row_nz = [sum(1 for c in range(self.w)
                                if self.grid[r][c] != 0)
                            for r in range(self.h)]
        return self._row_nz

    @property
    def col_nz(self):
        if self._col_nz is None:
            self._col_nz = [sum(1 for r in range(self.h)
                                if self.grid[r][c] != 0)
                            for c in range(self.w)]
        return self._col_nz

    @property
    def row_ndist(self):
        if self._row_ndist is None:
            self._row_ndist = [len(set(self.grid[r][c]
                                       for c in range(self.w)))
                               for r in range(self.h)]
        return self._row_ndist

    @property
    def col_ndist(self):
        if self._col_ndist is None:
            self._col_ndist = [len(set(self.grid[r][c]
                                       for r in range(self.h)))
                               for c in range(self.w)]
        return self._col_ndist

    @property
    def row_maj(self):
        if self._row_maj is None:
            result = []
            for r in range(self.h):
                nz = [self.grid[r][c] for c in range(self.w)
                      if self.grid[r][c] != 0]
                result.append(
                    Counter(nz).most_common(1)[0][0] if nz else 0)
            self._row_maj = result
        return self._row_maj

    @property
    def col_maj(self):
        if self._col_maj is None:
            result = []
            for c in range(self.w):
                nz = [self.grid[r][c] for r in range(self.h)
                      if self.grid[r][c] != 0]
                result.append(
                    Counter(nz).most_common(1)[0][0] if nz else 0)
            self._col_maj = result
        return self._col_maj


# ---------------------------------------------------------------------------
# Individual feature functions: (ctx, r, c) → hashable value
# ---------------------------------------------------------------------------

def _n_nz_4(ctx, r, c):
    """Count nonzero 4-neighbors."""
    g, h, w = ctx.grid, ctx.h, ctx.w
    return sum(1 for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
               if 0 <= r + dr < h and 0 <= c + dc < w
               and g[r + dr][c + dc] != 0)


def _n_nz_8(ctx, r, c):
    """Count nonzero 8-neighbors."""
    g, h, w = ctx.grid, ctx.h, ctx.w
    return sum(1 for dr in range(-1, 2) for dc in range(-1, 2)
               if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w
               and g[r + dr][c + dc] != 0)


def _majority_4(ctx, r, c):
    """Majority nonzero color among 4-neighbors."""
    g, h, w = ctx.grid, ctx.h, ctx.w
    nz = [g[r + dr][c + dc]
          for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
          if 0 <= r + dr < h and 0 <= c + dc < w
          and g[r + dr][c + dc] != 0]
    return Counter(nz).most_common(1)[0][0] if nz else 0


def _n_distinct_4(ctx, r, c):
    """Count distinct colors among 4-neighbors."""
    g, h, w = ctx.grid, ctx.h, ctx.w
    return len(set(g[r + dr][c + dc]
                   for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                   if 0 <= r + dr < h and 0 <= c + dc < w))


def _has_diag_nz(ctx, r, c):
    """1 if any diagonal neighbor is nonzero, else 0."""
    g, h, w = ctx.grid, ctx.h, ctx.w
    return int(any(0 <= r + dr < h and 0 <= c + dc < w
                   and g[r + dr][c + dc] != 0
                   for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1))))


def _min_nz_8(ctx, r, c):
    """Min nonzero color among 8-neighbors, -1 if none."""
    g, h, w = ctx.grid, ctx.h, ctx.w
    return min((g[r + dr][c + dc]
                for dr in range(-1, 2) for dc in range(-1, 2)
                if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w
                and g[r + dr][c + dc] != 0),
               default=-1)


def _nbr_set_4(ctx, r, c):
    """Sorted set of 4-neighbor colors (as tuple)."""
    g, h, w = ctx.grid, ctx.h, ctx.w
    return tuple(sorted(set(
        g[r + dr][c + dc]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        if 0 <= r + dr < h and 0 <= c + dc < w)))


def _row_nz(ctx, r, c):
    return ctx.row_nz[r]


def _col_nz(ctx, r, c):
    return ctx.col_nz[c]


def _row_ndist(ctx, r, c):
    return ctx.row_ndist[r]


def _col_ndist(ctx, r, c):
    return ctx.col_ndist[c]


def _row_maj(ctx, r, c):
    return ctx.row_maj[r]


def _col_maj(ctx, r, c):
    return ctx.col_maj[c]


# ---------------------------------------------------------------------------
# The feature pool: the ONE piece of manual input
# ---------------------------------------------------------------------------

FEATURE_POOL = [
    # Neighbor statistics
    ("n_nz_4",      _n_nz_4),
    ("n_nz_8",      _n_nz_8),
    ("majority_4",  _majority_4),
    ("n_distinct_4", _n_distinct_4),
    ("nbr_set_4",   _nbr_set_4),
    ("has_diag_nz", _has_diag_nz),
    ("min_nz_8",    _min_nz_8),
    # Row/col aggregates
    ("row_nz",      _row_nz),
    ("col_nz",      _col_nz),
    ("row_ndist",   _row_ndist),
    ("col_ndist",   _col_ndist),
    ("row_maj",     _row_maj),
    ("col_maj",     _col_maj),
    # Position modular
    ("r_mod2",      lambda ctx, r, c: r % 2),
    ("c_mod2",      lambda ctx, r, c: c % 2),
    ("r_mod3",      lambda ctx, r, c: r % 3),
    ("c_mod3",      lambda ctx, r, c: c % 3),
    # Direct neighbors
    ("left",        lambda ctx, r, c: ctx.grid[r][c - 1] if c > 0 else -1),
    ("right",       lambda ctx, r, c: (ctx.grid[r][c + 1]
                                       if c < ctx.w - 1 else -1)),
    ("above",       lambda ctx, r, c: ctx.grid[r - 1][c] if r > 0 else -1),
    ("below",       lambda ctx, r, c: (ctx.grid[r + 1][c]
                                       if r < ctx.h - 1 else -1)),
]


# ---------------------------------------------------------------------------
# Precomputation & rule building
# ---------------------------------------------------------------------------

def _precompute_features(examples, pool):
    """Compute all feature values for every pixel in every example.

    Returns list of (inp, out, PixelFeatureContext, {(r,c): feature_tuple}).
    """
    result = []
    for inp, out in examples:
        ctx = PixelFeatureContext(inp)
        feats = {}
        for r in range(ctx.h):
            for c in range(ctx.w):
                feats[(r, c)] = tuple(f(ctx, r, c) for _, f in pool)
        result.append((inp, out, ctx, feats))
    return result


def _build_rule(precomputed, indices, exclude_idx=None):
    """Build lookup table from precomputed features for given indices.

    Key: (center_pixel, f_i1, f_i2, ...) → output_pixel.
    Returns the dict, or None if any key maps to two different outputs.
    """
    rule = {}
    for idx, (inp, out, ctx, feats) in enumerate(precomputed):
        if idx == exclude_idx:
            continue
        for r in range(ctx.h):
            for c in range(ctx.w):
                fv = feats[(r, c)]
                key = (inp[r][c],) + tuple(fv[i] for i in indices)
                val = out[r][c]
                if key in rule and rule[key] != val:
                    return None
                rule[key] = val
    return rule


def _apply_rule(grid, rule, pool, indices):
    """Apply a learned feature-combo rule to a grid."""
    ctx = PixelFeatureContext(grid)
    result = []
    for r in range(ctx.h):
        row = []
        for c in range(ctx.w):
            fv = tuple(pool[i][1](ctx, r, c) for i in indices)
            key = (grid[r][c],) + fv
            row.append(rule.get(key, grid[r][c]))
        result.append(row)
    return result


# ---------------------------------------------------------------------------
# Generic feature search
# ---------------------------------------------------------------------------

def generic_feature_search(examples, pool=None):
    """Search for a consistent pixel→pixel rule using feature combinations.

    Tries in order of increasing complexity (simpler rules preferred):
      1. Singles: (center, f)           — N combos
      2. Pairs:   (center, f1, f2)      — C(N,2) combos
      3. Triples: (center, f1, f2, f3)  — C(N,3) combos (only if 1-2 fail)
      4. Raw 3×3: full 9-pixel neighborhood (1 combo)

    Returns (rule_name, apply_fn, rule_dict) or None.
    """
    if pool is None:
        pool = FEATURE_POOL

    precomputed = _precompute_features(examples, pool)
    n = len(pool)
    feature_names = [name for name, _ in pool]

    def _try_combo(indices):
        """Try a feature combo: build rule, then LOOCV."""
        rule = _build_rule(precomputed, indices)
        if rule is None:
            return None

        # LOOCV validation
        for hold_idx in range(len(examples)):
            rule_sub = _build_rule(precomputed, indices,
                                   exclude_idx=hold_idx)
            if rule_sub is None:
                return None
            inp = precomputed[hold_idx][0]
            expected = precomputed[hold_idx][1]
            result = _apply_rule(inp, rule_sub, pool, indices)
            if result != expected:
                return None

        # Build descriptive name
        name = "local_" + "_".join(feature_names[i] for i in indices)

        # Capture rule + indices in closure
        def _make_fn(r=rule, idx=indices, p=pool):
            def fn(grid):
                return _apply_rule(grid, r, p, idx)
            return fn

        return name, _make_fn(), rule

    # Stage 1: Singles
    for i in range(n):
        result = _try_combo((i,))
        if result is not None:
            return result

    # Stage 2: Pairs
    for combo in combinations(range(n), 2):
        result = _try_combo(combo)
        if result is not None:
            return result

    # Stage 3: Triples
    for combo in combinations(range(n), 3):
        result = _try_combo(combo)
        if result is not None:
            return result

    # Stage 4: Raw 3×3 neighborhood
    result = _try_raw3x3(examples)
    if result is not None:
        return result

    return None


# ---------------------------------------------------------------------------
# Raw 3×3 (special case: 9-pixel neighborhood key)
# ---------------------------------------------------------------------------

def _try_raw3x3(examples):
    """Try raw 3×3 neighborhood → output rule."""
    rule = {}
    for inp, out in examples:
        h, w = len(inp), len(inp[0])
        for r in range(h):
            for c in range(w):
                key = tuple(
                    inp[r + dr][c + dc]
                    if 0 <= r + dr < h and 0 <= c + dc < w else -1
                    for dr in range(-1, 2) for dc in range(-1, 2))
                val = out[r][c]
                if key in rule and rule[key] != val:
                    return None
                rule[key] = val

    # LOOCV
    for hold_idx in range(len(examples)):
        rule_sub = {}
        ok = True
        for i, (inp, out) in enumerate(examples):
            if i == hold_idx:
                continue
            h, w = len(inp), len(inp[0])
            for r in range(h):
                for c in range(w):
                    key = tuple(
                        inp[r + dr][c + dc]
                        if 0 <= r + dr < h and 0 <= c + dc < w else -1
                        for dr in range(-1, 2) for dc in range(-1, 2))
                    val = out[r][c]
                    if key in rule_sub and rule_sub[key] != val:
                        ok = False
                        break
                    rule_sub[key] = val
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            return None

        held_inp, held_exp = examples[hold_idx]
        h, w = len(held_inp), len(held_inp[0])
        for r in range(h):
            for c in range(w):
                key = tuple(
                    held_inp[r + dr][c + dc]
                    if 0 <= r + dr < h and 0 <= c + dc < w else -1
                    for dr in range(-1, 2) for dc in range(-1, 2))
                if rule_sub.get(key, held_inp[r][c]) != held_exp[r][c]:
                    return None

    def _make_fn(r=rule):
        def fn(grid):
            h, w = len(grid), len(grid[0])
            return [[r.get(tuple(
                grid[row + dr][col + dc]
                if 0 <= row + dr < h and 0 <= col + dc < w else -1
                for dr in range(-1, 2) for dc in range(-1, 2)),
                grid[row][col])
                for col in range(w)] for row in range(h)]
        return fn

    return "raw3x3_local_rule", _make_fn(), rule


# ---------------------------------------------------------------------------
# Composed search: feature search on pre-transformed inputs (depth-2)
# ---------------------------------------------------------------------------

def generic_feature_search_composed(examples, pool=None):
    """Try generic feature search after applying a pre-transform.

    Enables depth-2 compositions like fill_enclosed → local_rule.
    """
    if pool is None:
        pool = FEATURE_POOL

    from .transformation_primitives import fill_enclosed, dilate

    for transform_name, transform_fn in [
        ("fill_enclosed", fill_enclosed),
        ("dilate", dilate),
    ]:
        transformed_examples = []
        ok = True
        for inp, out in examples:
            try:
                t_inp = transform_fn(inp)
                if (not isinstance(t_inp, list) or not t_inp
                        or len(t_inp) != len(out)
                        or len(t_inp[0]) != len(out[0])):
                    ok = False
                    break
                transformed_examples.append((t_inp, out))
            except Exception:
                ok = False
                break
        if not ok or len(transformed_examples) < 2:
            continue

        result = generic_feature_search(transformed_examples, pool)
        if result is not None:
            base_name, base_fn, base_rule = result
            name = f"local_rule({transform_name})"

            def _make_composed(tf=transform_fn, bf=base_fn):
                def fn(grid):
                    return bf(tf(grid))
                return fn

            return name, _make_composed(), base_rule

    return None
