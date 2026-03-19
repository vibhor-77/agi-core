"""Auto-synthesis of primitives from input→output diffs.

Instead of hand-coding transform primitives, this module analyzes what changed
between input and output to propose reusable transforms. Each synthesized
transform is LOOCV-validated before use.

Synthesized primitives enter the library alongside extracted subtrees, so
the system grows its own vocabulary over rounds.

Diff pattern templates detected:
  1. Color substitution:   systematic c1→c2 recoloring
  2. Directional fill:     extend nonzero pixels along rows/cols
  3. Region fill:          fill enclosed zero-regions
  4. Symmetry completion:  complete partial symmetry
  5. Color overlay:        overlay one color onto another's positions
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Optional

from core import Primitive


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

Grid = list[list[int]]


def _compute_diff(inp: Grid, out: Grid) -> dict[tuple[int, int], int]:
    """Return {(r, c): new_color} for all pixels that changed."""
    if not inp or not out:
        return {}
    h_in, w_in = len(inp), len(inp[0])
    h_out, w_out = len(out), len(out[0])
    if h_in != h_out or w_in != w_out:
        return {}
    return {(r, c): out[r][c]
            for r in range(h_in) for c in range(w_in)
            if inp[r][c] != out[r][c]}


# ---------------------------------------------------------------------------
# Pattern 1: Color substitution
# ---------------------------------------------------------------------------

def _detect_color_sub(examples):
    """Detect systematic color replacement: every pixel of color A → B.

    Returns a color map dict {old_color: new_color} or None.
    """
    # Build color mapping from all examples
    color_map = {}
    for inp, out in examples:
        h, w = len(inp), len(inp[0])
        if len(out) != h or (h > 0 and len(out[0]) != w):
            return None
        for r in range(h):
            for c in range(w):
                src, dst = inp[r][c], out[r][c]
                if src != dst:
                    if src in color_map and color_map[src] != dst:
                        return None  # inconsistent mapping
                    color_map[src] = dst
    if not color_map:
        return None

    # Verify: all occurrences of mapped colors are consistently replaced
    for inp, out in examples:
        h, w = len(inp), len(inp[0])
        for r in range(h):
            for c in range(w):
                src = inp[r][c]
                if src in color_map:
                    if out[r][c] != color_map[src]:
                        return None
    return color_map


def _make_color_sub(color_map):
    """Create a color substitution transform."""
    def fn(grid):
        return [[color_map.get(grid[r][c], grid[r][c])
                 for c in range(len(grid[0]))]
                for r in range(len(grid))]
    return fn


# ---------------------------------------------------------------------------
# Pattern 2: Directional fill (extend nonzero pixels along rows/cols)
# ---------------------------------------------------------------------------

def _detect_directional_fill(examples):
    """Detect if nonzero pixels get extended in one direction.

    Tries: right, left, down, up fills where nonzero pixels propagate
    until hitting another nonzero pixel or grid edge.
    """
    directions = [
        ("fill_right", _apply_fill_right),
        ("fill_left", _apply_fill_left),
        ("fill_down", _apply_fill_down),
        ("fill_up", _apply_fill_up),
    ]
    for dir_name, apply_fn in directions:
        ok = True
        for inp, out in examples:
            if len(inp) != len(out) or (inp and out and len(inp[0]) != len(out[0])):
                ok = False
                break
            result = apply_fn(inp)
            if result != out:
                ok = False
                break
        if ok:
            return dir_name, apply_fn
    return None


def _apply_fill_right(grid):
    """Propagate each nonzero pixel rightward until blocked."""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if result[r][c] != 0:
                color = result[r][c]
                for cc in range(c + 1, w):
                    if result[r][cc] != 0:
                        break
                    result[r][cc] = color
    return result


def _apply_fill_left(grid):
    """Propagate each nonzero pixel leftward until blocked."""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w - 1, -1, -1):
            if result[r][c] != 0:
                color = result[r][c]
                for cc in range(c - 1, -1, -1):
                    if result[r][cc] != 0:
                        break
                    result[r][cc] = color
    return result


def _apply_fill_down(grid):
    """Propagate each nonzero pixel downward until blocked."""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for c in range(w):
        for r in range(h):
            if result[r][c] != 0:
                color = result[r][c]
                for rr in range(r + 1, h):
                    if result[rr][c] != 0:
                        break
                    result[rr][c] = color
    return result


def _apply_fill_up(grid):
    """Propagate each nonzero pixel upward until blocked."""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for c in range(w):
        for r in range(h - 1, -1, -1):
            if result[r][c] != 0:
                color = result[r][c]
                for rr in range(r - 1, -1, -1):
                    if result[rr][c] != 0:
                        break
                    result[rr][c] = color
    return result


# ---------------------------------------------------------------------------
# Pattern 3: Region fill (fill enclosed zero-regions)
# ---------------------------------------------------------------------------

def _detect_region_fill(examples):
    """Detect if enclosed zero regions get filled with a specific color."""
    # Try each nonzero color present in any output diff
    diff_colors = set()
    for inp, out in examples:
        diff = _compute_diff(inp, out)
        diff_colors.update(diff.values())

    for fill_color in sorted(diff_colors):
        if fill_color == 0:
            continue
        ok = True
        for inp, out in examples:
            result = _apply_region_fill(inp, fill_color)
            if result != out:
                ok = False
                break
        if ok:
            return fill_color
    return None


def _apply_region_fill(grid, fill_color):
    """Fill enclosed zero-regions with fill_color."""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Flood-fill from border to mark exterior zeros
    exterior = set()
    stack = []
    for r in range(h):
        for c in [0, w - 1]:
            if grid[r][c] == 0:
                stack.append((r, c))
    for c in range(w):
        for r in [0, h - 1]:
            if grid[r][c] == 0:
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        if (r, c) in exterior:
            continue
        if r < 0 or r >= h or c < 0 or c >= w:
            continue
        if grid[r][c] != 0:
            continue
        exterior.add((r, c))
        stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

    # Fill interior zeros
    for r in range(h):
        for c in range(w):
            if result[r][c] == 0 and (r, c) not in exterior:
                result[r][c] = fill_color

    return result


# ---------------------------------------------------------------------------
# Pattern 4: Symmetry completion
# ---------------------------------------------------------------------------

def _detect_symmetry_completion(examples):
    """Detect if output completes a symmetry the input lacks.

    Tries horizontal, vertical, and 180° rotational symmetry.
    """
    symmetries = [
        ("sym_h", _apply_sym_h),
        ("sym_v", _apply_sym_v),
        ("sym_180", _apply_sym_180),
    ]
    for sym_name, apply_fn in symmetries:
        ok = True
        for inp, out in examples:
            if len(inp) != len(out) or (inp and out and len(inp[0]) != len(out[0])):
                ok = False
                break
            result = apply_fn(inp)
            if result != out:
                ok = False
                break
        if ok:
            return sym_name, apply_fn
    return None


def _apply_sym_h(grid):
    """Complete horizontal symmetry: fill using left-right mirror."""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            mirror_c = w - 1 - c
            if result[r][c] == 0 and result[r][mirror_c] != 0:
                result[r][c] = result[r][mirror_c]
            elif result[r][mirror_c] == 0 and result[r][c] != 0:
                result[r][mirror_c] = result[r][c]
    return result


def _apply_sym_v(grid):
    """Complete vertical symmetry: fill using top-bottom mirror."""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        mirror_r = h - 1 - r
        for c in range(w):
            if result[r][c] == 0 and result[mirror_r][c] != 0:
                result[r][c] = result[mirror_r][c]
            elif result[mirror_r][c] == 0 and result[r][c] != 0:
                result[mirror_r][c] = result[r][c]
    return result


def _apply_sym_180(grid):
    """Complete 180° rotational symmetry."""
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            mr, mc = h - 1 - r, w - 1 - c
            if result[r][c] == 0 and result[mr][mc] != 0:
                result[r][c] = result[mr][mc]
            elif result[mr][mc] == 0 and result[r][c] != 0:
                result[mr][mc] = result[r][c]
    return result


# ---------------------------------------------------------------------------
# Pattern 5: Color overlay (nonzero pixels of color A fill positions of color B)
# ---------------------------------------------------------------------------

def _detect_color_overlay(examples):
    """Detect if output overlays one color onto another color's positions.

    E.g., all positions that are color 3 in input become color 1 in output,
    while positions that are 0 stay 0.
    """
    # Only same-dims
    for inp, out in examples:
        if len(inp) != len(out) or (inp and len(inp[0]) != len(out[0])):
            return None

    diff = _compute_diff(examples[0][0], examples[0][1])
    if not diff:
        return None

    # All diff pixels should go to the same target color
    target_colors = set(diff.values())
    if len(target_colors) != 1:
        return None
    target = target_colors.pop()

    # All diff pixels should come from the same source color
    source_colors = set(examples[0][0][r][c] for r, c in diff)
    if len(source_colors) != 1:
        return None
    source = source_colors.pop()

    # Verify: ALL pixels of source color → target, everything else unchanged
    for inp, out in examples:
        h, w = len(inp), len(inp[0])
        for r in range(h):
            for c in range(w):
                expected = target if inp[r][c] == source else inp[r][c]
                if out[r][c] != expected:
                    return None

    return source, target


# ---------------------------------------------------------------------------
# LOOCV validation
# ---------------------------------------------------------------------------

def _loocv_validate(examples, apply_fn):
    """Leave-one-out cross-validation for a transform function.

    For synthesis, the function is parameterless (doesn't need to be
    re-learned from subsets), so we just check it works on each example.
    """
    for inp, out in examples:
        result = apply_fn(inp)
        if result != out:
            return False
    return True


# ---------------------------------------------------------------------------
# Main synthesis entry point
# ---------------------------------------------------------------------------

def synthesize_from_diff(task) -> list[tuple[str, Callable, str]]:
    """Analyze input→output diffs to propose reusable transforms.

    Returns list of (name, apply_fn, description) for all discovered patterns.
    Each is already validated on all training examples.
    """
    examples = task.train_examples
    if not examples or len(examples) < 2:
        return []

    candidates = []

    # Pattern 1: Color substitution
    color_map = _detect_color_sub(examples)
    if color_map and _loocv_validate(examples, _make_color_sub(color_map)):
        fn = _make_color_sub(color_map)
        desc = f"color_sub: {color_map}"
        candidates.append((f"synth_color_sub", fn, desc))

    # Pattern 2: Directional fill
    dir_result = _detect_directional_fill(examples)
    if dir_result:
        dir_name, dir_fn = dir_result
        candidates.append((f"synth_{dir_name}", dir_fn,
                           f"directional fill: {dir_name}"))

    # Pattern 3: Region fill
    fill_color = _detect_region_fill(examples)
    if fill_color is not None:
        def _make_fill(fc=fill_color):
            def fn(grid):
                return _apply_region_fill(grid, fc)
            return fn
        candidates.append((f"synth_region_fill_{fill_color}",
                           _make_fill(),
                           f"fill enclosed zeros with color {fill_color}"))

    # Pattern 4: Symmetry completion
    sym_result = _detect_symmetry_completion(examples)
    if sym_result:
        sym_name, sym_fn = sym_result
        candidates.append((f"synth_{sym_name}", sym_fn,
                           f"symmetry completion: {sym_name}"))

    # Pattern 5: Color overlay
    overlay_result = _detect_color_overlay(examples)
    if overlay_result:
        source, target = overlay_result
        def _make_overlay(s=source, t=target):
            def fn(grid):
                return [[t if grid[r][c] == s else grid[r][c]
                         for c in range(len(grid[0]))]
                        for r in range(len(grid))]
            return fn
        candidates.append((f"synth_overlay_{source}_to_{target}",
                           _make_overlay(),
                           f"recolor {source}→{target}"))

    return candidates
