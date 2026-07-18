"""
Board-quality checker for Tutori.

Loads the layout engine's pure functions from engine.py via AST (no torch /
spaces imports needed), so it runs anywhere. Two uses:
  * validate FINAL ops from a live lesson (steps already laid out server-side)
  * simulate the full pipeline on raw ops (fuzz testing)

Verdict: list of violations [{kind, a, b, area}].
"""

import ast
import json
import random
import re as _re


def load_engine_ns(path="engine.py"):
    with open(path) as f:
        tree = ast.parse(f.read())
    ns = {"re": _re}
    names = {"_clamp_pt", "_op_bbox", "_overlaps", "_free_spot", "_overlap_area",
             "_score", "_solid_area", "_shift_pt", "_fix_text", "_seg_samples", "layout_pass", "_op_sig",
             "_snap_endpoint_to_edge", "_stroke_len", "_cap_arrow", "_target_bboxes", "_near_any_box",
             "_cap_callout", "dedup_ops", "_normalize_op", "_merge_text_groups", "sanitize_step"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))
             and (getattr(n, "name", "") in names
                  or (isinstance(n, ast.Assign)
                      and getattr(n.targets[0], "id", "").startswith(
                          ("_TEXT", "_LABEL", "_SHAPE", "_VALUE", "_OP_ALIAS", "_UESC"))))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns


def render_bboxes(ns, ops):
    """Bounding boxes of everything VISIBLE that shouldn't overlap."""
    out = []
    for op in ops:
        k = op.get("op")
        try:
            if k in ("text", "note", "title", "box", "ellipse", "axes", "graph",
                     "polygon", "notes", "callout"):
                bb = ns["_op_bbox"](op)
                if bb:
                    out.append((k, str(op.get("text") or op.get("label")
                                       or op.get("title") or k)[:30], bb))
            if k in ("arrow", "dot") and op.get("label") and op.get("label_at"):
                w = min(len(str(op["label"])) * 1.6, 30)
                x, y = op["label_at"]
                out.append((k + "-label", str(op["label"])[:30],
                            (x - w / 2, y - 2.5, x + w / 2, y + 2.5)))
        except Exception:
            pass
    return out


def check_final_ops(ns, all_ops, tol=2.5):
    """Violations among FINAL (already laid-out) ops."""
    boxes = render_bboxes(ns, all_ops)
    violations = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ka, ta, a = boxes[i]
            kb, tb, b = boxes[j]
            # a shape's own internal label is allowed inside it; skip
            # shape-vs-its-own-label comparisons by name match
            if {ka, kb} <= {"polygon", "box", "ellipse"} and ta == tb:
                continue
            area = ns["_overlap_area"](a, b)
            # a polygon's bbox is mostly empty interior — text sitting in a
            # triangle's empty corner is visually fine, so use a looser bar
            pair_tol = 15.0 if "polygon" in (ka, kb) else tol
            if area > pair_tol:
                violations.append({"a": f"{ka}:{ta}", "b": f"{kb}:{tb}",
                                   "area": round(area, 1)})
    for k, t, b in boxes:
        if b[2] > 100.5 or b[3] > 76 or b[0] < -0.5 or b[1] < -0.5:
            violations.append({"a": f"{k}:{t}", "b": "OFF-BOARD",
                               "area": round(max(b[2] - 100, b[3] - 75, 0), 1)})
    # a box labeled triangle should have been converted
    for op in all_ops:
        if op.get("op") in ("box", "ellipse") and \
                "triangle" in str(op.get("label", "")).lower():
            violations.append({"a": f"box:{op.get('label')}",
                               "b": "SHOULD-BE-POLYGON", "area": 0})
    return violations


def simulate_lesson(ns, raw_steps, tol=2.5):
    """Run raw step boards through the same pipeline run_turn uses."""
    placed, displaced, anchors, state, seen = [], [], [], {}, set()
    final_ops = []
    for idx, board in enumerate(raw_steps):
        step = ns["sanitize_step"]({"say": "x", "board": board}, idx)
        if not step:
            continue
        b = ns["dedup_ops"](step["board"], seen)
        b = ns["layout_pass"](b, placed, displaced, anchors, state)
        final_ops += [op for op in b if op.get("op") != "clear"]
    return check_final_ops(ns, final_ops, tol), final_ops


# ----------------------------------------------------------------------
def fuzz(n_lessons=300, seed=7, realistic=False):
    rng = random.Random(seed)
    ns = load_engine_ns()
    words = ["force", "energy", "cell", "graph", "node", "market", "orbit",
             "speed", "mass", "x", "supply", "demand", "loop", "root"]

    def rtext():
        n = rng.randint(1, 5)
        return " ".join(rng.choice(words) for _ in range(n))

    def rop():
        k = rng.choice(["text", "text", "box", "ellipse", "polygon", "notes",
                        "arrow", "dot", "highlight", "line"])
        x, y = rng.uniform(0, 110), rng.uniform(0, 85)
        if k == "text":
            return {"op": "text", "text": rng.choice(
                [rtext(), f"a = {rng.randint(1,99)}", "c^2 = a^2 + b^2"]),
                "at": [x, y], "size": rng.choice(["s", "m", "l"])}
        if k in ("box", "ellipse"):
            return {"op": k, "at": [x, y], "w": rng.uniform(8, 45),
                    "h": rng.uniform(6, 40), "label": rng.choice(
                        [rtext(), "Right Triangle", ""])}
        if k == "polygon":
            cx, cy = rng.uniform(15, 80), rng.uniform(20, 60)
            pts = [[cx + rng.uniform(-18, 18), cy + rng.uniform(-14, 14)]
                   for _ in range(rng.randint(3, 5))]
            return {"op": "polygon", "points": pts,
                    "side_labels": [rng.choice(words) for _ in pts]
                    if rng.random() < 0.5 else None}
        if k == "notes":
            return {"op": "notes", "title": rtext().title(),
                    "lines": [f"{rng.choice('abcxyz')} = {rng.randint(1,50)}"
                              for _ in range(rng.randint(2, 5))], "at": [x, y]}
        if k == "arrow":
            return {"op": "arrow", "from": [x, y],
                    "to": [x + rng.uniform(-60, 60), y + rng.uniform(-40, 40)],
                    "label": rtext() if rng.random() < 0.6 else None}
        if k == "dot":
            return {"op": "dot", "at": [x, y],
                    "label": rtext() if rng.random() < 0.7 else None}
        if k == "highlight":
            return {"op": "highlight", "at": [x, y], "w": rng.uniform(10, 30)}
        return {"op": "line", "from": [x, y],
                "to": [x + rng.uniform(-50, 50), y + rng.uniform(-40, 40)]}

    bad = 0
    worst = []
    for li in range(n_lessons):
        n_steps = rng.randint(2, 4)
        steps = []
        budget = rng.randint(4, 9)
        first = [{"op": "clear"}, {"op": "title", "text": rtext().title()}]
        per = max(1, budget // n_steps)
        ops = []
        for si in range(n_steps):
            ops.append((first if si == 0 else []) + [rop() for _ in range(per)])
        if realistic:
            # enforce the prompt's content budget: <=1 notes, <=1 large shape
            n_notes = n_big = 0
            for board in ops:
                kept = []
                for op in board:
                    if op.get("op") == "notes":
                        n_notes += 1
                        if n_notes > 1:
                            continue
                        op["lines"] = op.get("lines", [])[:4]
                    if op.get("op") in ("box", "ellipse") and \
                            max(op.get("w", 0), op.get("h", 0)) > 30:
                        n_big += 1
                        if n_big > 1:
                            op["w"] = min(op.get("w", 20), 26)
                            op["h"] = min(op.get("h", 14), 20)
                    if op.get("op") == "polygon":
                        n_big += 1
                        if n_big > 1:   # content budget: ONE main diagram
                            continue
                    kept.append(op)
                steps.append(kept)
        else:
            steps = ops
        violations, _ = simulate_lesson(ns, steps)
        if violations:
            bad += 1
            worst.append((li, violations[:3]))
    tier = "realistic" if realistic else "stress"
    print(f"fuzz[{tier}]: {n_lessons} lessons, {bad} with violations")
    for li, v in worst[:5]:
        print(f"  lesson {li}: {v}")
    return bad


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fuzz":
        stress_bad = fuzz(realistic=False)
        real_bad = fuzz(realistic=True)
        raise SystemExit(0 if real_bad == 0 else 1)
