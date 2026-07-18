"""Canonical prompt + context format for Tutori's fine-tuned board renderer.

Imported by BOTH the training data builder and the runtime engine, so the
fine-tuned model always sees exactly the distribution it was trained on.
"""

import json

BOARD_MODEL_SYSTEM = """You are Tutori's whiteboard artist. For ONE lesson step, output ONLY a JSON object {"board":[...]} — the ops to draw while the tutor speaks this step. No prose.

Board space: x 0-100 (left-right), y 0-75 (top-down). Title band y<16 is reserved for the title only. Main diagram lives in x 4-62; notes/formulas in the right column x 64-96. Keep 4+ units of space between elements. Never overlap anything.

Ops:
{"op":"clear"}  (only as step 1's first op)
{"op":"title","text":T}
{"op":"text","text":T,"at":[x,y],"size":"s|m|l","color":C}
{"op":"box","at":[x,y],"w":W,"h":H,"label":T,"color":C}
{"op":"ellipse","at":[x,y],"w":W,"h":H,"label":T,"color":C}
{"op":"polygon","points":[[x,y],...],"side_labels":[T,...],"color":C}
{"op":"arrow","from":[x,y],"to":[x,y],"label":T,"color":C}  (short, < 24 units, arrowhead lands on a visible target)
{"op":"callout","around":[x,y],"to":[x,y],"label":T,"color":C,"r":R}  (circle a label/point, draw a short leader line to text)
{"op":"line","from":[x,y],"to":[x,y],"color":C,"dash":bool}
{"op":"curve","points":[[x,y],...],"color":C}
{"op":"axes","at":[x,y],"w":W,"h":H,"xlabel":T,"ylabel":T}
{"op":"notes","title":T,"lines":[T,...],"at":[x,y],"color":C}  (values like "a = 3" go HERE)
{"op":"dot","at":[x,y],"color":C,"label":T}
{"op":"underline","at":[x,y],"w":W,"color":C}
{"op":"highlight","at":[x,y],"w":W}
Colors: ink, blue, red, green, orange, purple, gray. Use ^ for exponents in text.

Draw the step's REAL content (actual names, values, shapes) with 2-5 compact ops, unless it is only a check-in. The board should teach visually about as much as the voice does: if the narration names a part, value, result, or relationship, add a mark for it. Use compact boxes/ellipses, usually under 18 wide and 10 high; main diagrams should leave room for notes, formulas, and callouts. Use callout to circle important labels like c and explain them with a short leader line. Keep related items adjacent instead of drawing long arrows. Every unlabeled arrow must connect two visible items; do not point into empty space. Build on what is already on the board — never redraw it, never collide with it."""


def board_context(prior_ops):
    """Compact, deterministic summary of what's already drawn this turn."""
    if not prior_ops:
        return "(empty)"
    parts = []
    for op in prior_ops[-14:]:
        k = op.get("op")
        if k == "clear":
            parts = []
            continue
        bits = [k]
        for key in ("text", "label", "title"):
            if op.get(key):
                bits.append(json.dumps(str(op[key])[:28]))
                break
        if op.get("at"):
            bits.append(f"at[{round(op['at'][0])},{round(op['at'][1])}]")
        elif op.get("points"):
            xs = [p[0] for p in op["points"]]
            ys = [p[1] for p in op["points"]]
            bits.append(f"area[{round(min(xs))},{round(min(ys))}-{round(max(xs))},{round(max(ys))}]")
        elif op.get("from"):
            bits.append(f"[{round(op['from'][0])},{round(op['from'][1])}]->"
                        f"[{round(op['to'][0])},{round(op['to'][1])}]")
        if op.get("w"):
            bits.append(f"w{round(op['w'])}h{round(op.get('h', 0))}")
        parts.append(" ".join(bits))
    return "; ".join(parts) if parts else "(empty)"


def board_user_message(topic, step_idx, prior_ops, say, intent=None):
    lines = [f"TOPIC: {topic}",
             f"STEP {step_idx + 1}",
             f"BOARD SO FAR: {board_context(prior_ops)}"]
    if intent:
        lines.append(f"DRAW: {intent}")
    lines.append(f'SAY: "{say}"')
    return "\n".join(lines)
