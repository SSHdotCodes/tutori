"""Semantic whiteboard improvements for Tutori.

The teacher model is good at explaining, but 2D coordinate planning is a
weak spot for small models. This module keeps common instructional diagrams
off the model's shoulders by compiling high-confidence topics into stable
whiteboard ops. Unknown topics still use the model output, with only light
repairs and a restrained fallback note when a useful step would otherwise
draw nothing.
"""

import re


def _blob(*parts):
    return " ".join(str(p or "") for p in parts).lower()


def diagram_family(question, say=""):
    text = _blob(question, say)
    if ("rocket" in text or "spacecraft" in text or "launch" in text) and (
        "orbit" in text or "gravity turn" in text or "orbital" in text
    ):
        return "rocket_orbit"
    if ("fable 5" in text or "mythos 5" in text) and (
        "anthropic" in text or "government" in text or "gov" in text or
        "directive" in text or "ban" in text or "suspend" in text or
        "access" in text
    ):
        return "model_access_directive"
    if ("unqork" in text or
            (("no-code" in text or "low-code" in text or "drag-and-drop" in text
              or "visual builder" in text or "app builder" in text)
             and ("backend" in text or "database" in text or "app" in text))):
        return "no_code_platform"
    if ("gradient descent" in text or "learning rate" in text or
            ("gradient" in text and ("loss" in text or "minimum" in text))):
        return "gradient_descent"
    if "pythag" in text or ("right triangle" in text and "hypotenuse" in text):
        return "pythagorean"
    if "supply" in text and "demand" in text:
        return "supply_demand"
    if "binary search" in text:
        return "binary_search"
    if "recursion" in text or "recursive" in text or "call tree" in text:
        return "recursion"
    if "photosynthesis" in text:
        return "photosynthesis"
    if "neural network" in text or "neural networks" in text or "backprop" in text:
        return "neural_network"
    if "water cycle" in text:
        return "water_cycle"
    if "rainbow" in text or "rainbows" in text:
        return "rainbow"
    return None


def improve_step_board(question, step_index, say, board):
    """Return better board ops for a lesson step.

    High-confidence diagram families replace the model's coordinates entirely.
    Everything else keeps the model output, with generic curve repair and an
    empty-step fallback.
    """
    board = [dict(op) for op in (board or []) if isinstance(op, dict)]
    family = diagram_family(question, say)
    if family:
        templated = _template(family, step_index, question)
        if templated is not None:
            return templated
    return _repair_generic(question, step_index, say, board)


def _template(family, step_index, question=""):
    templates = {
        "rocket_orbit": _rocket_orbit,
        "model_access_directive": _model_access_directive,
        "no_code_platform": _no_code_platform,
        "gradient_descent": _gradient_descent,
        "pythagorean": _pythagorean,
        "supply_demand": _supply_demand,
        "binary_search": _binary_search,
        "recursion": _recursion,
        "photosynthesis": _photosynthesis,
        "neural_network": _neural_network,
        "water_cycle": _water_cycle,
        "rainbow": _rainbow,
    }
    if family == "no_code_platform":
        return templates[family](step_index, question)
    return templates[family](step_index)


def _topic_title(question, fallback="How It Works"):
    text = re.sub(r"[?!.\"]", "", str(question or "")).strip()
    text = re.sub(r"^(explain|teach me|show me|tell me about)\s+", "", text, flags=re.I)
    text = re.sub(r"^how\s+(does|do|is|are)\s+", "", text, flags=re.I)
    text = re.sub(r"\s+(work|works|function|functions)$", "", text, flags=re.I)
    if not text:
        return fallback
    words = text.split()[:4]
    return "How " + " ".join(w.capitalize() if w.islower() else w for w in words) + " Works"


def _rocket_orbit(i):
    steps = [
        [
            {"op": "title", "text": "How Rockets Reach Orbit"},
            {"op": "ellipse", "at": [7, 61], "w": 22, "h": 8,
             "label": "Earth", "color": "gray"},
            {"op": "line", "from": [8, 66], "to": [78, 66],
             "color": "gray", "dash": True},
            {"op": "curve", "points": [[18, 64], [21, 51], [30, 40],
                                        [43, 33], [58, 29], [72, 29]],
             "color": "blue"},
            {"op": "dot", "at": [18, 64], "color": "orange", "label": "launch"},
        ],
        [
            {"op": "ellipse", "at": [34, 55], "w": 24, "h": 8,
             "label": "thick air", "color": "gray"},
            {"op": "arrow", "from": [24, 45], "to": [30, 40],
             "label": "climb", "color": "blue"},
            {"op": "text", "text": "1 clear draggy air", "at": [32, 48],
             "size": "s", "color": "blue"},
            {"op": "dot", "at": [30, 40], "color": "blue", "label": "above air"},
        ],
        [
            {"op": "text", "text": "2 gravity turn", "at": [45, 40],
             "size": "s", "color": "green"},
            {"op": "arrow", "from": [43, 33], "to": [72, 29],
             "label": "build sideways speed", "color": "green"},
            {"op": "arrow", "from": [63, 28], "to": [63, 42],
             "label": "gravity pulls", "color": "red"},
        ],
        [
            {"op": "dot", "at": [72, 29], "color": "green", "label": "orbit"},
            {"op": "arrow", "from": [72, 29], "to": [86, 29],
             "label": "keeps moving", "color": "green"},
            {"op": "notes", "title": "Orbit =",
             "lines": ["above thick air", "fast sideways", "falling around Earth"],
             "at": [66, 45], "color": "green"},
        ],
    ]
    return steps[i] if i < len(steps) else None


def _model_access_directive(i):
    steps = [
        [
            {"op": "title", "text": "Fable/Mythos Access"},
            {"op": "box", "at": [8, 24], "w": 22, "h": 8,
             "label": "US directive", "color": "red"},
            {"op": "box", "at": [38, 24], "w": 24, "h": 8,
             "label": "Anthropic comply", "color": "orange"},
            {"op": "arrow", "from": [30, 28], "to": [38, 28],
             "label": "legal order", "color": "ink"},
        ],
        [
            {"op": "box", "at": [70, 24], "w": 18, "h": 8,
             "label": "Access off", "color": "red"},
            {"op": "arrow", "from": [62, 28], "to": [70, 28],
             "label": "suspend", "color": "red"},
        ],
        [
            {"op": "notes", "title": "Scope",
             "lines": ["Fable 5 + Mythos 5", "all customers disabled",
                       "foreign nationals too"],
             "at": [10, 43], "color": "blue", "compact": True},
            {"op": "notes", "title": "Reason cited",
             "lines": ["received Jun 11, 2026", "national security",
                       "Anthropic disputes scope"],
             "at": [57, 43], "color": "purple", "compact": True},
        ],
        [],
    ]
    return steps[i] if i < len(steps) else None


def _no_code_platform(i, question=""):
    title = _topic_title(question, "How No-Code Works")
    steps = [
        [
            {"op": "title", "text": title},
            {"op": "box", "at": [6, 28], "w": 19, "h": 10,
             "label": "User need", "color": "orange"},
            {"op": "box", "at": [32, 24], "w": 24, "h": 14,
             "label": "Visual canvas", "color": "blue"},
            {"op": "arrow", "from": [25, 33], "to": [32, 31],
             "label": "build", "color": "ink"},
        ],
        [
            {"op": "box", "at": [32, 50], "w": 24, "h": 11,
             "label": "Rules + workflows", "color": "purple"},
            {"op": "arrow", "from": [44, 40], "to": [44, 49],
             "label": "logic", "color": "purple"},
        ],
        [
            {"op": "box", "at": [67, 30], "w": 24, "h": 11,
             "label": "Backend services", "color": "green"},
            {"op": "arrow", "from": [56, 55], "to": [67, 36],
             "label": "connect", "color": "green"},
            {"op": "box", "at": [67, 53], "w": 24, "h": 10,
             "label": "Data + APIs", "color": "green"},
            {"op": "arrow", "from": [79, 41], "to": [79, 53],
             "label": "sync", "color": "green"},
        ],
        [
            {"op": "notes", "title": "Platform handles",
             "lines": ["interface", "business rules", "data flow"],
             "at": [8, 63], "color": "blue"},
            {"op": "text", "text": "running app", "at": [68, 69],
             "size": "s", "color": "orange"},
        ],
    ]
    return steps[i] if i < len(steps) else None


def _gradient_descent(i):
    steps = [
        [
            {"op": "title", "text": "Gradient Descent"},
            {"op": "graph", "at": [10, 24], "w": 50, "h": 36,
             "title": "Loss landscape", "xlabel": "weight", "ylabel": "loss",
             "x_range": [-3, 3], "y_range": [0, 9],
             "series": [{"label": "loss", "color": "blue",
                         "points": [[-3, 9], [-2.5, 6.25], [-2, 4], [-1.5, 2.25],
                                    [-1, 1], [-0.5, 0.25], [0, 0], [0.5, 0.25],
                                    [1, 1], [1.5, 2.25], [2, 4], [2.5, 6.25], [3, 9]]}],
             "markers": [{"at": [-2.5, 6.25], "label": "start", "color": "red"},
                         {"at": [0, 0], "label": "minimum", "color": "green"}]},
        ],
        [
            {"op": "arrow", "from": [14, 36], "to": [21, 48],
             "label": "slope points downhill", "color": "orange"},
            {"op": "dot", "at": [21, 48], "color": "orange", "label": "step 1"},
            {"op": "text", "text": "try weight -> measure loss", "at": [60, 24],
             "size": "s", "color": "gray"},
        ],
        [
            {"op": "arrow", "from": [21, 48], "to": [31, 59],
             "label": "update", "color": "green"},
            {"op": "dot", "at": [31, 59], "color": "green", "label": "step 2"},
        ],
        [
            {"op": "notes", "title": "Update rule",
             "lines": ["slope gives direction", "rate = step size",
                       "stop near flat bottom"],
             "at": [61, 38], "color": "gray"},
            {"op": "text", "text": "w <- w - rate * slope", "at": [60, 66],
             "size": "s", "color": "purple"},
        ],
    ]
    return steps[i] if i < len(steps) else None


def _pythagorean(i):
    steps = [
        [
            {"op": "title", "text": "Pythagorean Theorem"},
            {"op": "polygon", "points": [[11, 27], [11, 55], [41, 55]],
             "side_labels": ["a", "b", "c"], "label": "right triangle",
             "color": "blue"},
            {"op": "line", "from": [11, 51], "to": [15, 51],
             "color": "ink"},
            {"op": "line", "from": [15, 51], "to": [15, 55],
             "color": "ink"},
            {"op": "callout", "around": [31, 38], "to": [52, 30],
             "label": "c = hypotenuse", "color": "orange", "r": 2.7,
             "fit_w": 24},
        ],
        [
            {"op": "notes", "title": "Parts",
             "lines": ["legs: a and b", "hypotenuse: c", "c is longest"],
             "at": [61, 25], "color": "gray", "compact": True},
            {"op": "text", "text": "a^2 + b^2 = c^2", "at": [63, 47],
             "size": "l", "color": "purple", "fit_w": 34},
            {"op": "underline", "at": [63, 51], "w": 30, "color": "purple"},
        ],
        [
            {"op": "text", "text": "Example: a = 3, b = 4, c = ?",
             "at": [58, 55], "size": "s", "color": "green", "fit_w": 39},
        ],
        [
            {"op": "text", "text": "9 + 16 = 25, so c = 5",
             "at": [58, 66], "size": "s", "color": "green", "fit_w": 39},
            {"op": "callout", "around": [31, 38], "to": [52, 42],
             "label": "c = 5", "color": "green", "r": 2.7,
             "fit_w": 25},
        ],
    ]
    return steps[i] if i < len(steps) else None


def _supply_demand(i):
    steps = [
        [
            {"op": "title", "text": "Supply and Demand"},
            {"op": "graph", "at": [10, 21], "w": 54, "h": 42,
             "title": "Market curves", "xlabel": "quantity", "ylabel": "price",
             "x_range": [0, 10], "y_range": [0, 10],
             "series": [
                 {"label": "Supply", "color": "green",
                  "points": [[0, 1], [2, 2.6], [4, 4.2], [5, 5], [6, 5.8], [8, 7.4], [10, 9]]},
                 {"label": "Demand", "color": "red",
                  "points": [[0, 9], [2, 7.4], [4, 5.8], [5, 5], [6, 4.2], [8, 2.6], [10, 1]]},
             ],
             "markers": [{"at": [5, 5], "label": "equilibrium", "color": "purple"}]},
        ],
        [
            {"op": "text", "text": "Supply", "at": [55, 25],
             "size": "s", "color": "green"},
            {"op": "text", "text": "Demand", "at": [55, 60],
             "size": "s", "color": "red"},
            {"op": "notes", "title": "At the crossing",
             "lines": ["buyers accept price", "sellers accept quantity"],
             "at": [68, 25], "color": "gray"},
        ],
        [
            {"op": "arrow", "from": [42, 35], "to": [50, 30],
             "label": "more demand", "color": "orange"},
            {"op": "notes", "title": "Shift",
             "lines": ["curve moves", "new crossing", "new price"],
             "at": [68, 48], "color": "orange"},
        ],
    ]
    return steps[i] if i < len(steps) else None


def _binary_search(i):
    steps = [
        [
            {"op": "title", "text": "Binary Search"},
            {"op": "notes", "title": "Sorted array",
             "lines": ["2  5  8 | 13 | 21 34 55", "lo      mid       hi"],
             "at": [11, 27], "color": "ink"},
            {"op": "box", "at": [33, 33], "w": 15, "h": 11,
             "label": "mid = 13", "color": "purple"},
        ],
        [
            {"op": "arrow", "from": [49, 38], "to": [63, 38],
             "label": "target bigger", "color": "green"},
            {"op": "notes", "title": "Discard",
             "lines": ["left half too small", "keep right half"],
             "at": [67, 25], "color": "gray"},
        ],
        [
            {"op": "box", "at": [59, 45], "w": 18, "h": 11,
             "label": "new middle", "color": "green"},
            {"op": "text", "text": "repeat on half", "at": [59, 61],
             "size": "s", "color": "green"},
        ],
    ]
    return steps[i] if i < len(steps) else None


def _recursion(i):
    steps = [
        [
            {"op": "title", "text": "Recursion"},
            {"op": "box", "at": [36, 20], "w": 22, "h": 10,
             "label": "solve n", "color": "blue"},
            {"op": "box", "at": [18, 42], "w": 22, "h": 10,
             "label": "solve n-1", "color": "purple"},
            {"op": "box", "at": [56, 42], "w": 22, "h": 10,
             "label": "base case", "color": "green"},
            {"op": "arrow", "from": [43, 30], "to": [31, 42],
             "label": "calls", "color": "ink"},
        ],
        [
            {"op": "arrow", "from": [51, 30], "to": [64, 42],
             "label": "stops", "color": "green"},
            {"op": "notes", "title": "Two ingredients",
             "lines": ["smaller subproblem", "base case returns"],
             "at": [67, 57], "color": "gray"},
        ],
        [
            {"op": "arrow", "from": [64, 52], "to": [51, 30],
             "label": "returns", "color": "orange"},
        ],
    ]
    return steps[i] if i < len(steps) else None


def _photosynthesis(i):
    steps = [
        [
            {"op": "title", "text": "Photosynthesis"},
            {"op": "ellipse", "at": [29, 28], "w": 28, "h": 20,
             "label": "leaf", "color": "green"},
            {"op": "arrow", "from": [12, 23], "to": [31, 33],
             "label": "sunlight", "color": "orange"},
            {"op": "arrow", "from": [12, 48], "to": [31, 41],
             "label": "CO2 + water", "color": "blue"},
        ],
        [
            {"op": "arrow", "from": [57, 35], "to": [74, 28],
             "label": "O2", "color": "green"},
            {"op": "arrow", "from": [57, 42], "to": [74, 49],
             "label": "sugar", "color": "purple"},
            {"op": "notes", "title": "Inside chloroplasts",
             "lines": ["light energy", "becomes chemical energy"],
             "at": [67, 57], "color": "gray"},
        ],
        [
            {"op": "text", "text": "light + CO2 + H2O -> sugar + O2",
             "at": [12, 65], "size": "s", "color": "ink"},
        ],
    ]
    return steps[i] if i < len(steps) else None


def _neural_network(i):
    steps = [
        [
            {"op": "title", "text": "Neural Networks Learn"},
            {"op": "ellipse", "at": [9, 30], "w": 18, "h": 12,
             "label": "inputs", "color": "blue"},
            {"op": "ellipse", "at": [38, 25], "w": 20, "h": 22,
             "label": "hidden", "color": "purple"},
            {"op": "ellipse", "at": [70, 30], "w": 18, "h": 12,
             "label": "output", "color": "green"},
            {"op": "arrow", "from": [27, 36], "to": [38, 36],
             "label": "weights", "color": "ink"},
        ],
        [
            {"op": "arrow", "from": [58, 36], "to": [70, 36],
             "label": "prediction", "color": "green"},
            {"op": "notes", "title": "Training loop",
             "lines": ["compare to answer", "measure error", "adjust weights"],
             "at": [66, 48], "color": "gray"},
        ],
        [
            {"op": "arrow", "from": [70, 42], "to": [52, 47],
             "label": "error back", "color": "red"},
            {"op": "arrow", "from": [38, 47], "to": [23, 42],
             "label": "update", "color": "orange"},
        ],
    ]
    return steps[i] if i < len(steps) else None


def _water_cycle(i):
    steps = [
        [
            {"op": "title", "text": "Water Cycle"},
            {"op": "ellipse", "at": [10, 54], "w": 30, "h": 9,
             "label": "ocean", "color": "blue"},
            {"op": "ellipse", "at": [47, 22], "w": 28, "h": 12,
             "label": "cloud", "color": "gray"},
            {"op": "arrow", "from": [28, 54], "to": [50, 32],
             "label": "evaporation", "color": "orange"},
        ],
        [
            {"op": "arrow", "from": [61, 34], "to": [48, 54],
             "label": "rain", "color": "blue"},
            {"op": "box", "at": [42, 55], "w": 26, "h": 8,
             "label": "land", "color": "green"},
            {"op": "arrow", "from": [42, 60], "to": [28, 60],
             "label": "runoff", "color": "green"},
        ],
        [
            {"op": "notes", "title": "Cycle",
             "lines": ["sun lifts water", "clouds condense", "rain returns"],
             "at": [68, 45], "color": "gray"},
        ],
    ]
    return steps[i] if i < len(steps) else None


def _rainbow(i):
    steps = [
        [
            {"op": "title", "text": "How Rainbows Form"},
            {"op": "ellipse", "at": [38, 28], "w": 20, "h": 24,
             "label": "raindrop", "color": "blue"},
            {"op": "arrow", "from": [12, 36], "to": [38, 38],
             "label": "white light", "color": "orange"},
        ],
        [
            {"op": "line", "from": [58, 34], "to": [78, 24],
             "color": "red"},
            {"op": "line", "from": [58, 38], "to": [80, 38],
             "color": "orange"},
            {"op": "line", "from": [58, 42], "to": [78, 54],
             "color": "purple"},
            {"op": "notes", "title": "Inside the drop",
             "lines": ["light bends", "colors separate"],
             "at": [67, 58], "color": "gray"},
        ],
        [
            {"op": "arrow", "from": [45, 52], "to": [42, 40],
             "label": "reflects", "color": "ink"},
        ],
    ]
    return steps[i] if i < len(steps) else None


def _repair_generic(question, step_index, say, board):
    text = _blob(question, say)
    repaired = []
    for op in board:
        op = dict(op)
        if op.get("op") == "curve":
            if "orbit" in text and ("rocket" in text or "orbital" in text):
                op["points"] = [[14, 64], [18, 51], [26, 38],
                                [40, 31], [56, 28], [70, 28]]
            elif ("gradient" in text or "loss" in text) and (
                    "minimum" in text or "lowest" in text or "learning rate" in text):
                op["points"] = [[14, 28], [25, 43], [38, 57],
                                [50, 62], [62, 47], [75, 27]]
        repaired.append(op)
    if repaired or _is_checkin(say) or step_index == 0:
        return repaired
    lines = _note_lines(say)
    if not lines:
        return repaired
    return [{"op": "notes", "title": "Key point", "lines": lines,
             "at": [67, 35], "color": "gray"}]


def _is_checkin(say):
    low = str(say or "").lower()
    return bool(re.search(r"\b(does that|make sense|which part|would you like|want to)\b", low))


def _note_lines(say):
    text = re.sub(r"\s+", " ", str(say or "")).strip()
    text = re.sub(r"^(and|so|now|then|first|next|finally),?\s+", "", text, flags=re.I)
    chunks = re.split(r"[.;:]|\bwhile\b|\bbecause\b|\bwhich\b", text)
    lines = []
    for chunk in chunks:
        words = re.findall(r"[A-Za-z0-9^+-]+", chunk)
        if len(words) < 3:
            continue
        line = " ".join(words[:5])
        if len(line) > 24:
            line = " ".join(words[:4])
        if line and line.lower() not in {x.lower() for x in lines}:
            lines.append(line)
        if len(lines) == 3:
            break
    return lines
