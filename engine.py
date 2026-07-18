"""
Tutori engine — an OpenRouter-backed, long-horizon whiteboard tutor.

Models (all open-weight, all routed through OpenRouter):
  * tencent/hy3 ......................... lesson architect + teacher
  * google/gemma-4-31b-it:nitro ......... fast coach + whiteboard vision
  * nvidia/parakeet-tdt-0.6b-v3 ......... speech recognition
  * hexgrad/kokoro-82m .................. fast open-weight speech

run_turn() is a generator that yields event dicts; app.py turns those into
streaming UI updates. Events:
  {"type": "status",     "status": str, "detail": str}
  {"type": "transcript", "text": str}
  {"type": "step",       "step": {"say", "board", "audio", "dur"}}
  {"type": "memory",     "profile": dict}
  {"type": "final",      "text": str, "error": str | None}
"""

import base64
import datetime
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import requests
from board_quality import diagram_family, improve_step_board

if os.environ.get("TUTORI_DEBUG") == "1":
    import faulthandler
    faulthandler.dump_traceback_later(120, repeat=True)

OPENROUTER_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
LLM_ID = os.environ.get("TUTORI_INTELLIGENCE_MODEL", "tencent/hy3")
COACH_ID = os.environ.get("TUTORI_FAST_MODEL", "google/gemma-4-31b-it:nitro")
ASR_ID = os.environ.get("TUTORI_ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
TTS_ID = os.environ.get("TUTORI_TTS_MODEL", "hexgrad/kokoro-82m")
TTS_VOICE = os.environ.get("TUTORI_TTS_VOICE", "af_alloy")
USE_BOARD_MODEL = False
BOARD_ARTIST = None


def _api_key():
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    return key


def _headers(content_type="application/json"):
    h = {
        "Authorization": f"Bearer {_api_key()}",
        "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "https://tutori.ssh.codes"),
        "X-Title": os.environ.get("OPENROUTER_APP_NAME", "Tutori"),
    }
    if content_type:
        h["Content-Type"] = content_type
    return h


print(f"[tutori] OpenRouter stack: {LLM_ID} | {COACH_ID} | {ASR_ID} | {TTS_ID}", flush=True)


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

BOARD_REFERENCE = """\
The whiteboard coordinate space is x: 0-100 (left to right), y: 0-75 (top to bottom).
Available ops (use ONLY these — anything else is ignored):
  {"op":"clear"}                                          wipe the whole board for a genuinely new visual chapter
  {"op":"erase","targets":["id1","id2"]}              erase only earlier marks with these ids
  {"op":"title","text":T}                                 big heading, auto-centered at top
  {"op":"text","text":T,"at":[x,y],"size":"s|m|l","color":C,"align":"left|center"}
  {"op":"box","at":[x,y],"w":W,"h":H,"label":T,"color":C}        rectangle, at = top-left
  {"op":"ellipse","at":[x,y],"w":W,"h":H,"label":T,"color":C}    oval, at = top-left of bounding box
  {"op":"arrow","from":[x,y],"to":[x,y],"label":T,"color":C}    short connector between adjacent items
  {"op":"callout","around":[x,y],"to":[x,y],"label":T,"color":C,"r":R}
       circle a label or point, then draw a short leader line to the label text
       (for example circle "c" and label it "hypotenuse")
  {"op":"line","from":[x,y],"to":[x,y],"color":C,"dash":true|false}
  {"op":"curve","points":[[x,y],[x,y],...],"color":C}             smooth curve through 4-10 points
  {"op":"polygon","points":[[x,y],[x,y],[x,y]],"label":T,"side_labels":[T,...],"color":C}
       CLOSED shape through 3-8 points (auto-closes — ALWAYS use this for triangles;
       never assemble shapes from line ops). side_labels names each edge (edge i runs
       from point i to point i+1) and is auto-placed perfectly — ALWAYS label sides
       this way, never with separate text ops.
  {"op":"notes","title":T,"lines":[T,...],"at":[x,y],"color":C}   tidy stacked block —
       use ONE of these for any example, list of values, or takeaways
       (e.g. title "Example:", lines ["a = 3","b = 4","c = ?"]) instead of scattered text ops
  {"op":"axes","at":[x,y],"w":W,"h":H,"xlabel":T,"ylabel":T}      coordinate axes, at = top-left of plot area
  {"op":"graph","id":ID,"at":[x,y],"w":W,"h":H,"title":T,"xlabel":T,"ylabel":T,
   "x_range":[min,max],"y_range":[min,max],
   "series":[{"label":T,"color":C,"points":[[data_x,data_y],...]}],
   "markers":[{"at":[data_x,data_y],"label":T,"color":C}]}
       a complete quantitative chart with scale, light grid, ticks, legend, exact data-space
       series, and labeled points. Prefer this over loose axes + curves whenever values matter.
  {"op":"dot","at":[x,y],"color":C,"label":T}
  {"op":"underline","at":[x,y],"w":W,"color":C}    place at the text it emphasizes (it snaps to it)
  {"op":"highlight","at":[x,y],"w":W}              translucent yellow swipe, snaps to nearby text
Every visible op MUST have a short unique "id". Use those ids in later erase ops.
In any text/label, use ^ for exponents — "3^2 + 4^2 = 5^2" renders as real superscripts.
Colors C: "ink", "blue", "red", "green", "orange", "purple", "gray".
Layout rules: title around y 5-12; main content between y 16 and y 68; keep 4+ units
of whitespace between elements; boxes need w >= 4.5 per label character at size "m";
never place two elements at overlapping coordinates. Use compact diagrams: the main
shape should usually occupy less than half the board width so there is room for
formulas, examples, and callouts. Use callouts instead of large explanatory text
inside the diagram.
Curves must trace the TRUE shape of what you describe — and REMEMBER y grows
DOWNWARD: higher on the board means SMALLER y. Plan every point: an ascent has
DECREASING y; a valley's minimum sits at the LARGEST y; a rocket reaching orbit
rises steeply (y dropping), then bends sideways into a flat line — it never
comes back down. A wrong-shaped curve teaches the wrong idea: check the points
against your own words before writing them. For graph ops, give honest axis ranges,
monotonic data where the concept is monotonic, enough points to show curvature, and
mark the meaningful intersection, optimum, threshold, or inflection."""

LESSON_SYSTEM = """You are Tutori, a warm, brilliant one-on-one tutor who teaches at a whiteboard.
You SPEAK each step aloud while your pen draws on the board — the drawing must
illustrate exactly what you are saying in that step.

Respond with ONE JSON object and NOTHING else (no markdown fences, no prose outside JSON):
{{
  "steps": [ {{"say": "...", "board": [ <board ops> ]}}, ... ]
}}

Rules for "say" (spoken by a TTS voice):
- Natural, friendly spoken language. Follow TARGET LESSON LENGTH below.
- Never use markdown, symbols, URLs or formulas as symbols: say "x squared", "pi", "H two O".
- Sound like a human tutor mid-conversation: vary your openings, address the learner directly.

Rules for "steps":
- Use exactly the requested step range and spoken depth below.
- Each step has 1 to 6 board ops that draw exactly what that step says, as you say it.
- The visual lesson should carry roughly half the teaching. If the spoken step names
  a part, value, result, or relationship, draw a compact mark for it instead of only
  saying it in the transcript.
- THE DRAWING MUST BE SPECIFIC TO THIS TOPIC. Draw the idea's REAL structure — actual
  names, values, shapes and relationships — never a generic input/process/output chain
  and never the same layout you drew last time. Examples of being specific:
    * recursion -> the actual call tree with real arguments at each node
    * binary search -> a row of small boxes with the real sorted numbers, dots and
      underlines marking lo, mid, hi as they move
    * supply and demand -> axes with the two labeled curves crossing at a dot
    * photosynthesis -> a leaf (ellipse) with labeled arrows in (sun, water, CO2) and out (O2, sugar)
- The board PERSISTS across turns and across this whole lesson. Treat CURRENT BOARD as
  real ink already visible. Preserve useful context. Use targeted {{"op":"erase"}}
  by id to retire stale marks; use {{"op":"clear"}} only for a true topic/chapter
  change or a full board. After clearing, immediately establish the next visual.
- For lessons longer than three minutes, deliberately cycle through visual chapters:
  add detail, summarize, erase obsolete details, then reuse the space. The board should
  always contain a coherent, useful diagram—not empty space or accumulated clutter.
- Step 1 usually includes a short title (a fresh one for this answer).
- Never place two elements of THIS turn at overlapping coordinates.
- Plan the layout: main diagram on the left two-thirds (x 4-62), side notes in the
  right column (x 66-96). Keep related items adjacent.
- Write a label and its value as ONE text op ("Context window: 1 million tokens"),
  never as two ops in different places. Arrow labels: 3 words maximum.
- Lines and arrows must never pass THROUGH a box, ellipse or text — connect edge to
  edge with short strokes, and leave 3+ units of gap between separate shapes.
- Arrows must be SHORT (under 24 units) and connect adjacent elements. Every arrowhead
  must land on a visible box, dot, note, text label, or curve point; never point into
  empty space. Move the elements closer instead of shooting an arrow across the board.
- Compact, well-placed elements beat oversized scattered ones. Everything that belongs
  together (a shape and its labels, a list of values) must be ONE op, not several.
- Use callout when a learner needs to know what a label means: circle the small label
  or point, draw a short leader line, and put the explanation at the end.
- CONTENT BUDGET: the whole lesson must fit comfortably — at most ONE main diagram
  (left two-thirds), ONE formula with highlight, and ONE notes block (right column).
  Target 12 to 16 compact drawn elements per lesson. Shrink the main diagram before
  dropping important visual details.
- For "how X works" lessons, do not stop at two boxes. Draw the working system:
  3 to 5 topic-specific parts, causal arrows with short labels, and one compact
  note block that explains what the diagram means.
- Values like "a = 3" are lines inside the notes block, NEVER standalone text ops.
- Each step draws ONLY NEW marks. Never redraw the title, the diagram, or anything
  already on the board — duplicates are discarded.
- Keep boxes and ellipses at most ~22 units wide and ~12 units tall unless one is THE main diagram.
- CURRENT BOARD below is live and still visible. Refer to its ids for targeted erasing.
- Use color with meaning (e.g. red = the thing to watch, green = result, gray = notes).
- The last step usually asks one short check-in question matched to the learner's pace.

Conversation behavior:
- The chat history is the lesson so far. On follow-ups: skip greetings and recaps,
  answer the actual question, go deeper or give a new angle — never re-teach what the
  learner already confirmed they understood.
- If the learner answered your check-in question, react to THEIR answer first.
- CURRENT LEARNER PROFILE below is background for pace and personal touches ONLY.
  The learner's question always sets the topic: switch topics instantly and
  enthusiastically — never steer back to last_topic or old goals, never draw them.

{board_reference}

LEARNER PACE SETTING: {pace}
CURRENT LEARNER PROFILE: {profile}
CURRENT BOARD (still visible): {board_now}
TARGET LESSON LENGTH: {duration_guidance}
AGENT BLUEPRINT: {agent_plan}
TODAY: {today}{notes_block}{web_block}"""

LESSON_SYSTEM_LITE = """You are Tutori, a warm, brilliant one-on-one tutor who teaches at a whiteboard.
You SPEAK each step aloud while a whiteboard artist draws what you describe.

Respond with ONE JSON object and NOTHING else (no markdown fences, no prose outside JSON):
{{
  "title": "a short board heading for this answer (2-5 words)",
  "steps": [ {{"say": "...", "draw": "..."}}, ... ]
}}

Rules for "say" (spoken by a TTS voice):
- Natural, friendly spoken language. 1-3 sentences, 15 to 35 words per step.
- Never use markdown, symbols, URLs or formulas as symbols: say "x squared", "pi", "H two O".
- Sound like a human tutor mid-conversation: vary your openings, address the learner directly.

Rules for "steps":
- EXACTLY 3 or 4 steps forming one mini whiteboard lesson that directly answers the learner.
- "draw" is ONE short imperative sentence telling the artist exactly what to add to the
  board during this step — the idea's REAL structure with actual names and values
  (e.g. "draw a right triangle with sides labeled a, b and c" or "plot the loss curve
  dipping to a minimum and mark the lowest point" or "add a notes block with a = 3,
  b = 4, c = 5" or "circle c and label it hypotenuse with a short leader line").
  Step 1 establishes the main diagram; later steps ADD to it; the whole lesson fits
  one board (one diagram, one formula, one notes block at most).
- Prefer compact labels and adjacent items. Ask the artist for short connectors only;
  no long arrows, and no arrows pointing into empty space.
- The last step usually asks one short check-in question matched to the learner's pace.

Conversation behavior:
- The chat history is the lesson so far. On follow-ups: skip greetings and recaps,
  answer the actual question, go deeper or give a new angle — never re-teach what the
  learner already confirmed they understood.
- If the learner answered your check-in question, react to THEIR answer first.
- CURRENT LEARNER PROFILE below is background for pace and personal touches ONLY.
  The learner's question always sets the topic: switch topics instantly and
  enthusiastically — never steer back to last_topic or old goals, never draw them.

LEARNER PACE SETTING: {pace}
CURRENT LEARNER PROFILE: {profile}
TODAY: {today}{notes_block}{web_block}"""

PACE_DESCRIPTIONS = {
    1: "Total beginner — tiny steps, everyday analogies, zero jargon, frequent reassurance.",
    2: "Beginner — gentle pace, define every term, simple examples.",
    3: "Intermediate — steady pace, some technical vocabulary, concrete examples.",
    4: "Advanced — brisk pace, technical depth, edge cases welcome.",
    5: "Expert — fast and dense, formal definitions, assume strong background.",
}

SEARCH_SYSTEM = """You are the research planner for a tutor agent. Today is {today}.
Decide whether a quick web search would materially improve a whiteboard lesson on the
learner's request. Conceptual/timeless topics (math, physics, programming basics,
history before this year) need NO search. Current events, prices, versions, schedules,
records, laws, government directives, access changes, model/product releases,
niche/new algorithms or papers, named companies, or anything after your training data
DOES need one. If the learner provides a URL, that source must be used. If the EXISTING
NOTES below already cover the question, do NOT search again.
Answer with ONE JSON object only:
{{"search": false}}  or  {{"search": true, "queries": ["...", "..."]}}  (max 2 queries)"""


# --------------------------------------------------------------------------
# OpenRouter helpers
# --------------------------------------------------------------------------

def openrouter_chat(messages, model=LLM_ID, max_tokens=512, temperature=0.35,
                    json_mode=False, reasoning=None, timeout=240):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if reasoning:
        payload["reasoning"] = {"effort": reasoning, "exclude": True}
    r = requests.post(
        f"{OPENROUTER_URL}/chat/completions",
        headers=_headers(), json=payload, timeout=timeout,
    )
    if not r.ok:
        raise RuntimeError(f"OpenRouter {model} returned {r.status_code}: {r.text[:240]}")
    data = r.json()
    choice = data["choices"][0]
    message = choice.get("message") or {}
    content = str(message.get("content") or "").strip()
    if not content:
        refusal = str(message.get("refusal") or "")[:160]
        raise RuntimeError(
            f"OpenRouter {model} returned no content "
            f"(finish={choice.get('finish_reason')}, refusal={refusal or 'none'})"
        )
    return content


def openrouter_chat_stream(messages, model=LLM_ID, max_tokens=512,
                           temperature=0.35, json_mode=False,
                           reasoning=None, timeout=240):
    """Yield visible text from an OpenRouter SSE response as it arrives."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": True,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if reasoning:
        payload["reasoning"] = {"effort": reasoning, "exclude": True}
    with requests.post(
        f"{OPENROUTER_URL}/chat/completions",
        headers=_headers(), json=payload, timeout=timeout, stream=True,
    ) as response:
        if not response.ok:
            raise RuntimeError(
                f"OpenRouter {model} returned {response.status_code}: "
                f"{response.text[:240]}"
            )
        emitted = False
        for raw_line in response.iter_lines(decode_unicode=True):
            line = str(raw_line or "").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("error"):
                raise RuntimeError(f"OpenRouter {model} stream failed: {event['error']}")
            choices = event.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if isinstance(content, str) and content:
                emitted = True
                yield content
            elif isinstance(content, list):
                for block in content:
                    text = block.get("text") if isinstance(block, dict) else ""
                    if text:
                        emitted = True
                        yield str(text)
        if not emitted:
            raise RuntimeError(f"OpenRouter {model} streamed no visible content")


def llm_generate(messages, max_new_tokens=256, temperature=0.35):
    return openrouter_chat(
        messages, model=LLM_ID, max_tokens=max_new_tokens,
        temperature=temperature, reasoning="low",
    )


def cpm_generate(messages, max_new_tokens=220):
    """Compatibility name for the fast Gemma coach/planner path."""
    return openrouter_chat(
        messages, model=COACH_ID, max_tokens=max_new_tokens,
        temperature=0.2,
    )


# --------------------------------------------------------------------------
# streaming lesson parser
# --------------------------------------------------------------------------

class LessonStreamParser:
    """Pulls completed step objects out of the streamed JSON as it arrives."""

    def __init__(self):
        self.buf = ""
        self.started = False
        self.pos = 0
        self.stuck = False

    def feed(self, chunk):
        self.buf += chunk
        if self.stuck:
            return []
        if not self.started:
            m = re.search(r'"steps"\s*:\s*\[', self.buf)
            if not m:
                return []
            self.started = True
            self.pos = m.end()
        out = []
        i = self.pos
        while i < len(self.buf):
            ch = self.buf[i]
            if ch == "{":
                obj, end = self._balanced(i)
                if end is None:           # object not complete yet
                    break
                if obj is None:           # malformed — stop incremental parsing
                    self.stuck = True
                    break
                out.append(obj)
                i = end
                self.pos = i
            elif ch == "]":
                self.pos = i
                break
            else:
                i += 1
                self.pos = i
        return out

    def _balanced(self, start):
        depth, instr, esc = 0, False, False
        for j in range(start, len(self.buf)):
            c = self.buf[j]
            if instr:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    instr = False
            else:
                if c == '"':
                    instr = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(self.buf[start:j + 1]), j + 1
                        except Exception:
                            return None, j + 1
        return None, None


def parse_lesson_json(text):
    """Best-effort full parse of the final LLM output."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for candidate in (cleaned, cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def extract_lesson_steps(text):
    """Find a step array even when a model wraps it in `lesson` or `response`."""
    root = parse_lesson_json(text) or _first_json_object(text) or {}
    queue = [root]
    seen = set()
    while queue:
        obj = queue.pop(0)
        if not isinstance(obj, dict) or id(obj) in seen:
            continue
        seen.add(id(obj))
        steps = obj.get("steps")
        if isinstance(steps, list) and any(isinstance(s, dict) for s in steps):
            return steps
        for value in obj.values():
            if isinstance(value, dict):
                queue.append(value)
    return []


# ---- board layout hygiene: the model's spatial reasoning is approximate, ----
# ---- so clamp out-of-bounds coordinates and nudge overlapping text.       ----

_TEXT_H = {"s": 3.6, "m": 4.8, "l": 6.2, "xl": 7.4}


def _clamp_pt(pt, lo_x=2, hi_x=98, lo_y=3, hi_y=72):
    try:
        return [min(max(float(pt[0]), lo_x), hi_x), min(max(float(pt[1]), lo_y), hi_y)]
    except Exception:
        return [50, 38]


def _op_bbox(op):
    """Rough bounding box (x1, y1, x2, y2) for collision checks; None = skip."""
    kind = op.get("op")
    if kind == "title":
        return (8, 1, 92, 13)  # long titles render nearly full-width
    if kind in ("text", "note"):
        x, y = op.get("at", [10, 20])
        size = op.get("size", "m" if kind == "text" else "s")
        w = min(len(str(op.get("text", ""))) * {"s": 1.45, "m": 1.9, "l": 2.35, "xl": 2.9}.get(size, 1.9), 48)
        lines = max(1, int(w // 52) + 1)
        h = _TEXT_H.get(size, 5.5) * lines
        if op.get("align") == "center":
            x -= w / 2
        return (x, y - h / 2, x + w, y + h / 2)
    if kind in ("box", "ellipse", "axes", "graph"):
        x, y = op.get("at", [10, 20])
        return (x, y, x + float(op.get("w", 20) or 20), y + float(op.get("h", 10) or 10))
    if kind == "polygon":
        pts = op.get("points") or [[10, 20]]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pad = 6 if op.get("side_labels") else 0   # side labels live just outside
        return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    if kind == "notes":
        x, y = op.get("at", [66, 20])
        lines = ([str(op["title"])] if op.get("title") else []) + \
                [str(t) for t in (op.get("lines") or [])[:7]]
        w = min(max((len(s) for s in lines), default=8) * 1.7 + 3, 42)
        h = 4 + len(lines) * 5.4
        if op.get("compact"):
            w, h = w * 0.78, h * 0.74
        return (x - 1, y - 3, x + w, y + h)
    if kind == "callout" and op.get("label"):
        x, y = op.get("label_at") or op.get("to") or [55, 35]
        w = min(len(str(op.get("label", ""))) * 1.45, 28)
        return (x - w / 2, y - 2.5, x + w / 2, y + 2.5)
    return None


def _overlaps(a, b):
    return a[0] < b[2] - 0.8 and a[2] > b[0] + 0.8 and a[1] < b[3] - 0.8 and a[3] > b[1] + 0.8


# short NUMERIC value statements like "a = 3" or "c = ?" — these gravitate
# into one cluster. Word definitions ("a = height") stay where intended.
_VALUE_RE = re.compile(r"^\s*[\w^()]{1,6}\s*[=≈→]\s*[\d?][\w.?]{0,7}\s*$")


def _score(bbox, placed):
    """Weighted overlap: thin strokes barely count (text over a line is fine
    on a real whiteboard), the title zone counts extra, solids count fully."""
    s = 0.0
    for p in placed:
        a = _overlap_area(bbox, p)
        if a:
            s += a * (p[4] if len(p) > 4 else 1.0)
    return s


def _free_spot(bbox_fn, start, placed, candidates):
    """Try offsets around `start`; if nothing is fully free, take the spot
    with the least total overlap rather than giving up."""
    best = None
    for dx, dy in candidates:
        pos = _clamp_pt([start[0] + dx, start[1] + dy])
        bbox = bbox_fn(pos)
        score = _score(bbox, placed)
        if score <= 0.5:
            return pos, bbox
        if best is None or score < best[2]:
            best = (pos, bbox, score)
    return best[0], best[1]


_TEXT_CANDIDATES = [(0, 0), (0, 6), (0, -7), (0, 12), (14, 0), (-14, 0),
                    (16, 8), (-16, 8), (0, 18), (20, -8)]
_LABEL_CANDIDATES = [(0, -4), (0, 5), (8, -4), (-8, -4), (10, 4), (-10, 4), (0, -9)]
_SHAPE_CANDIDATES = [(0, 0), (0, 7), (9, 0), (-9, 0), (0, -8), (12, 9), (-12, 9),
                     (0, 15), (18, 0), (-18, 0)]


def _overlap_area(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * \
           max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _shift_pt(pt, displaced):
    """If a point targets a shape's ORIGINAL position, follow the shape."""
    for (x1, y1, x2, y2), dx, dy in reversed(displaced):
        if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
            return _clamp_pt([pt[0] + dx, pt[1] + dy])
    return pt


def _snap_endpoint_to_edge(pt, other, placed):
    """If an arrow endpoint sits inside a solid element, move it to the edge.

    LLMs often aim arrows at a box's center. That is semantically clear in
    text, but visually it draws through the shape. Snapping the endpoint to
    the rectangle boundary preserves the relationship without crossing labels.
    """
    try:
        x, y = float(pt[0]), float(pt[1])
        ox, oy = float(other[0]), float(other[1])
    except Exception:
        return pt
    for p in reversed(placed):
        if len(p) > 4 and p[4] < 0.5:
            continue
        x1, y1, x2, y2 = p[:4]
        w, h = x2 - x1, y2 - y1
        if y2 <= 16 or w < 8 or h < 6:
            continue
        if w > 44 and h > 24:
            # Plot/diagram regions such as axes are containers, not solid
            # targets. Arrows are allowed to live inside them.
            continue
        if not (x1 < x < x2 and y1 < y < y2):
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = ox - cx, oy - cy
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            return pt
        scale = min(
            abs((x2 - cx) / dx) if dx else 1e9,
            abs((y2 - cy) / dy) if dy else 1e9,
        )
        return _clamp_pt([cx + dx * scale, cy + dy * scale])
    return pt


def _seg_samples(a, b, step_len=7.0):
    """Points along a stroke — registered as small obstacles so shapes and
    text placed later don't sit on top of lines."""
    n = max(1, int((((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5) / step_len))
    return [(a[0] + (b[0] - a[0]) * t / n, a[1] + (b[1] - a[1]) * t / n)
            for t in range(n + 1)]


def _solid_area(placed):
    """Total area already claimed by solid elements (strokes excluded)."""
    s = 0.0
    for p in placed:
        if len(p) > 4 and p[4] < 0.5:
            continue
        s += max(0.0, (p[2] - p[0]) * (p[3] - p[1]))
    return s


def _stroke_len(a, b):
    try:
        return ((float(b[0]) - float(a[0])) ** 2 + (float(b[1]) - float(a[1])) ** 2) ** 0.5
    except Exception:
        return 0.0


def _cap_arrow(op, max_len=28.0):
    """Trim long arrows while preserving the arrowhead's intended target."""
    fx, fy = op.get("from", [0, 0])
    tx, ty = op.get("to", [0, 0])
    length = _stroke_len([fx, fy], [tx, ty])
    if length > max_len:
        s = max_len / length
        op["from"] = [tx - (tx - fx) * s, ty - (ty - fy) * s]


def _cap_callout(op, max_len=24.0):
    """Trim callout leaders while preserving the circled source."""
    ax, ay = op.get("around", [0, 0])
    tx, ty = op.get("to", [0, 0])
    length = _stroke_len([ax, ay], [tx, ty])
    if length > max_len:
        s = max_len / length
        op["to"] = [ax + (tx - ax) * s, ay + (ty - ay) * s]


def _target_bboxes(ops):
    """Visible non-stroke anchors that arrowheads/sources may connect to."""
    boxes = []
    for op in ops:
        if op.get("_drop"):
            continue
        kind = op.get("op")
        try:
            if kind in ("box", "ellipse", "polygon", "notes", "text", "note", "axes", "graph", "callout"):
                bb = _op_bbox(op)
                if bb and bb[3] > 15:
                    boxes.append(bb)
            elif kind == "dot" and op.get("at"):
                x, y = op["at"]
                boxes.append((x - 2.5, y - 2.5, x + 2.5, y + 2.5))
        except Exception:
            pass
    return boxes


def _near_any_box(pt, boxes, pad=5.0):
    try:
        x, y = float(pt[0]), float(pt[1])
    except Exception:
        return False
    return any(x1 - pad <= x <= x2 + pad and y1 - pad <= y <= y2 + pad
               for x1, y1, x2, y2 in boxes)


def layout_pass(board, placed, displaced, anchors, state):
    """Clamp coordinates into the visible board and resolve collisions.

    Pass 1: shapes claim space, dodging anything already drawn (title zone,
    earlier text, other shapes). When a shape moves, the displacement is
    remembered so later arrows/lines aimed at its original spot follow it.
    Pass 2: arrow/line endpoints get shifted, then floating text and arrow
    labels are placed into genuinely free spots.
    """
    # once a title exists this turn, free strokes may not invade its band —
    # curves/lines/arrows can't be relocated like shapes, so clamp instead
    if any(isinstance(op, dict) and op.get("op") == "title" for op in board):
        state["has_title"] = True
    stroke_ylo = 16.5 if state.get("has_title") else 3

    out = []
    for op in board:
        op = dict(op)
        kind = op.get("op")
        stroke = kind in ("curve", "line", "arrow")
        ylo = stroke_ylo if stroke else 3
        try:
            if "at" in op:
                op["at"] = _clamp_pt(op["at"])
            if "from" in op:
                op["from"] = _clamp_pt(op["from"], lo_y=ylo)
            if "to" in op:
                op["to"] = _clamp_pt(op["to"], lo_y=ylo)
            if "around" in op:
                op["around"] = _clamp_pt(op["around"], lo_y=ylo)
            if "points" in op and isinstance(op["points"], list):
                op["points"] = [_clamp_pt(p, lo_y=ylo) for p in op["points"][:12]]
            if kind in ("box", "ellipse", "axes", "graph"):
                x, y = op.get("at", [10, 20])
                op["w"] = min(float(op.get("w", 20) or 20), 96 - x)
                op["h"] = min(float(op.get("h", 10) or 10), 71 - y)
        except Exception:
            pass
        # board capacity accounting: when solid content would exceed what the
        # board can hold, drop the op (the narration still covers it) — an
        # over-full board cannot be laid out without overlaps by anyone.
        try:
            if kind in ("box", "ellipse", "axes", "graph", "polygon", "notes",
                        "text", "note", "title"):
                bb = _op_bbox(op)
                if bb:
                    est = (bb[2] - bb[0]) * (bb[3] - bb[1])
                    if kind == "notes":
                        est *= 0.6      # it will be compacted under pressure
                    used = state.get("area_used", 0.0)
                    if kind != "title" and used + est > 5200:
                        continue        # skip this op entirely
                    state["area_used"] = used + est
        except Exception:
            pass
        out.append(op)

    # pass 1: titles register; shapes dodge everything already on the board
    for op in out:
        try:
            kind = op.get("op")
            if kind == "title":
                tb = _op_bbox(op)
                placed.append((tb[0], tb[1], tb[2], tb[3], 2.5))
            elif kind in ("box", "ellipse", "axes", "graph", "polygon", "notes"):
                if kind == "notes" and not op.get("compact") and \
                        _solid_area(placed) > 2600:
                    op["compact"] = True   # busy board: start small
                min_w = len(str(op.get("label", ""))) * 2.2 + 6
                best = None
                for _attempt in range(6):
                    if kind in ("polygon", "notes"):
                        bx = _op_bbox(op)
                        x0, y0 = bx[0], bx[1]
                        w, h = bx[2] - bx[0], bx[3] - bx[1]
                    else:
                        x0, y0 = op.get("at", [10, 20])
                        w, h = float(op.get("w", 20) or 20), float(op.get("h", 10) or 10)
                    best = None
                    for dx, dy in _SHAPE_CANDIDATES:
                        px = min(max(x0 + dx, 2), 96 - w)
                        py = min(max(y0 + dy, 3), 71 - h)
                        bbox = (px, py, px + w, py + h)
                        score = _score(bbox, placed)
                        if score <= 0.5:
                            best = ([px, py], bbox, score)
                            break
                        if best is None or score < best[2]:
                            best = ([px, py], bbox, score)
                    if best[2] > 1.0:
                        # local candidates all collide — scan the whole board
                        # for the clearest spot, biased near the original
                        for py in range(16, max(17, int(72 - h)), 5):
                            for px in range(3, max(4, int(95 - w)), 5):
                                bbox = (px, py, px + w, py + h)
                                score = _score(bbox, placed)
                                score += 0.02 * (abs(px - x0) + abs(py - y0))
                                if score < best[2]:
                                    best = ([px, py], bbox, score)
                    if best[2] <= 1.0:
                        break
                    # board is crowded — draw the element smaller, like a human
                    if kind == "notes":
                        if not op.get("compact"):
                            op["compact"] = True   # smaller font, tighter spacing
                            continue
                        if len(op.get("lines") or []) > 2:
                            op["lines"] = op["lines"][:-1]   # shed a line
                            continue
                        break
                    if kind == "polygon":
                        cx = sum(p[0] for p in op["points"]) / len(op["points"])
                        cy = sum(p[1] for p in op["points"]) / len(op["points"])
                        if w * 0.72 < 12:
                            break
                        op["points"] = [[cx + (p[0] - cx) * 0.72,
                                         cy + (p[1] - cy) * 0.72] for p in op["points"]]
                    else:
                        nw, nh = w * 0.72, max(h * 0.72, 8)
                        if nw < max(11, min_w):
                            nw = w           # width locked by the label
                            if h * 0.72 < 8:
                                break        # can't shrink any further
                            nh = h * 0.72
                        op["w"], op["h"] = nw, nh
                pos, bbox, _ = best
                if best[2] > 6.0:
                    # even the least-bad spot overlaps visibly — a missing
                    # element reads far better than a pile-up. Drop it (and
                    # remember its footprint so arrows aimed at it drop too).
                    op["_drop"] = True
                    state.setdefault("dropped", []).append(
                        (x0 - 3, y0 - 3, x0 + w + 3, y0 + h + 3))
                    continue
                if abs(pos[0] - x0) > 0.5 or abs(pos[1] - y0) > 0.5:
                    displaced.append(((x0 - 2, y0 - 2, x0 + w + 2, y0 + h + 2),
                                      pos[0] - x0, pos[1] - y0))
                dx, dy = pos[0] - x0, pos[1] - y0
                if kind == "polygon":
                    op["points"] = [[p[0] + dx, p[1] + dy] for p in op["points"]]
                elif kind == "notes":
                    ax, ay = op.get("at", [66, 20])
                    op["at"] = [ax + dx, ay + dy]
                else:
                    op["at"] = pos
                if kind == "polygon":
                    # interior is mostly empty space; edges are sampled
                    # separately, so the bbox itself counts at reduced weight
                    placed.append((bbox[0], bbox[1], bbox[2], bbox[3], 0.6))
                else:
                    placed.append(bbox)
        except Exception:
            pass

    # pass 2a: connections follow any shapes that moved, then strokes register
    # as obstacles for everything placed after them
    for op in out:
        try:
            kind = op.get("op")
            if kind in ("arrow", "line"):
                op["from"] = _shift_pt(op["from"], displaced)
                op["to"] = _shift_pt(op["to"], displaced)
                if kind == "arrow":
                    op["from"] = _snap_endpoint_to_edge(op["from"], op["to"], placed)
                    op["to"] = _snap_endpoint_to_edge(op["to"], op["from"], placed)
                for (dx1, dy1, dx2, dy2) in state.get("dropped", []):
                    if (dx1 <= op["from"][0] <= dx2 and dy1 <= op["from"][1] <= dy2) or \
                            (dx1 <= op["to"][0] <= dx2 and dy1 <= op["to"][1] <= dy2):
                        op["_drop"] = True
                        break
                if op.get("_drop"):
                    continue
                if kind == "arrow":
                    # cap length, keeping the head on its target — prevents
                    # board-spanning arrows the prompt alone can't stop
                    _cap_arrow(op)
                for x, y in _seg_samples(op["from"], op["to"], 5.0):
                    placed.append((x - 2.2, y - 2.2, x + 2.2, y + 2.2, 0.25))
            elif kind == "callout":
                op["around"] = _shift_pt(op.get("around", [50, 38]), displaced)
                op["to"] = _shift_pt(op.get("to", [60, 32]), displaced)
                _cap_callout(op)
                r = float(op.get("r", 3.0) or 3.0)
                ax, ay = op["around"]
                placed.append((ax - r - 1.2, ay - r - 1.2,
                               ax + r + 1.2, ay + r + 1.2, 0.35))
                for x, y in _seg_samples(op["around"], op["to"], 5.0):
                    placed.append((x - 1.8, y - 1.8, x + 1.8, y + 1.8, 0.2))
            elif kind in ("curve", "polygon"):
                pts = list(op.get("points") or [])
                if kind == "curve" and len(pts) >= 2:
                    state["last_curve"] = [list(p) for p in pts]
                if kind == "polygon" and len(pts) >= 3:
                    pts = pts + [pts[0]]          # include the closing edge
                for i in range(len(pts) - 1):
                    for x, y in _seg_samples(pts[i], pts[i + 1], 6.0):
                        placed.append((x - 2.2, y - 2.2, x + 2.2, y + 2.2, 0.25))
            elif kind == "dot" and "at" in op:
                op["at"] = _shift_pt(op["at"], displaced)
        except Exception:
            pass

    # pass 2a'': "goal" and "start" dots mean specific points on a curve —
    # models often drop them somewhere decorative instead
    curve = state.get("last_curve")
    if curve and len(curve) >= 2:
        for op in out:
            try:
                if op.get("op") != "dot" or "at" not in op:
                    continue
                low = str(op.get("label") or "").lower()
                if any(k in low for k in ("goal", "minimum", "lowest")) or low.strip() == "min":
                    op["at"] = list(max(curve, key=lambda p: p[1]))  # y grows down
                elif "start" in low:
                    op["at"] = list(curve[0])
            except Exception:
                pass

    # pass 2b: floating text finds free space near where the model wanted it
    text_disp = []   # texts that moved — arrows pointing at them must follow
    for op in out:
        try:
            kind = op.get("op")
            if kind in ("text", "note"):
                model_at = list(op.get("at", [10, 20]))   # where the model put it
                orig_at = model_at
                # scattered "a = 3" style values stack under the first one
                is_value = bool(_VALUE_RE.match(str(op.get("text", ""))))
                if is_value and "value_cursor" in state:
                    orig_at = list(state["value_cursor"])


                raw_bbox = _op_bbox(op)
                est_w = raw_bbox[2] - raw_bbox[0]
                est_h = raw_bbox[3] - raw_bbox[1]

                def fit_x(pos, _w=est_w, _h=est_h, _c=op.get("align") == "center"):
                    # keep the whole text on the board — applied BEFORE the
                    # collision check (shifting after it caused real overlaps)
                    if _c:
                        x = min(max(pos[0], _w / 2 + 2), 98 - _w / 2)
                    else:
                        x = min(pos[0], max(3, 97 - _w))
                    y = min(max(pos[1], _h / 2 + 2), 73 - _h / 2)
                    return [x, y]

                def bbox_at(pos, _op=op):
                    probe = dict(_op); probe["at"] = fit_x(pos)
                    return _op_bbox(probe)
                pos, bbox = _free_spot(bbox_at, orig_at, placed, _TEXT_CANDIDATES)
                if _score(bbox, placed) > 0.75:
                    # neighborhood is full — scan the whole board, biased near
                    # where the model wanted the text
                    best = (pos, bbox, 1e9)
                    for py in range(16, 70, 5):
                        for px in range(3, 96, 5):
                            b = bbox_at([px, py])
                            score = _score(b, placed)
                            score += 0.03 * (abs(px - orig_at[0]) + abs(py - orig_at[1]))
                            if score < best[2]:
                                best = ([px, py], b, score)
                    pos, bbox = best[0], best[1]
                pos = fit_x(pos)
                if _score(bbox_at(pos), placed) > 6.0:
                    op["_drop"] = True       # unplaceable text is spoken, not drawn
                    continue
                op["at"] = pos
                final_bbox = bbox_at(pos)
                # register with a little breathing room around the text
                placed.append((final_bbox[0] - 1.8, final_bbox[1] - 1.8,
                               final_bbox[2] + 1.8, final_bbox[3] + 1.8))
                anchors.append({"orig": model_at, "bbox": final_bbox})
                if abs(pos[0] - model_at[0]) > 0.5 or abs(pos[1] - model_at[1]) > 0.5:
                    text_disp.append(((raw_bbox[0] - 4, raw_bbox[1] - 4,
                                       raw_bbox[2] + 4, raw_bbox[3] + 4),
                                      pos[0] - model_at[0], pos[1] - model_at[1]))
                if is_value:
                    top = state.setdefault("value_col_top", pos[1])
                    next_y = final_bbox[3] + 5.5
                    if next_y > 64:   # column full — wrap beside it, same top
                        state["value_cursor"] = [pos[0] + 15, top]
                    else:
                        state["value_cursor"] = [pos[0], next_y]
            elif kind in ("arrow", "dot", "callout") and op.get("label"):
                if kind == "arrow":
                    fx, fy = op.get("from", [10, 20]); tx, ty = op.get("to", [30, 20])
                    start = [(fx + tx) / 2, (fy + ty) / 2]
                elif kind == "callout":
                    start = list(op.get("to", [55, 35]))
                else:
                    ax, ay = op.get("at", [50, 38])
                    start = [ax + 3, ay - 3]
                w = min(len(str(op["label"])) * 1.6, 30)

                def lbl_fit(pos, _w=w):
                    return [min(max(pos[0], _w / 2 + 2), 98 - _w / 2),
                            min(max(pos[1], 4), 71)]

                def lbl_bbox(pos, _w=w):
                    p = lbl_fit(pos)
                    return (p[0] - _w / 2, p[1] - 2.5, p[0] + _w / 2, p[1] + 2.5)
                pos, bbox = _free_spot(lbl_bbox, start, placed,
                                       _LABEL_CANDIDATES + [(0, 10), (0, -14),
                                                            (16, 0), (-16, 0),
                                                            (14, 10), (-14, 10)])
                if _score(bbox, placed) > 1.5:
                    op["label"] = None        # nowhere readable — say it, don't draw it
                else:
                    op["label_at"] = lbl_fit(pos)
                    placed.append(bbox)
        except Exception:
            pass

    # pass 2b': arrows chase any text that moved (e.g. the label an arrow
    # points from), then re-cap their length so heads stay on target
    if text_disp:
        for op in out:
            try:
                if op.get("op") in ("arrow", "line") and not op.get("_drop"):
                    op["from"] = _shift_pt(op["from"], text_disp)
                    op["to"] = _shift_pt(op["to"], text_disp)
                    if op.get("op") == "arrow":
                        _cap_arrow(op)
            except Exception:
                pass

    # pass 2b'': an unlabeled connector whose head/source does not touch
    # visible content reads like it points to nowhere. Drop it instead of
    # leaving a confusing stray arrow on the board.
    current_targets = _target_bboxes(out)
    targets = list(state.get("target_boxes", [])) + current_targets
    for op in out:
        try:
            if op.get("op") == "arrow" and not op.get("_drop") and not op.get("label"):
                if not (_near_any_box(op.get("from", []), targets) and
                        _near_any_box(op.get("to", []), targets)):
                    op["_drop"] = True
        except Exception:
            pass
    if current_targets:
        state["target_boxes"] = (list(state.get("target_boxes", [])) + current_targets)[-80:]

    # pass 2c: highlights/underlines snap onto the text they emphasize —
    # model coordinates go stale the moment that text gets nudged. A marker
    # swipe that matches no text means nothing: drop it rather than leave an
    # orphaned yellow blob floating on the board.
    kept = []
    for op in out:
        try:
            kind = op.get("op")
            if kind not in ("underline", "highlight") or "at" not in op:
                kept.append(op)
                continue
            ox, oy = op["at"]
            best = None
            for a in anchors:
                d = abs(a["orig"][0] - ox) + abs(a["orig"][1] - oy)
                if d <= 30 and (best is None or d < best[0]):
                    best = (d, a)
            if best:
                x1, y1, x2, y2 = best[1]["bbox"]
                if kind == "highlight":
                    op["at"] = [x1 - 1, (y1 + y2) / 2]
                    op["w"] = (x2 - x1) + 2
                else:
                    op["at"] = [x1, y2 + 0.8]
                    op["w"] = (x2 - x1)
                kept.append(op)
            # no anchor: drop the op
        except Exception:
            kept.append(op)
    return [op for op in kept if not op.get("_drop")]


# models sometimes double-escape unicode in JSON ("\\u2192") — json.loads
# then leaves the literal 6-character sequence, which would be drawn/spoken
_UESC = re.compile(r"\\u([0-9a-fA-F]{4})")


def _fix_text(s):
    s = _UESC.sub(lambda m: chr(int(m.group(1), 16)), str(s))
    return s.replace("\\n", " ").replace("\\t", " ")


# the model occasionally invents reasonable op names — translate, don't drop
_OP_ALIASES = {"triangle": "polygon", "poly": "polygon", "shape": "polygon",
               "rect": "box", "rectangle": "box", "square": "box",
               "circle": "ellipse", "oval": "ellipse",
               "label": "text", "math": "text", "formula": "text",
               "leader": "callout", "annotation": "callout"}


def _normalize_op(op):
    kind = str(op.get("op", "")).lower()
    op["op"] = _OP_ALIASES.get(kind, kind)
    for key in ("text", "label", "title", "xlabel", "ylabel"):
        if isinstance(op.get(key), str):
            op[key] = _fix_text(op[key])
    for key in ("lines", "side_labels"):
        if isinstance(op.get(key), list):
            op[key] = [_fix_text(t) if isinstance(t, str) else t
                       for t in op[key]]
    if op["op"] == "notes" and "items" in op and "lines" not in op:
        op["lines"] = op.pop("items")
    if op["op"] == "notes":  # bound the block so it can always be placed
        if op.get("title"):
            op["title"] = str(op["title"])[:26]
        op["lines"] = [str(t)[:24] for t in (op.get("lines") or [])][:5]
    if op["op"] == "callout":
        if "around" not in op and "at" in op:
            op["around"] = op.get("at")
        if "to" not in op and "label_at" in op:
            op["to"] = op.get("label_at")
        try:
            op["r"] = min(max(float(op.get("r", 3.0) or 3.0), 1.8), 6.0)
        except Exception:
            op["r"] = 3.0
        if op.get("label"):
            op["label"] = str(op["label"])[:26]
    # very long display text at large sizes is a layout bomb — demote size
    if op["op"] == "text":
        tlen = len(str(op.get("text", "")))
        if op.get("size") == "l" and tlen > 12:
            op["size"] = "m"
        if op.get("size", "m") == "m" and tlen > 22:
            op["size"] = "s"
    # degenerate shapes (w or h near zero) render as slivers with leaking
    # labels — give every box/ellipse a sane minimum footprint
    if op["op"] in ("box", "ellipse"):
        try:
            w = float(op.get("w", 22) or 22)
            h = float(op.get("h", 10) or 10)
            label_len = len(str(op.get("label", "")))
            max_w = 34.0 if label_len > 16 else 28.0
            op["w"] = min(max(w, 8.0), max_w)
            op["h"] = min(max(h, 6.0), 15.0)
            if op.get("label") and len(str(op["label"])) <= 4 and w / max(h, 1) > 4:
                # a tiny label on an extreme sliver: the model meant a small
                # square-ish tag, not a banner
                op["w"] = min(op["w"], 14.0)
                op["h"] = max(op["h"], 8.0)
        except Exception:
            pass
    if op["op"] == "graph":
        try:
            op["w"] = min(88.0, max(28.0, float(op.get("w", 54) or 54)))
            op["h"] = min(48.0, max(20.0, float(op.get("h", 34) or 34)))
            for key in ("x_range", "y_range"):
                vals = op.get(key) or [0, 1]
                lo, hi = float(vals[0]), float(vals[1])
                if hi <= lo:
                    hi = lo + 1.0
                op[key] = [lo, hi]
            clean_series = []
            for s in (op.get("series") or [])[:4]:
                if not isinstance(s, dict):
                    continue
                pts = []
                for p in (s.get("points") or [])[:18]:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        pts.append([float(p[0]), float(p[1])])
                if len(pts) >= 2:
                    clean_series.append({
                        "label": str(s.get("label") or f"series {len(clean_series) + 1}")[:18],
                        "color": str(s.get("color") or ("blue", "red", "green", "purple")[len(clean_series)]),
                        "points": pts,
                    })
            op["series"] = clean_series
            clean_markers = []
            for m in (op.get("markers") or [])[:6]:
                if isinstance(m, dict) and isinstance(m.get("at"), (list, tuple)):
                    clean_markers.append({
                        "at": [float(m["at"][0]), float(m["at"][1])],
                        "label": str(m.get("label") or "")[:18],
                        "color": str(m.get("color") or "orange"),
                    })
            op["markers"] = clean_markers
            op["title"] = str(op.get("title") or "")[:28]
            op["xlabel"] = str(op["xlabel"] if "xlabel" in op else "x")[:14]
            op["ylabel"] = str(op["ylabel"] if "ylabel" in op else "y")[:14]
        except Exception:
            op["series"] = []

    # geometry semantics: on a triangle, the hypotenuse / "c" label belongs
    # on the longest side — models often rotate the labels off by one
    if op["op"] == "polygon":
        pts = op.get("points") or []
        labs = op.get("side_labels") or []
        if len(pts) == 3 and len(labs) == 3:
            def _elen(i):
                a, b = pts[i], pts[(i + 1) % 3]
                return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
            longest = max(range(3), key=_elen)
            def _is_hyp(s):
                s = str(s).lower().strip()
                return ("hyp" in s or s == "c" or s.startswith("c ")
                        or s.startswith("c=") or s.startswith("c("))
            hyp_idx = [i for i in range(3) if _is_hyp(labs[i])]
            if len(hyp_idx) == 1 and hyp_idx[0] != longest:
                i = hyp_idx[0]
                labs[i], labs[longest] = labs[longest], labs[i]
                op["side_labels"] = labs

    # a box labeled "...Triangle" is the model drawing the wrong shape —
    # convert it into an actual right triangle of the same footprint
    label = str(op.get("label", "")).lower()
    if op["op"] in ("box", "ellipse") and "triangle" in label:
        try:
            x, y = op.get("at", [20, 30])
            w = float(op.get("w", 26) or 26)
            h = float(op.get("h", 20) or 20)
            op["op"] = "polygon"
            op["points"] = [[x, y], [x, y + h], [x + w, y + h]]
            op.pop("w", None); op.pop("h", None)
        except Exception:
            pass
    if "r" in op and "w" not in op and "at" in op:   # circle given as center+radius
        try:
            r = float(op.pop("r"))
            op["at"] = [op["at"][0] - r, op["at"][1] - r]
            op["w"] = op["h"] = 2 * r
        except Exception:
            pass
    return op


def _op_sig(op):
    """Signature for duplicate detection — position-independent, since the
    layout engine relocates repeats, scattering copies across the board."""
    k = op.get("op")
    if k == "title":
        return ("title",)                      # one title per turn, period
    if k in ("box", "ellipse", "axes", "graph", "polygon"):
        bb = _op_bbox(op) or (0, 0, 0, 0)
        return (k, str(op.get("label", "")).lower(),
                tuple(str(s).lower() for s in (op.get("side_labels") or ())) or
                tuple(str(s.get("label", "")).lower() for s in (op.get("series") or ())),
                round((bb[2] - bb[0]) / 6), round((bb[3] - bb[1]) / 6))
    if k == "notes":
        return ("notes", str(op.get("title", "")).lower())
    if k in ("text", "note"):
        return ("text", str(op.get("text", "")).lower().strip())
    if k in ("arrow", "line"):
        f, t = op.get("from", [0, 0]), op.get("to", [0, 0])
        try:
            return (k, str(op.get("label", "")).lower(),
                    round(f[0] / 6), round(f[1] / 6), round(t[0] / 6), round(t[1] / 6))
        except Exception:
            return None
    if k == "callout":
        a, t = op.get("around", [0, 0]), op.get("to", [0, 0])
        try:
            return (k, str(op.get("label", "")).lower(),
                    round(a[0] / 4), round(a[1] / 4), round(t[0] / 6), round(t[1] / 6))
        except Exception:
            return None
    return None  # dots / highlights / underlines / clear may repeat


def dedup_ops(board, seen):
    """The model often re-describes the whole board each step; draw only what
    is actually new."""
    out = []
    for op in board:
        sig = _op_sig(op)
        if sig is not None:
            if sig in seen:
                continue
            seen.add(sig)
        out.append(op)
    return out


def _merge_text_groups(board):
    """A header text ("Focus Areas:") with items drawn under it becomes ONE
    notes block — the layout engine moves blocks whole, never scattering."""
    out, i = [], 0
    while i < len(board):
        op = board[i]
        txt = str(op.get("text", "")).strip() if op.get("op") == "text" else ""
        if txt.endswith(":") and 3 < len(txt) <= 26 and op.get("at"):
            members, last = [], op
            j = i + 1
            while j < len(board):
                nxt = board[j]
                if nxt.get("op") != "text" or not nxt.get("at"):
                    break
                dy = nxt["at"][1] - last["at"][1]
                dx = abs(nxt["at"][0] - op["at"][0])
                if -1.0 <= dy <= 15.0 and dx <= 16.0:
                    members.append(nxt)
                    last = nxt
                    j += 1
                else:
                    break
            if members:
                out.append({"op": "notes", "title": txt,
                            "lines": [str(m.get("text", ""))[:26] for m in members],
                            "at": list(op["at"]),
                            "color": op.get("color", "ink")})
                i = j
                continue
        out.append(op)
        i += 1
    return out


def sanitize_step(step, step_index=0):
    if not isinstance(step, dict):
        return None
    say = _fix_text(step.get("say", "")).strip()
    board = step.get("board") or []
    if not isinstance(board, list):
        board = []
    board = [_normalize_op(op) for op in board
             if isinstance(op, dict) and "op" in op][:8]
    board = _merge_text_groups(board)
    if not say and not board:
        return None
    # Clear and targeted erase are first-class agent actions. They let a long
    # lesson reuse the same physical board without turning it into a collage.
    controls = [op for op in board if op.get("op") in ("clear", "erase")]
    draws = [op for op in board if op.get("op") not in ("clear", "erase")]
    for i, op in enumerate(draws):
        if not op.get("id"):
            op["id"] = f"s{step_index + 1}_o{i + 1}"
        op["id"] = re.sub(r"[^a-zA-Z0-9_-]", "_", str(op["id"]))[:48]
    for op in controls:
        if op.get("op") == "erase":
            targets = op.get("targets") or ([op.get("target")] if op.get("target") else [])
            op["targets"] = [str(t)[:48] for t in targets if t][:20]
    return {"say": say[:650], "board": controls + draws}


# --------------------------------------------------------------------------
# speech
# --------------------------------------------------------------------------

def transcribe(audio_path):
    suffix = os.path.splitext(str(audio_path))[1].lower().lstrip(".") or "wav"
    with open(audio_path, "rb") as f:
        payload = {
            "model": ASR_ID,
            "input_audio": {
                "data": base64.b64encode(f.read()).decode("ascii"),
                "format": suffix,
            },
        }
    r = requests.post(
        f"{OPENROUTER_URL}/audio/transcriptions",
        headers=_headers(), json=payload, timeout=180,
    )
    if not r.ok:
        raise RuntimeError(f"Parakeet transcription failed ({r.status_code}): {r.text[:200]}")
    return str(r.json().get("text") or "").strip()


_SPOKEN_CLEAN = re.compile(r"[*_#`<>\[\]{}|\\~^]")
_TTS_BLOCKED_UNTIL = 0.0
_TTS_BLOCK_REASON = ""


def synthesize(text):
    """Text -> (base64 MP3, estimated duration). Browser corrects from decoded audio."""
    global _TTS_BLOCKED_UNTIL, _TTS_BLOCK_REASON
    spoken = _SPOKEN_CLEAN.sub("", text).strip()
    if not spoken:
        return None, 1.5
    if time.time() < _TTS_BLOCKED_UNTIL:
        return None, max(2.2, len(spoken.split()) * 0.36)
    try:
        payload = {
            "model": TTS_ID,
            "input": spoken[:1000],
            "voice": TTS_VOICE,
            "response_format": "mp3",
        }
        r = requests.post(
            f"{OPENROUTER_URL}/audio/speech",
            headers=_headers(), json=payload, timeout=45,
        )
        if not r.ok:
            raise RuntimeError(f"Kokoro speech failed ({r.status_code}): {r.text[:200]}")
        _TTS_BLOCK_REASON = ""
        return base64.b64encode(r.content).decode("ascii"), max(2.0, len(spoken.split()) / 2.25)
    except Exception as e:
        reason = str(e)
        _TTS_BLOCK_REASON = reason[:220]
        # OpenRouter account privacy rules can make a speech provider
        # unavailable. Avoid repeating a doomed request for every sentence;
        # the browser will narrate locally during this short retry window.
        if "ignored" in reason.lower() or "provider" in reason.lower() or "404" in reason:
            _TTS_BLOCKED_UNTIL = time.time() + 60
        print(f"[tutori] Kokoro unavailable; using browser voice: {reason[:220]}", flush=True)
        return None, max(2.2, len(spoken.split()) * 0.36)


# --------------------------------------------------------------------------
# web research and JSON helpers
# --------------------------------------------------------------------------


def _first_json_object(raw):
    """The first balanced {...} in raw, or None."""
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except Exception:  # noqa: BLE001
                    return None
    return None


COACH_SYSTEM = """You are Tutori's study coach, quietly watching a whiteboard tutoring session.
After each lesson you update the learner's file and suggest where to go next.
Respond in EXACTLY this format — four lines, nothing else:
PROFILE: {"name": null, "level": "...", "goals": [], "mastered": [], "struggling": [], "pace_notes": "...", "last_topic": "..."}
NEXT1: a short follow-up question digging deeper into this topic
NEXT2: a short question connecting it to something related
NEXT3: a short fun or surprising angle on it

PROFILE — start from CURRENT PROFILE, keep what is still true, fold in what this
turn revealed (level, what clicked, what confused them). Write REAL values from
THIS session: last_topic is what was just taught, pace_notes describes how this
learner likes to learn. "goals" holds ONLY aims the learner said out loud
("I want to pass calculus") — never lesson topics or study to-dos you invent.
Use null or [] when you don't know — never write "...". Keep it under 80 words.
The NEXT questions are about the topic just taught, in the learner's own voice,
4-10 words each — never a repeat of what the learner already asked."""


def coach_update(question, says, profile, pace):
    """Post-lesson: MiniCPM updates the learner profile + offers next steps."""
    lesson_text = " ".join(says)[:900]
    msgs = [{"role": "system", "content": COACH_SYSTEM},
            {"role": "user", "content":
             f"Learner asked: {question}\n"
             f"Tutor taught (spoken): {lesson_text}\n"
             f"CURRENT PROFILE: {json.dumps(profile or {}, ensure_ascii=False)[:500]}\n"
             f"Pace setting: {pace}/5"}]
    try:
        raw = cpm_generate(msgs, max_new_tokens=300)
        m = re.search(r"PROFILE:\s*(\{.*?\})\s*$", raw, re.M)
        prof = _first_json_object(m.group(1)) if m else None
        if isinstance(prof, dict):
            # a small model sometimes echoes placeholders — keep prior values
            prior = profile or {}
            prof = {k: (prior.get(k) if v in ("...", "…") else v)
                    for k, v in prof.items()}
            if prof.get("last_topic") in (None, "...", "…"):
                prof["last_topic"] = question[:48]
        qnorm = re.sub(r"[^a-z0-9 ]", "", question.lower()).strip()
        echoes = ("digging deeper", "connecting it to", "surprising angle",
                  "follow-up question", "learner's own voice")
        sugg = []
        for s in re.findall(r"NEXT\d:\s*(.+)", raw)[:3]:
            s = re.sub(r'^[A-Za-z ,\-]{2,28}:\s*', "", s.strip())  # "A fun angle: ..."
            s = s.strip().strip('"').strip()[:70]
            low = s.lower()
            if (8 <= len(s) <= 70
                    and re.sub(r"[^a-z0-9 ]", "", low).strip() != qnorm
                    and not any(e in low for e in echoes)):   # template parroting
                sugg.append(s)
        if sugg:
            print(f"[tutori] coach: {len(sugg)} suggestions, profile "
                  f"{'updated' if prof else 'unchanged'}")
        return (prof if isinstance(prof, dict) and prof else None), sugg
    except Exception as e:  # noqa: BLE001
        print(f"[tutori] coach failed: {e!r}")
        return None, []


def render_board(topic, step_idx, prior_ops, say, intent=None):
    """The fine-tuned board artist turns one lesson step into board ops."""
    msgs = [{"role": "system", "content": BOARD_MODEL_SYSTEM},
            {"role": "user",
             "content": board_user_message(topic, step_idx, prior_ops, say, intent)}]
    try:
        _ensure_board_adapter()  # mounts on first call, inside the GPU worker
        if BOARD_ARTIST == "gemma":
            raw = gemma_board_generate(msgs)
        else:
            raw = nemotron_generate(msgs, max_new_tokens=340, use_adapter=True)
        board = (_first_json_object(raw) or {}).get("board")
        return board if isinstance(board, list) else []
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"[tutori] board render failed: {e!r}")
        return []


_URL_RE = re.compile(r"https?://[^\s<>)\"']+")
_CURRENT_FACT_RE = re.compile(
    r"\b(latest|recent|today|this month|this year|release|released|launch|"
    r"ban|banned|directive|government|gov|export control|access|suspend|"
    r"suspended|removed|available|unavailable|pricing|version|model)\b",
    re.I,
)


def _extract_urls(text):
    urls = []
    for url in _URL_RE.findall(str(text or "")):
        url = url.rstrip(".,;:!?)]}")
        if url not in urls:
            urls.append(url)
    return urls[:3]


def _strip_urls(text):
    return _URL_RE.sub(" ", str(text or "")).strip()


def _forced_search_queries(question):
    """Cheap deterministic guardrails for questions where stale memory is risky."""
    q = _strip_urls(question)
    low = q.lower()
    queries = []
    if "anthropic" in low and ("fable 5" in low or "mythos 5" in low):
        queries.append("site:anthropic.com/news/fable-mythos-access Fable 5 Mythos 5 government directive")
    if _CURRENT_FACT_RE.search(q) and len(q.split()) >= 3:
        queries.append(q[:120])
    deduped = []
    for query in queries:
        if query and query not in deduped:
            deduped.append(query)
    return deduped[:2]


def decide_search(question, profile, notes):
    forced = _forced_search_queries(question)
    if forced:
        return forced
    today = datetime.date.today().isoformat()
    notes_hint = f"\nEXISTING NOTES (from earlier this session): {notes[:500]}" if notes else ""
    msgs = [
        {"role": "system", "content": SEARCH_SYSTEM.format(today=today)},
        {"role": "user",
         "content": f"Learner request: {question}\n"
                    f"Learner profile: {json.dumps(profile)[:400]}{notes_hint}"},
    ]
    try:
        raw = cpm_generate(msgs, max_new_tokens=160)
    except Exception as e:
        print(f"[tutori] MiniCPM planner failed ({e!r}); falling back to Gemma")
        try:
            raw = llm_generate(msgs, max_new_tokens=96, temperature=0.0)
        except Exception as e2:
            print(f"[tutori] search decision failed: {e2}")
            return []
    obj = parse_lesson_json(raw) or {}
    if obj.get("search") and isinstance(obj.get("queries"), list):
        return [str(q)[:120] for q in obj["queries"][:2]]
    return []


def _fetch_page(url, limit=1500):
    """Pull readable text from a page — snippets alone are too shallow to
    teach from (e.g. a newly published algorithm)."""
    try:
        import requests
        r = requests.get(url, timeout=4,
                         headers={"User-Agent": "Mozilla/5.0 (TutoriBot/1.0)"})
        txt = re.sub(r"<(script|style|nav|header|footer)[\s\S]*?</\1>", " ", r.text)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"&[a-z#0-9]+;", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt[:limit]
    except Exception:
        return ""


def web_research(queries, urls=None):
    snippets, page_blocks, first_url = [], [], None
    seen_urls = set()
    for url in urls or []:
        if url in seen_urls:
            continue
        seen_urls.add(url)
        page_text = _fetch_page(url, limit=2200)
        if page_text:
            page_blocks.append(f"FROM {url}:\n{page_text}")
    try:
        from ddgs import DDGS
        with DDGS() as ddg:
            for q in queries:
                try:
                    for r in ddg.text(q, max_results=4):
                        title = r.get("title", "")
                        body = r.get("body", "")
                        href = r.get("href", "")
                        if first_url is None and href:
                            first_url = href
                        if body:
                            snippets.append(f"- {title}: {body}")
                except Exception:
                    continue
    except Exception as e:
        print(f"[tutori] web research failed: {e}")
    if first_url and first_url not in seen_urls:
        page_text = _fetch_page(first_url)
        if page_text:
            page_blocks.append(f"FROM {first_url}:\n{page_text}")
    out = "\n\n".join(page_blocks + ["\n".join(snippets[:8])])
    return out.strip()[:4200]


# --------------------------------------------------------------------------
# the agent turn
# --------------------------------------------------------------------------

PLAN_SYSTEM = """You are the senior lesson architect for Tutori, a live whiteboard tutor.
Think about pedagogy, factual risk, misconceptions, and the best topic-specific diagram.
Return one JSON object with keys: teaching_goal, learner_gap, sequence (array),
visual_chapters (array), continuity_action (keep|erase|clear), search_queries (array,
maximum 2). Search only for current, niche, named-source, or time-sensitive facts.
For a long lesson, plan several visual chapters and explicitly reuse the board."""


def _duration_spec(minutes):
    minutes = min(10, max(1, int(minutes or 3)))
    steps = min(22, max(4, 2 + minutes * 2))
    target_words = minutes * 115
    words_per_step = min(62, max(26, round(target_words / steps)))
    return minutes, steps, (
        f"about {minutes} minute{'s' if minutes != 1 else ''}; exactly {steps} steps; "
        f"roughly {words_per_step} spoken words per step; approximately {target_words} "
        "spoken words total. Keep every step substantive and avoid filler."
    )


def analyze_board(data_url):
    if not data_url or not data_url.startswith("data:image/"):
        return ""
    messages = [
        {"role": "system", "content":
         "Read this tutoring whiteboard precisely. Describe the learner's ink, equations, "
         "labels, likely intent, and any mistake. Be concise and factual."},
        {"role": "user", "content": [
            {"type": "text", "text": "What should the tutor understand from this board?"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]
    return openrouter_chat(messages, model=COACH_ID, max_tokens=450, temperature=0.1)


def build_agent_plan(question, history, profile, notes, board_now, board_observation,
                     minutes, web_on):
    """Build a useful plan locally so teaching is not gated on a second LLM call.

    Hy3 still performs the high-intelligence lesson generation. The old version
    called Hy3 once to plan and again to teach, which doubled the time before the
    learner saw a single mark. Deterministic routing is both faster and more
    reliable for this small schema.
    """
    family = diagram_family(question)
    if minutes <= 1:
        sequence = ["answer directly", "show the mechanism", "name the key rule", "check understanding"]
        chapters = [family or "one clean topic-specific diagram"]
    elif minutes <= 3:
        sequence = ["orient", "show the mechanism", "explain why", "apply", "summarize", "check understanding"]
        chapters = [family or "main diagram", "evidence or worked example", "visual summary"]
    else:
        sequence = ["orient", "mechanism", "worked example", "misconception", "application", "synthesis"]
        chapters = [family or "main diagram", "worked example", "deeper model", "summary"]
    return {
        "teaching_goal": question[:160],
        "learner_gap": "answer the exact request with a visual mechanism",
        "sequence": sequence,
        "visual_chapters": chapters,
        "continuity_action": "keep" if board_now else "clear",
        "search_queries": _forced_search_queries(question) if web_on else [],
        "known_diagram_family": family,
        "learner_board_observation": board_observation[:500] if board_observation else "",
    }


def select_teacher_model(question, minutes, web_context="", board_observation=""):
    """Use Nitro for a quick answer; reserve Hy3 for deeper agentic lessons."""
    is_quick = int(minutes or 1) <= 1
    simple = len(str(question).split()) <= 30 and not web_context and not board_observation
    return COACH_ID if is_quick and simple else LLM_ID


def known_quick_fallback(question):
    """Reliable narration for the launch-chip lesson if a provider returns bad JSON."""
    if diagram_family(question) != "sky_blue":
        return []
    return [
        {"say": "Sunlight looks white because it contains all the visible colors. "
                "As it enters the atmosphere, that light meets tiny air molecules.",
         "board": []},
        {"say": "Those molecules redirect short blue wavelengths much more strongly "
                "than long red wavelengths, sending blue light toward your eyes.",
         "board": []},
        {"say": "This is Rayleigh scattering. Its strength rises sharply as wavelength "
                "gets shorter, which is why the graph is highest in the blue region.",
         "board": []},
        {"say": "Because scattered blue light reaches you from every direction, the whole "
                "sky appears blue. The direct redder light mostly keeps traveling onward.",
         "board": []},
    ]


def _build_messages(question, history, profile, pace, web_context, agent_plan,
                    board_observation, duration_guidance, notes="", board_now=None):
    today = datetime.date.today().isoformat()
    web_block = (
        "\n\nWEB CONTEXT (fresh source-of-truth; do not contradict it):\n" + web_context
        if web_context else ""
    )
    notes_block = (
        f"\nRESEARCH NOTES FROM THIS SESSION:\n{notes[:2200]}" if notes else ""
    )
    system = LESSON_SYSTEM.format(
        board_reference=BOARD_REFERENCE,
        pace=PACE_DESCRIPTIONS.get(int(pace), PACE_DESCRIPTIONS[3]),
        profile=json.dumps(profile or {}, ensure_ascii=False)[:900],
        board_now=(json.dumps(board_now, ensure_ascii=False)[:6500]
                   if board_now else "(empty board)"),
        duration_guidance=duration_guidance,
        agent_plan=json.dumps(agent_plan, ensure_ascii=False)[:4200],
        today=today,
        notes_block=notes_block,
        web_block=web_block,
    )
    msgs = [{"role": "system", "content": system}]
    for m in (history or [])[-12:]:
        msgs.append({"role": m["role"], "content": str(m["content"])[:900]})
    board_hint = (f"\nA fast vision pass read the learner's own ink as: {board_observation}"
                  if board_observation else "")
    msgs.append({"role": "user", "content": question + board_hint})
    return msgs


def _rebuild_layout(active_ops):
    placed, anchors, state = [], [], {}
    for op in active_ops[-80:]:
        try:
            kind = op.get("op")
            bb = _op_bbox(op)
            if bb:
                placed.append(bb)
                if kind in ("text", "note"):
                    anchors.append({"orig": list(op.get("at", [10, 20])), "bbox": bb})
                if kind != "title":
                    state["area_used"] = state.get("area_used", 0.0) + \
                        max(0, (bb[2] - bb[0]) * (bb[3] - bb[1]))
            if kind == "title":
                state["has_title"] = True
            if kind == "curve":
                state["last_curve"] = op.get("points") or []
        except Exception:
            continue
    state["target_boxes"] = _target_bboxes(active_ops)[-80:]
    return placed, [], anchors, state


def _apply_controls(active_ops, controls):
    active = list(active_ops)
    for op in controls:
        if op.get("op") == "clear":
            active = []
        elif op.get("op") == "erase":
            targets = set(op.get("targets") or [])
            if targets:
                active = [x for x in active if str(x.get("id")) not in targets]
    return active


def run_turn(audio_path, typed_text, board_snapshot, history, profile,
             notes, board_now, pace, lesson_minutes, web_on, voice_on):
    """One full agent turn. Yields event dicts (see module docstring)."""
    t0 = time.time()
    try:
        question = (typed_text or "").strip()
        if audio_path:
            yield {"type": "status", "status": "thinking", "detail": "Listening with Parakeet…"}
            heard = transcribe(audio_path)
            question = f"{heard} {question}".strip() if question else heard
            yield {"type": "transcript", "text": question}
        if not question and not board_snapshot:
            yield {"type": "final", "text": "", "error": "I didn't catch anything — try again?"}
            return
        if not question:
            question = "Take a look at what I drew on the board and help me with it."

        board_observation = ""
        if board_snapshot:
            yield {"type": "status", "status": "thinking", "detail": "Reading your whiteboard…"}
            board_observation = analyze_board(board_snapshot)

        minutes, _, duration_guidance = _duration_spec(lesson_minutes)
        yield {"type": "status", "status": "thinking", "detail": "Choosing the clearest teaching path…"}
        plan = build_agent_plan(question, history, profile, notes, board_now,
                                board_observation, minutes, web_on)

        web_context = ""
        urls = _extract_urls(question) if web_on else []
        queries = [str(q)[:140] for q in (plan.get("search_queries") or [])[:2]] if web_on else []
        forced = _forced_search_queries(question) if web_on else []
        for q in forced:
            if q not in queries:
                queries.append(q)
        queries = queries[:2]
        if urls or queries:
            bits = (["reading source"] if urls else []) + queries
            yield {"type": "status", "status": "searching",
                   "detail": "Researching: " + " · ".join(bits)}
            web_context = web_research(queries, urls=urls)
            if web_context:
                yield {"type": "research", "notes": web_context}

        teacher_model = select_teacher_model(question, minutes, web_context, board_observation)
        deep_future = None
        generation_pool = None
        if teacher_model == LLM_ID:
            # Hy3 is deliberately thoughtful but can take close to a minute
            # before returning visible tokens through its OpenRouter provider.
            # Start it now, then let Nitro teach a one-minute visual overview
            # immediately. The deeper chapter arrives while that overview is
            # already being spoken and drawn.
            deep_minutes = max(1, minutes - 1)
            _, _, deep_guidance = _duration_spec(deep_minutes)
            deep_plan = dict(plan)
            deep_plan["warm_start_context"] = (
                "A one-minute Gemma visual overview plays first. Begin with a "
                "fresh deeper chapter; avoid greetings and do not repeat the basic overview."
            )
            deep_msgs = _build_messages(
                question, history, profile, pace, web_context, deep_plan,
                board_observation, deep_guidance, notes=notes or "", board_now=board_now,
            )
            deep_tokens = min(18000, 3800 + deep_minutes * 1100)
            generation_pool = ThreadPoolExecutor(max_workers=1)
            deep_future = generation_pool.submit(
                openrouter_chat, deep_msgs, model=LLM_ID,
                max_tokens=deep_tokens, temperature=0.38,
                json_mode=True, reasoning="low", timeout=360,
            )
            _, _, quick_guidance = _duration_spec(1)
            quick_plan = dict(plan)
            quick_plan["sequence"] = [
                "answer directly", "show the mechanism", "name the key rule", "check understanding"
            ]
            quick_plan["visual_chapters"] = [
                plan.get("known_diagram_family") or "one clean overview diagram"
            ]
            msgs = _build_messages(
                question, history, profile, pace, web_context, quick_plan,
                board_observation, quick_guidance, notes=notes or "", board_now=board_now,
            )
            teacher_model = COACH_ID
            teacher_name = "Gemma Nitro"
            max_tokens = 3200
            reasoning = None
            detail = "Gemma starts now while Hy3 prepares the deeper chapter…"
        else:
            teacher_name = "Gemma Nitro"
            msgs = _build_messages(
                question, history, profile, pace, web_context, plan,
                board_observation, duration_guidance, notes=notes or "", board_now=board_now,
            )
            max_tokens = 3200
            reasoning = None
            detail = "Gemma Nitro is sketching the first panel…"
        yield {"type": "status", "status": "teaching", "detail": detail}
        pool = ThreadPoolExecutor(max_workers=3) if voice_on else None
        active_ops = list(board_now or [])[-80:]
        said = []
        n_steps = 0

        def prepare_step(raw_step, idx):
            nonlocal active_ops
            if not isinstance(raw_step, dict):
                return None
            candidate = dict(raw_step)
            raw_board = candidate.get("board") or []
            controls = [op for op in raw_board if isinstance(op, dict)
                        and op.get("op") in ("clear", "erase")]
            draws = [op for op in raw_board if isinstance(op, dict)
                     and op.get("op") not in ("clear", "erase")]
            candidate["board"] = controls + improve_step_board(
                question, idx, candidate.get("say", ""), draws,
            )
            step = sanitize_step(candidate, idx)
            if not step:
                return None
            controls = [op for op in step["board"] if op.get("op") in ("clear", "erase")]
            draws = [op for op in step["board"] if op.get("op") not in ("clear", "erase")]
            active_ops = _apply_controls(active_ops, controls)
            placed, displaced, anchors, lstate = _rebuild_layout(active_ops)
            seen = {sig for sig in (_op_sig(op) for op in active_ops) if sig is not None}
            draws = dedup_ops(draws, seen)
            draws = layout_pass(draws, placed, displaced, anchors, lstate)
            step["board"] = controls + draws
            active_ops.extend(draws)
            active_ops = active_ops[-80:]

            audio_b64, dur = None, None
            if pool and step["say"]:
                future = pool.submit(synthesize, step["say"])
                try:
                    # Never hold the first visible panel hostage to a slow TTS
                    # provider. The client narrates locally if this deadline wins.
                    audio_b64, dur = future.result(timeout=6 if idx == 0 else 3)
                except FutureTimeout:
                    print(f"[tutori] Kokoro step {idx + 1} exceeded the fast path; "
                          "using browser voice", flush=True)
                except Exception as exc:
                    print(f"[tutori] voice step {idx + 1} failed: {exc!r}", flush=True)
            step["audio"] = audio_b64
            step["dur"] = dur or max(2.2, len(step["say"].split()) / 2.25)
            step["voice_fallback"] = bool(voice_on and not audio_b64)
            return step

        # Nitro usually returns this four-step chapter in a few seconds. Its
        # provider occasionally emits partial/cumulative SSE chunks, so a
        # normal JSON response is measurably more reliable than reconstructing
        # the lesson client-side and is still dramatically faster than Hy3.
        full_text = ""
        raw_steps = []
        best_text, best_steps = "", []
        for attempt, temperature in enumerate((0.34, 0.20, 0.10), start=1):
            candidate_text = openrouter_chat(
                msgs, model=teacher_model, max_tokens=max_tokens,
                temperature=temperature, json_mode=True, reasoning=reasoning,
                timeout=180,
            )
            candidate_steps = extract_lesson_steps(candidate_text)
            if len(candidate_steps) > len(best_steps):
                best_text, best_steps = candidate_text, candidate_steps
            if len(candidate_steps) >= 3:
                break
            print(f"[tutori] Nitro attempt {attempt} returned "
                  f"{len(candidate_steps)} lesson steps; retrying", flush=True)
        full_text, raw_steps = best_text, best_steps
        reliable = known_quick_fallback(question)
        if len(raw_steps) < 3 and reliable:
            print("[tutori] using the verified quick lesson fallback", flush=True)
            raw_steps = reliable
        elif not raw_steps:
            plain = re.sub(r"[{}\[\]\"]", "", full_text).strip()[:650]
            raw_steps = [{"say": plain or "Let's build that idea one clean step at a time.",
                          "board": []}]
        print(f"[tutori] quick chapter: {len(raw_steps)} raw steps, "
              f"{len(full_text)} chars", flush=True)
        for idx, raw_step in enumerate(raw_steps[:8]):
            step = prepare_step(raw_step, idx)
            if not step:
                continue
            if n_steps == 0:
                print(f"[tutori] first panel ready in {time.time() - t0:.1f}s "
                      f"via {teacher_name}", flush=True)
            said.append(step["say"])
            n_steps += 1
            yield {"type": "step", "step": step}
        if deep_future is not None:
            yield {"type": "status", "status": "thinking",
                   "detail": "Hy3 is preparing the deeper whiteboard chapter…"}
            try:
                deep_text = deep_future.result(timeout=360)
                deep_steps = extract_lesson_steps(deep_text)
                print(f"[tutori] Hy3 deeper chapter: {len(deep_steps)} raw steps, "
                      f"{len(deep_text)} chars", flush=True)
                for deep_idx, raw_step in enumerate(deep_steps):
                    if n_steps >= 24:
                        break
                    raw_step = dict(raw_step) if isinstance(raw_step, dict) else raw_step
                    if deep_idx == 0 and isinstance(raw_step, dict):
                        raw_board = list(raw_step.get("board") or [])
                        if not any(isinstance(op, dict) and op.get("op") == "clear"
                                   for op in raw_board):
                            raw_step["board"] = [{"op": "clear"}] + raw_board
                    # Indices 0-7 are reserved for trusted quick templates.
                    # Starting the deeper chapter at 8 lets Hy3 use its own
                    # coordinates after the explicit chapter clear.
                    step = prepare_step(raw_step, 8 + deep_idx)
                    if not step:
                        continue
                    said.append(step["say"])
                    n_steps += 1
                    yield {"type": "step", "step": step}
            except Exception as exc:
                # The learner already received a complete quick lesson. A deep
                # model hiccup should not turn that successful turn into an error.
                print(f"[tutori] Hy3 deeper chapter unavailable: {exc!r}", flush=True)
            finally:
                if generation_pool:
                    generation_pool.shutdown(wait=False)
        if pool:
            pool.shutdown(wait=False)

        new_profile, suggestions = coach_update(question, said, profile, pace)
        if new_profile:
            yield {"type": "memory", "profile": new_profile}
        if suggestions:
            yield {"type": "coach", "suggestions": suggestions}

        yield {"type": "final", "text": " ".join(said).strip(), "error": None,
               "question": question, "elapsed": round(time.time() - t0, 1),
               "steps": n_steps}
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        yield {"type": "final", "text": "",
               "error": f"Something went wrong on my side: {type(e).__name__}. Try once more?"}


def tts_only(text):
    b64, dur = synthesize(str(text)[:650])
    return json.dumps({"audio": b64, "dur": dur})


MODELS_INFO = {
    "llm": LLM_ID, "coach": COACH_ID, "tts": TTS_ID, "asr": ASR_ID,
    "mode": "openrouter",
}
