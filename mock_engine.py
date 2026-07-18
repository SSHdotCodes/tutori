"""
Tutori mock engine — runs anywhere, no GPU, no model downloads.

Used automatically when the app is not running on a Space (local dev), so the
whole UI — whiteboard animation, audio-synced playback, chat flow — can be
exercised end to end. Speaks with soft synthesized tones instead of TTS.
"""

import base64
import io
import json
import os
import time

import numpy as np
import soundfile as sf

from board_quality import diagram_family, improve_step_board

# Replay mode: stream captured real-lesson event traces with original (or
# capped) timing — used to record authentic demo footage locally. Accepts a
# comma-separated list; the Nth turn replays the Nth trace.
REPLAY = os.environ.get("TUTORI_REPLAY")
REPLAY_WAITCAP = float(os.environ.get("TUTORI_REPLAY_WAITCAP", "1.4"))
_replay_turn = {"i": 0}

SR = 24000


def _tone_audio(seconds):
    """A quiet, pleasant placeholder 'voice' so playback timing is real."""
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    f = 196 + 30 * np.sin(2 * np.pi * 0.7 * t)
    wave = 0.06 * np.sin(2 * np.pi * f * t) * (0.55 + 0.45 * np.sin(2 * np.pi * 2.1 * t))
    env = np.minimum(1, np.minimum(t / 0.15, (seconds - t) / 0.25).clip(0))
    buf = io.BytesIO()
    sf.write(buf, (wave * env).astype(np.float32), SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _step(say, board):
    dur = max(2.4, len(say.split()) * 0.34)
    return {"say": say, "board": board, "audio": _tone_audio(dur), "dur": dur}


def _templated_lesson(topic):
    if diagram_family(topic) != "pythagorean":
        return None
    says = [
        "Let's identify the right triangle first: the two legs make the L shape, and c is the hypotenuse across from it.",
        "The theorem says the square on c equals the two leg squares added together.",
        "For a concrete example, set a to three and b to four, then substitute those values into the formula.",
        "That gives twenty five, so c is five. Notice five is the longest side, which matches the drawing.",
    ]
    out = []
    for i, say in enumerate(says):
        board = improve_step_board(topic, i, say, [])
        if i == 0:
            board = [{"op": "clear"}] + board
        out.append(_step(say, board))
    return out


def _demo_lesson(topic):
    t = topic.strip().rstrip("?!.") or "how rainbows form"
    templated = _templated_lesson(t)
    if templated:
        return templated
    return [
        _step(f"Great question! Let's break down {t} together, step by step.",
              [{"op": "clear"}, {"op": "title", "text": t.title()[:42]}]),
        _step("First, picture the big idea as three connected parts.",
              [{"op": "box", "at": [8, 22], "w": 24, "h": 11, "label": "Input", "color": "blue"},
               {"op": "box", "at": [38, 22], "w": 24, "h": 11, "label": "Process", "color": "purple"},
               {"op": "box", "at": [68, 22], "w": 24, "h": 11, "label": "Result", "color": "green"}]),
        _step("Each part feeds the next one, like a little assembly line.",
              [{"op": "arrow", "from": [32, 27.5], "to": [38, 27.5], "color": "ink"},
               {"op": "arrow", "from": [62, 27.5], "to": [68, 27.5], "color": "ink"}]),
        _step("And here's how the effect grows over time — slowly at first, then quickly.",
              [{"op": "axes", "at": [14, 40], "w": 44, "h": 26, "xlabel": "time", "ylabel": "effect"},
               {"op": "curve", "points": [[15, 64], [26, 62], [36, 56], [46, 46], [55, 42]], "color": "red"},
               {"op": "dot", "at": [46, 46], "color": "orange", "label": "tipping point"}]),
        _step("Quick check: which of the three parts would you like to zoom into first?",
              [{"op": "highlight", "at": [36, 33], "w": 28},
               {"op": "text", "text": "your pick? →", "at": [66, 52], "size": "m", "color": "orange"}]),
    ]


def run_turn(audio_path, typed_text, board_snapshot, history, profile,
             notes, board_now, pace, lesson_minutes, web_on, voice_on):
    if REPLAY:
        files = REPLAY.split(",")
        path = files[min(_replay_turn["i"], len(files) - 1)]
        _replay_turn["i"] += 1
        trace = json.load(open(path))
        t_prev = 0.0
        for entry in trace["events"]:
            dt = max(0.0, entry["t"] - t_prev)
            t_prev = entry["t"]
            # compress thinking AND generation gaps — the browser queues steps
            # and paces playback by the audio itself
            dt = min(dt, REPLAY_WAITCAP)
            if dt:
                time.sleep(dt)
            yield entry["event"]
        return
    question = (typed_text or "").strip()
    if audio_path:
        yield {"type": "status", "status": "thinking", "detail": "Listening… (mock)"}
        time.sleep(0.5)
        question = question or "How do rainbows form?"
        yield {"type": "transcript", "text": question}
    if not question:
        question = "Tell me about what's on the board" if board_snapshot else "Teach me something fun"

    if web_on and not notes:
        yield {"type": "status", "status": "searching",
               "detail": f"Researching: {question[:60]} (mock)"}
        time.sleep(0.7)
        yield {"type": "research", "notes": f"- (mock) background notes about {question[:50]}"}

    yield {"type": "status", "status": "teaching", "detail": "Preparing your whiteboard lesson… (mock)"}
    time.sleep(0.6)

    says = []
    for step in _demo_lesson(question):
        if not voice_on:
            step = dict(step, audio=None)
        says.append(step["say"])
        yield {"type": "step", "step": step}
        time.sleep(0.4)

    yield {"type": "memory", "profile": {
        **(profile or {}),
        "last_topic": question[:80],
        "pace_notes": f"pace slider at {pace}",
    }}
    yield {"type": "memory", "profile": {"level": "curious beginner", "last_topic": "mock lesson"}}
    yield {"type": "coach", "suggestions": ["What happens if we go deeper?", "How does this connect to music?", "Could this work underwater?"]}
    yield {"type": "final", "text": " ".join(says), "error": None,
           "question": question, "elapsed": 3.0}


MODELS_INFO = {
    "llm": "mock", "tts": "mock", "asr": "mock",
    "total_params": "0 (mock mode — run on a Space for the real models)",
    "mode": "mock",
}
