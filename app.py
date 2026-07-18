"""
Tutori — your personal whiteboard tutor.

Speak (or type) a question. Tutori listens, gathers whatever context it needs,
then teaches you out loud while sketching the idea on a whiteboard in real
time. This deployment preserves the Build Small-winning Gradio experience
while routing an upgraded open-weight model team through OpenRouter.
"""

import json
import os
import uuid
from pathlib import Path

import gradio as gr

# The API engine is the default everywhere; mock mode is explicit for offline UI work.
if os.environ.get("TUTORI_MOCK") != "1":
    try:
        import engine as ENGINE
    except Exception:  # e.g. Space running on CPU hardware
        import traceback
        traceback.print_exc()
        print("[tutori] API engine unavailable — falling back to mock")
        import mock_engine as ENGINE
        try:
            # ZeroGPU kills Spaces that register no @spaces.GPU function;
            # give it one so the mock UI stays reachable for debugging.
            import spaces

            @spaces.GPU
            def _zerogpu_probe():
                return "ok"
        except Exception:
            pass
else:
    import mock_engine as ENGINE

ROOT = Path(__file__).parent
BOARD_JS = (ROOT / "static" / "board.js").read_text()
CSS = (ROOT / "static" / "style.css").read_text()


FONT_FACES = (ROOT / "static" / "fonts" / "faces.css").read_text()

HEAD = f"""
<style>{FONT_FACES}</style>
<script>{BOARD_JS}</script>
"""

HEADER_HTML = f"""
<div id="tutori-header">
  <div class="head-left">
    <div class="logo"><span class="mark">✏️</span> Tutori
      <span class="sub">your whiteboard tutor</span>
    </div>
    <svg class="squiggle" viewBox="0 0 320 14" preserveAspectRatio="none" aria-hidden="true">
      <path d="M4 9 Q 44 2, 84 8 T 164 8 T 244 9 T 316 6"/>
    </svg>
    <div class="tag">Ask anything out loud — Tutori researches it, then teaches you while sketching it live.</div>
  </div>
  <a class="award-badge" href="https://huggingface.co/build-small-hackathon"
     target="_blank" rel="noopener" aria-label="Tutori won fourth place at the Hugging Face Build Small Hackathon">
    <span class="trophy">🏆</span><span><b>#4</b> · 4th place at Build Small</span>
  </a>
  <div class="badges">
    <span class="badge hot">⚡ OpenRouter · open-weight only</span>
    <span class="badge">🧠 Tencent Hy3 teacher</span>
    <span class="badge">🧭 Gemma 4 31B coach</span>
    <span class="badge">🗣️ Kokoro 82M voice</span>
    <span class="badge">👂 NVIDIA Parakeet ears</span>
    <span class="badge">⏱️ Lessons up to 10 minutes</span>
  </div>
</div>
"""

CHIPS = [
    "Why is the sky blue?",
    "Teach me the Pythagorean theorem",
    "How do neural networks learn?",
    "What happened in space exploration this month?",
]

EMPTY_PROFILE = {}


def _payload(turn_id, voice_on, status, detail, steps):
    # CUMULATIVE: every payload carries all steps so far (browser dedupes by
    # "i"). Gradio coalesces fast generator yields into latest-state — an
    # incremental protocol silently loses steps when that happens.
    n_ops = sum(len(s.get("board") or []) for s in steps)
    heavy = sum(1 for s in steps for op in (s.get("board") or [])
                if op.get("op") in ("axes", "graph", "polygon", "notes", "curve"))
    return json.dumps({
        "turn": turn_id,
        "voice": bool(voice_on),
        "status": status,
        "status_detail": detail,
        "big": bool(n_ops >= 9 or heavy >= 4),
        "steps": steps,
    })


def _track_board(board_ops, step_board):
    """Mirror what the lesson drew so the next turn knows the board state."""
    for op in step_board or []:
        if op.get("op") == "clear":
            board_ops = []
        elif op.get("op") == "erase":
            targets = set(op.get("targets") or ([op.get("target")] if op.get("target") else []))
            board_ops = [old for old in board_ops if str(old.get("id")) not in targets]
        else:
            board_ops.append(op)
    return board_ops[-80:]


N_CHIPS = len(CHIPS)


def run_turn(audio_path, typed_text, snapshot, chat, convo, profile,
             notes, board_ops, pace, lesson_minutes, web_on, voice_on):
    """Bridges ENGINE.run_turn events into streaming Gradio updates."""
    chat = list(chat or [])
    convo = list(convo or [])
    profile = dict(profile or {})
    notes = notes or ""
    board_ops = list(board_ops or [])
    turn_id = uuid.uuid4().hex[:10]

    has_input = bool(audio_path or (typed_text or "").strip() or snapshot)
    if not has_input:
        yield (gr.update(), chat, convo, profile, profile, notes, board_ops,
               gr.update(), gr.update(),
               *([gr.update()] * (2 * N_CHIPS)))
        return

    user_label = (typed_text or "").strip() or ("🎙️ …" if audio_path else "🖼️ (my whiteboard)")
    chat.append({"role": "user", "content": user_label})
    chat.append({"role": "assistant", "content": "✏️ *composing the first clean panel…*"})
    question_for_context = user_label
    says = []
    sent_steps = []
    chip_vals = None  # set when the study coach suggests follow-ups

    def render(status="thinking", detail=""):
        n = N_CHIPS
        if chip_vals:
            chips = ([gr.update(value=s) for s in chip_vals[:n]] +
                     [gr.update()] * (n - min(n, len(chip_vals))))
            states = (list(chip_vals[:n]) +
                      [gr.update()] * (n - min(n, len(chip_vals))))
        else:
            chips = [gr.update()] * n
            states = [gr.update()] * n
        return (_payload(turn_id, voice_on, status, detail, sent_steps),
                chat, convo, profile, profile, notes, board_ops,
                gr.update(value=None), gr.update(value=""),
                *chips, *states)

    yield render(detail="Thinking…")

    seq = 0
    error = None
    print(f"[tutori] turn {turn_id}: starting engine", flush=True)
    try:
        for ev in ENGINE.run_turn(audio_path, typed_text, snapshot, convo, profile,
                                  notes, board_ops, pace, lesson_minutes, web_on, voice_on):
            kind = ev.get("type")
            if seq == 0 and kind != "step":
                print(f"[tutori] turn {turn_id}: event {kind}", flush=True)
            if kind == "status":
                yield render(ev.get("status", "thinking"), ev.get("detail", ""))
            elif kind == "transcript":
                question_for_context = ev["text"]
                chat[-2]["content"] = f"🎙️ {ev['text']}"
                yield render("thinking", "Heard you!")
            elif kind == "research":
                fresh = ev.get("notes", "")
                if fresh:
                    notes = (notes + "\n" + fresh)[-4000:]
            elif kind == "step":
                step = dict(ev["step"], i=seq)
                says.append(step.get("say", ""))
                board_ops = _track_board(board_ops, step.get("board"))
                chat[-1]["content"] = "✏️ " + " ".join(s for s in says if s)
                sent_steps.append(step)
                yield render("teaching", f"Teaching — step {seq + 1}")
                seq += 1
            elif kind == "memory":
                profile = dict(ev.get("profile") or profile)
            elif kind == "coach":
                chip_vals = ev.get("suggestions") or None
            elif kind == "final":
                error = ev.get("error")
                question_for_context = ev.get("question") or question_for_context
    except Exception as e:  # ZeroGPU quota / GPU-time-cap errors land here
        msg = str(e)
        print(f"[tutori] turn {turn_id}: engine raised: {msg[:300]}", flush=True)
        if seq > 0:
            error = None  # we already taught something — end the turn gracefully
        elif "quota" in msg.lower():
            error = ("ZeroGPU quota reached for your session. Sign in to "
                     "Hugging Face (free) for much more GPU time, then retry.")
        else:
            error = "The GPU hiccuped on that one — give it another try?"

    if error:
        chat[-1]["content"] = f"⚠️ {error}"
        yield render("error", error)
        return

    final_text = " ".join(s for s in says if s).strip() or chat[-1]["content"]
    chat[-1]["content"] = "✏️ " + final_text
    convo.append({"role": "user", "content": question_for_context})
    convo.append({"role": "assistant", "content": final_text[:1200]})
    yield render("done", "")


def reset_session():
    # chat, convo, typed text, board mirror, mic, research notes
    return [], [], "", [], gr.update(value=None), ""


def forget_me():
    return {}, {}


with gr.Blocks(
    title="Tutori — your whiteboard tutor",
    theme=gr.themes.Soft(
        primary_hue="indigo", secondary_hue="amber", neutral_hue="stone",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ),
    css=CSS,
    head=HEAD,
) as demo:

    profile_state = gr.BrowserState(EMPTY_PROFILE, storage_key="tutori_profile_v1")
    convo_state = gr.State([])
    notes_state = gr.State("")     # web research carried across the session
    board_state = gr.State([])     # ops currently on the board (server mirror)

    payload_box = gr.Textbox(
        visible=True, elem_id="tutori-payload", elem_classes="payload-sink",
        show_label=False, container=False,
    )
    snap_box = gr.Textbox(visible=False)

    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False, elem_classes="studio-row"):
        # ---------------- whiteboard ----------------
        with gr.Column(scale=15, elem_classes="board-col"):
            gr.HTML('<div id="tutori-board-mount"></div>')
            with gr.Row(elem_classes="board-actions"):
                ask_board_btn = gr.Button(
                    "🖐 Ask Tutori about the board", elem_id="ask-board-btn", scale=3
                )
                new_btn = gr.Button("🧽 New lesson", scale=1, elem_id="new-btn")
            if ENGINE.MODELS_INFO["mode"] == "mock":
                gr.HTML(
                    '<div id="mode-banner">🧪 <b>Mock mode</b> — running without GPU models '
                    "(local dev). Configure OPENROUTER_API_KEY to use the full model team.</div>"
                )

        # ---------------- conversation ----------------
        with gr.Column(scale=8, elem_classes="side-col"):
            chatbot = gr.Chatbot(
                type="messages", height=320, elem_id="tutori-chat",
                label="Lesson transcript",
                avatar_images=(None, None),
                show_copy_button=False,
            )
            mic = gr.Audio(
                sources=["microphone"], type="filepath", format="wav",
                label="🎤 Talk to Tutori (recording stops = message sent)",
                elem_id="mic-box", show_download_button=False,
            )
            with gr.Row():
                text_in = gr.Textbox(
                    placeholder="…or type your question and press Enter",
                    show_label=False, scale=5, container=False,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=80, elem_id="send-btn")
            with gr.Row(elem_classes="chip-row"):
                chip_btns = [gr.Button(c, size="sm") for c in CHIPS]
            chip_states = [gr.State(c) for c in CHIPS]

            with gr.Accordion("⚙️ How Tutori teaches you", open=False):
                pace = gr.Slider(
                    1, 5, value=3, step=1, label="Pace & depth",
                    info="1 = total beginner, tiny steps · 5 = expert, fast and dense",
                )
                lesson_minutes = gr.Slider(
                    1, 10, value=1, step=1, label="Lesson length (minutes)",
                    info="1 = quick Gemma lesson · 2–10 = deeper Hy3 whiteboard session.",
                )
                web_on = gr.Checkbox(True, label="🔎 Let Tutori research the web when useful")
                voice_on = gr.Checkbox(
                    True,
                    label="🔊 Voice replies (Kokoro 82M, with an instant device-voice fallback)",
                )

            with gr.Accordion("🧠 What Tutori remembers about you", open=False):
                gr.Markdown(
                    "Tutori keeps gentle notes — your level, goals, what clicked, what "
                    "didn't — saved **only in your browser**, and uses them to pick the "
                    "right pace next time."
                )
                profile_view = gr.JSON(value=EMPTY_PROFILE, label="Learner profile")
                forget_btn = gr.Button("🗑️ Forget everything about me", size="sm")

    gr.HTML(
        '<div id="tutori-foot"><b>Our open-weight commitment:</b> Tutori will only use open-weight models, '
        'a principle we carry forward thanks to the '
        '<a href="https://huggingface.co/build-small-hackathon" target="_blank">'
        "Hugging Face Build Small Hackathon</a>. "
        'Visit the <a href="https://huggingface.co/spaces/build-small-hackathon/tutori" '
        'target="_blank">original award-winning Space</a>. · '
        "Hy3 teaches · Gemma coaches · Kokoro speaks · Parakeet listens."
        '<div id="made-by">Made by SSH/ProCreations</div></div>'
    )


    # ---------------- wiring ----------------
    # The `js` hook runs client-side BEFORE fn and its return replaces the
    # input values — we use it to inject a fresh whiteboard snapshot into
    # `snap_box`. (gr.State arrives as null in js; pass it through untouched.)
    def _snap_js(fn_name):
        return (
            "(audio, text, snap, chat, convo, profile, notes, board, pace, minutes, web, voice) => "
            f"[audio, text, (window.{fn_name} ? window.{fn_name}() : ''), "
            "chat, convo, profile, notes, board, pace, minutes, web, voice]"
        )

    turn_io = dict(
        fn=run_turn,
        inputs=[mic, text_in, snap_box, chatbot, convo_state, profile_state,
                notes_state, board_state, pace, lesson_minutes, web_on, voice_on],
        outputs=[payload_box, chatbot, convo_state, profile_state, profile_view,
                 notes_state, board_state, mic, text_in,
                 *chip_btns, *chip_states],
        show_progress="hidden",
    )

    turn_events = [
        mic.stop_recording(js=_snap_js("tutoriSnapshotIfInk"), **turn_io),
        text_in.submit(js=_snap_js("tutoriSnapshotIfInk"), **turn_io),
        send_btn.click(js=_snap_js("tutoriSnapshotIfInk"), **turn_io),
        ask_board_btn.click(js=_snap_js("tutoriSnapshot"), **turn_io),
    ]
    for btn, chip_state in zip(chip_btns, chip_states):
        ev = btn.click(lambda s: s, chip_state, text_in)
        turn_events.append(ev.then(js=_snap_js("tutoriSnapshotIfInk"), **turn_io))

    payload_box.change(fn=None, inputs=payload_box, outputs=None,
                       js="(p) => { window.tutoriOnPayload(p); }")
    new_btn.click(
        reset_session, None,
        [chatbot, convo_state, text_in, board_state, mic, notes_state],
        js="() => { window.tutoriClearAll && window.tutoriClearAll(); return []; }",
        cancels=turn_events,   # a mid-stream turn must die with the old lesson
    )
    forget_btn.click(forget_me, None, [profile_state, profile_view])
    demo.load(lambda p: p or {}, profile_state, profile_view)

    if os.environ.get("TUTORI_TTS_DEMO") == "1":
        # dev-only narration endpoint (never set on the public Space)
        import spaces as _spaces

        @_spaces.GPU(duration=45)
        def _tts_demo(text):
            audio_b64, dur = ENGINE.synthesize(str(text)[:300])
            return json.dumps({"audio": audio_b64, "dur": dur})

        _tts_in = gr.Textbox(visible=False)
        _tts_out = gr.Textbox(visible=False)
        _tts_btn = gr.Button("tts", visible=False)
        _tts_btn.click(_tts_demo, _tts_in, _tts_out, api_name="tts_demo")



if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860"))),
        ssr_mode=False, allowed_paths=[str(ROOT / "static")])
