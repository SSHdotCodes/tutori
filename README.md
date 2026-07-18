---
title: Tutori — Your Whiteboard Tutor
emoji: ✏️
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
fullWidth: true
header: mini
pinned: false
license: apache-2.0
short_description: Voice tutor that sketches on a whiteboard while it talks
tags:
  - build-small-hackathon
  - backyard ai
  - off the grid
  - off-brand
  - sharing is caring
  - agent
  - education
  - speech
  - track:backyard
  - sponsor:openbmb
  - sponsor:openai
  - achievement:offgrid
  - achievement:offbrand
  - achievement:sharing
  - achievement:fieldnotes
models:
  - google/gemma-4-12B-it
  - bosonai/higgs-audio-v2-generation-3B-base
  - openbmb/MiniCPM5-1B
  - openai/whisper-large-v3-turbo
  - nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
  - ProCreations/tutori-board-nemotron
  - ProCreations/tutori-board-gemma
datasets:
  - ProCreations/tutori-whiteboard-lessons
---

# ✏️ Tutori — your whiteboard tutor

## The `ssh.codes` hosted edition

This repository now also powers [tutori.ssh.codes](https://tutori.ssh.codes),
an upgraded Gradio deployment that preserves the original studio UI and canvas
renderer while using an open-weight OpenRouter model team:

- `tencent/hy3` (low reasoning) for lesson architecture and teaching
- `google/gemma-4-31b-it:nitro` for fast coaching and whiteboard vision
- `nvidia/parakeet-tdt-0.6b-v3` for transcription
- `sesame/csm-1b` for conversational text-to-speech

The hosted edition adds 1–10 minute continuous lessons, agent-controlled clear
and targeted erase actions, persistent whiteboard state, and a native scaled
graph primitive with grids, ticks, legends, multiple data series, and markers.
In gratitude for the competition, Tutori's hosted model stack is committed to
open-weight models only. The original Build Small Space remains available at
[build-small-hackathon/tutori](https://huggingface.co/spaces/build-small-hackathon/tutori).

The remainder of this README documents the original award-winning ZeroGPU
submission and is preserved as its historical build record.

## 🎬 Demo

<video src="https://huggingface.co/spaces/build-small-hackathon/tutori/resolve/main/tutori_demo.mp4" controls width="100%"></video>

*([demo video file](./tutori_demo.mp4) — narrated by Tutori's own voice; music synthesized from scratch.)*

**Speak a question. Tutori researches it, then teaches you out loud while
sketching the idea on a whiteboard — in real time, stroke by stroke, in sync
with its voice.**

Built for the [HF Build Small Hackathon](https://huggingface.co/build-small-hackathon):
every model runs **on this Space itself** via ZeroGPU. No cloud APIs, no keys.

## Why Backyard AI 🏡

I built Tutori for my parents. They're quite behind on today's technology —
especially AI — and the firehose of new models and jargon is impenetrable
from the outside. With Tutori they can just **ask**: "what is Gemma?",
"what happened in AI this month?" — and get a patient, spoken explanation,
drawn out on a whiteboard, at their pace, with the research done for them.
They've actually been using it, and they find it genuinely useful for
finally keeping up.

## The stack (Σ 16.9B params — well under the 32B cap)

| Role | Model | Params |
|---|---|---|
| 🧠 Teacher + vision | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | 12B |
| 🧭 Research planner + study coach | [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | 1B |
| 🗣️ Expressive voice | [bosonai/higgs-audio-v2-generation-3B-base](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) | 3B |
| 👂 Speech recognition | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 0.8B |

**Gemma teaches. MiniCPM plans and coaches.** Every turn, MiniCPM5 1B decides
whether the question needs fresh facts and writes the search queries — the
agentic step that turns a chatbot into a researcher. Gemma 4 then teaches
from what was found, Higgs speaks it, Whisper listens. And after every
lesson, MiniCPM comes back as the **study coach**: it updates the learner's
profile and writes three personalized follow-up questions that land on the
sticky notes under the chat — tap one and the lesson continues where your
curiosity points.

### Engineering notes (the honest kind)

- We tried to ship **Nemotron ASR** as the ears — three separate times. NeMo
  in the main process crashes ZeroGPU's forked workers ("GPU task aborted");
  lazy-loading inside the worker costs a fresh worker its whole turn (we
  re-measured on June 12 with a dedicated probe Space: **57.1 s just to
  restore the 0.6B streaming model**, against a 59 s turn budget, paid every
  turn because workers are disposable); and a CPU sidecar measured RTF ≈ 24.
  So Whisper turbo keeps the ears — it preloads with everything else and
  transcribes in about a second.
- Higgs Audio **v3** TTS ships only for the SGLang-Omni serving stack (needs
  a persistent GPU), so we use v2 — same family, natively in transformers.
- The live whiteboard now has a deterministic **diagram specialist** between
  Gemma and the renderer. When the lesson lands on a known teaching family
  (rockets to orbit, gradient descent, Pythagorean theorem, neural networks,
  photosynthesis, supply/demand, binary search, recursion, rainbows, water
  cycle), Tutori compiles the board from hand-authored diagram ops instead of
  asking a language model to freehand coordinates. Unknown topics still use
  the model's drawing, then pass through the same no-overlap layout engine.
- We **LoRA-fine-tuned two dedicated whiteboard artists** on a purpose-built
  dataset ([tutori-whiteboard-lessons](https://huggingface.co/datasets/ProCreations/tutori-whiteboard-lessons):
  7,109 gold lesson steps, 8 diagram families, 78 topics, every one validated
  to render with zero overlapping elements). Both are integrated behind a
  `TUTORI_BOARD_MODEL` flag — and the flag is **off**. The honest scorecard:
  - [**Nemotron 3 Nano 4B artist**](https://huggingface.co/ProCreations/tutori-board-nemotron)
    (eval loss 0.021): can't run here — Nemotron-H's Mamba-2 layers need the
    fused `mamba-ssm` Triton kernels, which ZeroGPU's fresh-per-turn workers
    can't JIT inside the 59 s turn budget (the pure-PyTorch fallback OOMs at
    prefill instead). The same constraint broke Nemotron's earlier role as
    research planner — which is how MiniCPM5 1B got the job, and it turned
    out to be excellent at it.
  - [**Gemma 4 12B artist**](https://huggingface.co/ProCreations/tutori-board-gemma)
    (eval loss 0.024, rides the already-loaded teacher as a LoRA): this one
    DID run live on ZeroGPU, and drew textbook diagrams on eval topics —
    correct loss-curve axes, properly labelled hypotenuses. But in real use
    it lost to the boring pipeline: the artist never sees the researched
    facts (only the topic and narration), so current-events lessons got
    improvised geometry, and sharing one model for narration + rendering
    serialized the turn past the GPU window. We kept the better teacher.
  On a persistent GPU (`TUTORI_REAL=1` locally) both artists work — flip the
  flag and they draw. ZeroGPU giveth, ZeroGPU taketh away.

## How a turn works

1. **You talk** (or type). Whisper turbo transcribes you on-device.
2. **MiniCPM plans.** MiniCPM5 1B decides whether the question needs fresh
   facts; if so it writes the search queries and Tutori pulls snippets +
   page text from the web (DuckDuckGo, keyless). Timeless topics skip
   straight to teaching.
3. **It teaches in steps.** Gemma 4 emits a JSON lesson script — each step is
   a sentence to *say* plus whiteboard ops to *draw* (boxes, arrows, curves,
   axes, highlights in a 100×75 coordinate space). A semantic board layer
   upgrades high-confidence topics into deterministic textbook diagrams before
   the layout pass.
4. **Steps stream.** The moment the first step's JSON closes, Higgs Audio
   voices it and ships it to your browser — Tutori starts talking while the
   rest of the lesson is still being generated.
5. **The board draws itself in sync.** A hand-drawn canvas renderer animates
   each stroke across exactly the duration of that step's audio — and when a
   lesson is drawing-heavy, the agent expands the whiteboard to take over the
   page, nudging the rest of the UI aside until the next lesson.
6. **The coach debriefs.** MiniCPM5 updates your learner profile and swaps
   the sticky notes for three follow-up questions tailored to what you just
   learned.

## The smart context system

- **Learner profile** — Tutori keeps gentle notes (level, goals, what clicked,
  what confused you) that it updates every turn and folds into the next
  lesson. Stored **only in your browser** (`BrowserState`), fully inspectable
  and erasable in the UI.
- **Pace dial** — 1 (total beginner, tiny steps, analogies) → 5 (expert,
  dense). Injected straight into the teaching prompt.
- **It can see your drawings** — sketch on the board with the pen tools and
  hit “🖐 Ask Tutori about the board”: Gemma 4’s vision reads your strokes.

## Running it

- **On Spaces (the real thing):** select **ZeroGPU** hardware. All four
  models load at startup and get packed by ZeroGPU; each turn runs in a
  single `@spaces.GPU(duration=59)` generator call, sized so even
  **logged-out visitors** (120s/day ZeroGPU quota) get a full lesson. Sign in
  to Hugging Face for much more daily GPU time.
- **Locally (no GPU needed):** `pip install gradio soundfile numpy && python app.py`
  runs a mock engine so you can play with the whiteboard, voice flow, and UI.
  Set `TUTORI_REAL=1` on a CUDA machine to use the real models.

## Submission

- **Space:** you're looking at it 🙂
- **Demo:** the video at the top of this card ([file](./tutori_demo.mp4))
- **Social Media Post:** https://x.com/SSHTheDev/status/2065159474671653005

## Merit badges claimed

- 🔌 **Off the Grid** — zero cloud model APIs; the only network egress is
  optional keyless web search, and you can switch that off too.
- 🎨 **Off-Brand** — a full "teacher's studio" design system over Gradio:
  chalkboard scene, hand-chalked header, paper cards with washi tape,
  sticky-note suggestion chips, marker-cap toolbar, self-hosted fonts
  (zero external requests) — plus the custom hand-drawn whiteboard renderer
  (vanilla canvas, ~600 lines of sketchy-stroke drawing) and an
  agent-controlled expanding board.
- 🤝 **Sharing is Caring** — verbatim agent traces from live sessions are
  published in [`traces/`](./traces): the Nemotron planner's search queries,
  every spoken sentence, and every whiteboard op with coordinates, timestamped.
- 📓 **Field Notes** — a full write-up of what we built and learned,
  published on the hackathon org blog:
  [Building Tutori, a Whiteboard Tutor That Draws While It Talks](https://huggingface.co/blog/build-small-hackathon/tutori)
  (source: [docs/field-notes.md](./docs/field-notes.md)).
- 🎯 **Well-Tuned** — we LoRA-fine-tuned TWO whiteboard artists
  ([Nemotron 3 Nano 4B](https://huggingface.co/ProCreations/tutori-board-nemotron)
  and [Gemma 4 12B](https://huggingface.co/ProCreations/tutori-board-gemma))
  on a purpose-built, programmatically validated dataset
  ([tutori-whiteboard-lessons](https://huggingface.co/datasets/ProCreations/tutori-whiteboard-lessons))
  and published all three. Both run flag-gated (`TUTORI_BOARD_MODEL=nemotron|gemma`);
  the Gemma artist was A/B-tested live on this very Space and honestly
  retired — full scorecard in the engineering notes above.
