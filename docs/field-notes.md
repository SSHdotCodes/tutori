# Field Notes: Building Tutori, a Whiteboard Tutor That Draws While It Talks

*What we built for the HF Build Small Hackathon, what broke, and what 17B
parameters taught us about engineering around small models.*

[**Tutori**](https://huggingface.co/spaces/build-small-hackathon/tutori) is a
voice tutor. You ask it anything out loud; it researches the question if it
needs to, then teaches you in spoken steps while sketching the idea on a
whiteboard — stroke by stroke, in sync with its own voice, like a teacher who
never gets tired of your questions.

I built it for my parents. They're behind on today's technology, and the
firehose of AI news is impenetrable from the outside. With Tutori they just
ask — "what is Gemma?" — and get a patient, drawn-out explanation at their
pace. They actually use it. That's the only benchmark that ever mattered.

Everything runs on the Space itself via ZeroGPU: Gemma 4 12B teaches,
MiniCPM5 1B plans research and coaches, Higgs Audio v2 speaks, Whisper turbo
listens. No cloud APIs, no keys. Here is what we learned building it.

## Lesson 1: The model is 20% of a drawing tutor. Geometry is the other 80%.

Our first whiteboards were a disaster — overlapping labels, triangles
assembled from three unrelated lines, values scattered like confetti. We
tried prompting harder. It did not work, and it never works: a language
model freehanding 2D coordinates is a model being asked to do geometry
without eyes.

What worked was splitting the job:

- **A composite op vocabulary.** The model stopped placing twelve primitives
  and started saying `{"op": "polygon", "side_labels": ["a", "b", "c"]}` —
  one op, with the geometry math done deterministically in the renderer.
- **A layout engine with zero tolerance.** Every op the model emits passes
  through a placement pass: weighted obstacle maps, candidate scanning,
  shrink-on-crowding, a capacity accountant that drops what won't fit, and a
  hard guarantee that nothing overlapping ever reaches the screen.
- **A fuzz harness.** We generate hundreds of synthetic "worst-case" lessons
  and require zero violations before any deploy. The harness caught ten real
  bugs that user screenshots later would have.

The model proposes; the engine disposes.

## Lesson 2: ZeroGPU is a hardware constraint disguised as a free GPU.

ZeroGPU gives every Space an H200 slice with one catch: your GPU code runs in
disposable forked workers with a per-turn time budget. Almost every hard
problem we hit traces back to this:

- **NeMo crashes the workers** if imported in the main process, and costs
  30–60s per fresh worker if loaded lazily — so our planned Nemotron ASR
  became Whisper (preloads with everything, transcribes in a second).
- **Mamba-hybrid models can't really run.** Nemotron-H's Mamba-2 layers need
  fused Triton kernels that won't JIT-compile inside a 59-second window, and
  the pure-PyTorch fallback allocates multi-gigabyte tensors at prefill and
  OOMs the worker. We learned this the expensive way — see Lesson 4.
- **Fast generators get coalesced.** Gradio collapses rapid streaming yields
  into the latest state, which silently dropped lesson steps until we made
  every payload cumulative and let the browser deduplicate.

None of this is in a tutorial. All of it is reproducible on a $0 Space, which
is exactly why the constraint is worth embracing.

## Lesson 3: Small models need formats, not freedom.

Three times we needed structured output from a model ≤1B parameters, and
three times the same arc played out. Ask MiniCPM5 1B for nested JSON and it
writes brilliant content with one missing bracket. Give it a few-shot example
and it copies the example verbatim. What finally worked was a *line
protocol*:

```
PROFILE: {"level": "beginner", "last_topic": "rainbows", ...}
NEXT1: What happens to the colors inside a raindrop?
NEXT2: Why do we see bands instead of a smear?
NEXT3: Could a rainbow form at night?
```

One regex per line, code-side guards against parroting, and the 1B model
became a reliable study coach — it updates the learner's profile after every
lesson and writes the three sticky-note follow-up questions you see in the
app. Small models are superb employees and terrible freelancers: define the
job precisely and they shine.

## Lesson 4: Fine-tune honestly, and let the boring pipeline win.

For the Well-Tuned badge we built a
[dataset](https://huggingface.co/datasets/ProCreations/tutori-whiteboard-lessons)
of 7,109 gold whiteboard lesson steps — programmatically generated across 8
diagram families and 78 topics, every single lesson validated to render with
zero overlaps *before* it could enter the training set. Then we LoRA-tuned
two dedicated "board artists":
[Nemotron 3 Nano 4B](https://huggingface.co/ProCreations/tutori-board-nemotron)
and [Gemma 4 12B](https://huggingface.co/ProCreations/tutori-board-gemma).

Both learned the job almost perfectly on paper (99% held-out token accuracy).
The Gemma artist drew the best gradient-descent diagram we've ever gotten
from any model — correct axes, slope arrows, the actual update rule. And we
still shipped neither. The Nemotron artist can't execute on ZeroGPU at all
(Lesson 2), and the Gemma artist lost a live A/B for a humbling reason: the
artist never sees the researched facts, only the narration — so a lesson
about that week's space news got confidently improvised geometry. Meanwhile
the boring pipeline — the big teacher model plus the deterministic layout
engine — kept quietly producing good boards.

Publishing the models, the dataset, and the scorecard felt better than
flipping the flag. The README says exactly why it's off.

## Lesson 5: A custom face is mostly CSS — and one JS gotcha.

The "stunning UI" pass that chases the Off-Brand badge is not a rewrite. It's
a design system (chalkboard scene, paper cards, sticky notes, two bundled
fonts), about 400 lines of CSS over Gradio's components, a hand-drawn canvas
renderer we already had, and one hard-won discovery: Gradio's CSS
preprocessor silently drops `:has()` selectors and everything after them —
tag elements with a class from JavaScript instead.

## The scorecard

| | |
|---|---|
| Models on the Space | 4 (Gemma 4 12B · Higgs v2 3B · MiniCPM5 1B · Whisper 0.8B) — Σ ~16.9B |
| Models fine-tuned & published | 2 + 1 dataset |
| Cloud APIs | 0 |
| Layout-engine fuzz violations at ship | 0 of 600 |
| GPU budget per lesson | 59 seconds, every model included |

Built small. It was the constraint that made it good.

*— SSH/ProCreations, June 2026*
