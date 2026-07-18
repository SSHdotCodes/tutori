# Agent traces — Sharing is Caring 🤝

Three **verbatim event traces** from live sessions on this Space, exactly as
the agent streamed them (timestamps in seconds from turn start):

| Trace | Shows |
|---|---|
| `trace-space-exploration.json` | The full research path: Nemotron 3 Nano decides to search, its queries appear verbatim in the `Researching:` status, and the lesson teaches from what it found |
| `trace-rockets-orbit.json` | A conceptual lesson — the planner correctly skips the web |
| `trace-gradient-descent.json` | A quantitative lesson (axes / curve / descent arrow board ops) |

Each event is one of: `status` (the agent's live state, including the
planner's search queries), `step` (a lesson step: the exact spoken sentence
plus every whiteboard op with coordinates, post layout-engine), or `final`.
TTS audio payloads are stripped for repo size; everything else is untouched.
