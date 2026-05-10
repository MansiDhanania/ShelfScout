# ShelfScout — Architecture & Design Decisions

This document describes the backend architecture of ShelfScout and the reasoning behind key technical decisions. It is intended to give a complete picture of the system for anyone who wants to understand how it was built and why.

---

## What ShelfScout Does

ShelfScout is a real-time AI assistant that helps blind and visually impaired (BVI) users navigate complex environments such as grocery stores, independently. A user speaks a query ("where are the apples?"), the system captures an image from their device, and responds with spatially grounded audio guidance within seconds.

The system is frontend-agnostic by design: the same backend serves a web app, an iOS/Android mobile app, and (in future) smart glasses — without any backend changes. All intelligence lives server-side.

---

## System Architecture Overview

```
Frontend (Web / iOS / Android / Smart Glasses)
        │
        │  POST /webhook  (audio transcript + image + session_id)
        ▼
   n8n Workflow Engine  ◄──────────────────────────────────────┐
        │                                                       │
        ├─ Redis (session state)                                │
        ├─ Intent Classification  (GPT-OSS-120B via Groq)      │
        ├─ Intent Router  ──────────────────────────────────────┤
        │     ├─ Scene Description    → Groq LMM               │
        │     ├─ Object Detection     → Qwen Vision Pipeline    │
        │     │                       + Groq Guidance LMM       │
        │     ├─ Navigation (SLAM)    → RTAB-Map API            │
        │     │                       + Groq Guidance LMM       │
        │     ├─ Reaching (ARKit)     → Qwen Vision Pipeline    │
        │     │                       + Reaching Microservice   │
        │     ├─ Validation           → Groq LMM               │
        │     └─ Chat                 → Groq LMM               │
        │                                                       │
        ├─ Response Synthesizer  (GPT-OSS-120B via Groq)        │
        └─ Webhook Response  →  Frontend  →  TTS  →  User      │
                                                                │
   [Every Groq call has a Gemini fallback on inference failure]─┘
```

---

## The n8n Workflow: Routing Logic

The core of the backend is an n8n workflow that handles every incoming request. Here is the full decision flow:

### Stage 1 — Session Initialisation
On every request, the payload (transcript, image, session flags, mode, reaching flags, image dimensions) is written to a Redis hash keyed by `session:{session_id}`. This happens before any model inference, so all subsequent nodes can retrieve exactly what they need without passing large payloads through the workflow graph.

### Stage 2 — Intent Classification
The transcript is passed to **GPT-OSS-120B** (via Groq) with a structured output parser enforcing a JSON schema:
```json
{ "reasoning": "...", "intent": "...", "content": "specific object if mentioned" }
```
Intent is classified into one of: `scene`, `object`, `guidance`, `validation`, `chat`. The reasoning and extracted object name are written back to Redis immediately.

### Stage 3 — Presence Check (for object/guidance intents)
Before running expensive vision inference, the system checks whether the target object is visible in the current frame. A lightweight LMM call (LLaMA-4-Scout-17B via Groq) with a strict JSON schema (`{"evidence": "...", "visible": true/false}`) determines this. If the object is not present, the system routes to a "no object found" path with either a RTAB-Map based cross-aisle navigation guidance (for guidance intents, if the object is present in the mapped environment) or a natural language fallback (for object description intents), avoiding unnecessary Qwen inference.

### Stage 4 — Intent Routing
A conditional router reads `presence` and `intent` from Redis and branches into one of the following paths:

| Intent | Path |
|--------|------|
| `scene` | Groq Scene Description → Redis → Synthesize |
| `object` | Qwen Detection → Groq Guidance → Redis → Synthesize |
| `guidance` but `object` NOT present (SLAM active) | RTAB-Map API → Groq Turn-Priority Guidance → Synthesize |
| `guidance` when `object` IS present | Qwen Object Detection → Groq High-level Guidance + Hand-level Reaching Microservice → Synthesize |
| `validation` | Groq Validation LMM → Synthesize |
| `chat` | Groq Chat LMM → Synthesize |

### Stage 5 — Response Synthesis
All paths converge at a synthesis node. **GPT-OSS-120B** receives the raw model output and distills it into a concise, conversational response calibrated to the user's profile (blind/low vision) and preferred style (chatty/efficient). The synthesised response is written to Redis and returned to the webhook.

---

## Redis Session Schema

Redis stores all per-session state as a hash under `session:{session_id}`. This was a deliberate design choice over passing state through the workflow graph — it keeps nodes decoupled, enables selective retrieval (each node only fetches the fields it needs), and makes the system resumable mid-session.

### Key fields

| Field | Purpose |
|-------|---------|
| `transcript` | Current user query |
| `image` | Base64-encoded frame from device |
| `intent` | Classified intent for this turn |
| `object` | Target object name (if applicable) |
| `prev_intent` | Intent from previous turn (for context) |
| `prev_object` | Object from previous turn |
| `prev_reasoning` | LLM reasoning from previous turn |
| `mode` | Current interaction mode |
| `navigation` | SLAM navigation state flag |
| `reaching_flag` | Whether reaching mode is active |
| `reaching_ios` | ARKit reaching state |
| `presence` | Whether target object is visible in frame |
| `vision_model` | Configurable per-session vision model |
| `bbox` | Bounding box from detection |
| `depth` | Depth estimate from Depth-Anything-V2 |
| `confidence` | Detection confidence |
| `annotated_image` | Image with detection overlays |
| `response` | Final synthesised response |
| `reached` | Whether user has reached the object |
| `segments` | Stepwise navigation segments from RTAB-Map |

### Why separate navigation state from reaching state?
Cross-aisle navigation (SLAM-based) and fine-grained reaching (ARKit-based) operate on fundamentally different data and timescales. Navigation state is persistent across multiple turns and updates incrementally as the user moves. Reaching state is frame-specific and gives hand-level guidance to assist the user in grasping the object. Storing them under separate Redis keys with separate retrieval nodes means neither path ever loads data it doesn't need.

---

## Qwen Vision Pipeline: Three Deployment Targets

Object detection uses **Qwen3-VL** (open-vocabulary vision-language model) deployed across three targets, selected at runtime:

| Target | URL | When used |
|--------|-----|-----------|
| Local GPU server | `cybersight-vision-pipeline-1:5000` | Primary — lowest latency |
| Pegasus (lab server) | `ollama.pegasus.cim.mcgill.ca` | Secondary GPU fallback |
| Fireworks API | `api.fireworks.ai` | Cloud fallback when both local servers unavailable |

Images must be preprocessed to match Qwen's `smart_resize` logic (the `/qwen-preprocessing` endpoint handles this), and returned bounding boxes are rescaled back to original image coordinates via stored scale factors.

Gemini is available as a final fallback on every detection path if all Qwen targets fail.

### Why Qwen over OmDet/YOLO?
The original architecture used OmDet for object detection. OmDet was replaced with Qwen3-VL for two reasons:
..........
1. **Open-vocabulary detection**: OmDet, like YOLO-based detectors, operates on a fixed set of trained object classes. A user might ask for "the Heinz ketchup" or "the gluten-free bread" — arbitrary natural language object names that a fixed-class detector cannot handle. Qwen3-VL takes the object name directly as a text prompt and detects it regardless of whether it appeared in training data.

2. **Integrated reasoning**: Qwen can simultaneously detect, localise, and describe an object — reducing the number of model calls needed for a single user query.

The tradeoff is reliability: LMM outputs are less structured than traditional detector outputs, which is why the multi-strategy JSON repair pipeline in `vision_app.py` exists.

---

## Model Selection: The Full Evaluation Journey

The current model choices emerged from systematic evaluation over the course of the project. The following models were tested and rejected or adopted at various stages:

### Object Detection / Vision
| Model | Outcome | Reason |
|-------|---------|--------|
| YOLOv11 + DPT | Replaced | Fixed object classes — cannot handle arbitrary user queries |
| OmDet | Replaced | Same limitation as YOLO; open-vocabulary but weaker than LMM |
| SAMURAI / SAM2 | Rejected | Designed for video tracking, not real-time single-frame detection; 1hr43min on CPU for 9s clip |
| BLIP-2 | Rejected | Image captioning only, no spatial grounding |
| **Qwen2.5-VL** | **Adopted** | Open-vocabulary, spatially grounded, integrated with depth estimation |

### OCR (for product label reading)
| Model | Outcome | Reason |
|-------|---------|--------|
| Tesseract 5.5 | Rejected | Poor performance on real-world grocery packaging |
| EasyOCR | Rejected | Better but still unreliable on skewed/printed labels |
| EAST + EasyOCR | Rejected | Improved detection regions but still brittle |
| Google Vision AI | Not deployed | API key constraints; tested only |
| Azure OCR / Read | Not deployed | Preferred Read over Vision (better text ordering) but not integrated |
| LLaMA-3.2-90B Vision (Groq) | Adopted via VLM path | Prompt-sensitive OCR via natural language — more robust than dedicated OCR |

### Scene Understanding / LMM
| Model | Outcome | Reason |
|-------|---------|--------|
| LLaMA-3.2-90B | Early prototype | Replaced by faster, more capable models |
| LLaMA-4-Scout-17B | Adopted for presence check | Lightweight, fast, sufficient for binary visible/not-visible decision |
| **LLaMA-4-Maverick** | **Adopted for guidance/description** | Best balance of reasoning quality and latency via Groq |
| GPT-OSS-120B | Adopted for intent classification + synthesis | Strongest instruction-following for structured JSON output |
| Gemini (via Google API) | Adopted as universal fallback | Reliable, multimodal, available when Groq is unavailable |
| DeepSeek-V3 | Evaluated | Strong but requires local GPU; not suitable for cloud path |

### Depth Estimation
| Model | Outcome | Reason |
|-------|---------|--------|
| DPT (Dense Prediction Transformer) | Replaced | Used with YOLO; replaced when switching to Qwen |
| **Depth-Anything-V2-Small** | **Adopted** | ~100MB, fast, accurate enough for spatial guidance; integrates directly with Qwen pipeline |

### Navigation
| Model/System | Outcome | Reason |
|-------|---------|--------|
| RL agent (PPO, custom gym env) | Prototype only | 7D state space, trained for ~2M timesteps — unreliable performance; limited to simulated environments |
| **RTAB-Map** | **Adopted** | Production-grade SLAM; provides real-world spatial coordinates, map-based localisation, and route guidance |

---

## User Profiles and Response Personalisation

The system supports configurable user profiles that modify how responses are generated:

- **Blind (Tactile)**: Responses avoid colour references; use shape, size, and positional language instead
- **Low Vision (Color)**: Colour descriptions included as they aid navigation for users with partial sight
- **Style: Chatty**: Longer, more conversational responses
- **Style: Efficient**: Concise, action-oriented responses

These profiles are applied at the synthesis stage, not by modifying individual model prompts — keeping the routing logic profile-agnostic.

---

## Infrastructure

### Docker Network
All services run as Docker containers on a centralised network:
- `vision-pipeline` — Qwen2.5-VL + Depth-Anything-V2 (FastAPI, port 5000)
- `n8n` — workflow engine
- `redis` — session state
- `rtabmap-api` — SLAM navigation (port 8000)
- `reaching` — ARKit reaching microservice (port 8000)
- `kokoro-tts` — local text-to-speech
- `faster-whisper` — local speech-to-text

### Traefik
Traefik acts as the reverse proxy and TLS termination layer, routing external HTTPS traffic to internal services. This allows the system to be accessible at `https://cybersight.cim.mcgill.ca` from any device (web, mobile, smart glasses) without exposing internal ports.

### MCP Servers
SearXNG (self-hosted search) and Crawl4AI are deployed as MCP servers on the same Docker network, available for future agent-based extensions requiring web retrieval.

---

## Evaluation

ShelfScout was benchmarked against three commercial systems across four real grocery store environments (produce, dry goods, bakery, cold section):

| System | Hallucination Rate | Actionable Guidance |
|--------|--------------------|---------------------|
| Be My AI (GPT-4V) | 0.007/word | Partial |
| Meta Ray-Ban glasses | 0.030/word | Inconsistent |
| Gemini Live | 0.004/word | Poor |
| **ShelfScout** | **0.007/word** | **Consistent** |

ShelfScout matched the lowest hallucination rates while being the only system to consistently produce spatially grounded, step-wise guidance instructions. Findings submitted to ACM 2026.

---

## What's Next

- Integration with smart glasses frontend (currently in progress)
- Expanded user study with BVI participants in real shopping scenarios
- Customisable response profiles (user-controlled verbosity, colour descriptions toggle)
- Context-aware memory across sessions (not just within a single session)
