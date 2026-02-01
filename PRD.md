# Product Requirements Document (PRD)

| Project Name | Ghost.ai |
| :--- | :--- |
| **Version** | 1.0 |
| **Status** | Draft |
| **Last Updated** | February 1, 2026 |
| **Category** | Agentic AI / Behavioral Modelling |

---

## 1. Executive Summary
**Ghost.ai** is a **Behavioral Compilation Pipeline** designed to transform static narrative text (novels, screenplays) into high-fidelity, autonomous AI agents. Unlike traditional approaches that rely on expensive model fine-tuning, Ghost.ai utilizes a **"Frozen Model, Fluid State"** architecture. It compiles raw text into a portable JSON specification (`Ghost_Spec`) using Inverse Reinforcement Learning (IRL) and DSPy-driven optimization, effectively "reading" a character into existence.

### 1.1 Core Value Proposition
*   **For Developers/Game Studios:** Create deeply realistic NPCs that adhere to canonical behavior without manual scripting.
*   **For Researchers:** A standardized framework for "Cognitive Architectures" that separates the Model (Brain) from the Personality (Soul).

---

## 2. Problem Statement
Current "Character AI" solutions suffer from:
1.  **RLHF Bias:** Models are tuned to be helpful assistants, making it difficult to simulate flawed, emotional, or irrational characters (e.g., a jealous child).
2.  **Context Amnesia:** Agents lose track of complex narrative history or core personality traits over long sessions.
3.  **Homogenization:** All characters sound like "Corporate English" due to lack of stylometric enforcement.

Ghost.ai solves this by treating character creation as an **Optimization Problem**, not a creative writing prompt.

---

## 3. System Architecture & Scope

The system is divided into three distinct phases.

### Phase 1: The Narrator (Ingestion)
*   **Goal:** Convert unstructured text into a structured, labeled "Ground Truth" dataset.
*   **Input:** Raw files (PDF/TXT) of literary works (Target: *Anne of Green Gables*).
*   **Output:** `Training_Set.jsonl` containing structured **Micro-Scenarios**.

### Phase 2: The Crucible (Evolution/Training)
*   **Goal:** Iteratively tune the Agent's configuration until behavior matches the Ground Truth.
*   **Mechanism:** A closed-loop simulation using **DSPy** to optimize System Prompts, Trait Weights, and Memory selection.
*   **Output:** The `Ghost_Spec_v3.json` (The "Ghost Bundle").

### Phase 3: The Ghost Engine (Runtime)
*   **Goal:** Execute the `Ghost_Spec` in real-time interactions.
*   **Mechanism:** A **LangGraph** orchestration layer using Mixture of Experts (MoE) and Retrieval-Augmented Style Transfer (RAST).

---

## 4. Functional Requirements

### 4.1 Ingestion Layer ("The Narrator")
*   **FR-1.1 Semantic Chunking:** System must split text based on *Narrative Scenes* (change of time/location), not arbitrary tokens.
*   **FR-1.2 Event Extraction:** Must extract `(Context, Action, Internal_State)` tuples from scenes.
    *   *Context:* "Gilbert pulls braid."
    *   *Action:* "Anne smashes slate."
    *   *Internal_State:* "Humiliation, Rage."
*   **FR-1.3 Stylometry Profiling:** System must statistically analyze the text (TF-IDF) to identify "Signature Phrases" and "Syntactic Density" (e.g., usage of adjectives vs. verbs) to populate the `linguistic_bio`.

### 4.2 Training Layer ("The Crucible")
*   **FR-2.1 Simulation Loop:** System must instantiate a "Headless" version of the Runtime agent to generate responses to historical contexts.
*   **FR-2.2 Auto-Evaluation ("The Judge"):** A scoring function that compares Agent Output vs. Book Truth.
    *   *Metric:* Cosine Similarity of Intent + Emotional Alignment Score.
    *   *Pass Threshold:* >90% Fidelity.
*   **FR-2.3 Optimization Strategy:**
    *   If Score < 90%, the system must use **DSPy (MIPROv2)** to propose updates to the `Ghost_Spec`.
    *   *Mutable Parameters:* Trait Vectors (Big Five), System Prompt Instructions, Memory Salience Weights.
    *   *Immutable Parameters:* The Base LLM Weights.

### 4.3 Runtime Layer ("The Ghost Engine")
*   **FR-3.1 MoE Routing:** A router must classify user intent and delegate generation to specific "Expert" chains:
    *   *Narrative Expert:* Access to deep lore/memory.
    *   *Spatial Expert:* Access to physical location data.
    *   *Psych Expert:* Management of Maslow’s Needs.
*   **FR-3.2 Dynamic State Management:** The agent must maintain a `Needs_State` (e.g., Social Belonging, Hunger) that decays over time and influences behavior.
*   **FR-3.3 RAST (Retrieval-Augmented Style Transfer):** The inference prompt must be dynamically seeded with 3-5 "Style Exemplars" (quotes from the book) relevant to the current topic to enforce "Voice."

---

## 5. Data Model: The Ghost Spec
The core artifact is `Ghost_Spec_v3.json`. The system must strictly adhere to this schema.

| Field | Description | Type |
| :--- | :--- | :--- |
| `core_personality.traits` | The "Big Five" + Custom traits (0.0 - 1.0 floats). | Mutable (Optimized) |
| `core_personality.needs` | Maslow's hierarchy states (e.g., `loneliness`). | Dynamic (Runtime) |
| `linguistic_bio.signatures` | Catchphrases and idiolect markers. | Static (extracted) |
| `skills` | "Unlockable" capabilities (e.g., `slate_combat`). | Boolean / Level |
| `memory_graph` | Reference to the Vector DB Cluster ID. | Reference |

---

## 6. Assumptions & Constraints

### 6.1 Technical Constraints
*   **Base Model:** Gemini 3 Pro (or equivalent SOTA model).
*   **Orchestration:** **LangGraph** is required for the cyclic state management.
*   **Optimization:** **DSPy** is required for the feedback loop.
*   **Vector DB:** ChromaDB or MemGPT-compatible store.

### 6.2 Logic Assumptions
*   **The "Method Actor" Assumption:** We assume the base model is intelligent but misaligned. We are not "teaching it to speak"; we are "directing it to act."
*   **The "90% Fidelity" Rule:** Fidelity is measured by **Intent**, not verbatim text. (e.g., "I hate you" $\approx$ "You are detestable" = PASS).
*   **The Latency Trade-off:** We accept higher latency (1-3s) for the sake of behavioral depth (MoE + RAST overhead).

---

## 7. Success Metrics (KPIs)

*   **Behavioral Fidelity Score (BFS):** Average semantic similarity score between Agent Actions and Book Actions across the validation set (Target: >0.85).
*   **Character Consistency:** Rate of contradiction in "Core Beliefs" over a 50-turn conversation (Target: <2 contradictions).
*   **Stylometric Match:** Kullback–Leibler (KL) divergence between Agent's word distribution and the original Author's distribution (Target: Low divergence).

---

## 8. Future Scope (Post-MVP)
*   **Visual Imagination:** Integrating Stable Diffusion to allow the agent to "imagine" scenes visually.
*   **Multi-Agent Society:** Simulating an entire village (Avonlea) where distinct Ghost.ai agents interact without user input.
*   **Real-time Voice:** Integrating low-latency TTS with emotional modulation.
