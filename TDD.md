# Technical Design Document (TDD)

| Project Name | Ghost.ai |
| :--- | :--- |
| **Version** | 1.2 (Final Architecture with References) |
| **Status** | Approved for Implementation |
| **Date** | February 1, 2026 |
| **Target Audience** | AI Coding Agents / Senior Developers |

---

## 1. System Overview & Metaphor
**Ghost.ai** is a **Behavioral Compilation Pipeline**. It treats a literary character as a dataset to be "compiled" into an executable agent state.
*   **The Input:** Unstructured narrative text (The "Source Code").
*   **The Compiler:** A DSPy-driven optimization loop ("The Crucible") that employs Inverse Reinforcement Learning.
*   **The Binary:** `Ghost_Spec.json` (A portable, frozen state definition).
*   **The Runtime:** A LangGraph-based engine ("The Ghost Engine") that executes the Spec using a Mixture of Experts (MoE) topology.

**Core Architectural Invariant:**
We do **not** fine-tune the LLM weights. We optimize the **Contextual State** (System Prompts, Trait Vectors, Memory Salience, Needs Decay Rates).

---

## 2. Technology Stack

### 2.1 Core Frameworks
| Layer | Library | Minimum Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.11+ | Primary ecosystem. |
| **Orchestration** | **LangGraph** | 0.0.15+ | Cyclic state management, MoE routing. |
| **Optimization** | **DSPy** | 2.1.0+ | `MIPROv2` optimizer for Prompt/Weight tuning. |
| **Ingestion** | **LlamaIndex** | 0.9+ | `SemanticSplitterNodeParser` for scene chunking. |
| **Storage** | **ChromaDB** | 0.4+ | Vector store for Episodic Memory & Style Exemplars. |
| **Validation** | **Pydantic** | 2.6+ | Strict schema enforcement for IO. |
| **Serving** | **FastAPI** | 0.109+ | Async REST API for the runtime agent. |
| **Math** | **NumPy** | Latest | Vector operations and Needs decay limits. |

### 2.2 Models
*   **Inference (Runtime):** `gemini-3-pro` (Vertex AI) or `gpt-4-turbo`.
*   **Inference (Judge):** `gpt-4o` (Must be high-reasoning capability).
*   **Embedding:** `text-embedding-3-small` (OpenAI) or `gecko` (Google).
    *   *Constraint:* Dimension must be consistent (1536) across Ingestion and Runtime.

---

## 3. Data Architecture

### 3.1 The Artifact: `Ghost_Spec_v3.json`
This file acts as the "ROM" for the agent.

```json
{
  "meta": {
    "name": "Anne Shirley",
    "version": "1.0.0",
    "base_model_compatibility": "gemini-3-pro"
  },
  "cognitive_state": {
    "traits": {
      "openness": 0.98,
      "conscientiousness": 0.35,
      "extraversion": 0.70,
      "agreeableness": 0.40,
      "neuroticism": 0.85
    },
    "needs_config": {
      "social_belonging": { "baseline": 0.5, "decay_rate": 0.05 },
      "safety": { "baseline": 0.8, "decay_rate": 0.01 },
      "esteem": { "baseline": 0.4, "decay_rate": 0.02 }
    },
    "core_beliefs": [
      "Imagination is the only escape from reality.",
      "Red hair is a curse that defines my tragedy."
    ]
  },
  "linguistic_bio": {
    "catchphrases": [
      { "phrase": "depths of despair", "weight": 0.9, "sentiment": "negative" },
      { "phrase": "scope for imagination", "weight": 0.8, "sentiment": "positive" }
    ],
    "syntax_preferences": {
      "adjective_frequency": 1.4,
      "sentence_complexity": "high"
    }
  },
  "skill_tree": {
    "slate_combat": {
      "unlocked": false,
      "trigger_condition": "user_sentiment < -0.8 AND topic == 'hair'"
    },
    "poetry_recitation": {
      "unlocked": true,
      "trigger_condition": "topic == 'nature'"
    }
  },
  "memory_ref": {
    "collection_id": "anne_shirley_mem_001",
    "style_collection_id": "anne_shirley_style_001"
  }
}
```

### 3.2 Vector Database Schema (ChromaDB)
We use two collections to separate "What happened" from "How she talks".

**Collection 1: `episodic_memory`**
*   `id`: UUID
*   `document`: "Gilbert pulled my braid and called me Carrots." (Text)
*   `metadata`: `{"timestamp": int, "emotion": "rage", "entities": "Gilbert"}`

**Collection 2: `style_exemplars`**
*   `id`: UUID
*   `document`: "You mean, hateful boy! I hate you!" (Direct Quote)
*   `metadata`: `{"tone": "aggressive", "context_tags": "conflict, humiliation"}`

---

## 4. Module Specifications

### 4.1 Module 1: The Narrator (Ingestion)
**Path:** `src/ingestion/`

**Pipeline Flow:**
1.  **Raw Load:** Read `.txt` / `.pdf`.
2.  **Semantic Chunking:**
    *   Use `LlamaIndex` `SemanticSplitterNodeParser`.
    *   *Parameters:* `buffer_size=1`, `breakpoint_percentile_threshold=95` (Force split on major topic shifts).
3.  **Extraction (The "Ground Truth" Generator):**
    *   For each chunk, call LLM to extract a list of `TrainingEpisode` objects.
    *   *Schema:* `Stimulus` (Input), `Reaction` (Output), `InternalState` (Hidden).
    *   *Critical:* Extract direct quotes into the `style_exemplars` Chroma collection.
4.  **Stylometry Analysis:**
    *   Run `TfidfVectorizer` on the extracted dialogue.
    *   Compare against `wikitext-103`.
    *   Store top 20 distinct n-grams in `Ghost_Spec.linguistic_bio`.

### 4.2 Module 2: The Crucible (Optimization)
**Path:** `src/crucible/`

**Concept:** Uses **DSPy** to optimize the `Ghost_Spec` values using the `TrainingEpisode` data as the validation set.

**Classes:**
*   `HeadlessAgent(dspy.Module)`: A wrapper that mimics the Runtime MoE but is fully differentiable/traceable by DSPy.
    *   *Forward Pass:* Input Context $\to$ Apply Traits $\to$ Generate Action.
*   **The Optimizer:** `dspy.MIPROv2`.
    *   *Target:* Optimize `system_prompt_template` and `trait_values`.
*   **The Metric (The Judge):**
    ```python
    def assess_fidelity(gold: TrainingEpisode, pred: Prediction, trace=None):
        # 1. Intent Matching (Cosine Sim of Action Description)
        intent_score = embedding_similarity(gold.reaction_desc, pred.action)
        
        # 2. Stylistic Matching (if dialogue exists)
        style_score = 0.0
        if gold.dialogue:
             style_score = lexical_overlap(gold.dialogue, pred.dialogue)
        
        # 3. Factuality Check (Negative Constraint)
        hallucination_penalty = 0.0
        if pred.contains_info_not_in(gold.context):
             hallucination_penalty = 1.0
        
        return (0.6 * intent_score) + (0.4 * style_score) - hallucination_penalty
    ```

### 4.3 Module 3: The Ghost Engine (Runtime)
**Path:** `src/runtime/`

**Architecture:** **LangGraph StateGraph**
*   **Global State:** `AgentState` (Messages, Needs, Active Skills, Location).

**Node Logic:**

1.  **`node_perception`**:
    *   Updates `Needs`: $Need_t = Need_{t-1} - (DecayRate \times \Delta Time)$.
    *   If `Social < 0.2`, push `Force_Conversation` flag to Context.
2.  **`node_router` (MoE)**:
    *   Classify User Input.
    *   *Routing Table:*
        *   "Where/Go/Map" $\to$ `SpatialExpert`.
        *   "Feel/Think/Want" $\to$ `PsychExpert`.
        *   "Recall/Remember/Who" $\to$ `NarrativeExpert`.
3.  **Experts (Prompt Generators)**:
    *   **Spatial:** Injects `Ghost_Spec.world_knowledge` + current coordinates.
    *   **Psych:** Injects `Ghost_Spec.needs_config` + current trait vectors.
    *   **Narrative:** Performs RAG on `episodic_memory`.
4.  **`node_rast` (Style Transfer)**:
    *   Performs Vector Search on `style_exemplars` using user input.
    *   Retrieves top 3 quotes.
    *   Constructs Final System Prompt:
        > "You are [Name]. Your voice references: [Quote 1], [Quote 2]. Adopt this syntax."
5.  **`node_action`**: calls LLM.

---

## 5. Implementation Guidelines (Best Practices)

### 5.1 DSPy Integration
*   **Do not** wrap the entire LangGraph in DSPy. Only wrap the *Decision/Generation* steps that need optimization.
*   Use `dspy.InputField` for `Stimulus` and `dspy.OutputField` for `Action`.
*   The `Ghost_Spec` traits should be injected as `dspy.InputField` (as context) during the forward pass so MIPRO can tune the values fed into them.

### 5.2 RAG & VectorDB
*   **Metadata is King:** Never store just the text. Always store `who`, `when`, and `emotion` in metadata.
*   **Hybrid Search:** If possible, enable Keyword Search + Vector Search in Chroma (or use a re-ranker like `FlashRank`) to ensure specific names ("Diana Barry") are caught even if embeddings drift.

### 5.3 Safety & Guardrails
*   **Sandwich Prompting:**
    *   *Top:* Identity/Traits (Protected).
    *   *Middle:* Retrieval/Needs/User Input.
    *   *Bottom:* "Stay in character. Do not break the fourth wall." (Protected).
*   **Skill Locking:** Ensure `skills` in `Ghost_Spec` act as *permission gates*. The agent cannot output a combat action unless `slate_combat` is unlocked in the spec.

---

## 6. Directory Structure
```text
ghost-ai/
├── data/
│   ├── raw/                  # Source PDFs/TXTs
│   ├── processed/            # training_episodes.jsonl
│   └── chromadb/             # Local Vector Store
├── profiles/                 # The Output Artifacts
│   └── anne_shirley_v1.json
├── src/
│   ├── ingestion/
│   │   ├── splitter.py       # LlamaIndex Logic
│   │   ├── extractor.py      # Ground Truth Gen
│   │   └── stylometry.py     # TF-IDF Logic
│   ├── crucible/
│   │   ├── simulator.py      # Headless Agent Wrapper
│   │   ├── optimizer.py      # DSPy MIPRO Script
│   │   └── metrics.py        # The Judge
│   └── runtime/
│       ├── graph.py          # LangGraph Definition
│       ├── experts.py        # MoE Prompt Logic
│       ├── rast.py           # Style Retrieval
│       └── server.py         # FastAPI App
└── requirements.txt
```

---

## 7. References & Project URLs

### Research Papers (Foundational)
*   **CoALA: Cognitive Architectures for Language Agents** (Sumers et al., 2023)
    *   *Paper:* [arXiv:2309.02427](https://arxiv.org/abs/2309.02427)
    *   *Relevance:* Defines the "Memory vs. Action" separation standard.
*   **Generative Agents: Interactive Simulacra of Human Behavior** (Park et al., 2023)
    *   *Paper:* [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
    *   *Repo:* [joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents)
    *   *Relevance:* Foundational architecture for Memory Streams and Reflection.
*   **DSPy: Compiling Declarative Language Model Calls** (Khattab et al., 2023)
    *   *Paper:* [arXiv:2310.03714](https://arxiv.org/abs/2310.03714)
    *   *Relevance:* The optimization logic used in "The Crucible".
*   **MemGPT: Towards LLMs as Operating Systems** (Packer et al., 2023)
    *   *Paper:* [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
    *   *Relevance:* Inspiration for the OS-level memory hierarchy.

### Libraries & Frameworks
*   **DSPy (StanfordNLP):**
    *   *Repo:* [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
    *   *Docs:* [dspy.ai](https://dspy.ai/)
*   **LangGraph (LangChain):**
    *   *Repo:* [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
    *   *Docs:* [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
*   **LlamaIndex:**
    *   *Repo:* [run-llama/llama_index](https://github.com/run-llama/llama_index)
    *   *Docs:* [docs.llamaindex.ai](https://docs.llamaindex.ai/)
*   **ChromaDB:**
    *   *Repo:* [chroma-core/chroma](https://github.com/chroma-core/chroma)
    *   *Docs:* [docs.trychroma.com](https://docs.trychroma.com/)

### Related Works & Inspirations
*   **Voyager (MineDojo):** An Open-Ended Embodied Agent with LLMs.
    *   *Project:* [voyager.minedojo.org](https://voyager.minedojo.org/)
    *   *Repo:* [MineDojo/Voyager](https://github.com/MineDojo/Voyager)
*   **SillyTavern:** Open-source frontend for complex character interactions.
    *   *Repo:* [SillyTavern/SillyTavern](https://github.com/SillyTavern/SillyTavern)
    *   *Relevance:* Reference for "Character Card" data structures.
