# Ghost.ai

**The Behavioral Compiler: Transforming Static Narrative into Dynamic Agency.**

![Ghost.ai Architecture](https://github.com/sztimhdd/Ghost.ai/blob/main/Arch-diagram-v1.jpeg?raw=true)

> **"Frozen Model, Fluid State."**

Ghost.ai is not just another chatbot. It is a **Large Behavioral Model (LBM) pipeline** designed to "compile" literary characters (from novels, scripts, or psychological profiles) into high-fidelity, autonomous agents.

Instead of fine-tuning Large Language Models (which is costly and rigid), Ghost.ai uses **Inverse Reinforcement Learning** and **DSPy-driven Optimization** to iteratively tune the *contextual state* (Traits, Memories, Styles) of a SOTA base model until it behaves exactly like the target character.

---

## 🏗 System Architecture

The Ghost.ai pipeline consists of three distinct phases, moving from raw data to optimized runtime execution.

### Phase 1: The Narrator (Ingestion)
*The Data Preprocessing Layer.*
*   **Input:** Raw textual material (e.g., *Anne of Green Gables*, Movie Scripts).
*   **Semantic Chunking:** Uses **LlamaIndex** to segment text not by paragraph, but by "Micro-Scenarios" (Context + Action + Reaction).
*   **Stylometry Extraction:** Mathematically profiles the character's "Linguistic Bio" (TF-IDF catchphrases, sentence structure density) to ensure the agent speaks with the correct texture, not just the correct content.

### Phase 2: The Crucible (Evolution)
*The Optimization Engine.*
*   **The Loop:** A recursive simulation environment where the "Newborn" agent attempts to react to scenarios from the book.
*   **The Judge:** A "Unit Test for Personality." It compares the agent's output against the ground truth from the book using Semantic Cosine Similarity and Intent Fidelity.
*   **The Optimizer (DSPy):** If the agent fails (e.g., "Too polite" or "Forgot a memory"), the system automatically backpropagates the error to update the **Ghost Spec** (adjusting trait weights, rewriting system prompts) rather than updating model weights.

### Phase 3: The Ghost Engine (Runtime)
*The Execution Kernel.*
*   **Mixture of Experts (MoE) Router:** A **LangGraph** orchestrator that dynamically routes user inputs to specialized "expert" prompts:
    *   *Narrative Expert:* Handles lore and memory.
    *   *Spatial Expert:* Handles physical navigation and object interaction.
    *   *Psych Expert:* Manages Maslow's Needs and emotional state.
*   **RAST (Retrieval-Augmented Style Transfer):** Injects specific "Style Exemplars" from the original text into the context window to force the LLM to mimic the character's idiolect.

---

## 🧬 The Data Model: `Ghost_Spec_v3`

We define a character not by a simple prompt, but by a portable, evolvable JSON specification.

```json
{
  "name": "Anne Shirley",
  "base_model": "gemini-3-pro",
  "core_personality": {
    "traits": {
      "openness": 0.98,
      "neuroticism": 0.85,
      "agreeableness": 0.45 
    },
    "needs_state": {
      "social_belonging": 0.2, 
      "esteem": 0.4
    }
  },
  "linguistic_bio": {
    "signatures": ["scope for imagination", "depths of despair"],
    "syntax_weights": { "adjective_density": 1.4 }
  },
  "skills": [
    {"name": "slate_combat", "level": 1},
    {"name": "flowery_prose", "level": 5}
  ]
}
```

---

## 🛠 Tech Stack

We stand on the shoulders of giants to build the next generation of Agentic AI.

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | **LangGraph** | Managing the cyclic state of the agent (Perception $\to$ Reflection $\to$ Action). |
| **Optimization** | **DSPy** (MIPROv2) | Automating the "Prompt Engineering" via mathematical optimization. |
| **Memory** | **MemGPT / Chroma** | OS-level memory management (Core Block vs. Archival Storage). |
| **Ingestion** | **LlamaIndex** | Hierarchical parsing of narrative structures. |
| **Evaluation** | **DeepEval / Ragas** | LLM-as-a-Judge frameworks for fidelity scoring. |

---

## 🗺 Roadmap

- [ ] **v0.1 (Prototype):** "The Narrator" pipeline to extract Scene Tuples from *Anne of Green Gables*.
- [ ] **v0.2 (The Loop):** Implement the `Comparator` function to score Agent Output vs. Book Truth.
- [ ] **v0.5 (The Crucible):** Full DSPy integration to auto-tune the `Ghost_Spec.json`.
- [ ] **v1.0 (Release):** A finalized `Anne_Shirley.json` bundle running on the Ghost Engine runtime.

---

## 🤝 Contributing

Ghost.ai is an experimental project exploring the frontiers of **Cognitive Architectures**. We welcome contributions in:
1.  **Inverse RL strategies** for personality extraction.
2.  **Stylometry analysis** algorithms.
3.  **LangGraph** workflow optimizations.

*Project Lead: [sztimhdd]*
