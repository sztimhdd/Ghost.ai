# Product Requirements Document (PRD)

| Project Name | Ghost.ai |
| :--- | :--- |
| **Version** | 1.0 (Final) |
| **Status** | Approved |
| **Date** | February 1, 2026 |
| **Document Owner** | Product Management |

---

## 1. Executive Summary
**Ghost.ai** is a **Behavioral Compilation Pipeline** designed to automate the creation of high-fidelity, autonomous AI characters. By ingesting static narrative text (novels, scripts), the system "compiles" a character into a portable, executable specification.

Unlike traditional Chatbots that rely on generic prompting, or Fine-tuned Models that are rigid and expensive, Ghost.ai employs a **"Frozen Model, Fluid State"** approach. It treats character personality as an optimization problem, iteratively tuning the agent's memory, traits, and behavioral logic until it statistically aligns with the source material.

**Core Value Proposition:**
*   **Fidelity:** Agents that adhere to the canonical behavior, voice, and psychology of their source material.
*   **Autonomy:** Agents capable of navigating complex environments and maintaining long-term internal states (mood, needs) without manual scripting.
*   **Portability:** The output is a lightweight configuration file ("The Ghost Spec") compatible with modern SOTA LLMs.

---

## 2. Target Audience & Personas
1.  **Game Developers (Primary):** Need to populate open worlds with hundreds of unique, deeply interactive NPCs without writing thousands of lines of dialogue trees.
2.  **AI Researchers:** Need a standardized framework to experiment with Cognitive Architectures and behavioral alignment.
3.  **Creative Writers:** Need to "test" their characters in simulated scenarios to check for consistency and voice.

---

## 3. User Stories

| ID | As a... | I want to... | So that... |
| :--- | :--- | :--- | :--- |
| **US-1** | **Developer** | Upload a raw text file (e.g., novel, script) | The system can automatically extract the character's history and personality. |
| **US-2** | **Developer** | Define a "Target Character" (e.g., "Anne") | The system knows which entity to profile and ignore others. |
| **US-3** | **Developer** | View a "Fidelity Score" during training | I know how accurately the agent is currently mimicking the source material. |
| **US-4** | **End User** | Chat with the agent about past events in the book | The agent recalls specific details accurately. |
| **US-5** | **End User** | Observe the agent taking initiative (e.g., leaving a conversation) | The agent feels like a living entity with its own needs, not just a reactive chatbot. |
| **US-6** | **Writer** | See a breakdown of the agent's psychological traits | I can verify if the AI captured the nuance (e.g., High Neuroticism) correctly. |

---

## 4. Functional Requirements

The system is divided into three functional phases.

### 4.1 Phase 1: Ingestion ("The Narrator")
The system must parse unstructured text to create a Ground Truth dataset.

*   **FR-1.1 Narrative Segmentation:** The system shall automatically split raw text into discrete "Micro-Scenarios" based on narrative context (e.g., scene changes), ensuring no context bleed.
*   **FR-1.2 Behavioral Extraction:** For each scenario, the system must identify:
    *   **Stimulus:** The external event or dialogue.
    *   **Response:** The target character's action and dialogue.
    *   **Internal State:** The inferred emotion or thought process of the character.
*   **FR-1.3 Stylometric Profiling:** The system must analyze the character's speech patterns to identify signature phrases, sentence complexity, and vocabulary preferences (Idiolect).

### 4.2 Phase 2: Optimization ("The Crucible")
The system must iteratively train the agent's configuration.

*   **FR-2.1 Simulation Loop:** The system shall run the agent through the extracted Micro-Scenarios without providing the "correct" answer, generating a predicted response.
*   **FR-2.2 Automated Evaluation:** The system must compare the Predicted Response against the Ground Truth using semantic and emotional metrics.
*   **FR-2.3 Auto-Tuning:** The system must automatically adjust the agent's configurable parameters (Traits, System Instructions, Memory Weights) to minimize the deviation from the Ground Truth.
*   **FR-2.4 Convergence Criteria:** The training process shall stop when the Fidelity Score reaches a predefined threshold (e.g., 90%) or progress plateaus.

### 4.3 Phase 3: Runtime ("The Ghost Engine")
The system must serve the agent in real-time.

*   **FR-3.1 Context Routing:** The system shall analyze user input to route the request to the most relevant processing module (e.g., Narrative retrieval vs. Spatial reasoning vs. Emotional reaction).
*   **FR-3.2 Dynamic Needs System:** The agent must possess an internal state of "Needs" (e.g., Social, Safety, Esteem) that decays over time and influences behavioral output.
*   **FR-3.3 Style Enforcement:** The agent's output must dynamically incorporate the "Stylometric Profile" extracted in Phase 1 to ensure the voice matches the source material.
*   **FR-3.4 Skill Execution:** The agent must be able to recognize when a specific capability is required (e.g., "Recite Poetry", "Move to Location") and execute that logic.

---

## 5. Information Architecture: The "Ghost Spec"

The output of the platform is the **Ghost Spec**, a structured profile that defines the agent. The PRD requires this artifact to contain:

1.  **Core Personality:** Psychological attributes (e.g., Big Five traits) and moral values.
2.  **Needs Matrix:** Baseline drivers for the character's autonomy (e.g., baseline loneliness tolerance).
3.  **Linguistic Bio:** Data definition of the character's voice, including catchphrases and syntax preferences.
4.  **Memory Graph:** A reference to the episodic knowledge base extracted from the text.
5.  **Skill Registry:** A list of specific capabilities the agent has "unlocked" based on the narrative.

---

## 6. Non-Functional Requirements (NFRs)

*   **NFR-1 Scalability:** The Ingestion pipeline must handle literary works of up to 500,000 words without degradation in extraction quality.
*   **NFR-2 Latency:** The Runtime Engine must generate a response within 3 seconds (P90) to maintain conversational immersion.
*   **NFR-3 Modularity:** The "Ghost Spec" must be decoupled from the underlying LLM; switching from Gemini to GPT-4 should require configuration changes, not re-architecture.
*   **NFR-4 Hallucination Control:** The agent must prioritize "I don't know" or plausible deflections over fabricating facts that contradict the source material.

---

## 7. Success Metrics (KPIs)

*   **Behavioral Fidelity Score (BFS):** The semantic similarity between the Agent's simulated actions and the Book's actual actions across a hold-out test set. **Target: >0.85**.
*   **Idiolect Alignment:** The statistical closeness of the Agent's vocabulary distribution to the original character's.
*   **Training Efficiency:** Time required to "compile" a standard novel into a Ghost Spec. **Target: <2 hours**.
*   **User Consistency Rating:** In subjective testing, users should not detect "personality breaks" (contradicting core beliefs) in a 50-turn conversation.

---

## 8. Assumptions & Dependencies
*   **Source Quality:** We assume the input text is of high literary quality with consistent characterization. The system cannot fix poorly written characters.
*   **Model Capability:** We rely on the reasoning capabilities of SOTA models (Gemini 3 Pro, GPT-4 class) for the internal logic. Smaller models (e.g., 7B parameter) are assumed insufficient for the "Crucible" phase.
*   **Semantic Measurement:** We assume that current Embedding Models are sufficient to measure the "sameness" of intent between two different phrasings.

---

## 9. References & Inspiration

*   **Foundational Architecture:**
    *   *Generative Agents: Interactive Simulacra of Human Behavior* (Park et al., 2023) - [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
    *   *Cognitive Architectures for Language Agents (CoALA)* (Sumers et al., 2023) - [arXiv:2309.02427](https://arxiv.org/abs/2309.02427)
*   **Core Libraries:**
    *   *DSPy (Stanford NLP):* For programmatic optimization of the prompt/weight pipelines. - [GitHub](https://github.com/stanfordnlp/dspy)
    *   *LangGraph (LangChain):* For stateful, cyclic multi-agent orchestration. - [Documentation](https://python.langchain.com/docs/langgraph)
    *   *MemGPT:* For OS-level memory hierarchy management. - [Website](https://memgpt.ai/)
    *   *LlamaIndex:* For hierarchical data ingestion and semantic chunking. - [Website](https://www.llamaindex.ai/)
*   **Related Projects:**
    *   *Voyager:* An Open-Ended Embodied Agent with Large Language Models. - [Website](https://voyager.minedojo.org/)
    *   *SillyTavern:* Open-source frontend for character interactions (Inspiration for Character Cards). - [GitHub](https://github.com/SillyTavern/SillyTavern)
