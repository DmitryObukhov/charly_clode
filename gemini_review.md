# Gemini's Review of Project Charly Prompts

This document contains a review of the prompt files for Project Charly. As per my role of "criticism and verification," I have analyzed the project's prompts and propose the following improvements.

## 1. Project-Level Suggestions

### Prompt-Driven Development Process
The prompt-driven development approach is innovative. To make it more robust, I suggest the following:

*   **Add a validation step:** After the AI generates the code, a script could be run to check for basic correctness, such as syntax, and maybe even run a suite of basic tests. This would provide a faster feedback loop. The `build.bat` could be extended to include this.
*   **Version control for prompts:** The prompts are the source of truth. They should be under version control with the same rigor as source code. It seems you are already doing this, which is great.

### File Path Inconsistencies
Some markdown files refer to file paths under a `cl_01/` directory, which is inconsistent with the `src/` directory used in `CLAUDE.md` and the actual project structure.

*   **Recommendation:** Unify all file paths in the documentation to point to `src/` for consistency. This will prevent confusion for both human and AI developers.

## 2. `physics.md` - Physical Model

### Agent Movement
The agent's movement is defined as `(left_activation - right_activation)`.

*   **Critique:** This model implies that if both flagella are active, their forces cancel out. While this is a valid model, it might not be the most energy-efficient or biologically plausible. An alternative could be that the agent has a maximum speed, and the actuators control the direction.
*   **Recommendation:** Add a sentence to the `physics.md` prompt to clarify the rationale behind this choice of movement dynamics. For example: "This model simulates a simple push-pull mechanism where opposing forces cancel each other out."

## 3. `neuron.md` - Neuron Structure and Processing

### Elastic Parameter Degradation
The `elastic_trigger` and `elastic_recharge` parameters currently degrade by a constant value (`ELASTIC_TRIGGER_DEGRADATION`, `ELASTIC_RECHARGE_DEGRADATION`).

*   **Critique:** A constant degradation might lead to a less stable system, where the values can quickly drop to zero. A proportional degradation (e.g., multiplying by a factor like 0.99) would provide a smoother decay and might be more stable in the long run.
*   **Recommendation:** Consider changing the degradation mechanism to be proportional rather than subtractive. The prompt should be updated to reflect this: `next_neuron.elastic_trigger = current_neuron.elastic_trigger * ELASTIC_TRIGGER_DECAY_FACTOR`.

## 4. `substrate.md` - Neural Substrate

### Learning Rule
The learning rule in `run_night()` is a simple Hebbian rule based on the EQ of the "star" neurons.

*   **Critique:** This rule reinforces any synapse that was active when a positive EQ star neuron fired. This might lead to reinforcing spurious correlations. For example, if a neuron fires for a reason unrelated to the agent's goal, but a positive EQ star neuron also happens to fire, that synapse will be strengthened. More advanced learning rules could provide more robust learning.
*   **Recommendation:** Propose exploring more advanced learning rules. A good starting point would be to incorporate the pre-synaptic and post-synaptic activity in the weight update. For example, a basic STDP (Spike-Timing-Dependent Plasticity) rule could be implemented. The prompt could be updated to say: "Strengthen the synapse if the pre-synaptic neuron fired shortly before the post-synaptic star neuron." This would be a more direct causal link.

## 5. `sequence.md` - SequenceGenerator

### Component Usage
The `SequenceGenerator` DSL is well-documented but does not appear to be used in the current `config/config.yaml`.

*   **Critique:** This component might be an example of over-engineering if it's not being used. Maintaining a complex component that is not part of the core functionality adds to the cognitive load of the project.
*   **Recommendation:**
    1.  If the `SequenceGenerator` is planned for future use, add a note in `sequence.md` and `CLAUDE.md` to clarify its status.
    2.  If it is deprecated, consider removing the `sequence.md` file and the corresponding source code to simplify the project.

## 6. `application.md` - CLI Application

### Logging
The CLI has a `--log-level` argument, but the progress output is printed to `stdout`.

*   **Recommendation:** For better separation of concerns, consider using a proper logging library (like Python's `logging` module) to handle all output. This would allow routing progress information to `stderr` and results to `stdout`, which is a common practice for command-line tools. The prompt should be updated to specify this.

## 7. `config/config.yaml`

### Recharge Rate
The `RECHARGE_RATE` is set to `30`.

*   **Critique:** With a `CHARGE_MIN` of `100`, a neuron would need about 4 iterations to recharge. This might be too fast, making neurons fire too frequently.
*   **Recommendation:** Consider reducing the `RECHARGE_RATE` to a lower value (e.g., 10) to allow for more complex temporal dynamics to emerge. This is a parameter to experiment with, but the current value seems high.

---
End of review. These suggestions are intended to improve the clarity, consistency, and robustness of Project Charly.