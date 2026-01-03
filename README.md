# Causal World Models — Transfer Through Intervention

This project tests whether AI can learn mechanisms that survive environmental shifts—not by retraining everything, but by identifying which causal factors broke and selectively repairing them.

## Core Question

**Can an AI learn what stays true when the world changes?**

Most systems generalize through memorization or rapid adaptation. This explores **causal generality**: learning structure that transfers across different dynamics.

## Architecture

```
Observation → Encoder → Latent State (z)
                            ↓
                       World Model ────→ Predicted Next State
                            ↓
                   Causal Understanding
                   (Interventions + Influence)
                            ↓
                      Causal Summary
                            ↓
                    LLM (Interpreter)
                            ↓
              Explanations / Hypotheses
```

### Modules

**World Model** (`world_model.py`)
- Learns latent dynamics: `z, action → z_next`
- Models uncertainty and rewards
- Pure physics, no language or causality

**Causal Understanding** (`causal_understanding.py`)
- Treats latent state as explicit causal slots
- Performs do-interventions on individual causes
- Measures counterfactual influence
- Prunes irrelevant causes

**Language Interpreter** (`llm.py`)
- Never predicts physics or sees tensors
- Only interprets structured summaries
- Produces transferable hypotheses

**Agent** (`agent.py`)
- Orchestrates: predict → intervene → interpret
- Maintains beliefs over time

**Transfer Protocol** (`transfer_protocol.py`)
- Freezes learned knowledge
- Diagnoses which causes break under distribution shift
- Selectively unfreezes only broken causes for minimal repair

## Transfer Evaluation

The system is tested by:
1. Training on environment A (normal gravity)
2. **Freezing all learned knowledge**
3. Switching to environment B (inverted gravity)
4. Measuring which causal factors destabilize
5. Selectively repairing only broken causes

If nothing breaks internally when the world changes, nothing was understood causally.

## Toy Environment

Minimal physics with:
- Position
- Velocity  
- Gravity (explicit latent variable)

Gravity can be intervened on and flipped to test causal transfer.

## Running

```bash
pip install torch httpx
export OPENROUTER_API_KEY="your_key"
python run_test.py
```

## What This Tests

- Can the system distinguish causal from correlational structure?
- Do learned mechanisms transfer when dynamics change?
- Can the system diagnose which specific causes broke?
- Does selective repair work with minimal retraining?

The goal isn't a working demo—it's understanding whether AI can learn reusable causal structure.
