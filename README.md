General Intelligence — Causal World Model Experiments

This repository explores causal understanding and transfer through learned world models, explicit interventions, and causal influence analysis.

The goal is not to build a flashy agent or a demo environment, but to answer a harder question:

Can an AI learn mechanisms that survive when the world changes?

Core Idea

Most AI systems generalize by:

memorizing correlations

adapting quickly

relearning silently

This project instead focuses on causal generality:

Learning what stays true when dynamics, forces, or rules change.

To do this, the system is structured into three strictly separated modules:

Architecture Overview
Observation
   ↓
Encoder
   ↓
Latent State (z)
   ↓
World Model ───────────────┐
   ↓                        │
Predicted Next State        │
                             ↓
Causal Understanding        │
(Interventions + Influence) │
   ↓                        │
Causal Summary              │
                             ↓
LLM (Interpreter)
   ↓
Explanations / Rules / Hypotheses


Each module has a single responsibility.

Modules
1. World Model (world_model.py)

Learns latent dynamics:
z, action → z_next

Models uncertainty (logvar)

Predicts reward and termination

Contains no language, no causality, no explanations

This is the system’s physics engine.

2. Causal Understanding (causal_understanding.py)

Treats latent state as explicit causal slots

Performs do-interventions

Measures counterfactual influence

Prunes irrelevant causes

This is the system’s internal scientist.

3. Language Interpreter (llm.py)

Never predicts physics

Never sees tensors

Only interprets structured summaries

Produces explanations and transferable rules

This is the system’s theorist, not its controller.

4. Agent (agent.py)

Orchestrates the full cognitive loop:

predict → intervene → interpret

Decides what happens next

Stores beliefs (optional)

This is the mind loop, not a policy.


Transfer is evaluated by internal signals:

causal influence changes

uncertainty spikes

selective destabilization

re-stabilization of invariant structure

If nothing breaks internally when the world changes, nothing was understood.

Toy Environment

A minimal physics environment (toy_env.py) is used:

Position

Velocity

Gravity (explicit latent variable)

Gravity is treated as a latent causal factor, allowing:

intervention

regime changes

causal transfer testing

This simplicity is intentional.

Running the Experiment
1. Install requirements
pip install torch httpx


Set your OpenRouter API key


2. Run the test
python run_test.py

Please try this and tell how this current project is.