# State-Space Reconstruction

This document defines the explicit state-space artifact released with TIMELY-Bench.

## Purpose

Condition graphs and physiology templates define expected clinical structure.  
State-space reconstruction adds an executable patient-level trajectory:
- each episode is mapped to ordered latent states over the 0-24h window,
- each transition is linked to a triggering detected pattern.

## Files

- `state_space_schema.json`: state and transition-rule specification.
- `episode_state_trajectory.jsonl`: transition-level records (`stay_id`, `hour`, `state`, `from_state`, trigger fields).
- `state_transition_summary.json`: aggregate counts and quality summary.

## State Definitions

- `baseline`: no salient pattern.
- `at_risk`: mild abnormalities.
- `active`: moderate syndrome activity.
- `severe`: severe/critical syndrome activity.
- `recovering`: de-escalation after active/severe phase.

## Reconstruction Rules

1. Initialize each episode at `baseline` (hour 0).
2. Map each detected pattern severity to state:
   - `mild -> at_risk`
   - `moderate -> active`
   - `severe/critical -> severe`
3. If state de-escalates after severe/active, emit `recovering`.
4. If final observed state is `active` or `severe` before hour 24, append an end-of-window `recovering` transition.

## Generator

Run:

`python3 code/state_space/reconstruct_state_space.py`

Optional:

`python3 code/state_space/reconstruct_state_space.py --max-episodes 2000`
