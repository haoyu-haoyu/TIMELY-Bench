# Canonical Trajectories Documentation

## Overview

This document defines canonical (expected) temporal trajectories for clinical conditions based on established clinical guidelines. These trajectories describe the **complete clinical course** of conditions, while TIMELY-Bench episodes capture only the **first 24 hours** of ICU stay.

## Relationship to TIMELY-Bench Data

| Aspect | Canonical Trajectory | TIMELY-Bench Episode |
|--------|---------------------|---------------------|
| Time Range | 0-336 hours (condition-dependent) | 0-24 hours (fixed) |
| Purpose | Define expected clinical evolution | Capture early indicators |
| Use Case | Trajectory prediction target | Model input features |

## Clinical Interpretation

The 24-hour observation window in TIMELY-Bench is designed for **early prediction**:
- **Early phase detection**: Identify which trajectory a patient is likely to follow
- **Risk stratification**: Use early markers to predict outcomes
- **Clinical decision support**: Inform interventions before trajectory is fully established

## Trajectory Definitions

### Sepsis (Sepsis-3 / Surviving Sepsis Campaign 2021)
- **Recovery trajectory**: 0-6h (resuscitation) -> 6-24h (stabilization) -> 24-72h (resolution)
- **Worsening trajectory**: 0-12h (sepsis) -> 12-24h (refractory shock) -> 24-72h (MOF)

### AKI (KDIGO 2012)
- **Recovery trajectory**: 0-24h (injury) -> 24-72h (plateau) -> 72-168h (recovery)
- **Progression trajectory**: 0-24h (Stage 1) -> 24-48h (Stage 2) -> 48-96h (Stage 3/RRT)

### ARDS (Berlin Definition)
- **Recovery trajectory**: 0-72h (exudative) -> 72-168h (proliferative) -> 168-336h (resolution)

### Heart Failure (ESC 2021)
- **Compensation trajectory**: 0-24h (decompensation) -> 24-72h (diuresis) -> 72-168h (optimization)

## Usage in Models

Models trained on TIMELY-Bench can use these trajectories to:
1. **Multi-task learning**: Predict both immediate outcome and trajectory type
2. **Interpretability**: Explain predictions in terms of expected clinical course
3. **Evaluation**: Assess if model predictions align with known trajectories
