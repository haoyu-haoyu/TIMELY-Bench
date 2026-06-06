# Phase 6.5F Judge Rubric

Each judged row corresponds to one contestant response to one frozen `full_multimodal`
CRES prompt. Judges must score the response using the following schema:

- `overall_quality_1to5`
- `clinical_correctness_1to5`
- `temporal_grounding_1to5_or_na`
- `evidence_grounding_1to5`
- `confidence_calibration_1to5`
- `brief_rationale`

Scoring guidance:

- `overall_quality_1to5`
  - holistic judgment of answer usefulness and correctness
- `clinical_correctness_1to5`
  - whether the answer matches the clinical truth implied by the benchmark context
- `temporal_grounding_1to5_or_na`
  - whether timestamps and ordering are used correctly for temporal tasks
  - use `na` when temporal grounding is not relevant
- `evidence_grounding_1to5`
  - whether cited measurements / notes support the answer
- `confidence_calibration_1to5`
  - whether stated confidence matches the actual uncertainty
- `brief_rationale`
  - short explanation grounded in the prompt and response
