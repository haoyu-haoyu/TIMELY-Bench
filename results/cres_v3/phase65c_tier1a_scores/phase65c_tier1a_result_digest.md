# Phase 6.5C Tier 1A Result Digest

## Run Completeness
- `gpt54`: 53070/53070 parse_success, avg_latency=11.40s, total_tokens=264064840
- `gemini31pro`: 53070/53070 parse_success, avg_latency=15.94s, total_tokens=343643633

## Auto-Scoring Coverage
- scored_prompt_rows: `41325`
- supported_task_dimensions: `20` / `47`
- deferred_task_dimensions: `27`

## Key Provider-Level Metrics
- gemini31pro AKI-S1 D1 (event_time) n=935, binary_accuracy=0.9561, event_presence_auprc=1.0000, positive_tolerance_1h_rate=0.3701, median_abs_hour_error=4.0000
- gemini31pro AKI-S1 D2 (event_time) n=938, binary_accuracy=0.9019
- gemini31pro AKI-T1 D1 (event_time) n=1830, binary_accuracy=0.6656
- gemini31pro AKI-T1 D2 (event_time) n=1868, binary_accuracy=0.9920
- gemini31pro AKI-T1 D5 (categorical) n=1115, accuracy=0.6520, macro_f1=0.4136
- gemini31pro DEL-S1 D1 (event_time) n=929, binary_accuracy=1.0000, event_presence_auprc=1.0000, positive_tolerance_1h_rate=0.9656, median_abs_hour_error=0.0000
- gemini31pro DEL-T1 D1 (event_time) n=1859, binary_accuracy=1.0000, event_presence_auprc=1.0000, positive_tolerance_1h_rate=0.9758, median_abs_hour_error=0.0000
- gemini31pro DEL-T1 D2 (event_time) n=1875, binary_accuracy=1.0000, event_presence_auprc=1.0000, positive_tolerance_1h_rate=0.9984, median_abs_hour_error=0.0000
- gemini31pro S-R1 D4 (categorical) n=312, accuracy=0.3429, macro_f1=0.2343
- gemini31pro S-R2 D4 (categorical) n=158, accuracy=0.2975, macro_f1=0.2860
- gemini31pro S-R3 D2 (numeric_exact) n=4, exact_match=0.7500, tolerance_1=0.7500
- gemini31pro S-R4 D4 (binary_from_yesno) n=362, binary_accuracy=0.4834, event_presence_auroc=0.4625, event_presence_auprc=0.1073
- gemini31pro S-T1 D1 (event_time) n=931, binary_accuracy=0.0279
- gemini31pro S-T1 D3 (binary_from_trend) n=912, binary_accuracy=0.7610, event_presence_auroc=0.5590, event_presence_auprc=0.1785
- gemini31pro S-T2 D4 (categorical) n=272, accuracy=0.1949, macro_f1=0.1960
- gemini31pro S-T3 D4 (categorical) n=299, accuracy=0.3545, macro_f1=0.2301
- gemini31pro SEP-S1 D1 (event_time) n=448, binary_accuracy=1.0000, event_presence_auprc=1.0000, positive_tolerance_1h_rate=0.2946, median_abs_hour_error=3.0000
- gemini31pro SEP-T1 D1 (event_time) n=1803, binary_accuracy=0.9956, event_presence_auprc=1.0000, positive_tolerance_1h_rate=0.3300, median_abs_hour_error=3.0000
- gemini31pro SEP-T1 D2 (event_time) n=1875, binary_accuracy=0.8299
- gemini31pro SEP-T1 D5 (categorical) n=1875, accuracy=0.4208, macro_f1=0.3807
- gpt54 AKI-S1 D1 (event_time) n=884, binary_accuracy=0.8348, event_presence_auprc=1.0000, positive_tolerance_1h_rate=0.2670, median_abs_hour_error=5.0000
- gpt54 AKI-S1 D2 (event_time) n=938, binary_accuracy=0.9126
- gpt54 AKI-T1 D1 (event_time) n=1874, binary_accuracy=0.8298
- gpt54 AKI-T1 D2 (event_time) n=1691, binary_accuracy=0.9692
- gpt54 AKI-T1 D5 (categorical) n=1116, accuracy=0.6317, macro_f1=0.3984
- gpt54 DEL-S1 D1 (event_time) n=898, binary_accuracy=1.0000, event_presence_auprc=1.0000, positive_tolerance_1h_rate=0.9989, median_abs_hour_error=0.0000
- gpt54 DEL-T1 D1 (event_time) n=1821, binary_accuracy=1.0000, event_presence_auprc=1.0000, positive_tolerance_1h_rate=0.9956, median_abs_hour_error=0.0000
- gpt54 DEL-T1 D2 (event_time) n=1875, binary_accuracy=1.0000, event_presence_auprc=1.0000, positive_tolerance_1h_rate=1.0000, median_abs_hour_error=0.0000
- gpt54 S-R1 D4 (categorical) n=397, accuracy=0.2594, macro_f1=0.1735
- gpt54 S-R2 D4 (categorical) n=167, accuracy=0.2455, macro_f1=0.2251
- gpt54 S-R3 D2 (numeric_exact) n=4, exact_match=0.7500, tolerance_1=0.7500
- gpt54 S-R4 D4 (binary_from_yesno) n=468, binary_accuracy=0.2115, event_presence_auroc=0.3345, event_presence_auprc=0.0734
- gpt54 S-T1 D1 (event_time) n=926, binary_accuracy=0.1080
- gpt54 S-T1 D3 (binary_from_trend) n=935, binary_accuracy=0.8064, event_presence_auroc=0.5274, event_presence_auprc=0.1667
- gpt54 S-T2 D4 (categorical) n=316, accuracy=0.2057, macro_f1=0.1484
- gpt54 S-T3 D4 (categorical) n=352, accuracy=0.2131, macro_f1=0.1464
- gpt54 SEP-S1 D1 (event_time) n=463, binary_accuracy=1.0000, event_presence_auprc=1.0000, positive_tolerance_1h_rate=0.3650, median_abs_hour_error=2.0000
- gpt54 SEP-T1 D1 (event_time) n=1854, binary_accuracy=0.9887, event_presence_auprc=1.0000, positive_tolerance_1h_rate=0.3646, median_abs_hour_error=2.0000
- gpt54 SEP-T1 D2 (event_time) n=1868, binary_accuracy=0.7018
- gpt54 SEP-T1 D5 (categorical) n=1875, accuracy=0.3200, macro_f1=0.3103

## Audit Notes
- Gemini manual recovery: tail4=4, parsefix=1
- Representation-branch comparisons are not computed in Tier 1A because this run uses only the full_multimodal prompt variant.
- D6 evidence attribution remains deferred to later judge/evidence analyses.
- Some task-dimension combinations are scored with conservative coarse mappings where only partial automatic ground truth is available (e.g. S-T1 D3 worsening vs non-worsening).