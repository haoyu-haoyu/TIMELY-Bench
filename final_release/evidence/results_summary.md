| step | task | model | cohort | window | auroc_mean_std | auprc_mean_std | test_auroc | test_auprc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| structured | mortality | LogisticRegression | all | 6h | 0.7833 +/- 0.0013 | 0.3578 +/- 0.0120 | 0.7809 | 0.3645 |
| structured | mortality | XGBoost | all | 6h | 0.8052 +/- 0.0036 | 0.3867 +/- 0.0194 | 0.8072 | 0.4072 |
| structured | mortality | LogisticRegression | sepsis | 6h | 0.7427 +/- 0.0076 | 0.4055 +/- 0.0134 | 0.7418 | 0.4056 |
| structured | mortality | XGBoost | sepsis | 6h | 0.7491 +/- 0.0020 | 0.4159 +/- 0.0178 | 0.7541 | 0.4309 |
| structured | mortality | LogisticRegression | aki | 6h | 0.7686 +/- 0.0036 | 0.3810 +/- 0.0152 | 0.7662 | 0.3966 |
| structured | mortality | XGBoost | aki | 6h | 0.7831 +/- 0.0047 | 0.3998 +/- 0.0191 | 0.7833 | 0.4178 |
| structured | prolonged_los | LogisticRegression | all | 6h | 0.7047 +/- 0.0038 | 0.3089 +/- 0.0102 | 0.7020 | 0.3038 |
| structured | prolonged_los | XGBoost | all | 6h | 0.7256 +/- 0.0049 | 0.3378 +/- 0.0102 | 0.7273 | 0.3352 |
| structured | prolonged_los | LogisticRegression | sepsis | 6h | 0.6757 +/- 0.0083 | 0.3966 +/- 0.0096 | 0.6749 | 0.3975 |
| structured | prolonged_los | XGBoost | sepsis | 6h | 0.6971 +/- 0.0047 | 0.4286 +/- 0.0136 | 0.6970 | 0.4371 |
| structured | prolonged_los | LogisticRegression | aki | 6h | 0.6839 +/- 0.0059 | 0.3419 +/- 0.0108 | 0.6818 | 0.3404 |
| structured | prolonged_los | XGBoost | aki | 6h | 0.7034 +/- 0.0052 | 0.3703 +/- 0.0090 | 0.7019 | 0.3683 |
| structured | mortality | LogisticRegression | all | 12h | 0.8177 +/- 0.0028 | 0.4086 +/- 0.0122 | 0.8130 | 0.4257 |
| structured | mortality | XGBoost | all | 12h | 0.8385 +/- 0.0052 | 0.4468 +/- 0.0155 | 0.8348 | 0.4602 |
| structured | mortality | LogisticRegression | sepsis | 12h | 0.7749 +/- 0.0055 | 0.4515 +/- 0.0110 | 0.7754 | 0.4676 |
| structured | mortality | XGBoost | sepsis | 12h | 0.7871 +/- 0.0039 | 0.4736 +/- 0.0180 | 0.7841 | 0.4807 |
| structured | mortality | LogisticRegression | aki | 12h | 0.8002 +/- 0.0028 | 0.4273 +/- 0.0144 | 0.7983 | 0.4488 |
| structured | mortality | XGBoost | aki | 12h | 0.8188 +/- 0.0051 | 0.4628 +/- 0.0172 | 0.8156 | 0.4725 |
| structured | prolonged_los | LogisticRegression | all | 12h | 0.7462 +/- 0.0056 | 0.3525 +/- 0.0144 | 0.7524 | 0.3569 |
| structured | prolonged_los | XGBoost | all | 12h | 0.7620 +/- 0.0067 | 0.3796 +/- 0.0121 | 0.7702 | 0.3900 |
| structured | prolonged_los | LogisticRegression | sepsis | 12h | 0.7141 +/- 0.0105 | 0.4409 +/- 0.0155 | 0.7184 | 0.4479 |
| structured | prolonged_los | XGBoost | sepsis | 12h | 0.7336 +/- 0.0106 | 0.4692 +/- 0.0114 | 0.7431 | 0.4920 |
| structured | prolonged_los | LogisticRegression | aki | 12h | 0.7236 +/- 0.0073 | 0.3847 +/- 0.0148 | 0.7279 | 0.3951 |
| structured | prolonged_los | XGBoost | aki | 12h | 0.7408 +/- 0.0074 | 0.4125 +/- 0.0116 | 0.7477 | 0.4246 |
| structured | mortality | LogisticRegression | all | 24h | 0.8517 +/- 0.0024 | 0.4870 +/- 0.0082 | 0.8481 | 0.5076 |
| structured | mortality | XGBoost | all | 24h | 0.8679 +/- 0.0049 | 0.5236 +/- 0.0079 | 0.8677 | 0.5414 |
| structured | mortality | LogisticRegression | sepsis | 24h | 0.8095 +/- 0.0019 | 0.5174 +/- 0.0091 | 0.8037 | 0.5309 |
| structured | mortality | XGBoost | sepsis | 24h | 0.8207 +/- 0.0060 | 0.5451 +/- 0.0173 | 0.8218 | 0.5547 |
| structured | mortality | LogisticRegression | aki | 24h | 0.8343 +/- 0.0019 | 0.5025 +/- 0.0125 | 0.8337 | 0.5268 |
| structured | mortality | XGBoost | aki | 24h | 0.8489 +/- 0.0036 | 0.5358 +/- 0.0078 | 0.8491 | 0.5520 |
| structured | prolonged_los | LogisticRegression | all | 24h | 0.7888 +/- 0.0053 | 0.4094 +/- 0.0130 | 0.7966 | 0.4219 |
| structured | prolonged_los | XGBoost | all | 24h | 0.8059 +/- 0.0063 | 0.4468 +/- 0.0104 | 0.8145 | 0.4604 |
| structured | prolonged_los | LogisticRegression | sepsis | 24h | 0.7568 +/- 0.0097 | 0.4920 +/- 0.0142 | 0.7638 | 0.5104 |
| structured | prolonged_los | XGBoost | sepsis | 24h | 0.7794 +/- 0.0098 | 0.5283 +/- 0.0166 | 0.7814 | 0.5421 |
| structured | prolonged_los | LogisticRegression | aki | 24h | 0.7653 +/- 0.0066 | 0.4392 +/- 0.0140 | 0.7708 | 0.4534 |
| structured | prolonged_los | XGBoost | aki | 24h | 0.7833 +/- 0.0069 | 0.4754 +/- 0.0132 | 0.7888 | 0.4815 |
| structured | mortality | LogisticRegression | all | D0 | 0.7969 +/- 0.0027 | 0.3818 +/- 0.0122 | 0.7898 | 0.3870 |
| structured | mortality | XGBoost | all | D0 | 0.8111 +/- 0.0045 | 0.4006 +/- 0.0171 | 0.8100 | 0.4175 |
| structured | mortality | LogisticRegression | sepsis | D0 | 0.7551 +/- 0.0054 | 0.4288 +/- 0.0220 | 0.7473 | 0.4292 |
| structured | mortality | XGBoost | sepsis | D0 | 0.7624 +/- 0.0068 | 0.4366 +/- 0.0218 | 0.7571 | 0.4470 |
| structured | mortality | LogisticRegression | aki | D0 | 0.7802 +/- 0.0012 | 0.4040 +/- 0.0176 | 0.7735 | 0.4122 |
| structured | mortality | XGBoost | aki | D0 | 0.7916 +/- 0.0030 | 0.4177 +/- 0.0210 | 0.7902 | 0.4385 |
| structured | prolonged_los | LogisticRegression | all | D0 | 0.7311 +/- 0.0041 | 0.3333 +/- 0.0127 | 0.7341 | 0.3331 |
| structured | prolonged_los | XGBoost | all | D0 | 0.7432 +/- 0.0018 | 0.3563 +/- 0.0034 | 0.7531 | 0.3574 |
| structured | prolonged_los | LogisticRegression | sepsis | D0 | 0.6983 +/- 0.0085 | 0.4193 +/- 0.0171 | 0.6909 | 0.4111 |
| structured | prolonged_los | XGBoost | sepsis | D0 | 0.7134 +/- 0.0069 | 0.4440 +/- 0.0109 | 0.7167 | 0.4468 |
| structured | prolonged_los | LogisticRegression | aki | D0 | 0.7081 +/- 0.0060 | 0.3661 +/- 0.0124 | 0.7102 | 0.3668 |
| structured | prolonged_los | XGBoost | aki | D0 | 0.7216 +/- 0.0036 | 0.3913 +/- 0.0035 | 0.7289 | 0.3926 |
| text | mortality | XGBoost (ClinicalBERT) | all | 24h | 0.8123 +/- 0.0056 | 0.4140 +/- 0.0101 | 0.8168 | 0.4437 |
| text | mortality | LogisticRegression (ClinicalBERT) | all | 24h | 0.8287 +/- 0.0050 | 0.4291 +/- 0.0129 | 0.8318 | 0.4439 |
| text | prolonged_los | XGBoost (ClinicalBERT) | all | 24h | 0.7894 +/- 0.0087 | 0.4365 +/- 0.0124 | 0.7997 | 0.4559 |
| text | prolonged_los | LogisticRegression (ClinicalBERT) | all | 24h | 0.7901 +/- 0.0091 | 0.4376 +/- 0.0104 | 0.8000 | 0.4521 |
| text | mortality | XGBoost (MedCAT) | all | 24h | 0.5571 +/- 0.0032 | 0.1497 +/- 0.0041 | 0.5520 | 0.1506 |
| text | mortality | LogisticRegression (MedCAT) | all | 24h | 0.5575 +/- 0.0031 | 0.1465 +/- 0.0049 | 0.5519 | 0.1501 |
| text | prolonged_los | XGBoost (MedCAT) | all | 24h | 0.5554 +/- 0.0027 | 0.1975 +/- 0.0040 | 0.5495 | 0.1946 |
| text | prolonged_los | LogisticRegression (MedCAT) | all | 24h | 0.5552 +/- 0.0030 | 0.1967 +/- 0.0038 | 0.5491 | 0.1922 |
| text | mortality | XGBoost | all | 24h | 0.7433 +/- 0.0048 | 0.3069 +/- 0.0157 | 0.7551 | 0.3266 |
| text | mortality | LogisticRegression | all | 24h | 0.7371 +/- 0.0050 | 0.3068 +/- 0.0146 | 0.7481 | 0.3287 |
| text | prolonged_los | XGBoost | all | 24h | 0.6875 +/- 0.0080 | 0.2967 +/- 0.0060 | 0.7007 | 0.3107 |
| text | prolonged_los | LogisticRegression | all | 24h | 0.6774 +/- 0.0072 | 0.2923 +/- 0.0054 | 0.6849 | 0.3031 |
| fusion | mortality | Early Fusion (AnnotFeatures) | all | 24h | 0.8693 +/- 0.0052 | 0.5288 +/- 0.0095 | 0.8725 | 0.5568 |
| fusion | mortality | Early Fusion (ClinicalBERT) | all | 24h | 0.8816 +/- 0.0058 | 0.5562 +/- 0.0089 | 0.8848 | 0.5844 |
| fusion | prolonged_los | Early Fusion (AnnotFeatures) | all | 24h | 0.8097 +/- 0.0055 | 0.4542 +/- 0.0115 | 0.8182 | 0.4677 |
| fusion | prolonged_los | Early Fusion (ClinicalBERT) | all | 24h | 0.8267 +/- 0.0060 | 0.4918 +/- 0.0099 | 0.8353 | 0.5089 |
| fusion | mortality | Late Fusion (Alpha=1.0 Structured XGB) | all | 24h | 0.8680 +/- 0.0052 | 0.5243 +/- 0.0080 | 0.8686 | 0.5374 |
| fusion | mortality | Late Fusion (Tuned Alpha XGB Preds) | all | 24h | 0.8682 +/- 0.0051 | 0.5224 +/- 0.0083 | 0.8688 | 0.5354 |
| fusion | mortality | Late Fusion (Stacking LR on OOF preds) | all | 24h | 0.8681 +/- 0.0050 | 0.5218 +/- 0.0088 | 0.8689 | 0.5348 |
| fusion | mortality | Late Fusion (Alpha=1.0 Structured XGB) [clinicalbert] | all | 24h | 0.8677 +/- 0.0045 | 0.5232 +/- 0.0087 | 0.8675 | 0.5389 |
| fusion | mortality | Late Fusion (Tuned Alpha XGB Preds) [clinicalbert] | all | 24h | 0.8783 +/- 0.0038 | 0.5313 +/- 0.0098 | 0.8805 | 0.5508 |
| fusion | mortality | Late Fusion (Stacking LR on OOF preds) [clinicalbert] | all | 24h | 0.8783 +/- 0.0038 | 0.5330 +/- 0.0121 | 0.8803 | 0.5524 |
| fusion | prolonged_los | Late Fusion (Alpha=1.0 Structured XGB) | all | 24h | 0.8049 +/- 0.0070 | 0.4442 +/- 0.0115 | 0.8134 | 0.4585 |
| fusion | prolonged_los | Late Fusion (Tuned Alpha XGB Preds) | all | 24h | 0.8062 +/- 0.0071 | 0.4443 +/- 0.0096 | 0.8146 | 0.4579 |
| fusion | prolonged_los | Late Fusion (Stacking LR on OOF preds) | all | 24h | 0.8060 +/- 0.0071 | 0.4439 +/- 0.0095 | 0.8146 | 0.4582 |
| fusion | prolonged_los | Late Fusion (Alpha=1.0 Structured XGB) [clinicalbert] | all | 24h | 0.8055 +/- 0.0064 | 0.4466 +/- 0.0136 | 0.8144 | 0.4602 |
| fusion | prolonged_los | Late Fusion (Tuned Alpha XGB Preds) [clinicalbert] | all | 24h | 0.8249 +/- 0.0074 | 0.4898 +/- 0.0155 | 0.8338 | 0.5062 |
| fusion | prolonged_los | Late Fusion (Stacking LR on OOF preds) [clinicalbert] | all | 24h | 0.8246 +/- 0.0074 | 0.4899 +/- 0.0156 | 0.8336 | 0.5063 |
| gru | mortality | temporal_gru_v2 | all | 24h | 0.8526 +/- 0.0044 | 0.4933 +/- 0.0096 | 0.8419 | 0.4832 |
| readmission | readmission_30d | LogisticRegression | all | 24h | 0.5604 +/- 0.0064 | 0.2495 +/- 0.0047 | 0.5600 | 0.2465 |
| readmission | readmission_30d | GradientBoosting | all | 24h | 0.5661 +/- 0.0096 | 0.2533 +/- 0.0070 | 0.5639 | 0.2525 |
| readmission | readmission_30d | LogisticRegression | all | 24h | 0.5682 +/- 0.0060 | 0.2535 +/- 0.0058 | 0.5680 | 0.2503 |
| readmission | readmission_30d | GradientBoosting | all | 24h | 0.5695 +/- 0.0069 | 0.2551 +/- 0.0057 | 0.5653 | 0.2526 |
| readmission | readmission_30d | LogisticRegression | all | 24h | 0.5841 +/- 0.0076 | 0.2646 +/- 0.0082 | 0.5775 | 0.2573 |
| readmission | readmission_30d | GradientBoosting | all | 24h | 0.5762 +/- 0.0077 | 0.2576 +/- 0.0061 | 0.5764 | 0.2568 |
| readmission | readmission_30d | LogisticRegression | all | 24h | 0.5734 +/- 0.0050 | 0.2602 +/- 0.0062 | 0.5737 | 0.2525 |
| readmission | readmission_30d | GradientBoosting | all | 24h | 0.5788 +/- 0.0065 | 0.2606 +/- 0.0047 | 0.5754 | 0.2534 |
| readmission | readmission_30d | LogisticRegression | all | 24h | 0.5867 +/- 0.0074 | 0.2677 +/- 0.0082 | 0.5828 | 0.2578 |
| readmission | readmission_30d | GradientBoosting | all | 24h | 0.5789 +/- 0.0073 | 0.2612 +/- 0.0069 | 0.5764 | 0.2557 |
