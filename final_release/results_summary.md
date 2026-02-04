| step       | task          | model                                  | cohort   | window   | auroc_mean_std    | auprc_mean_std    |   test_auroc |   test_auprc |
|:-----------|:--------------|:---------------------------------------|:---------|:---------|:------------------|:------------------|-------------:|-------------:|
| structured | mortality     | LogisticRegression                     | all      | 6h       | 0.7798 +/- 0.0037 | 0.3496 +/- 0.0151 |       0.7794 |       0.3664 |
| structured | mortality     | XGBoost                                | all      | 6h       | 0.8012 +/- 0.0036 | 0.3771 +/- 0.0208 |       0.8046 |       0.4013 |
| structured | mortality     | LogisticRegression                     | sepsis   | 6h       | 0.7382 +/- 0.0078 | 0.3961 +/- 0.0160 |       0.7535 |       0.4243 |
| structured | mortality     | XGBoost                                | sepsis   | 6h       | 0.7480 +/- 0.0077 | 0.4088 +/- 0.0064 |       0.763  |       0.4343 |
| structured | mortality     | LogisticRegression                     | aki      | 6h       | 0.7670 +/- 0.0042 | 0.3807 +/- 0.0066 |       0.7583 |       0.3638 |
| structured | mortality     | XGBoost                                | aki      | 6h       | 0.7850 +/- 0.0060 | 0.4086 +/- 0.0109 |       0.7775 |       0.3872 |
| structured | prolonged_los | LogisticRegression                     | all      | 6h       | 0.7079 +/- 0.0076 | 0.2952 +/- 0.0064 |       0.7072 |       0.2955 |
| structured | prolonged_los | XGBoost                                | all      | 6h       | 0.7318 +/- 0.0062 | 0.3332 +/- 0.0079 |       0.7315 |       0.3396 |
| structured | prolonged_los | LogisticRegression                     | sepsis   | 6h       | 0.6954 +/- 0.0153 | 0.4025 +/- 0.0221 |       0.7038 |       0.4134 |
| structured | prolonged_los | XGBoost                                | sepsis   | 6h       | 0.7173 +/- 0.0102 | 0.4397 +/- 0.0187 |       0.7206 |       0.44   |
| structured | prolonged_los | LogisticRegression                     | aki      | 6h       | 0.6891 +/- 0.0075 | 0.3332 +/- 0.0076 |       0.7068 |       0.3529 |
| structured | prolonged_los | XGBoost                                | aki      | 6h       | 0.7131 +/- 0.0018 | 0.3719 +/- 0.0088 |       0.7306 |       0.3949 |
| structured | mortality     | LogisticRegression                     | all      | 12h      | 0.8122 +/- 0.0030 | 0.3957 +/- 0.0130 |       0.8108 |       0.4127 |
| structured | mortality     | XGBoost                                | all      | 12h      | 0.8348 +/- 0.0040 | 0.4405 +/- 0.0131 |       0.8351 |       0.4581 |
| structured | mortality     | LogisticRegression                     | sepsis   | 12h      | 0.7669 +/- 0.0040 | 0.4402 +/- 0.0131 |       0.7858 |       0.4647 |
| structured | mortality     | XGBoost                                | sepsis   | 12h      | 0.7791 +/- 0.0076 | 0.4610 +/- 0.0058 |       0.8027 |       0.4988 |
| structured | mortality     | LogisticRegression                     | aki      | 12h      | 0.7983 +/- 0.0052 | 0.4284 +/- 0.0092 |       0.7857 |       0.4043 |
| structured | mortality     | XGBoost                                | aki      | 12h      | 0.8169 +/- 0.0041 | 0.4615 +/- 0.0076 |       0.8083 |       0.4424 |
| structured | prolonged_los | LogisticRegression                     | all      | 12h      | 0.7520 +/- 0.0076 | 0.3568 +/- 0.0063 |       0.7536 |       0.3548 |
| structured | prolonged_los | XGBoost                                | all      | 12h      | 0.7716 +/- 0.0063 | 0.3917 +/- 0.0070 |       0.7728 |       0.4045 |
| structured | prolonged_los | LogisticRegression                     | sepsis   | 12h      | 0.7348 +/- 0.0143 | 0.4579 +/- 0.0206 |       0.7375 |       0.4704 |
| structured | prolonged_los | XGBoost                                | sepsis   | 12h      | 0.7567 +/- 0.0109 | 0.4965 +/- 0.0204 |       0.7589 |       0.4978 |
| structured | prolonged_los | LogisticRegression                     | aki      | 12h      | 0.7332 +/- 0.0061 | 0.3924 +/- 0.0068 |       0.7439 |       0.4153 |
| structured | prolonged_los | XGBoost                                | aki      | 12h      | 0.7562 +/- 0.0061 | 0.4348 +/- 0.0063 |       0.7588 |       0.4423 |
| structured | mortality     | LogisticRegression                     | all      | 24h      | 0.8450 +/- 0.0022 | 0.4732 +/- 0.0096 |       0.8442 |       0.4924 |
| structured | mortality     | XGBoost                                | all      | 24h      | 0.8644 +/- 0.0042 | 0.5153 +/- 0.0076 |       0.8651 |       0.5305 |
| structured | mortality     | LogisticRegression                     | sepsis   | 24h      | 0.7996 +/- 0.0044 | 0.5061 +/- 0.0098 |       0.8048 |       0.5222 |
| structured | mortality     | XGBoost                                | sepsis   | 24h      | 0.8150 +/- 0.0053 | 0.5302 +/- 0.0056 |       0.8169 |       0.5485 |
| structured | mortality     | LogisticRegression                     | aki      | 24h      | 0.8311 +/- 0.0063 | 0.5002 +/- 0.0157 |       0.8209 |       0.4861 |
| structured | mortality     | XGBoost                                | aki      | 24h      | 0.8459 +/- 0.0058 | 0.5319 +/- 0.0158 |       0.8434 |       0.5212 |
| structured | prolonged_los | LogisticRegression                     | all      | 24h      | 0.7972 +/- 0.0082 | 0.4233 +/- 0.0125 |       0.8008 |       0.425  |
| structured | prolonged_los | XGBoost                                | all      | 24h      | 0.8149 +/- 0.0045 | 0.4647 +/- 0.0043 |       0.823  |       0.4803 |
| structured | prolonged_los | LogisticRegression                     | sepsis   | 24h      | 0.7814 +/- 0.0075 | 0.5222 +/- 0.0190 |       0.7808 |       0.5364 |
| structured | prolonged_los | XGBoost                                | sepsis   | 24h      | 0.8020 +/- 0.0090 | 0.5659 +/- 0.0167 |       0.804  |       0.5723 |
| structured | prolonged_los | LogisticRegression                     | aki      | 24h      | 0.7784 +/- 0.0025 | 0.4555 +/- 0.0052 |       0.7798 |       0.4659 |
| structured | prolonged_los | XGBoost                                | aki      | 24h      | 0.7991 +/- 0.0042 | 0.4960 +/- 0.0030 |       0.803  |       0.5123 |
| text       | mortality     | XGBoost                                | nan      | nan      | 0.7314 +/- 0.0038 | 0.2974 +/- 0.0075 |       0.7434 |       0.3169 |
| text       | mortality     | LogisticRegression                     | nan      | nan      | 0.7316 +/- 0.0044 | 0.3020 +/- 0.0101 |       0.7434 |       0.3237 |
| text       | prolonged_los | XGBoost                                | nan      | nan      | 0.6619 +/- 0.0036 | 0.2681 +/- 0.0027 |       0.6723 |       0.2788 |
| text       | prolonged_los | LogisticRegression                     | nan      | nan      | 0.6583 +/- 0.0032 | 0.2658 +/- 0.0026 |       0.6701 |       0.2811 |
| fusion     | mortality     | Early Fusion                           | nan      | nan      | 0.7614 +/- 0.0060 | 0.3471 +/- 0.0161 |       0.7642 |       0.3569 |
| fusion     | mortality     | Late Fusion (Alpha=1.0 Structured XGB) | all      | 24h      | 0.8634 +/- 0.0054 | 0.5132 +/- 0.0091 |       0.8653 |       0.532  |
| fusion     | mortality     | Late Fusion (Tuned Alpha XGB Preds)    | all      | 24h      | 0.8636 +/- 0.0054 | 0.5133 +/- 0.0099 |       0.8653 |       0.5318 |
| gru        | mortality     | temporal_gru_v2                        | nan      | nan      | 0.8484 +/- 0.0041 | 0.4881 +/- 0.0148 |       0.8392 |       0.4714 |