# TIMELY-Bench 审计目录说明（Canonical）

本目录用于存放审计过程输出。对外汇报与交付请优先使用：

- `final_release/evidence/final_qa_32045137.json`
- `final_release/evidence/final_qa_32045137.md`
- `final_release/evidence/FINAL_AUDIT_SUMMARY.md`

## 当前目录用途

- `FINAL_AUDIT_SUMMARY.md/.json`：与 `final_release/evidence/` 对齐的镜像摘要。
- `OBSOLETE_NOTICE.md`：标记历史快照归档位置。
- 其余 JSON/CSV：过程诊断材料（不作为最终对外口径）。

## 历史快照归档

- 旧版审计快照：`documentation/archive_legacy/`
- 2026-02-20 18:30 旧脚本产物：`results/audit/archive_legacy/final_delivery_audit_run_20260220_183022/`

## 注意

- 不要再引用已归档文件名（例如旧 `final_anchor_inventory.json`、`episodes_parse_smoketest.json`）。
- 若需最新发布级审计结论，以 `final_release/evidence/final_qa_32045137.*` 为准。
