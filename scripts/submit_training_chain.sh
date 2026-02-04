#!/bin/bash
set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final

smoke_id=$(sbatch scripts/smoke_structured_mortality_hpc.sh | awk '{print $4}')
structured_id=$(sbatch --dependency=afterok:${smoke_id} scripts/train_structured_hpc.sh | awk '{print $4}')
text_id=$(sbatch --dependency=afterok:${structured_id} scripts/train_text_hpc.sh | awk '{print $4}')
fusion_id=$(sbatch --dependency=afterok:${text_id} scripts/train_fusion_hpc.sh | awk '{print $4}')
gru_id=$(sbatch --dependency=afterok:${fusion_id} scripts/train_gru_hpc.sh | awk '{print $4}')

echo "Submitted chain:"
echo "  smoke_structured: ${smoke_id}"
echo "  structured:       ${structured_id}"
echo "  text:             ${text_id}"
echo "  fusion:           ${fusion_id}"
echo "  gru:              ${gru_id}"
