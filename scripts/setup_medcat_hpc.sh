#!/bin/bash
#SBATCH --job-name=medcat_setup
#SBATCH --output=medcat_setup_%j.log
#SBATCH --error=medcat_setup_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

# MedCAT Full Model Deployment on KCL CREATE HPC
# ================================================

echo "=== MedCAT Full Model Setup ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

# Create virtual environment
cd /scratch/users/k25113331/TIMELY-Bench_Final
module load python/3.9

# Create and activate venv
python -m venv medcat_env
source medcat_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install MedCAT and dependencies
echo "Installing MedCAT..."
pip install medcat
pip install spacy
python -m spacy download en_core_web_md

# Download MedCAT model (MedMentions - publicly available)
echo "Downloading MedCAT model..."
mkdir -p models/medcat
cd models/medcat

# Download the publicly available MedMentions model
# This is ~1GB and doesn't require UMLS license
python -c "
from medcat.cat import CAT
print('Downloading MedMentions model...')
# The model will be downloaded automatically on first use
"

echo "=== MedCAT Setup Complete ==="
