#!/bin/bash
#SBATCH --job-name=download_medcat
#SBATCH --output=download_medcat_%j.log
#SBATCH --error=download_medcat_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

# Download MedCAT Pretrained Model with UMLS License
echo "=== MedCAT Model Download ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

cd /scratch/users/k25113331/TIMELY-Bench_Final

# Create models directory
mkdir -p models/medcat
cd models/medcat

# Activate environment
source /scratch/users/k25113331/TIMELY-Bench_Final/medcat_env/bin/activate

# Download MedCAT model using the official method
echo "Downloading MedCAT pretrained model..."

python << 'EOF'
import os
from pathlib import Path

# Try to download using MedCAT's built-in model download
try:
    from medcat.cat import CAT
    
    # Download the publicly available mc_modelpack_snomed_int_16_mar_2022 model
    # This is a SNOMED model that doesn't require UMLS license
    print("Attempting to download SNOMED model pack...")
    
    # Check if wget is available for direct download
    import subprocess
    
    # MedCAT public models available at:
    # https://github.com/CogStack/MedCAT/tree/master#available-models
    
    # Download the model directly
    model_url = "https://cogstack-medcat-example-models.s3.eu-west-2.amazonaws.com/medcat-example-models/mc_modelpack_snomed_int_16_mar_2022_25be3857ba34bdd5.zip"
    model_path = Path("mc_modelpack_snomed.zip")
    
    if not model_path.exists():
        print(f"Downloading from {model_url}...")
        result = subprocess.run([
            "wget", "-O", str(model_path), model_url
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Download successful!")
        else:
            print(f"wget failed: {result.stderr}")
            # Try with curl
            result = subprocess.run([
                "curl", "-L", "-o", str(model_path), model_url
            ], capture_output=True, text=True)
            if result.returncode == 0:
                print("curl download successful!")
            else:
                print(f"curl failed: {result.stderr}")
    else:
        print("Model file already exists")
    
    # Verify the download
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"Model file size: {size_mb:.1f} MB")
        
        # Test loading
        if size_mb > 10:
            print("Testing model load...")
            cat = CAT.load_model_pack(str(model_path))
            print("Model loaded successfully!")
            
            # Test extraction
            test_text = "Patient has sepsis and acute kidney injury with fever."
            entities = cat.get_entities(test_text)
            print(f"Test extraction found {len(entities['entities'])} entities")
        else:
            print("Model file too small, download may have failed")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
EOF

echo ""
echo "=== Download Complete ==="
echo "Date: $(date)"
ls -la
