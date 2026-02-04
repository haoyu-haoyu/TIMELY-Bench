#!/bin/bash
#SBATCH --job-name=dl_medcat_hf
#SBATCH --output=dl_medcat_hf_%j.log
#SBATCH --error=dl_medcat_hf_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

# Download MedCAT Model from Hugging Face
echo "=== MedCAT Model Download via Hugging Face ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

cd /scratch/users/k25113331/TIMELY-Bench_Final

# Activate environment
source medcat_env/bin/activate

# Install huggingface_hub
pip install huggingface_hub --quiet

# Create models directory
mkdir -p models/medcat
cd models/medcat

# Download using Python
python << 'EOF'
import os
from pathlib import Path

print("Downloading MedCAT model from Hugging Face...")

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    
    # Try to download from CogStack organization on HuggingFace
    # Available models: https://huggingface.co/cogstack
    
    # Method 1: Try the SNOMED model
    try:
        print("Attempting to download MedCAT SNOMED model...")
        model_path = snapshot_download(
            repo_id="cogstack/medcat-snomed",
            local_dir="./medcat_snomed",
            local_dir_use_symlinks=False
        )
        print(f"Downloaded to: {model_path}")
    except Exception as e1:
        print(f"SNOMED model failed: {e1}")
        
        # Method 2: Try alternative model
        try:
            print("Trying alternative: CogStack/MedCAT...")
            model_path = snapshot_download(
                repo_id="CogStack/MedCAT", 
                local_dir="./medcat_hf",
                local_dir_use_symlinks=False
            )
            print(f"Downloaded to: {model_path}")
        except Exception as e2:
            print(f"Alternative also failed: {e2}")
            
            # Method 3: Create minimal vocab and CDB for keyword extraction
            print("Creating minimal MedCAT setup for keyword-based extraction...")
            
            from medcat.vocab import Vocab
            from medcat.cdb import CDB
            from medcat.cat import CAT
            from medcat.config import Config
            
            # Create minimal vocab
            vocab = Vocab()
            
            # Create minimal CDB with medical concepts
            cdb = CDB(config=Config())
            
            # Add some basic medical concepts
            concepts = {
                'C0036690': {'name': 'sepsis', 'cui': 'C0036690'},
                'C0032285': {'name': 'pneumonia', 'cui': 'C0032285'},
                'C0022660': {'name': 'acute kidney injury', 'cui': 'C0022660'},
                'C0035229': {'name': 'respiratory failure', 'cui': 'C0035229'},
                'C0018801': {'name': 'heart failure', 'cui': 'C0018801'},
                'C0015967': {'name': 'fever', 'cui': 'C0015967'},
                'C0020649': {'name': 'hypotension', 'cui': 'C0020649'},
                'C0039231': {'name': 'tachycardia', 'cui': 'C0039231'},
                'C0242184': {'name': 'hypoxia', 'cui': 'C0242184'},
                'C0036974': {'name': 'shock', 'cui': 'C0036974'},
            }
            
            for cui, info in concepts.items():
                try:
                    cdb.add_names(cui=cui, names={info['name']})
                except:
                    pass
            
            # Save
            vocab.save("./minimal_vocab.dat")
            cdb.save("./minimal_cdb.dat")
            print("Created minimal MedCAT resources")
    
    # List files
    print("\nFiles in models/medcat:")
    for f in Path(".").iterdir():
        if f.is_file():
            size = f.stat().st_size / (1024*1024)
            print(f"  {f.name}: {size:.1f} MB")
        elif f.is_dir():
            print(f"  {f.name}/ (directory)")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

EOF

echo ""
echo "=== Download Complete ==="
echo "Date: $(date)"
