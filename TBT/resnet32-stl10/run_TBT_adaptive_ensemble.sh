#!/bin/bash
#SBATCH --job-name=tbt_stl10_adap_ens
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%j.out

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda/etc/profile.d/conda.sh 2>/dev/null \
    || { echo "ERROR: conda not found."; exit 1; }
conda activate bfa

# CRITICAL: Change to fast local SSD (required by cluster policy)
cd "$TMPDIR" || exit 1

echo "[SETUP] Staging files to $TMPDIR ..."

# 1. Copy ONLY the TBT scripts directory — not the whole Aegis repo
mkdir -p Aegis/TBT
cp -r ~/Aegis/TBT/resnet32-stl10 Aegis/TBT/

# 2. Copy ONLY the model checkpoint needed by the script
mkdir -p Aegis/stl10/resnet32/save_finetune
cp ~/Aegis/stl10/resnet32/save_finetune/model_best.pth.tar \
   Aegis/stl10/resnet32/save_finetune/

# 3. Copy STL-10 dataset into the TBT script's working directory
# The Python script uses root='./data' relative to resnet32-stl10/
mkdir -p Aegis/TBT/resnet32-stl10/data
cp -r ~/Aegis/stl10/resnet32/data/stl10_binary \
      Aegis/TBT/resnet32-stl10/data/

echo "[SETUP] Done. Total staged: $(du -sh Aegis/ | cut -f1)"

cd Aegis/TBT/resnet32-stl10
python -u TBT_adaptive_ensemble.py

# CRITICAL: Copy all results back before SLURM wipes TMPDIR
mkdir -p "$SLURM_SUBMIT_DIR/result_stl10_adap_ens"
cp -r result/* "$SLURM_SUBMIT_DIR/result_stl10_adap_ens/"
echo "=== DONE: Adaptive Ensemble TBT (STL-10) results saved to $SLURM_SUBMIT_DIR/result_stl10_adap_ens/ ==="
