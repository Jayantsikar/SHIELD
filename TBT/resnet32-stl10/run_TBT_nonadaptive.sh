#!/bin/bash
#SBATCH --job-name=tbt_stl10_nonadp
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%j_tbt_stl10_nonadaptive.out

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda/etc/profile.d/conda.sh 2>/dev/null \
    || { echo "ERROR: conda not found."; exit 1; }
conda activate bfa

# CRITICAL: Change to fast local SSD (required by cluster policy)
cd "$TMPDIR" || exit 1

echo "[SETUP] Staging files to $TMPDIR ..."

# 1. Copy ONLY the TBT scripts directory
mkdir -p Aegis/TBT
cp -r ~/Aegis/TBT/resnet32-stl10 Aegis/TBT/

# 2. Copy the model checkpoint
# Baseline script loads: ../../stl10/resnet32/save_finetune/model_best.pth.tar
mkdir -p Aegis/stl10/resnet32/save_finetune
cp ~/Aegis/stl10/resnet32/save_finetune/model_best.pth.tar \
   Aegis/stl10/resnet32/save_finetune/

# 3. Copy the STL-10 dataset
# Baseline script loads: ../../stl10/resnet32/data (relative to TBT/resnet32-stl10/)
mkdir -p Aegis/stl10/resnet32/data
cp -r ~/Aegis/stl10/resnet32/data/stl10_binary \
      Aegis/stl10/resnet32/data/

echo "[SETUP] Done. Total staged: $(du -sh Aegis/ | cut -f1)"

cd Aegis/TBT/resnet32-stl10
mkdir -p result
python -u TBT_nonadaptive.py

# CRITICAL: Copy all results back before SLURM wipes TMPDIR
mkdir -p "$SLURM_SUBMIT_DIR/result_stl10_nonadaptive"
cp -r result/* "$SLURM_SUBMIT_DIR/result_stl10_nonadaptive/" 2>/dev/null || true
echo "=== DONE: Non-Adaptive Baseline TBT (STL-10) results saved to $SLURM_SUBMIT_DIR/result_stl10_nonadaptive/ ==="
