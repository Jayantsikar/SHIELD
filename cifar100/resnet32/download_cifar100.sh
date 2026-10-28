#!/bin/bash
# NOTE: Run this directly on the login node (e.g. clusterlogin):
#   bash download_cifar100.sh
# DO NOT run via 'sbatch' because compute nodes do not have internet access!

set -e

DATA_DIR="$HOME/Aegis/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading CIFAR-100 into ${DATA_DIR} ==="
wget -c https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

echo "=== Extracting CIFAR-100 archive ==="
tar -xzvf cifar-100-python.tar.gz

echo "=== Verification: Checking extracted files ==="
ls -lh "${DATA_DIR}/cifar-100-python"

echo "=== CIFAR-100 download and extraction successful! ==="
