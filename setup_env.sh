##################################################
## setup_env.sh                                 ##
## Installs all dependencies in editable mode.  ##
## This script is not finished, don't use it.   ##
## -------------------------------------------- ##
## Author:   Mathias Otnes                      ##
## Date:     09/15/2025                         ##
##################################################

#!/usr/bin/env bash
set -e  # stop on first error

cd "$(dirname "$0")"

echo "[PerAct-MISO Setup]"

# Detect OS
OS=$(uname -s)

# Create conda env if missing
if ! conda info --envs | grep -q peract-miso; then
    conda create -y -n peract-miso python=3.10
fi
conda activate peract-miso

# Install pip basics
pip install --upgrade pip
pip install scipy ftfy regex tqdm einops trimesh pyrender pycollada
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/openai/CLIP.git

# Submodules
git submodule update --init --recursive

for mod in RLBench PerAct MISO; do
    if [ -d "mod/$mod" ]; then
        echo "→ Installing $mod"
        pip install -e "mod/$mod"
    fi
done

if [ "$OS" = "Linux" ]; then
    echo "→ Linux detected: setting up PyRep + CoppeliaSim"
    # user must set COPPELIASIM_ROOT manually!
    cd /tmp
    git clone https://github.com/stepjam/PyRep.git
    cd PyRep
    pip install -r requirements.txt
    pip install .
fi

echo "✅ Setup complete!"
