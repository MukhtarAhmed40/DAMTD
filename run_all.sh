#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Activating environment..."
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate damtd-env || conda env create -f "$ROOT_DIR/environment.yml"
  conda activate damtd-env
else
  python3 -m venv "$ROOT_DIR/damtd-venv"
  source "$ROOT_DIR/damtd-venv/bin/activate"
  pip install -r "$ROOT_DIR/requirements.txt"
fi
echo "Generating subsets..."
python3 "$ROOT_DIR/code/preprocess_general.py" --dataset LSNM2024 --samples 1000
python3 "$ROOT_DIR/code/preprocess_general.py" --dataset DoHBrw2020 --samples 800
python3 "$ROOT_DIR/code/preprocess_general.py" --dataset CICIoT2023 --samples 1200
echo "Training and evaluating..."
for ds in LSNM2024 DoHBrw2020 CICIoT2023; do
  cfg="$ROOT_DIR/configs/damtd_${ds}.yaml"
  python3 "$ROOT_DIR/code/train.py" --config "$cfg"
  python3 "$ROOT_DIR/code/evaluate.py" --model_path "$ROOT_DIR/pretrained_models/damtd_${ds}.h5" --dataset "$ds"
done
echo "Pipeline finished. Results in $ROOT_DIR/results and pretrained_models."
