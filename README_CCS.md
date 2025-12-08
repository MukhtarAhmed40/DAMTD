# DAMTD Artifact - Representative subsets for CCS 2026

This artifact contains a runnable implementation of DAMTD (Domain-Adaptive Malicious Traffic Detection).
It includes representative anonymized subsets for the three datasets used in the paper: LSNM2024, CIRA-CIC-DoHBrw2020, and CICIoT2023.
The subsets are synthetic but preserve key statistical properties to enable reviewers to run the full pipeline.

## Reproducibility
- Create environment:
  ```
  conda env create -f environment.yml
  conda activate damtd-env
  ```
- Or use venv:
  ```
  python3 -m venv damtd-venv
  source damtd-venv/bin/activate
  pip install -r requirements.txt
  ```

- Generate / regenerate subsets:
  ```
  python3 code/preprocess_general.py --dataset LSNM2024 --samples 1000
  python3 code/preprocess_general.py --dataset DoHBrw2020 --samples 800
  python3 code/preprocess_general.py --dataset CICIoT2023 --samples 1200
  ```

- Train (example):
  ```
  python3 code/train.py --config configs/damtd_LSNM2024.yaml
  ```

- Evaluate:
  ```
  python3 code/evaluate.py --model_path pretrained_models/damtd_LSNM2024.h5 --dataset LSNM2024
  ```

- Run full pipeline:
  ```
  bash run_all.sh
  ```

## Open Science
See `OpenScience_Appendix.tex` for the exact LaTeX appendix to include in the paper. After acceptance we will replace the anonymized links with permanent repositories.
