# DAMTD Artifact – Representative Subsets

This repository provides a runnable implementation of **DAMTD (Domain-Adaptive Malicious Traffic Detection)**, accompanying the paper:

> **DAMTD: A Domain-Adaptive Framework for Scalable Malicious Network Traffic Detection**  
> *Submitted to ACM CCS 2026*

The artifact enables reviewers to reproduce the core experimental pipeline reported in the paper.

Due to dataset licensing restrictions, **raw datasets are not redistributed**. Instead, this repository includes **representative anonymized and synthetic subsets** derived from the original datasets. These subsets preserve key statistical characteristics (e.g., feature distributions and class imbalance patterns) to allow end-to-end execution of the DAMTD framework.

---

## Contents

- Complete DAMTD source code  
- Configuration files for all experiments  
- Scripts for preprocessing, training, and evaluation  
- Representative synthetic subsets for reproducibility  
- Documentation for running the full pipeline  

---

## Reproducibility

### Environment Setup

#### Option 1: Conda (recommended)
```bash
conda env create -f environment.yml
conda activate damtd-env
```

#### Option 2: Python venv
```bash
python3 -m venv damtd-venv
source damtd-venv/bin/activate
pip install -r requirements.txt
```

---

### Subset Generation (Optional)

Representative subsets are provided by default.  
To regenerate them, run:

```bash
python3 code/preprocess_general.py --dataset LSNM2024 --samples 1000
python3 code/preprocess_general.py --dataset DoHBrw2020 --samples 800
python3 code/preprocess_general.py --dataset CICIoT2023 --samples 1200
```

---

### Training (Example)

```bash
python3 code/train.py --config configs/damtd_LSNM2024.yaml
```

---

### Evaluation

```bash
python3 code/evaluate.py   --model_path pretrained_models/damtd_LSNM2024.h5   --dataset LSNM2024
```

---

### Run Full Pipeline

```bash
bash run_all.sh
```

---

## Description

This artifact provides a complete and executable Python-based experimental framework for reproducing the main results reported in the DAMTD paper, including:

- Detection performance measured by **Accuracy**, **Precision**, **Recall**, **F1-score**, **False Positive Rate (FPR)**, and **False Negative Rate (FNR)**
- Cross-domain evaluation across heterogeneous network traffic datasets

All scripts perform data loading, preprocessing, training, evaluation, and result storage automatically.  
A fixed random seed is used to ensure deterministic execution.

---

## How to Access

The anonymous artifact repository is available at:

🔗 https://anonymous.4open.science/r/Artifact-ACMCCS/

---

## Hardware Dependencies

- **Minimum**: 4-core CPU, 16 GB RAM, 10 GB disk space  
- **Recommended**: 32 GB RAM for large-scale datasets  
- **Optional**: NVIDIA GPU with CUDA support for accelerated training  

---

## Software Dependencies

- Python **3.10** or **3.11**
- **TensorFlow ≥ 2.12**
- NumPy, Pandas, scikit-learn, Matplotlib
- PyYAML, tqdm, psutil
- Tested on **Windows 10/11** and **Ubuntu 22.04**

All dependencies are listed in `requirements.txt`.

---

## Benchmarks and Datasets

Experiments are based on three publicly available intrusion detection datasets released by the **Canadian Institute for Cybersecurity (CIC)**:

- **LSNM2024**  
  https://data.mendeley.com/datasets/7pzyfvv9jn/1

- **CIRA-CIC-DoHBrw2020**  
  https://www.unb.ca/cic/datasets/dohbrw-2020.html

- **CICIoT2023**  
  https://www.unb.ca/cic/datasets/iotdataset-2023.html

> **Note:** Evaluators must download the original datasets separately and place them in the directory structure specified in `configs/*.yaml` if full-scale reproduction is desired.

---

## Configuration

- Dataset paths and label mappings are defined in `configs/*.yaml`
- Hyperparameters (epochs, window size, stride, attention heads) are configurable via YAML files or command-line arguments

Example:
```bash
python3 code/train.py --config configs/damtd_LSNM2024.yaml
python3 code/evaluate.py --model_path pretrained_models/damtd_LSNM2024.h5 --dataset LSNM2024
```

---

## Experiment Workflow

Each experiment is executed via a standalone Python script located in `scripts/` or `code/` and follows this workflow:

1. Load raw CSV network traffic data  
2. Preprocess and normalize features  
3. Train the DAMTD model  
4. Evaluate detection performance  
5. Save outputs to the `outputs/` directory  

Experiments can be run from the command line or using IDEs such as **VS Code** with the integrated terminal.

---

## Notes for Reviewers

- The provided subsets are intended for **functional and methodological validation**, not for absolute performance comparison.
- All quantitative results reported in the paper were obtained using the **full datasets**.
- This artifact demonstrates the **correctness, reproducibility, and usability** of the DAMTD framework.

---

## License

This project is released for **academic and research use only**.
