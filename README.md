# UGAP: Uncertainty-Guided Adaptive Perturbation for Reliable Molecular Property Prediction
## Overview
This repository is the implementation of the IJCAI 2026 paper ***Uncertainty-Guided Adaptive Local Adversarial Perturbation for Reliable Molecular Property Prediction***. It contains the official implementation of UGAP, a framework that integrates Evidential Deep Learning (EDL) with Adaptive Adversarial Perturbations to enhance both the generalization and the uncertainty fidelity of molecular property predictions.
---



- **Uncertainty-Guided Perturbation**: Dynamically targets salient atoms with perturbation intensities modulated by predictive uncertainty.
- **Stability-aware Calibration**: Refines uncertainty by regularizing the discrepancy between clean and perturbed predictions, ensuring evidence magnitude aligns with structural robustness.
---

## Dependencies

The project is built with **Python 3.12** and requires the following core libraries:

### Deep Learning & GNNs
* **PyTorch (2.7.0+cu128)**: Core deep learning framework.
* **PyTorch Geometric (2.7.0)**: Library for deep learning on graphs.
* **Supporting GNN Libraries**:
    * `torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`
    * `pyg-lib`

### Cheminformatics
* **RDKit (2025.9.2)**: Essential toolkit for molecular informatics and SMILES processing.

### Data Science
* **Pandas (2.3.3) & NumPy (2.2.6)**: Data manipulation and numerical computing.
* **Scikit-learn (1.7.2)**: Machine learning utilities and evaluation metrics.

---
## Environment Setup
We recommend using Conda to manage your environment. To install dependencies, run:

```bash
# Create the environment from file
conda env create -f environment.yml

# Activate the environment
conda activate ugap
```

## Data Preparation
UGAP is designed to work with molecular property prediction.

1. Data Acquisition  
You can download datasets such as BBBP, BACE, ClinTox, and ESOL directly from [MoleculeNet](https://moleculenet.org/).

2. Data Directory  
All necessary dataset files (SMILES and labels) are already stored in the `/data/` directory. The scripts are pre-configured to locate and load files directly from this path.

---

## Training
### Quick Start
To reproduce the results on the BBBP dataset, use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```
### Custom Training
You can manually trigger the training process using train.py. The script supports both classification and regression tasks:
```bash
python train.py \
    --data_path data/bbbp.csv  --dataset_type classification  --epochs 90  --clip_norm 5.0   --topk 0.5 \
    --hidden_size 240     --depth 2  --lam 1   --dropout 0.4     --activation LeakyReLU     --ffn_num_layers 3 \
    --ffn_hidden_size 240     --batch_size 96     --max_lr 0.0009     --init_lr 0.0001     --final_lr 5e-05 \
    --adv_w 1.0     --beta 0.9     --eps_max 0.07     --atten_head 4     --atten_dropout 0.3
```
The `train.py` script accepts various arguments to control the training process and model architecture. Below are the key parameters used in the UGAP framework:

* `--adv_w`: Weight λ<sub>1</sub> for the adversarial learning loss.
* `--beta`: Weight λ<sub>1</sub> for the **stability-aware calibration loss** ($\mathcal{L}_{calib}$).
* `--eps_max`: Maximum perturbation intensity $\epsilon$ for the adaptive attack.
* `--topk`: The ratio of salient atoms (e.g., `0.5` for top-50%) targeted for perturbation.
* `--hidden_size`: Dimensionality of the graph hidden representations.
* `--depth`: Number of message-passing layers in the Graph Encoder.
* `--atten_head`: Number of attention heads for the uncertainty-guided saliency mechanism.
* `--ffn_num_layers` & `--ffn_hidden_size`: Depth and width of the Feed-Forward Network after graph encoding.
* `--activation`: Activation function (e.g., `LeakyReLU`, `ReLU`).
* `--dropout` & `--atten_dropout`: Dropout rates for the global features and attention layers respectively.
* `--batch_size`: Number of molecules per training batch.
* `--max_lr`, `--init_lr`, `--final_lr`: Learning rate schedule parameters.
* `--clip_norm`: Maximum gradient norm for gradient clipping to prevent explosion.
* `--epochs`: Total number of training iterations.

---
## Training Results
Training outputs, including model checkpoints (.pt files) and score logs, are automatically saved in the `/ckpt` directory.