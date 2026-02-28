# COVID-19 Detection from Cough Audio using Convolutional Neural Networks

## Overview

This repository contains the implementation developed for a Bachelor Thesis focused on automatic COVID-19 detection from cough audio recordings using Deep Learning techniques.

The project implements a complete and reproducible machine learning pipeline, starting from raw Coswara dataset recordings and ending with quantitative evaluation of a Convolutional Neural Network (CNN) classifier. The goal is to investigate whether cough signals contain discriminative acoustic patterns that can be learned by a compact deep neural network model.

The system is designed for research and educational purposes.

---

## Methodology

The implemented pipeline consists of the following stages:

1. **Cough extraction** from the Coswara dataset (heavy and shallow cough recordings).
2. **Preprocessing**, including resampling, padding/truncation to fixed duration and normalization.
3. **Time–frequency representation** using log-Mel spectrograms.
4. **Binary classification** (COVID vs Non-COVID) using a compact CNN architecture.
5. **Evaluation** using clinically relevant metrics and threshold optimization.

The dataset split is performed **by subject**, not by recording, in order to prevent data leakage and ensure realistic generalization.

---

## Dataset

The system is built on the Coswara dataset, which contains crowdsourced recordings of cough, breathing, and speech sounds collected during the COVID-19 pandemic.

Only cough-heavy and cough-shallow recordings are used in this implementation.

Users must download the dataset separately and comply with its license terms.

---

## Installation

Python 3.9 or newer is required.

Install dependencies:

```bash
pip install torch torchaudio numpy pandas scikit-learn matplotlib soundfile
```

GPU acceleration is optional but recommended for faster training.

---

## Usage

### 1. Extract cough recordings

```bash
python extract_coswara_cough.py
```

### 2. Generate the manifest file

```bash
python prepare_manifest.py
```

This creates the metadata file used for training, including subject ID, file path, label and dataset split.

### 3. Train the model

Baseline training:

```bash
python train_cnn.py --epochs 20
```

Training with data augmentation:

```bash
python train_cnn.py --epochs 20 --augment
```

Optional parameters:

```bash
--seed 42
--batch_size 16
--n_mels 64
--max_seconds 6
--lr 0.001
```

---

## Model

The classifier is a compact Convolutional Neural Network designed to balance representational capacity and generalization ability, given the size and variability of the dataset.

Input:  
Log-Mel spectrogram of shape `(1 × n_mels × T)`

The architecture consists of stacked convolutional blocks with Batch Normalization and ReLU activations, followed by adaptive pooling and a fully connected classification layer.

The model is intentionally lightweight to reduce overfitting risk.

---

## Evaluation

Performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score (positive class)
- ROC-AUC
- PR-AUC
- Confusion Matrix

The decision threshold is optimized on the validation set to maximize the F1-score of the positive class.

Experiments can be repeated with different random seeds to assess stability.

---

## Reproducibility

The training pipeline supports reproducibility through:

- Controlled random seeds
- Deterministic configuration in PyTorch
- Explicit logging of hyperparameters
- Structured experiment tracking

---

## Limitations

This implementation is based on crowdsourced audio recordings and is intended strictly for research purposes.

The model is not a medical diagnostic system and must not be used for clinical decision-making.

---

## Author

Bachelor Thesis Project  
Deep Learning for COVID-19 Detection from Cough Audio
