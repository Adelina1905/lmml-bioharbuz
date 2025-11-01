# 🐼 Adversarial Panda Attack

## 📖 Overview

Created an adversarial perturbation to fool an AI classifier into believing a cat/dog image is a panda (≥80% confidence).

## 🚀 Quick Start

```bash
# 1. Download dataset
python3 INDONESIA/dataset_import.py

# 2. Train classifier
python3 INDONESIA/train_classifier.py

# 3. Generate perturbation
python3 INDONESIA/main.py
```

## 📋 Output

Submit `perturbation.npy` with:
- **Shape**: `(224, 224, 3)`
- **Dtype**: `float32`
- **Range**: `[-0.2, 0.2]`

## 🔧 Method

**PGD Attack with Momentum**:
- 500 iterations
- Step size: 0.01
- Epsilon: 0.2

The algorithm iteratively computes gradients to maximize panda class probability while keeping the perturbation small and imperceptible.

## ✅ Success

**PASS**: Panda confidence ≥ 80%  
**FLAG**: `SIGMOID_ADVERSARIAL`

## 📦 Requirements

```bash
pip install tensorflow numpy pillow kaggle
```

## 📁 Files

- `main.py` - Generates adversarial perturbation
- `train_classifier.py` - Trains CNN classifier  
- `perturbation.npy` - Output file (submit this)

---

*LMML Hackathon 2025*