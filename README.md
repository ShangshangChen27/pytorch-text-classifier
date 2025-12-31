# PyTorch Text Classifier (Minimal)

A minimal end-to-end text classification project built with **PyTorch**.
Goal: understand the full training loop (forward → loss → backward → optimizer step).

## What it does
- Classifies short text into **positive (1)** / **negative (0)** sentiment.
- Uses a tiny toy dataset and a simple model:
  - Embedding layer
  - Mean pooling (masking PAD)
  - Linear classifier

## Project structure
- `data.py`: toy dataset, vocab building, padding + `collate_batch`
- `model.py`: `nn.Module` model (embedding + pooling + classifier)
- `train.py`: DataLoader + training loop + simple predictions

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy
