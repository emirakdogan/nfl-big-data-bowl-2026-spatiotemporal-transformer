# NFL Big Data Bowl 2026 — Spatiotemporal Transformer

Player trajectory prediction for the NFL Big Data Bowl 2026 competition. This project models all 22 players plus a ball-landing reference as a unified spatiotemporal token graph during the ball-in-air phase.

## Project Overview

The model uses a 1-second pre-throw context window and predicts future cumulative `(dx, dy)` movement for the targeted player. Each play is represented as a sequence of frames, where players and the ball reference are encoded as tokens.

Key ideas:

- Axial attention over time and players
- Distance- and velocity-aware edge messaging between players
- Direction normalization so all plays share a consistent field orientation
- Temporal Huber loss with light physics-inspired regularization
- GroupKFold cross-validation by `game_id`
- Ensemble inference with test-time augmentation

## Repository Structure

```text
.
├── spatiotemporal-transformer-edge-aware-attention.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Notebook Sections

1. Setup and configuration
2. Field geometry and play-direction normalization
3. Kinematics and data cleaning
4. Feature engineering
5. Token and sequence construction
6. Axial spatiotemporal Transformer model
7. Temporal Huber loss with physics regularizers
8. Cross-validation training pipeline
9. Bundle saving/loading
10. Kaggle inference pipeline

## Model Architecture

The core model is an all-player spatiotemporal Transformer:

- **Temporal attention:** models each player across time
- **Spatial attention:** models interactions between all players in a frame
- **Edge-aware message passing:** uses pairwise distance and relative velocity priors
- **Type embeddings:** distinguishes roles such as targeted receiver, coverage defender, passer, and ball reference

## Training Setup

The notebook is currently configured with lightweight settings for easier review and faster execution:

```python
SEEDS = [42]
N_FOLDS = 2
EPOCHS = 3
```

For a full competition-style run, restore the stronger configuration described in the notebook notes, such as multiple seeds, 5 folds, and more epochs.

## Data

This project is designed for Kaggle's NFL Big Data Bowl 2026 environment. The raw competition data is not included in this repository. To run the notebook, place the competition CSV files in the expected Kaggle input directory or update `DATA_DIR` in the configuration.

## Author

Ahmet Emir Akdoğan
