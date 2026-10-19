Research code for multimodal MRI brain tumor segmentation.

## Overview

Rician3D explores correction-guided feature refinement for 3D brain tumor segmentation,
with a focus on whole tumor (WT), tumor core (TC) and enhancing tumor (ET).

## Installation

Requirements: Python, PyTorch, NumPy and SciPy.
The included example was tested with Python 3.8 and PyTorch 1.11.

```bash
pip install -r requirements.txt
```

## Data Preparation

Obtain BraTS data from the [official dataset website].
The project uses four MRI modalities: FLAIR, T1, T1ce and T2.
The included preprocessing helpers support normalization and WT/TC/ET label conversion.
Dataset files are not distributed here.

## Repository Structure

```text
models/       # Baseline 3D U-Net
utils/        # Preprocessing and segmentation metrics
examples/     # Synthetic-data example
```

