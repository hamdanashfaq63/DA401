# Finding Order in Disorder: Detecting Fold Symmetries in Metallic Glass Nanodiffraction Patterns

**Hamdan Ashfaq**
DA 401: Senior Research Seminar, Denison University, Spring 2026
Professor Mason Shero

## Overview

This repository contains the code and documentation for my senior capstone research project on automated detection and classification of fold symmetries in metallic glass nanodiffraction patterns collected via 4D-STEM. The project has two phases:

1. **Rule-based speckle detection pipeline** (complete): Classifies 65,536 nanodiffraction patterns from CuZr-based metallic glass by detecting circular bright spots and measuring their angular spacing against expected fold geometries.
2. **Self-supervised feature learning** (in progress): SimSiam contrastive learning with ResNet-18 backbone followed by K-means clustering, to be compared against the rule-based baseline.

## Repository Structure

```
DA401/
├── README.md                    # This file
├── Hamdan_DA401_Work.ipynb      # Main analysis notebook
├── requirements.txt             # Python dependencies
├── data/                        # Data directory (see note below)
│   └── .gitkeep
├── results/                     # Output figures and classification results
│   ├── tsne_visualization.png
│   ├── intensity_histogram.png
│   └── fold_classification_results.csv
└── paper/
    └── Hamdan_DA401_First_Draft.pdf
```

## Data

**Primary dataset:** 65,536 4D-STEM nanodiffraction patterns (128x128 pixels each) from CuZr-based metallic glass samples, collected by the Hwang Lab at The Ohio State University. This dataset is private and used with explicit permission from Dr. Jinwoo Hwang. It is not included in this repository.

**Public benchmark dataset:** For method verification, you can run the pipeline on the publicly available dataset from Zhu et al. (2024), available on Zenodo at [DOI: 10.5281/zenodo.8349542](https://doi.org/10.5281/zenodo.8349542). This dataset contains 4D-STEM diffraction patterns from epitaxial LaFeO3 films with domain labels.

### Running with the public dataset

1. Download the Zhu et al. dataset from Zenodo
2. Place the TIFF stack in the `data/` directory
3. Update the file path in the notebook's loading cell
4. Run all cells - the pipeline functions work on any stack of 128x128 diffraction pattern images

## Pipeline Description

The analysis notebook (`Hamdan_DA401_Work.ipynb`) contains the full rule-based classification pipeline:

### Core Functions

**`detect_circular_speckles(image, min_radius=6, max_radius=10)`**
- Applies central mask (radius 8 pixels) to block the bright center beam
- Enhances contrast using 2nd-98th percentile rescaling
- Thresholds at the 92nd-95th percentile of nonzero pixel intensity
- Labels connected components and filters by area (radius bounds) and size consistency (0.5x-1.5x median)
- Returns array of qualifying speckles as (y, x, radius) tuples

**`classify_pattern(blobs, image_shape)`**
- Computes azimuthal angle of each speckle relative to pattern center
- Checks angular spacing against expected fold values:
  - 2-fold: 180 degrees
  - 3-fold: 120 degrees
  - 4-fold: 90 degrees
  - 5-fold: 72 degrees
  - 6-fold: 60 degrees
- Angular tolerance: 15% for 2-fold, 3-fold, 5-fold; 25% for 4-fold, 6-fold
- Verifies opposite-pair geometry for even-fold patterns
- Returns fold label string or "None"/"Other"

**`process_image(image)`**
- Wraps detection and classification
- Computes feature vector (mean/std of outer ring intensities, speckle spatial distribution)
- Returns classification label and feature vector

### Visualization

The notebook also includes t-SNE dimensionality reduction (perplexity=30, 1000 iterations) and intensity histogram generation for the figures in the paper.

## Requirements

```
numpy>=1.24
scikit-image>=0.21
scikit-learn>=1.3
matplotlib>=3.7
tqdm>=4.65
```

For the planned SimSiam phase (not yet in this repo):
```
torch>=2.0
torchvision>=0.15
umap-learn>=0.5
```

### Install

```bash
pip install numpy scikit-image scikit-learn matplotlib tqdm
```

## Results Summary

| Category | Count | Percent |
|---|---|---|
| 2-fold | 3,221 | 4.91% |
| 3-fold (Odd) | 59 | 0.09% |
| 4-fold | 91 | 0.14% |
| 5-fold (Odd) | 0 | 0.00% |
| 6-fold | 1 | < 0.01% |
| None | 27,662 | 42.19% |
| Other | 34,502 | 52.61% |

The pipeline classified 5.2% of patterns with a definite fold label. The remaining 94.8% in None/Other categories represent the target for the self-supervised learning phase.

## Compute Environment

- Local development: Python 3.11, macOS/Linux
- Training (planned SimSiam phase): Ohio Supercomputer Center (OSC), GPU nodes

## References

Key papers for this project:

- Im et al. (2021). Medium range ordering in Zr-Cu-Co-Al metallic glasses. *Physical Review Materials*, 5(11), 115601.
- Zhu et al. (2024). Structural degeneracy in LaFeO3 films via ML-assisted 4D-STEM. *Scientific Reports*, 14, 4198.
- Chen and He (2021). Exploring simple Siamese representation learning. *CVPR 2021*.
- Ophus (2019). 4D-STEM: From scanning nanodiffraction to ptychography and beyond. *Microscopy and Microanalysis*, 25(3), 563-582.

## Contact

Hamdan Ashfaq - Denison University
Advisor: Professor Mason Shero (DA 401)
Research collaborator: Dr. Jinwoo Hwang, The Ohio State University
