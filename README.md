# Patch-based Querying for Electron Microscopy Data

A patch-based similarity search system for identifying structures of interest in electron microscopy images using autoencoder embeddings.

## Overview

This repository implements a comprehensive framework for patch-based querying in electron microscopy data. The system uses pre-trained autoencoders (AE) and variational autoencoders (VAE) to encode image patches into a latent space, then performs similarity search to identify structures of interest.

### Key Features

- **Patch-based similarity search** using learned embeddings
- **Autoencoder support** for both AE and VAE models
- **Approximate nearest neighbor search** using Annoy indexing
- **Precision-recall evaluation** with comprehensive metrics
- **t-SNE visualization** for latent space analysis
- **Multi-dataset support** (EMBL, EPFL, VIB datasets)

### System Architecture

```
├── search_framework.py    # Main similarity search framework
├── clustering.py          # t-SNE clustering and visualization
├── models/               # Autoencoder model definitions
│   ├── ae.py            # Autoencoder implementation
│   └── vae.py           # Variational autoencoder implementation
├── weights/             # Pre-trained model weights
│   ├── ae_pretrained.pth.tar
│   ├── ae_finetuned.pth.tar
│   ├── vae_pretrained.pth.tar
│   └── vae_finetuned.pth.tar
└── images/              # Dataset directory structure
    ├── EMBL/
    ├── EPFL/
    └── VIB/
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended)
- PyTorch with CUDA support

### Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Required packages:
- `torch` (with CUDA support)
- `numpy`
- `scikit-image`
- `opencv-python`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `annoy`
- `tqdm`
- `imageio`
- `tabulate`

## Usage

### Dataset Structure

Organize your datasets in the following structure:

```
images/
├── DATASET_NAME/
│   ├── raw/          # Raw microscopy images (.png)
│   └── labels/       # Corresponding label masks (.png)
```

### 1. Patch-based Querying Framework

Run the main search pipeline:

```bash
# Run complete evaluation pipeline
python search_framework.py run_search

# Run single query experiments
python search_framework.py single_query

# Run multiple query experiments
python search_framework.py multiple_queries
```

The framework will:
1. Load pre-trained models
2. Extract and encode patches from images
3. Build search trees for positive/negative examples
4. Compute similarity scores for all patches
5. Generate precision-recall curves
6. Save results and visualizations

### 2. t-SNE Clustering and Visualization

Generate t-SNE visualizations of the latent space:

```bash
python clustering.py <experiment>
```

Available experiments:

#### Pretraining vs. Finetuning Comparison
- `exp1`: Pretrained AE on 10 slices of EMBL dataset
- `exp2`: Pretrained VAE on 10 slices of EMBL dataset  
- `exp3`: Finetuned AE on 10 slices of EMBL dataset
- `exp4`: Finetuned VAE on 10 slices of EMBL dataset

#### Local vs. Global Analysis
- `exp5`: Finetuned AE on full EMBL dataset and subgroups
- `exp6`: Finetuned VAE on full EMBL dataset and subgroups

#### Multi-structure Analysis
- `exp7`: Finetuned AE on multiple structures (VIB dataset)
- `exp8`: Finetuned VAE on multiple structures (VIB dataset)

### 3. Custom Usage

#### Basic similarity search:

```python
from search_framework import loadModel, construct_query_trees, run_search_pipeline

# Load model
model = loadModel("ae_finetuned")

# Run search pipeline
results, query_idxs = run_search_pipeline(
    input_dir="EMBL",
    encoder="ae_finetuned", 
    batch_size=10,
    structure=1,
    dims=[80]
)
```

#### Generate t-SNE visualization:

```python
from clustering import run_clustering

# Run clustering analysis
run_clustering(
    model='ae_finetuned',
    dataset='EMBL',
    start=0,
    end=10,
    dims=[64],
    stride=2,
    show_image=True,
    binary=True
)
```

## Model Details

### Autoencoder Architecture
- **Input**: 64×64 grayscale patches
- **Latent dimension**: 32
- **Activation**: ReLU
- **Training**: Pre-trained and fine-tuned versions available

### Search Algorithm
- **Indexing**: Annoy (Approximate Nearest Neighbors Oh Yeah)
- **Distance metric**: Euclidean distance
- **Trees**: 750 trees for high accuracy
- **Query strategy**: Positive/negative example pairs

## Output and Results

### Generated Files

- `data/DATASET/MODEL_BATCH_DIM_STRUCT_K/similarities.csv`: Similarity scores
- `results/PR/DATASET/PR_DATASET_BATCH.png`: Precision-recall curves
- `results/retrieved/MODEL_DATASET.png`: Retrieved patch visualizations
- `results/TSNE/MODEL_START_END.png`: t-SNE visualizations
- `results/evaluation_results.csv`: Comprehensive evaluation metrics

### Evaluation Metrics

- **Average Precision (AP)**: Area under precision-recall curve
- **Precision-Recall curves**: Performance at different retrieval thresholds
- **Structure overlap analysis**: Quantitative overlap with ground truth

## Configuration

### Key Parameters

- `structure`: Target structure label (default: 1)
- `dims`: Patch dimensions (default: [80])
- `min_overlap`: Minimum overlap for positive examples (default: 0.5)
- `num_neighbors`: Number of nearest neighbors (default: 1)
- `batch_size`: Batch size for query selection (default: 10, 20, 30, 50, 100)

### Supported Datasets

- **EMBL**: European Molecular Biology Laboratory dataset
- **EPFL**: École Polytechnique Fédérale de Lausanne dataset  
- **VIB**: Vlaams Instituut voor Biotechnologie dataset

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or patch dimensions
2. **Missing dataset directories**: Ensure proper dataset structure
3. **Model loading errors**: Verify model weights are in `weights/` directory
4. **Visualization issues**: Check matplotlib backend configuration

### Performance Tips

- Use CUDA-enabled GPU for faster encoding
- Adjust `dataset_size` parameter to limit number of processed images
- Use appropriate `stride` values for patch extraction density

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper documentation
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@article{patch_based_querying,
  title={Patch-based Querying Identifies Structures of Interest in Electron Microscopy Data},
  author={Niels Vyncke},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the author.
