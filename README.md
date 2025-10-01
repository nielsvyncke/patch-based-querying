# Patch-based Querying for Electron Microscopy Data

(this work is currently in submission)

A patch-based similarity search system for identifying structures of interest in electron microscopy images using autoencoder embeddings.

**Website:** [gaim.ugent.be/software/patch-based-querying](https://gaim.ugent.be/software/patch-based-querying)

## Overview

This repository implements a comprehensive framework for patch-based querying in electron microscopy data. The system uses pre-trained autoencoders (AE) and variational autoencoders (VAE) to encode image patches into a latent space, then performs similarity search to identify structures of interest.

### System Architecture

```
├── src/                            # Main source code
│   ├── core/                       # Core framework classes
│   │   ├── patch_info.py           # PatchInfoRecord, PatchInfoList
│   │   ├── search_tree.py          # SearchTree (Annoy-based)
│   │   ├── model_utils.py          # Model loading and encoding
│   │   └── framework.py            # SearchFramework class
│   ├── models/                     # Neural network models
│   │   ├── ae.py                   # Autoencoder implementation
│   │   └── vae.py                  # Variational autoencoder implementation
│   ├── experiments/                # Experiment scripts
│   │   ├── single_query.py         # Two-query retrieval experiment
│   │   ├── multiple_queries.py     # Multiple queries experiment
│   │   ├── evaluation_pipeline.py  # Main evaluation pipeline
│   │   └── clustering.py           # t-SNE clustering and visualization
│   ├── utils/                      # Utility functions
│   │   ├── query_construction.py   # Query building utilities
│   │   └── evaluation.py           # Precision-recall calculations
│   └── visualization/              # Plotting utilities
│       └── plotting.py             # Visualization functions
├── scripts/                        # Entry point scripts
│   └── run_experiments.py          # CLI for experiments
├── main.py                         # Simple main entry point
├── weights/                        # Pre-trained model weights
│   ├── ae_pretrained.pth.tar
│   ├── ae_finetuned.pth.tar
│   ├── vae_pretrained.pth.tar
│   └── vae_finetuned.pth.tar
├── images/                         # Dataset directory structure
│   ├── EMBL/
│   ├── EPFL/
│   └── VIB/
├── data/                           # Generated data and results
└── results/                        # Output visualizations and metrics
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

### Quick Start

The framework provides multiple entry points for different experiments:

```bash
# Main entry point (recommended)
python main.py <experiment_type>

# Alternative CLI script
python scripts/run_experiments.py <experiment_type>
```

### Available Experiments

#### 1. Evaluation Pipeline
Run the complete evaluation with precision-recall analysis:
```bash
python main.py evaluation
```

#### 2. Two-Query Retrieval
Run patch retrieval using two positive and two negative query examples:
```bash
python main.py two_queries
```

#### 3. Multiple Queries
Run robust similarity search using multiple query patches:
```bash
python main.py multiple_queries
```

#### 4. t-SNE Clustering Experiments
Generate t-SNE visualizations of the latent space:

```bash
# Run specific clustering experiments
python main.py clustering-exp1    # Pretrained AE on EMBL (0-10)
python main.py clustering-exp2    # Pretrained VAE on EMBL (0-10)
python main.py clustering-exp3    # Finetuned AE on EMBL (0-10)
python main.py clustering-exp4    # Finetuned VAE on EMBL (0-10)
python main.py clustering-exp5    # Finetuned AE on EMBL (batch processing)
python main.py clustering-exp6    # Finetuned VAE on EMBL (batch processing)
python main.py clustering-exp7    # Finetuned AE on VIB (multi-class)
python main.py clustering-exp8    # Finetuned VAE on VIB (multi-class)
```

### Experiment Details

#### Pretraining vs. Finetuning Comparison
- **exp1**: Pretrained AE on 10 slices of EMBL dataset
- **exp2**: Pretrained VAE on 10 slices of EMBL dataset  
- **exp3**: Finetuned AE on 10 slices of EMBL dataset
- **exp4**: Finetuned VAE on 10 slices of EMBL dataset

#### Local vs. Global Analysis
- **exp5**: Finetuned AE on full EMBL dataset and subgroups
- **exp6**: Finetuned VAE on full EMBL dataset and subgroups

#### Multi-structure Analysis
- **exp7**: Finetuned AE on multiple structures (VIB dataset)
- **exp8**: Finetuned VAE on multiple structures (VIB dataset)

### Processing Pipeline

The framework will:
1. Load pre-trained models from `weights/` directory
2. Extract and encode patches from images using autoencoders
3. Build search trees for positive/negative examples
4. Compute similarity scores for all patches
5. Generate precision-recall curves and visualizations
6. Save results to `results/` and `data/` directories

### Programmatic Usage

#### Basic similarity search using the new modular structure:

```python
from src.core import SearchFramework, loadModel
from src.utils import construct_query_trees

# Load model
model = loadModel("ae_finetuned")

# Build query trees
pos_tree, neg_tree = construct_query_trees(
    query_idxs=[0, 1, 2], 
    model=model,
    raw_files=raw_files,
    label_files=label_files
)

# Create framework and run search
framework = SearchFramework(pos_tree, neg_tree, model)
results = framework.search("EMBL")
```

#### Generate t-SNE visualization:

```python
from src.experiments.clustering import run_clustering

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

#### Run evaluation pipeline:

```python
from src.experiments.evaluation_pipeline import run_search_pipeline

# Run complete evaluation
results, query_idxs = run_search_pipeline(
    input_dir="EMBL",
    encoder="ae_finetuned", 
    batch_size=10,
    structure=1,
    dims=[80]
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

The framework generates organized output in the following structure:

- **Similarity Data**: `data/DATASET/MODEL_BATCH_DIM_STRUCT_K/similarities.csv`
- **Precision-Recall Curves**: `results/PR/DATASET/PR_DATASET_BATCH.png`
- **Retrieved Patches**: `results/retrieved/MODEL_DATASET_SLICE.png`
- **Query Visualizations**: `results/retrieved/DATASET_pos_0.png`, `DATASET_neg_0.png`
- **t-SNE Visualizations**: `results/TSNE/MODEL_START_END.png`
- **Multiple Queries**: `results/multiple_queries/query_slice_DATASET.png`
- **Evaluation Summary**: `results/evaluation_results.csv` (CSV and LaTeX formats)

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

## Citation

If you use this code in your research, please cite:

```bibtex
@article{patch_based_querying,
  title={Patch-based Querying Identifies Structures of Interest in Electron Microscopy Data},
  author={Vyncke, Niels and Nadisic, Nicolas and Saeys, Yvan and Pižurica, Aleksandra},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the author.

## Acknowledgments

This work is partially funded by the Flanders AI Research Program, grant 174B09119; and the Belgian Federal Science Policy (BELSPO), grant Prf-2022-050.
