# Patch-based Querying Identifies Structures of Interest in Electron Microscopy Data

This repository contains the code for the patch-based querying system for identifying structures of interest in electron microscopy data.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

### t-SNE visualizations

To run the t-SNE visualizations, run the following command:

```bash
python clustering.py <experiment>
```

Where `<experiment>` takes `exp1`, `exp2`, `exp3`, `exp4`, `exp5`, `exp6`, `exp7`, or `exp8`, defined as follows:

#### Pretraining vs. finetuning

- `exp1`: pretrained AE on 10 slices of the EMBL dataset
- `exp2`: pretrained VAE on 10 slices of the EMBL dataset
- `exp3`: finetuned AE on 10 slices of the EMBL dataset
- `exp4`: finetuned VAE on 10 slices of the EMBL dataset

#### Local vs. global

- `exp5`: finetuned AE on full EMBL dataset, and groups of 10 slices
- `exp6`: finetuned VAE on full EMBL dataset, and groups of 10 slices

#### Multiple structures

- `exp7`: finetuned AE on multiple structures in 5 slices of the VIB dataset
- `exp8`: finetuned VAE on multiple structures in 5 slices of the VIB dataset

### Patch-based querying framework

To run the patch-based querying framework, run the following command:

```bash
python search_framework.py
```

## License

This repository is licensed under the MIT License - see the LICENSE file for details.
