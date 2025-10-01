"""Main entry point for running experiments.

This script provides a command-line interface for running various experiments
in the patch-based querying framework.

Usage:
    python scripts/run_experiments.py <experiment_type>
    
    experiment_type options:
    - evaluation: Run full evaluation pipeline
    - two_queries: Run two-query retrieval experiment
    - multiple_queries: Run multiple queries experiment
    - clustering: Show available clustering experiments
    - exp1-exp8: Run specific clustering experiments

Author: Niels Vyncke
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.experiments import retrieved_patches_two_queries, main
from src.experiments.multiple_queries import multiple_queries_experiment
from src.experiments.clustering import main as clustering_main


def main_cli():
    """Main command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_experiments.py <experiment_type>")
        print("Available experiments: evaluation, two_queries, multiple_queries, clustering, exp1-exp8")
        sys.exit(1)
    
    experiment_type = sys.argv[1]
    
    if experiment_type == "evaluation":
        main()
    elif experiment_type == "two_queries":
        retrieved_patches_two_queries("VIB", dims=[80])
        retrieved_patches_two_queries("EMBL", dims=[100])
        retrieved_patches_two_queries("EPFL", dims=[80])
    elif experiment_type == "multiple_queries":
        # Run multiple queries experiments on different datasets
        multiple_queries_experiment("VIB", dims=[85], encoder="ae_finetuned")
        multiple_queries_experiment("VIB", dims=[85], encoder="vae_finetuned")
    elif "clustering" in experiment_type:
        experiment_type = experiment_type.replace("clustering-", "")
        if experiment_type in ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8"]:
            # Handle specific clustering experiments
            sys.argv = ["clustering.py", experiment_type]  # Simulate clustering.py command line
            clustering_main()
        else:
            print(f"Unknown clustering experiment type: {experiment_type}")
            print("Available clustering experiments: exp1-exp8")
            sys.exit(1)
    else:
        print(f"Unknown experiment type: {experiment_type}")
        print("Available experiments: evaluation, two_queries, multiple_queries, clustering, exp1-exp8")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
