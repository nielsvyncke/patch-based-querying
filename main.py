"""Main entry point for the patch-based querying framework.

This script provides a simple interface to run the main experiments.

Usage:
    python main.py <experiment_type>

Author: Niels Vyncke
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from experiments import retrieved_patches_two_queries, main
from experiments.multiple_queries import multiple_queries_experiment
from experiments.clustering import main as clustering_main


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <experiment_type>")
        print("Available experiments:")
        print("  - evaluation: Run full evaluation pipeline")
        print("  - two_queries: Run two-query retrieval experiment")
        print("  - multiple_queries: Run multiple queries experiment")
        print("  - clustering: Show available clustering experiments")
        print("  - exp1-exp8: Run specific clustering experiments")
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
