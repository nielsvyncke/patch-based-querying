"""Evaluation pipeline for precision-recall analysis.

This module implements the main evaluation pipeline that runs experiments
across different datasets, batch sizes, and models.

Author: Niels Vyncke
"""

import os
import pandas as pd
from tabulate import tabulate

from src.core import SearchFramework, PatchInfoList, loadModel
from src.utils import construct_query_trees, calculate_precision_recall, compute_average_precision
from src.visualization import plot_pr_curve


def run_search_pipeline(input_dir, encoder, batch_size, structure=1, num_neighbors=1, dims=[80], min_overlap=0.5):
    """Run the search pipeline for a given configuration using SearchFramework.
    
    Args:
        input_dir (str): Dataset directory name
        encoder (str): Model name (e.g., 'ae_finetuned')
        batch_size (int): Batch size for query selection
        structure (int): Target structure label (default: 1)
        num_neighbors (int): Number of nearest neighbors (default: 1)
        dims (list): Patch dimensions (default: [80])
        min_overlap (float): Minimum overlap for positive examples (default: 0.5)
        
    Returns:
        tuple: (all_patches, query_idxs) or None if error
    """
    print(f"Running search pipeline for {encoder}, batch_size={batch_size}, dataset={input_dir}")
    
    # Set up directories
    raw_dir = os.path.join("images", input_dir, "raw")
    label_dir = os.path.join("images", input_dir, "labels")
    
    if not os.path.exists(raw_dir) or not os.path.exists(label_dir):
        print(f"Warning: Dataset directories not found for {input_dir}")
        return None
    
    raw_files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".png")])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".png")])
    
    num_slices = len(raw_files)
    latent_size = 32
    
    # Load model
    model = loadModel(encoder)

    # Select middle slice from every batch
    query_idxs = [batch_size * i + batch_size // 2 for i in range(num_slices // batch_size)]
    # Take the middle of the remaining slices
    if num_slices % batch_size != 0:
        query_idxs.append(num_slices - (num_slices % batch_size) // 2 - 1)
    
    pos_search_tree, neg_search_tree = construct_query_trees(query_idxs, model, raw_files, label_files, structure, dims, min_overlap, latent_size)

    # Create SearchFramework and perform search on non-query slices
    framework = SearchFramework(pos_search_tree, neg_search_tree, model, latent_size)
    
    # Process only non-query slices
    all_patches = PatchInfoList()
    for slice_idx in range(num_slices):
        if slice_idx in query_idxs:
            continue
        
        slice_patches = framework.search_single_slice(input_dir, slice_idx, structure, dims, num_neighbors)
        for record in slice_patches.recordList:
            all_patches.addRecord(record)

    # Create output directory and save results
    output_dir = os.path.join("data", input_dir, f"{encoder}_{batch_size}_{dims[0]}_{structure}_{num_neighbors}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to file
    all_patches.writeToFile(os.path.join(output_dir, "similarities.csv"))
    
    return all_patches, query_idxs


def main():
    """Run the main evaluation pipeline."""
    # Experiment configuration
    structures = [1]
    k = 1
    input_dirs = ["EMBL", "EPFL", "VIB"]
    batch_sizes = [10, 20, 30, 50, 100]
    models = ["ae_pretrained", "vae_pretrained", "ae_finetuned", "vae_finetuned"]
    
    # Results storage
    results_table = []
    
    print("Starting evaluation pipeline...")
    print("=" * 60)
    
    for input_dir in input_dirs:
        print(f"\nProcessing dataset: {input_dir}")
        print("-" * 40)
        
        # Check if dataset exists
        dataset_dir = f'images/{input_dir}/labels'
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory not found: {dataset_dir}")
            continue
        
        for batch_size in batch_sizes:
            print(f"\n  Batch size: {batch_size}")
            
            # Get query indices for this batch size
            num_slices = len(os.listdir(dataset_dir))
            query_idxs = [batch_size * i + batch_size // 2 for i in range(num_slices // batch_size)]
            if num_slices % batch_size != 0:
                query_idxs.append(num_slices - (num_slices % batch_size) // 2 - 1)
            
            batch_results = {}
            
            for model in models:
                print(f"    Processing model: {model}")
                
                # Check if results already exist
                result_file = f'data/{input_dir}/{model}_{batch_size}_80_1_1/similarities.csv'
                
                if os.path.exists(result_file):
                    # Load existing results
                    df = pd.read_csv(result_file)
                    df = df.sort_values(by='similarity', ascending=False)
                    print(f"      Loaded existing results from {result_file}")
                else:
                    # Run search pipeline
                    print(f"      Running search pipeline for {model}...")
                    try:
                        all_patches, _ = run_search_pipeline(
                            input_dir=input_dir,
                            encoder=model,
                            batch_size=batch_size,
                            structure=1,
                            num_neighbors=k,
                            dims=[80]
                        )
                        if all_patches is None:
                            continue
                        
                        # Convert to DataFrame for consistency
                        df_data = []
                        for record in all_patches.recordList:
                            slice_idx, x, y, dim = record.getLoc()
                            df_data.append({
                                'slice': slice_idx,
                                'x': x,
                                'y': y,
                                'dim': dim,
                                'similarity': record.get_similarity()
                            })
                        df = pd.DataFrame(df_data)
                        df = df.sort_values(by='similarity', ascending=False)
                    except Exception as e:
                        print(f"      Error running search pipeline: {e}")
                        continue
            
                # Calculate precision and recall
                print(f"      Calculating precision and recall...")
                try:
                    precisions, recalls = calculate_precision_recall(df, dataset_dir, query_idxs, structure=1)
                    
                    # Compute average precision
                    ap = compute_average_precision(precisions, recalls)
                    batch_results[model] = {
                        'precisions': precisions,
                        'recalls': recalls,
                        'ap': ap
                    }
                    
                    print(f"      Average Precision: {ap:.5f}")
                    
                    # Plot PR curve for finetuned AE
                    if model == "ae_finetuned":
                        print(f"      Generating PR curve...")
                        plot_pr_curve(precisions, recalls, input_dir, batch_size, save_path=f"results/PR/{input_dir}")
                        
                except Exception as e:
                    print(f"      Error calculating precision/recall: {e}")
                    continue
            
            # Add results to table
            results_table.append([
                input_dir,
                batch_size,
                *(f"{batch_results[model]['ap']:.5f}" for model in models if model in batch_results)
            ])
    
    # Print results table
    print("\n" + "=" * 80)
    print("AVERAGE PRECISION RESULTS")
    print("=" * 80)
    
    if results_table:
        headers = ["Dataset", "Batch Size", "Pretrained AE", "Pretrained VAE", "Finetuned AE", "Finetuned VAE"]
        print(tabulate(results_table, headers=headers, tablefmt="grid"))
        
        # Save results to CSV
        os.makedirs("results", exist_ok=True)
        results_df = pd.DataFrame(results_table, columns=headers)
        results_df.to_csv("results/evaluation_results.csv", index=False)
        print(f"\nResults saved to: results/evaluation_results.csv")

        # save result to LATEX
        with open("results/evaluation_results.tex", "w") as f:
            f.write(tabulate(results_table, headers=headers, tablefmt="latex"))
            print(f"\nResults saved to: results/evaluation_results.tex")
    else:
        print("No results to display. Check dataset paths and model weights.")
