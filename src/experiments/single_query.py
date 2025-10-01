"""Single query experiment for patch retrieval and visualization.

This module implements the two-query experiment that uses two positive and 
two negative query examples for similarity search.

Author: Niels Vyncke
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2 as cv

from src.core import SearchFramework, loadModel
from src.utils import get_two_queries


def retrieved_patches_two_queries(dataset, dataset_size=5, structure=1, num_neighbors=1, dims=[80], min_overlap=0.5):
    """Retrieve and visualize patches using two query examples.
    
    Performs patch-based similarity search using two positive and two negative
    query examples, then visualizes the top retrieved patches with color-coded borders.
    
    Args:
        dataset (str): Dataset name (e.g., 'EMBL', 'EPFL', 'VIB')
        dataset_size (int): Number of slices to use from dataset (default: 5)
        structure (int): Label value for target structure (default: 1)
        num_neighbors (int): Number of nearest neighbors for search (default: 1)
        dims (list): Patch dimensions to extract (default: [80])
        min_overlap (float): Minimum overlap threshold for positive examples (default: 0.5)
        
    Returns:
        None: Saves visualization results to results/retrieved/ directory
    """
    input_dir = dataset
    models = ["ae_finetuned", "vae_finetuned"]
    
    for encoder in models:
        random.seed(42)  # Ensure reproducible results
        print(f"Processing {encoder}")
        
        # Set up dataset directories
        raw_dir = os.path.join("images", input_dir, "raw")
        label_dir = os.path.join("images", input_dir, "labels")
        
        if not os.path.exists(raw_dir) or not os.path.exists(label_dir):
            print(f"Warning: Dataset directories not found for {input_dir}")
            return None
    
        raw_files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".png")])[:dataset_size]
        label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".png")])[:dataset_size]
        
        num_slices = len(raw_files)
        latent_size = 32
        
        # Load model
        model = loadModel(encoder)

        query_idxs = range(dataset_size)

        pos_search_tree, neg_search_tree = get_two_queries(query_idxs, model, raw_files, label_files, structure, dims, min_overlap, latent_size)
        pos_queries = pos_search_tree.items
        neg_queries = neg_search_tree.items
        
        # Create SearchFramework instance and perform search
        framework = SearchFramework(pos_search_tree, neg_search_tree, model, latent_size)
        all_patches = framework.search(dataset, structure=structure, dims=dims, num_neighbors=num_neighbors, num_slices=dataset_size)

        # sort by similarity
        all_patches.recordList.sort(key=lambda record: record.get_similarity(), reverse=True)   

        retrieved = []

        for patch in all_patches.recordList:
            slice_idx, x, y, dim = patch.getLoc()
            image = io.imread(label_files[slice_idx]) == structure
            if image[x:x+dim, y:y+dim].shape != (dim, dim):
                continue
            num_labels, labels_img = cv.connectedComponents((image).astype(np.uint8))
            for label in range(1, num_labels):
                if (labels_img == label)[x:x+dim, y:y+dim].any():
                    for retrieved_patch in retrieved:
                        retrieved_slice_idx, retrieved_x, retrieved_y, retrieved_dim = retrieved_patch.getLoc()
                        if retrieved_slice_idx == slice_idx:
                            if (labels_img == label)[retrieved_x:retrieved_x+retrieved_dim, retrieved_y:retrieved_y+retrieved_dim].any():
                                print("Patch overlaps with retrieved patch")
                                break
                    else:
                        print("Patch does not overlap with retrieved patch")
                        retrieved.append(patch)
                        break
            else:
                if patch.getOverlap() > 0:
                    continue
                print("Patch does not overlap with structure")
                retrieved.append(patch)
                
            if len(retrieved) == 12:
                break
        print("Retrieved", len(retrieved), "patches")
        
        # Create output directory
        os.makedirs("results/retrieved", exist_ok=True)
        
        # plot retrieved patches on each slice
        for index, slice in enumerate(raw_files):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            image = io.imread(slice)
            # convert to rgb
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            patches = filter(lambda patch: patch.getLoc()[0] == index, retrieved)
            for patch in patches:
                slice_idx, x, y, dim = patch.getLoc()
                # compute overlap
                overlap = patch.getOverlap()
                
                # add bounding box green if overlap > 0 and red if overlap == 0
                if overlap > 0:
                    cv.rectangle(image, (y, x), (y + dim, x + dim), (0, 255, 0), 5)
                else:
                    cv.rectangle(image, (y, x), (y + dim, x + dim), (255, 0, 0), 5)
            ax.imshow(image)
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(f"results/retrieved/{encoder}_{input_dir}_{index}.png", bbox_inches='tight', pad_inches=0)
            plt.close()

        # plot positive query patches
        for i, pos_query in enumerate(pos_queries):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            slice_idx, x, y, dim = pos_query[0].getLoc()
            patch = io.imread(raw_files[slice_idx])[x:x+dim, y:y+dim]
            # convert to rgb
            patch = cv.cvtColor(patch, cv.COLOR_GRAY2RGB)
            # add green border
            cv.rectangle(patch, (0, 0), (dim, dim), (0, 255, 0), 5)
            ax.imshow(patch)
            ax.axis("off")
            plt.savefig(f"results/retrieved/{input_dir}_pos_{i}.png", bbox_inches="tight", pad_inches=0)
            plt.close()

        # plot negative query patches
        for i, neg_query in enumerate(neg_queries):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            slice_idx, x, y, dim = neg_query[0].getLoc()
            patch = io.imread(raw_files[slice_idx])[x:x+dim, y:y+dim]
            # convert to rgb
            patch = cv.cvtColor(patch, cv.COLOR_GRAY2RGB)
            # add red border
            cv.rectangle(patch, (0, 0), (dim, dim), (255, 0, 0), 5)
            ax.imshow(patch)
            ax.axis("off")
            plt.savefig(f"results/retrieved/{input_dir}_neg_{i}.png", bbox_inches="tight", pad_inches=0)
            plt.close()
