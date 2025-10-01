"""Multiple queries experiment for robust similarity search.

This module implements experiments using multiple positive and negative
query patches for more robust similarity search.

Author: Niels Vyncke
"""

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2 as cv

from src.core import SearchFramework, SearchTree, PatchInfoRecord, PatchInfoList, loadModel, encodeImage


def multiple_queries_experiment(dataset="VIB", structure=1, num_neighbors=7, dims=[85], 
                               num_query_slices=25, min_overlap=0.3, encoder="ae_finetuned", 
                               num_pos_queries=100, num_neg_queries=100):
    """Run multiple queries experiment for robust similarity search.
    
    Uses multiple positive and negative query patches for more robust similarity search,
    with denser patch sampling and visualization of results on full images.
    
    Args:
        dataset (str): Dataset name (default: "VIB")
        structure (int): Target structure label (default: 1)
        num_neighbors (int): Number of nearest neighbors (default: 7)
        dims (list): Patch dimensions (default: [85])
        num_query_slices (int): Number of query slices to consider (default: 25)
        min_overlap (float): Minimum overlap for positive examples (default: 0.3)
        encoder (str): Model name (default: "ae_finetuned")
        num_pos_queries (int): Number of positive query patches (default: 100)
        num_neg_queries (int): Number of negative query patches (default: 100)
    """
    print(f"Running multiple queries experiment on {dataset}")
    
    # Set up directories
    raw_dir = os.path.join("images", dataset, "raw")
    label_dir = os.path.join("images", dataset, "labels")
    
    if not os.path.exists(raw_dir) or not os.path.exists(label_dir):
        print(f"Warning: Dataset directories not found for {dataset}")
        return None
    
    raw_files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".png")])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".png")])
    num_slices = len(raw_files)
    latent_size = 32
    
    # Load model
    model = loadModel(encoder)
    
    # Determine query slice indices
    if num_query_slices == 1:
        query_idxs = [0]
    else:
        offset = (num_slices-1) / (num_query_slices-1)
        query_idxs = [math.ceil(index * offset) for index in range(num_query_slices)]
    
    # Randomly select 1 query slice and 1 example slice
    random.seed(42)  # Preserve randomness from original
    query_idxs = [random.choice(query_idxs)]
    example_idx = [query_idxs[0] + 7] if query_idxs[0] + 7 < num_slices else [query_idxs[0] - 7]
    
    # Extract query patches with denser sampling
    queries = PatchInfoList()
    
    for slice_idx in query_idxs:
        print(f"Processing query slice {slice_idx}...")
        label_img = io.imread(label_files[slice_idx])
        
        for dim in dims:
            # Use overlapping stride (dim//4) for denser coverage
            for x in range(0, label_img.shape[0], dim//4):
                for y in range(0, label_img.shape[1], dim//4):
                    # Calculate overlap
                    label = label_img[x:x+dim, y:y+dim]
                    # Skip if patch is not square
                    if label.shape[0] != label.shape[1] or label.shape[0] != dim or label.shape[1] != dim:
                        continue
                    overlap = np.mean(label == structure)
                    record = PatchInfoRecord(slice_idx, x, y, dim, overlap)
                    queries.addRecord(record)
    
    # Collect positive and negative queries
    pos_queries = []
    neg_queries = []
    
    for index, overlap in queries:
        if index % 100 == 0:
            print(f"Processing query {index} of {queries.getLength()}...")
        
        record = queries.getRecord(index)
        (slice_idx, x, y, dim) = record.getLoc()
        patch = io.imread(raw_files[slice_idx])[x:x+dim, y:y+dim]
        
        # Encode patch
        patch_encoding = encodeImage(patch, model)
        
        if overlap > min_overlap:
            pos_queries.append((patch_encoding, record))
        elif overlap == 0:
            neg_queries.append((patch_encoding, record))
    
    # Subsample to specified number of queries
    random.seed(42)  # Preserve randomness
    if len(pos_queries) > num_pos_queries:
        pos_queries = random.sample(pos_queries, num_pos_queries)
    if len(neg_queries) > num_neg_queries:
        neg_queries = random.sample(neg_queries, num_neg_queries)
    
    print(f"Selected {len(pos_queries)} positive and {len(neg_queries)} negative queries")
    
    # Build search trees
    pos_search_tree = SearchTree(dim=latent_size)
    neg_search_tree = SearchTree(dim=latent_size)
    
    for patch_encoding, record in pos_queries:
        pos_search_tree.addVector(patch_encoding, record)
    for patch_encoding, record in neg_queries:
        neg_search_tree.addVector(patch_encoding, record)
    
    # Create SearchFramework instance
    framework = SearchFramework(pos_search_tree, neg_search_tree, model, latent_size)
    
    # Process example slice using framework
    all_patches = PatchInfoList()
    
    for slice_idx in example_idx:
        if slice_idx in query_idxs:
            continue
        print(f"Processing example slice {slice_idx}...")
        
        # Use framework to search single slice with denser sampling
        data_img = io.imread(raw_files[slice_idx])
        label_img = io.imread(label_files[slice_idx])
        
        for dim in dims:
            for x in range(0, data_img.shape[0], dim//4):
                for y in range(0, data_img.shape[1], dim//4):
                    # Calculate overlap
                    label = label_img[x:x+dim, y:y+dim]
                    # Skip if patch is not square
                    if label.shape[0] != label.shape[1] or label.shape[0] != dim or label.shape[1] != dim:
                        continue
                    overlap = np.mean(label == structure)
                    record = PatchInfoRecord(slice_idx, x, y, dim, overlap)
                    all_patches.addRecord(record)
                    
                    # Compute similarity using framework's encoding method
                    patch = data_img[x:x+dim, y:y+dim]
                    patch_encoding = encodeImage(patch, model)
                    
                    # Query search trees
                    pos_dist = pos_search_tree.queryVector(patch_encoding, num_neighbors)
                    neg_dist = neg_search_tree.queryVector(patch_encoding, num_neighbors)
                    
                    # Compute similarity score
                    similarity = 1/np.exp(np.mean(pos_dist)) - 1/np.exp(np.mean(neg_dist))
                    record.add_similarity(similarity)
    
    print(f"Processed {len(all_patches.recordList)} patches")
    
    # Get most similar patches
    most_similar = all_patches.mostSimilar(100)  # Top 100%
    most_similar = most_similar[:25]  # Take top 25
    
    # Create output directory
    os.makedirs("results/multiple_queries", exist_ok=True)
    
    # Visualize query slice
    fig = plt.figure(figsize=(12, 8))
    query_img = io.imread(raw_files[query_idxs[0]])
    # Convert to RGB
    query_img = query_img[..., np.newaxis].repeat(3, axis=2)
    
    # Mark positive queries in green, negative in red
    for patch_encoding, record in pos_queries:
        (slice_idx, x, y, dim) = record.getLoc()
        cv.rectangle(query_img, (y, x), (y+dim, x+dim), (0, 255, 0), 2)
    for patch_encoding, record in neg_queries:
        (slice_idx, x, y, dim) = record.getLoc()
        cv.rectangle(query_img, (y, x), (y+dim, x+dim), (255, 0, 0), 2)
    
    plt.axis("off")
    plt.imshow(query_img)
    plt.savefig(f"results/multiple_queries/query_slice_{dataset}.png", 
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    # Visualize most similar patches on example slice
    fig = plt.figure(figsize=(12, 8))
    data_img = io.imread(raw_files[example_idx[0]])
    # Convert to RGB
    data_img = data_img[..., np.newaxis].repeat(3, axis=2)
    
    # Mark retrieved patches
    for record in most_similar:
        (slice_idx, x, y, dim) = record.getLoc()
        overlap = record.getOverlap()
        if overlap > 0:
            cv.rectangle(data_img, (y, x), (y+dim, x+dim), (0, 255, 0), 2)  # Green for structure
        else:
            cv.rectangle(data_img, (y, x), (y+dim, x+dim), (255, 0, 0), 2)  # Red for background
    
    plt.axis("off")
    plt.imshow(data_img)
    plt.savefig(f"results/multiple_queries/retrieved_{dataset}_{encoder}.png", 
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    print(f"Results saved to results/multiple_queries/")
    print(f"Query visualization: query_slice_{dataset}.png")
    print(f"Retrieved patches: retrieved_{dataset}_{encoder}.png")
