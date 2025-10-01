"""Query construction utilities for building positive and negative search trees.

This module provides functions for constructing query examples from labeled data.

Author: Niels Vyncke
"""

import random
import numpy as np
from skimage import io
from src.core import PatchInfoRecord, PatchInfoList, SearchTree, encodeImage


def construct_query_trees(query_idxs, model, raw_files, label_files, structure=1, dims=[80], min_overlap=0.5, latent_size=32):
    """Construct positive and negative query search trees from labeled data.
    
    Extracts patches from query slices, encodes them using the model, and builds
    separate search trees for positive (structure present) and negative (no structure) examples.
    
    Args:
        query_idxs (list): Indices of slices to use for queries
        model (torch.nn.Module): Pre-trained encoder model
        raw_files (list): Paths to raw image files
        label_files (list): Paths to label image files
        structure (int): Label value for target structure (default: 1)
        dims (list): Patch dimensions to extract (default: [80])
        min_overlap (float): Minimum overlap threshold for positive examples (default: 0.5)
        latent_size (int): Dimensionality of latent space (default: 32)
        
    Returns:
        tuple: (pos_search_tree, neg_search_tree) containing positive and negative examples
    """
    queries = PatchInfoList()
    
    # Extract all patches from query slices
    for slice_idx in query_idxs:
        label_img = io.imread(label_files[slice_idx])
        for dim in dims:
            for x in range(0, label_img.shape[0], dim):
                for y in range(0, label_img.shape[1], dim):
                    # Calculate overlap with target structure
                    label = label_img[x:x+dim, y:y+dim]
                    overlap = np.mean(label == structure)
                    record = PatchInfoRecord(slice_idx, x, y, dim, overlap)
                    queries.addRecord(record)

    # Build positive and negative search trees
    pos_search_tree = SearchTree(dim=latent_size)
    neg_search_tree = SearchTree(dim=latent_size)
    slice_idx_prev = -1
    
    for index, overlap in queries:
        record = queries.getRecord(index)
        (slice_idx, x, y, dim) = record.getLoc()
        
        # Load slice image only when needed (optimization)
        if slice_idx != slice_idx_prev:
            slice = io.imread(raw_files[slice_idx])
        patch = slice[x:x+dim, y:y+dim]

        # Encode patch and add to appropriate tree
        patch_encoding = encodeImage(patch, model)
        if overlap > min_overlap:
            pos_search_tree.addVector(patch_encoding, record)
        elif overlap == 0:
            neg_search_tree.addVector(patch_encoding, record)
        slice_idx_prev = slice_idx
        
    return pos_search_tree, neg_search_tree


def get_two_queries(query_idxs, model, raw_files, label_files, structure=1, dims=[80], min_overlap=0.5, latent_size=32):
    """Select two positive and two negative query patches for similarity search.
    
    Constructs query trees and randomly selects valid positive and negative
    examples with correct patch dimensions for use as query examples.
    
    Args:
        query_idxs (list): Indices of slices to use for queries
        model (torch.nn.Module): Pre-trained encoder model
        raw_files (list): Paths to raw image files
        label_files (list): Paths to label image files
        structure (int): Label value for target structure (default: 1)
        dims (list): Patch dimensions to extract (default: [80])
        min_overlap (float): Minimum overlap threshold for positive examples (default: 0.5)
        latent_size (int): Dimensionality of latent space (default: 32)
        
    Returns:
        tuple: (pos_search_tree, neg_search_tree) with two query examples each
    """
    pos_search_tree, neg_search_tree = construct_query_trees(query_idxs, model, raw_files, label_files, structure, dims, min_overlap, latent_size)

    # Select random valid queries with correct dimensions
    dim = dims[0]
    counter = 0
    pos_queries = []
    neg_queries = []
    while True:
        pos_query = pos_search_tree.items[random.randint(0, len(pos_search_tree.items) - 1)]
        neg_query = neg_search_tree.items[random.randint(0, len(neg_search_tree.items) - 1)]
        
        # Verify patch dimensions are correct
        pos_slice_idx, pos_x, pos_y, pos_dim = pos_query[0].getLoc()
        neg_slice_idx, neg_x, neg_y, neg_dim = neg_query[0].getLoc()
        slice_pos = io.imread(raw_files[pos_slice_idx])
        slice_neg = io.imread(raw_files[neg_slice_idx])
        patch_pos = slice_pos[pos_x:pos_x+pos_dim, pos_y:pos_y+pos_dim]
        patch_neg = slice_neg[neg_x:neg_x+neg_dim, neg_y:neg_y+neg_dim]
        
        if patch_pos.shape == (dim, dim) and patch_neg.shape == (dim, dim):
            counter += 1
            pos_queries.append(pos_query)
            neg_queries.append(neg_query)
            if counter == 2:
                break

    # Create new search trees with selected queries
    pos_search_tree = SearchTree(latent_size)
    neg_search_tree = SearchTree(latent_size)
    for pos_query in pos_queries:
        pos_search_tree.addVector(pos_query[1], pos_query[0])
    for neg_query in neg_queries:
        neg_search_tree.addVector(neg_query[1], neg_query[0])

    return pos_search_tree, neg_search_tree
