"""Evaluation utilities for precision-recall analysis.

This module provides functions for calculating precision-recall metrics
and average precision scores.

Author: Niels Vyncke
"""

import os
import numpy as np
import pandas as pd
from skimage import io
import cv2 as cv


def calculate_precision_recall(df, dataset_dir, query_idxs, structure=1):
    """Calculate precision and recall from similarity results.
    
    Args:
        df: DataFrame or PatchInfoList with similarity results
        dataset_dir (str): Path to dataset label directory
        query_idxs (list): Indices of query slices to exclude
        structure (int): Label value for target structure (default: 1)
        
    Returns:
        tuple: (precisions, recalls) lists
    """
    precisions = []
    recalls = []
    
    # Get total number of structures
    total_structures = 0
    for index, slice in enumerate(sorted(os.listdir(dataset_dir))):
        if index in query_idxs:
            continue
        img = io.imread(os.path.join(dataset_dir, slice))
        total_structures += np.sum(img == structure)

    true_positives = 0
    false_positives = 0
    structures_detected = 0
    masks = {}
    
    # Sort by similarity
    df_sorted = df.sort_values(by='similarity', ascending=False) if isinstance(df, pd.DataFrame) else df
    
    # Iterate over patches
    for i, (_, row) in enumerate(df_sorted.iterrows() if isinstance(df_sorted, pd.DataFrame) else enumerate(df_sorted.recordList)):
        if isinstance(df_sorted, pd.DataFrame):
            slice_idx = int(row['slice'])
            x = int(row['x'])
            y = int(row['y'])
            dim = int(row['dim'])
        else:
            slice_idx, x, y, dim = row.getLoc()

        # Open slice
        slice_path = os.path.join(dataset_dir, sorted(os.listdir(dataset_dir))[slice_idx])
        img = io.imread(slice_path)
        
        # Check if mask exists
        if slice_idx not in masks:
            masks[slice_idx] = img == structure

        mask = masks[slice_idx]
        
        num_labels, labels_img = cv.connectedComponents((mask).astype(np.uint8))
        # Get image patch
        patch = img[x:x+dim, y:y+dim]
        # Compute overlap
        overlap = np.mean(patch == structure)
        if overlap > 0:
            true_positives += 1
        else:
            false_positives += 1
            
        # Iterate over structures
        for label in range(1, num_labels):
            if (labels_img == label)[x:x+dim, y:y+dim].any():
                structures_detected += np.sum((labels_img == label))
                masks[slice_idx][labels_img == label] = False
                
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives)
        recall = structures_detected / total_structures
        precisions.append(precision)
        recalls.append(recall)
        
    return precisions, recalls


def compute_average_precision(precisions, recalls):
    """Compute average precision from precision-recall curve.
    
    Args:
        precisions (list): Precision values
        recalls (list): Recall values
        
    Returns:
        float: Average precision score
    """
    average_precision = 0
    for i in range(1, len(precisions)):
        average_precision += (recalls[i] - recalls[i-1]) * precisions[i]
    return average_precision
