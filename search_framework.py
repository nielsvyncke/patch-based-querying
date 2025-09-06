"""Patch-based querying framework for electron microscopy data.

This module implements a patch-based similarity search system for identifying
structures of interest in electron microscopy images using autoencoder embeddings.

The framework includes:
- Patch extraction and encoding using pre-trained autoencoders
- Nearest neighbor search using Annoy indexing
- Precision-recall evaluation metrics
- Visualization of retrieved patches

Author: Niels Vyncke
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from skimage import io, transform
import cv2 as cv
import torch
from annoy import AnnoyIndex
from models.ae import AE
from models.vae import BetaVAE
import time
import matplotlib.pyplot as plt
from tabulate import tabulate

class PatchInfoRecord:
    """Record containing metadata for an image patch.
    
    Stores location information, overlap with target structure,
    and computed similarity score for a patch.
    
    Attributes:
        slice (int): Index of the image slice
        x (int): X coordinate of patch top-left corner
        y (int): Y coordinate of patch top-left corner
        dim (int): Patch dimensions (assumed square)
        overlap (float): Overlap ratio with target structure [0, 1]
        similarity (float): Computed similarity score
    """
    
    def __init__(self, slice_idx, x, y, dim, overlap):
        """Initialize patch record.
        
        Args:
            slice_idx (int): Index of the image slice
            x (int): X coordinate of patch top-left corner
            y (int): Y coordinate of patch top-left corner
            dim (int): Patch dimensions (square patches)
            overlap (float): Overlap ratio with target structure [0, 1]
        """
        self.slice = slice_idx
        self.x = x
        self.y = y
        self.dim = dim
        self.overlap = overlap
        self.similarity = None
    
    def getOverlap(self):
        """Get overlap ratio with target structure.
        
        Returns:
            float: Overlap ratio [0, 1]
        """
        return self.overlap

    def getLoc(self):
        """Get patch location information.
        
        Returns:
            tuple: (slice_idx, x, y, dim)
        """
        return (self.slice, self.x, self.y, self.dim)
    
    def add_similarity(self, similarity):
        """Set similarity score for this patch.
        
        Args:
            similarity (float): Computed similarity score
        """
        self.similarity = similarity

    def get_similarity(self):
        """Get similarity score.
        
        Returns:
            float: Similarity score (None if not computed)
        """
        return self.similarity

class PatchInfoList:
    """Container for managing collections of patch records.
    
    Provides methods for adding, removing, and querying patch records,
    as well as sorting by similarity scores.
    
    Attributes:
        recordList (list): List of PatchInfoRecord objects
    """
    
    def __init__(self):
        """Initialize empty patch list."""
        self.recordList = []
    
    def addRecord(self, record):
        """Add a patch record to the list.
        
        Args:
            record (PatchInfoRecord): Patch record to add
        """
        self.recordList.append(record)

    def removeRecord(self, index):
        """Remove a patch record by index.
        
        Args:
            index (int): Index of record to remove
        """
        self.recordList.pop(index)

    def getRecord(self, index):
        """Get patch record by index.
        
        Args:
            index (int): Index of record to retrieve
            
        Returns:
            PatchInfoRecord: The requested patch record
        """
        return self.recordList[index]
    
    def getEncodings(self, slice_idx):
        """Get all records from a specific slice.
        
        Args:
            slice_idx (int): Index of the slice
            
        Returns:
            list: List of PatchInfoRecord objects from the slice
        """
        return [record for record in self.recordList if record.slice == slice_idx]

    def getLength(self):
        """Get number of records in the list.
        
        Returns:
            int: Number of patch records
        """
        return len(self.recordList)
    
    def __iter__(self):
        """Iterate over records yielding (index, overlap) tuples.
        
        Yields:
            tuple: (index, overlap) for each record
        """
        return iter([(index, record.getOverlap()) for index, record in enumerate(self.recordList)])
    
    def mostSimilar(self, percent):
        """Get top percentage of most similar patches.
        
        Args:
            percent (float): Percentage of top patches to return (0-100)
            
        Returns:
            list: Top percentage of patches sorted by similarity
        """
        percent = percent / 100
        # Sort by similarity in descending order
        self.recordList.sort(key=lambda record: record.get_similarity(), reverse=True)
        # Return top percent
        return self.recordList[:int(len(self.recordList) * percent)]
    
    def writeToFile(self, filename):
        """Write patch records to CSV file.
        
        Args:
            filename (str): Output CSV filename
        """
        with open(filename, "w") as f:
            # Write header
            f.write("slice,x,y,dim,similarity\n")
            for record in self.recordList:
                f.write(f"{record.slice},{record.x},{record.y},{record.dim},{record.similarity}\n")

class SearchTree:
    """Approximate nearest neighbor search tree using Annoy indexing.
    
    Provides efficient similarity search for high-dimensional vectors
    using Euclidean distance metric.
    
    Attributes:
        tree (AnnoyIndex): Annoy index for nearest neighbor search
        dim (int): Dimensionality of vectors
        items (list): List of (record, vector) tuples
        index (int): Current index counter
        build (bool): Whether the tree has been built
    """
    
    def __init__(self, dim=32):
        """Initialize search tree.
        
        Args:
            dim (int): Dimensionality of vectors to index (default: 32)
        """
        self.tree = AnnoyIndex(dim, 'euclidean')
        self.dim = dim
        self.items = []
        self.index = 0
        self.build = False
    
    def resetTree(self):
        """Reset and rebuild the search tree with current items."""
        self.tree = AnnoyIndex(self.dim, 'euclidean')
        for index, item in enumerate(self.items):
            self.tree.add_item(index, item[1])
        self.build = False

    def addVector(self, vector, record):
        """Add a vector to the search tree.
        
        Args:
            vector (np.ndarray): Feature vector to add
            record (PatchInfoRecord): Associated patch record
        """
        if self.build:
            self.resetTree()
        self.items.append((record, vector))
        self.tree.add_item(self.index, vector)
        self.index += 1

    def queryVector(self, vector, num):
        """Query for nearest neighbors of a vector.
        
        Args:
            vector (np.ndarray): Query vector
            num (int): Number of nearest neighbors to return
            
        Returns:
            list: Distances to nearest neighbors
        """
        if not self.build:
            self.tree.build(750)  # Build with 750 trees for good accuracy
            self.build = True
        return self.tree.get_nns_by_vector(vector, num, include_distances=True)[1]

def loadModel(name="ae"):
    """Load pre-trained autoencoder model with weights.
    
    Loads either an Autoencoder (AE) or Variational Autoencoder (VAE) 
    with pre-trained weights from the weights directory.
    
    Args:
        name (str): Model name, should contain 'ae' or 'vae' (default: "ae")
        
    Returns:
        torch.nn.Module: Loaded model in evaluation mode on CUDA device
        
    Raises:
        NotImplementedError: If model name is not recognized
    """
    weights_dir = 'weights'

    if "ae" in name.lower() and not "vae" in name.lower():
        weights_path = os.path.join(weights_dir, f'{name}.pth.tar')
        model = AE(32, activation_str='relu')
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))['state_dict'])
    
    elif "vae" in name.lower():
        weights_path = os.path.join(weights_dir, f'{name}.pth.tar')
        model = BetaVAE(32, activation_str='relu')
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))['state_dict'])    
    else:
        raise NotImplementedError(("Model '{}' is not a valid model. " +
            "Argument 'name' must be in ['ae', 'vae'].").format(name))
    
    return model.cuda().eval()

def encodeImage(patch, model):
    """Encode an image patch using a pre-trained autoencoder.
    
    Resizes the patch to 64x64, converts to tensor format, and computes
    the latent encoding using the provided model.
    
    Args:
        patch (np.ndarray): Input image patch
        model (torch.nn.Module): Pre-trained AE or VAE model
        
    Returns:
        np.ndarray: Flattened latent encoding vector
    """
    # Resize patch to 64x64
    patch = transform.resize(patch, (64, 64))
    # Convert to tensor format (batch_size=1, channels=1, height, width)
    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().cuda()
    # Encode patch (VAE returns mean, logvar, sample - we use mean)
    encoding = model.encode(patch) if isinstance(model, AE) else model.encode(patch)[1]
    # Return flattened encoding as numpy array
    return encoding.detach().cpu().numpy().flatten()

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

def get_single_queries(query_idxs, model, raw_files, label_files, structure=1, dims=[80], min_overlap=0.5, latent_size=32):
    """Select single positive and negative query patches for similarity search.
    
    Constructs query trees and randomly selects one positive and one negative
    example with valid patch dimensions for use as query examples.
    
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
        tuple: (pos_search_tree, neg_search_tree) with single query examples
    """
    pos_search_tree, neg_search_tree = construct_query_trees(query_idxs, model, raw_files, label_files, structure, dims, min_overlap, latent_size)

    # Select random valid queries with correct dimensions
    dim = dims[0]
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
            break

    # Create new search trees with single queries
    pos_search_tree = SearchTree(latent_size)
    neg_search_tree = SearchTree(latent_size)
    pos_search_tree.addVector(pos_query[1], pos_query[0])
    neg_search_tree.addVector(neg_query[1], neg_query[0])

    return pos_search_tree, neg_search_tree

def retrieved_patches_single_query(dataset, structure=1, num_neighbors=1, dims=[80], min_overlap=0.5):
    """Retrieve and visualize patches using single query examples.
    
    Performs patch-based similarity search using single positive and negative
    query examples, then visualizes the top retrieved patches with color-coded borders.
    
    Args:
        dataset (str): Dataset name (e.g., 'EMBL', 'EPFL', 'VIB')
        structure (int): Label value for target structure (default: 1)
        num_neighbors (int): Number of nearest neighbors for search (default: 1)
        dims (list): Patch dimensions to extract (default: [80])
        min_overlap (float): Minimum overlap threshold for positive examples (default: 0.5)
        
    Returns:
        None: Saves visualization results to results/retrieved/ directory
    """
    input_dir = dataset
    dataset_size = 50
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

        query_idxs = range(num_slices//2 - 1, num_slices//2 + 2)

        pos_search_tree, neg_search_tree = get_single_queries(query_idxs, model, raw_files, label_files, structure, dims, min_overlap, latent_size)
        pos_query = pos_search_tree.items[0]
        neg_query = neg_search_tree.items[0]
        
        # Get encodings for all patches
        all_patches = PatchInfoList()

        for slice_idx in range(0, num_slices, 3):
            if slice_idx in query_idxs:
                continue

            data_img = io.imread(raw_files[slice_idx])
            label_img = io.imread(label_files[slice_idx])

            for dim in dims:
                for x in range(0, data_img.shape[0], dim):
                    for y in range(0, data_img.shape[1], dim):
                        # Calculate overlap
                        label = label_img[x:x+dim, y:y+dim]
                        overlap = np.mean(label == structure)
                        # Add to encodings
                        record = PatchInfoRecord(slice_idx, x, y, dim, overlap)
                        all_patches.addRecord(record)

                        # Compute similarity
                        patch = data_img[x:x+dim, y:y+dim]
                        patch_encoding = encodeImage(patch, model)
                        # Query search tree
                        pos_dist = pos_search_tree.queryVector(patch_encoding, num_neighbors)
                        neg_dist = neg_search_tree.queryVector(patch_encoding, num_neighbors)
                        # Compute similarity
                        similarity = 1/np.exp(np.mean(pos_dist)) - 1/np.exp(np.mean(neg_dist))
                        record.add_similarity(similarity)

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
                
            if len(retrieved) == 10:
                break
        print("Retrieved", len(retrieved), "patches")
        # plot retrieved patches in single row
        fig, axs = plt.subplots(1, len(retrieved), figsize=(len(retrieved) * 1.1, 1.1))
        for ax, record in zip(axs, retrieved):
            slice_idx, x, y, dim = record.getLoc()
            # compute overlap
            overlap = record.getOverlap()
            patch = io.imread(raw_files[slice_idx])[x:x+dim, y:y+dim]
            # convert to rgb
            patch = cv.cvtColor(patch, cv.COLOR_GRAY2RGB)
            # add green border if overlap > 0 and red border if overlap == 0
            if overlap > 0:
                cv.rectangle(patch, (0, 0), (dim, dim), (0, 255, 0), 5)
            else:
                cv.rectangle(patch, (0, 0), (dim, dim), (255, 0, 0), 5)
            ax.imshow(patch)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"results/retrieved/{encoder}_{input_dir}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # plot positive query patch
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
        plt.savefig(f"results/retrieved/{input_dir}_pos.png", bbox_inches="tight", pad_inches=0)
        plt.close()

        # plot negative query patch
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
        plt.savefig(f"results/retrieved/{input_dir}_neg.png", bbox_inches="tight", pad_inches=0)
        plt.close()


def run_search_pipeline(input_dir, encoder, batch_size, structure=1, num_neighbors=1, dims=[80], min_overlap=0.5):
    """
    Run the search pipeline for a given configuration
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

    # Get encodings for all patches
    all_patches = PatchInfoList()

    for slice_idx in range(num_slices):
        if slice_idx in query_idxs:
            continue

        data_img = io.imread(raw_files[slice_idx])
        label_img = io.imread(label_files[slice_idx])

        for dim in dims:
            for x in range(0, data_img.shape[0], dim):
                for y in range(0, data_img.shape[1], dim):
                    # Calculate overlap
                    label = label_img[x:x+dim, y:y+dim]
                    overlap = np.mean(label == structure)
                    # Add to encodings
                    record = PatchInfoRecord(slice_idx, x, y, dim, overlap)
                    all_patches.addRecord(record)

                    # Compute similarity
                    patch = data_img[x:x+dim, y:y+dim]
                    patch_encoding = encodeImage(patch, model)
                    # Query search tree
                    pos_dist = pos_search_tree.queryVector(patch_encoding, num_neighbors)
                    neg_dist = neg_search_tree.queryVector(patch_encoding, num_neighbors)
                    # Compute similarity
                    similarity = 1/np.exp(np.mean(pos_dist)) - 1/np.exp(np.mean(neg_dist))
                    record.add_similarity(similarity)

    # Create output directory and save results
    output_dir = os.path.join("data", input_dir, f"{encoder}_{batch_size}_{dims[0]}_{structure}_{num_neighbors}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to file
    all_patches.writeToFile(os.path.join(output_dir, "similarities.csv"))
    
    return all_patches, query_idxs

def calculate_precision_recall(df, dataset_dir, query_idxs, structure=1):
    """
    Calculate precision and recall from similarity results
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
    """
    Compute average precision from precision-recall curve
    """
    average_precision = 0
    for i in range(1, len(precisions)):
        average_precision += (recalls[i] - recalls[i-1]) * precisions[i]
    return average_precision

def plot_pr_curve(precisions, recalls, dataset, batch_size, save_path=None):
    """
    Plot precision-recall curve for finetuned AE
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, linewidth=2)
    
    # Add percentage points
    points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]
    for point in points:
        idx = int(len(precisions) * (1-point))
        if idx < len(recalls) and idx < len(precisions):
            plt.scatter(recalls[idx], precisions[idx], color='red', s=30)
            plt.text(recalls[idx], precisions[idx], f'{int(point*100)}%', fontsize=9, ha='right')

    # Set axis range from 0 to 1
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title(f'Precision-Recall Curve - {dataset} (Batch Size: {batch_size})')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/PR_{dataset}_{batch_size}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'PR_{dataset}_{batch_size}.png', dpi=300, bbox_inches='tight')
    
    # plt.show()
    plt.close()

def main():
    # Experiment configuration (from experiments.sh)
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
        
        # Check if dataset exists - use original path structure
        dataset_dir = f'images/{input_dir}/labels'
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory not found: {dataset_dir}")
            # Try alternative path structure
            dataset_dir = os.path.join("images", input_dir, "labels")
            if not os.path.exists(dataset_dir):
                print(f"Alternative dataset directory not found: {dataset_dir}")
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
        results_df = pd.DataFrame(results_table, columns=headers)
        results_df.to_csv("results/evaluation_results.csv", index=False)
        print(f"\nResults saved to: results/evaluation_results.csv")

        # save result to LATEX
        with open("results/evaluation_results.tex", "w") as f:
            f.write(tabulate(results_table, headers=headers, tablefmt="latex"))
            print(f"\nResults saved to: results/evaluation_results.tex")
    else:
        print("No results to display. Check dataset paths and model weights.")

def multiple_queries_experiment(dataset="VIB", structure=1, num_neighbors=7, dims=[85], 
                               num_query_slices=25, min_overlap=0.3, encoder="ae_finetuned", 
                               num_pos_queries=100, num_neg_queries=100):
    """Run multiple queries experiment similar to examples2.py functionality.
    
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
    import math
    import matplotlib.pyplot as plt
    
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
    
    # Process example slice
    all_patches = PatchInfoList()
    
    for slice_idx in example_idx:
        if slice_idx in query_idxs:
            continue
        print(f"Processing example slice {slice_idx}...")
        
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
                    
                    # Compute similarity
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
    # plt.title(f"Query Slice {query_idxs[0]} - Green: Positive, Red: Negative")
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
    # plt.title(f"Retrieved Patches on Slice {example_idx[0]} - Green: Structure, Red: Background")
    plt.savefig(f"results/multiple_queries/retrieved_{dataset}_{encoder}.png", 
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    print(f"Results saved to results/multiple_queries/")
    print(f"Query visualization: query_slice_{dataset}_{encoder}.png")
    print(f"Retrieved patches: retrieved_{dataset}_{encoder}.png")

if __name__ == "__main__":
    if sys.argv[1] == "run_search":
        main()
    elif sys.argv[1] == "single_query":
        retrieved_patches_single_query("VIB", dims=[80])
        retrieved_patches_single_query("EMBL", dims=[100])
        retrieved_patches_single_query("EPFL", dims=[80])
    elif sys.argv[1] == "multiple_queries":
        # Run multiple queries experiments on different datasets
        multiple_queries_experiment("VIB", dims=[85], encoder="ae_finetuned")
        multiple_queries_experiment("VIB", dims=[85], encoder="vae_finetuned")
