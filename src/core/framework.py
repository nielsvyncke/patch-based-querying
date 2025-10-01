"""Main SearchFramework class for patch-based querying.

This module contains the core SearchFramework class that orchestrates
the similarity search process.

Author: Niels Vyncke
"""

import os
import numpy as np
from skimage import io
from .patch_info import PatchInfoRecord, PatchInfoList
from .model_utils import encodeImage


class SearchFramework:
    """Main search framework for patch-based querying.
    
    This class encapsulates the core functionality for performing similarity search
    on electron microscopy data using positive and negative query examples.
    
    Attributes:
        pos_search_tree (SearchTree): Tree containing positive query examples
        neg_search_tree (SearchTree): Tree containing negative query examples
        model (torch.nn.Module): Pre-trained encoder model
        latent_size (int): Dimensionality of latent space
    """
    
    def __init__(self, pos_search_tree, neg_search_tree, model=None, latent_size=32):
        """Initialize the search framework.
        
        Args:
            pos_search_tree (SearchTree): Tree with positive query examples
            neg_search_tree (SearchTree): Tree with negative query examples
            model (torch.nn.Module, optional): Pre-trained encoder model
            latent_size (int): Dimensionality of latent space (default: 32)
        """
        self.pos_search_tree = pos_search_tree
        self.neg_search_tree = neg_search_tree
        self.model = model
        self.latent_size = latent_size
    
    def search(self, dataset_name, structure=1, dims=[80], num_neighbors=1, num_slices=None):
        """Perform similarity search on a dataset.
        
        Args:
            dataset_name (str): Name of the dataset to search
            structure (int): Label value for target structure (default: 1)
            dims (list): Patch dimensions to extract (default: [80])
            num_neighbors (int): Number of nearest neighbors for search (default: 1)
            num_slices (int, optional): Number of slices to process (default: all)
            
        Returns:
            PatchInfoList: List of all patches with computed similarities
        """
        # Set up dataset directories
        raw_dir = os.path.join("images", dataset_name, "raw")
        label_dir = os.path.join("images", dataset_name, "labels")
        
        if not os.path.exists(raw_dir) or not os.path.exists(label_dir):
            raise FileNotFoundError(f"Dataset directories not found for {dataset_name}")
        
        raw_files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".png")])
        label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".png")])
        
        if num_slices is not None:
            raw_files = raw_files[:num_slices]
            label_files = label_files[:num_slices]
        
        # Process all patches
        all_patches = PatchInfoList()
        
        for slice_idx in range(len(raw_files)):
            print(f"Processing slice {slice_idx + 1}/{len(raw_files)}")
            
            data_img = io.imread(raw_files[slice_idx])
            label_img = io.imread(label_files[slice_idx])
            
            for dim in dims:
                for x in range(0, data_img.shape[0], dim):
                    for y in range(0, data_img.shape[1], dim):
                        # Calculate overlap
                        label = label_img[x:x+dim, y:y+dim]
                        overlap = np.mean(label == structure)
                        
                        # Create record
                        record = PatchInfoRecord(slice_idx, x, y, dim, overlap)
                        all_patches.addRecord(record)
                        
                        # Compute similarity
                        patch = data_img[x:x+dim, y:y+dim]
                        patch_encoding = encodeImage(patch, self.model)
                        
                        # Query search trees
                        pos_dist = self.pos_search_tree.queryVector(patch_encoding, num_neighbors)
                        neg_dist = self.neg_search_tree.queryVector(patch_encoding, num_neighbors)
                        
                        # Compute similarity score
                        similarity = 1/np.exp(np.mean(pos_dist)) - 1/np.exp(np.mean(neg_dist))
                        record.add_similarity(similarity)
        
        return all_patches
    
    def search_single_slice(self, dataset_name, slice_idx, structure=1, dims=[80], num_neighbors=1):
        """Perform similarity search on a single slice.
        
        Args:
            dataset_name (str): Name of the dataset
            slice_idx (int): Index of the slice to search
            structure (int): Label value for target structure (default: 1)
            dims (list): Patch dimensions to extract (default: [80])
            num_neighbors (int): Number of nearest neighbors for search (default: 1)
            
        Returns:
            PatchInfoList: List of patches from the slice with computed similarities
        """
        # Set up dataset directories
        raw_dir = os.path.join("images", dataset_name, "raw")
        label_dir = os.path.join("images", dataset_name, "labels")
        
        if not os.path.exists(raw_dir) or not os.path.exists(label_dir):
            raise FileNotFoundError(f"Dataset directories not found for {dataset_name}")
        
        raw_files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".png")])
        label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".png")])
        
        if slice_idx >= len(raw_files):
            raise IndexError(f"Slice index {slice_idx} out of range for dataset {dataset_name}")
        
        # Process single slice
        slice_patches = PatchInfoList()
        
        data_img = io.imread(raw_files[slice_idx])
        label_img = io.imread(label_files[slice_idx])
        
        for dim in dims:
            for x in range(0, data_img.shape[0], dim):
                for y in range(0, data_img.shape[1], dim):
                    # Calculate overlap
                    label = label_img[x:x+dim, y:y+dim]
                    overlap = np.mean(label == structure)
                    
                    # Create record
                    record = PatchInfoRecord(slice_idx, x, y, dim, overlap)
                    slice_patches.addRecord(record)
                    
                    # Compute similarity
                    patch = data_img[x:x+dim, y:y+dim]
                    patch_encoding = encodeImage(patch, self.model)
                    
                    # Query search trees
                    pos_dist = self.pos_search_tree.queryVector(patch_encoding, num_neighbors)
                    neg_dist = self.neg_search_tree.queryVector(patch_encoding, num_neighbors)
                    
                    # Compute similarity score
                    similarity = 1/np.exp(np.mean(pos_dist)) - 1/np.exp(np.mean(neg_dist))
                    record.add_similarity(similarity)
        
        return slice_patches
