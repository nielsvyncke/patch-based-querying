"""Search tree implementation using Annoy for nearest neighbor search.

This module provides the SearchTree class for efficient similarity search
using approximate nearest neighbors.

Author: Niels Vyncke
"""

import numpy as np
from annoy import AnnoyIndex


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
