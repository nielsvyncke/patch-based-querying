"""Patch information classes for managing patch metadata and collections.

This module contains classes for storing and managing patch information
including location, overlap, and similarity scores.

Author: Niels Vyncke
"""


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
