"""Core framework classes and utilities.

This module provides the main classes and functions for the patch-based
querying framework.
"""

from .patch_info import PatchInfoRecord, PatchInfoList
from .search_tree import SearchTree
from .model_utils import loadModel, encodeImage
from .framework import SearchFramework

__all__ = [
    'PatchInfoRecord',
    'PatchInfoList', 
    'SearchTree',
    'loadModel',
    'encodeImage',
    'SearchFramework'
]
