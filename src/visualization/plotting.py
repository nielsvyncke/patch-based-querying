"""Plotting and visualization utilities.

This module provides functions for creating precision-recall curves and
other visualizations.

Author: Niels Vyncke
"""

import os
import matplotlib.pyplot as plt


def plot_pr_curve(precisions, recalls, dataset, batch_size, save_path=None):
    """Plot precision-recall curve.
    
    Args:
        precisions (list): Precision values
        recalls (list): Recall values
        dataset (str): Dataset name for title
        batch_size (int): Batch size for title
        save_path (str, optional): Directory to save plot
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
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/PR_{dataset}_{batch_size}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'PR_{dataset}_{batch_size}.png', dpi=300, bbox_inches='tight')
    
    plt.close()
