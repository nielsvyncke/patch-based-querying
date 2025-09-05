import math
import os
import random
import numpy as np
import pandas as pd
from skimage import io, transform
import cv2 as cv
import torch
from annoy import AnnoyIndex
from models.ae import AE
from models.vae import BetaVAE
import argparse
import json
import shutil
import time
import matplotlib.pyplot as plt
from tabulate import tabulate

# Define PatchInfoRecord
class PatchInfoRecord:
    def __init__(self, slice, x, y, dim, overlap):
        self.slice = slice
        self.x = x
        self.y = y
        self.dim = dim
        self.overlap = overlap
        self.similarity = None
    
    def getOverlap(self):
        return self.overlap

    def getLoc(self):
        return (self.slice, self.x, self.y, self.dim)
    
    def add_similarity(self, similarity):
        self.similarity = similarity

    def get_similarity(self):
        return self.similarity

# Define PatchInfoList
class PatchInfoList:
    def __init__(self):
        self.recordList = []
    
    def addRecord(self, record):
        self.recordList.append(record)

    def removeRecord(self, index):
        self.recordList.pop(index)

    def getRecord(self, index):
        return self.recordList[index]
    
    def getEncodings(self, slice_idx):
        return [record for record in self.recordList if record.slice == slice_idx]

    def getLength(self):
        return len(self.recordList)
    
    def __iter__(self):
        return iter([(index, record.getOverlap()) for index, record in enumerate(self.recordList)])
    
    def mostSimilar(self, percent):
        percent = percent / 100
        # sort by similarity
        self.recordList.sort(key=lambda record: record.get_similarity(), reverse=True)
        # get top percent
        return self.recordList[:int(len(self.recordList) * percent)]
    
    def writeToFile(self, filename):
        # write as csv
        with open(filename, "w") as f:
            # write header
            f.write("slice,x,y,dim,similarity\n")
            for record in self.recordList:
                f.write(f"{record.slice},{record.x},{record.y},{record.dim},{record.similarity}\n")

# Define SearchTree
class SearchTree:
    def __init__(self, dim=32):
        self.tree = AnnoyIndex(dim, 'euclidean')
        self.dim = dim
        self.items = []
        self.index = 0
        self.build = False
    
    def resetTree(self):
        self.tree = AnnoyIndex(self.dim, 'euclidean')
        for index, item in enumerate(self.items):
            self.tree.add_item(index, item[1])
        self.build = False

    def addVector(self, vector, record):
        if self.build:
            self.resetTree()
        self.items.append((record, vector))
        self.tree.add_item(self.index, vector)
        self.index += 1

    def queryVector(self, vector, num):
        if not self.build:
            self.tree.build(750)
            self.build = True
        return self.tree.get_nns_by_vector(vector, num, include_distances=True)[1]

def loadModel(name="ae"):
    """
    Loads the model with its precomputed parameters.
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
    # resize patch to 64x64
    patch = transform.resize(patch, (64, 64))
    # convert to tensor
    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().cuda()
    # encode patch
    encoding = model.encode(patch) if isinstance(model, AE) else model.encode(patch)[1]
    # return encoding
    return encoding.detach().cpu().numpy().flatten()

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
    
    # Encode all patches
    queries = PatchInfoList()
    
    # Select middle slice from every batch
    query_idxs = [batch_size * i + batch_size // 2 for i in range(num_slices // batch_size)]
    # Take the middle of the remaining slices
    if num_slices % batch_size != 0:
        query_idxs.append(num_slices - (num_slices % batch_size) // 2 - 1)
    
    # Start timer
    start = time.time()
    for slice_idx in query_idxs:
        # Get label image
        label_img = io.imread(label_files[slice_idx])
        for dim in dims:
            for x in range(0, label_img.shape[0], dim):
                for y in range(0, label_img.shape[1], dim):
                    # Calculate overlap
                    label = label_img[x:x+dim, y:y+dim]
                    overlap = np.mean(label == structure)
                    # Add to encodings
                    record = PatchInfoRecord(slice_idx, x, y, dim, overlap)
                    queries.addRecord(record)

    # Add query patches to search tree
    pos_search_tree = SearchTree(dim=latent_size)
    neg_search_tree = SearchTree(dim=latent_size)
    slice_idx_prev = -1
    for index, overlap in queries:
        # Get patch
        record = queries.getRecord(index)
        (slice_idx, x, y, dim) = record.getLoc()
        if slice_idx != slice_idx_prev:
            slice = io.imread(raw_files[slice_idx])
        patch = slice[x:x+dim, y:y+dim]

        # Encode patch
        patch_encoding = encodeImage(patch, model)
        # Add to search tree
        if overlap > min_overlap:
            pos_search_tree.addVector(patch_encoding, record)
        elif overlap == 0:
            neg_search_tree.addVector(patch_encoding, record)
        slice_idx_prev = slice_idx

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

    # End timer
    end = time.time()
    print(f"Time elapsed: {end - start}")

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
    points = [0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]
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

if __name__ == "__main__":
    main()
