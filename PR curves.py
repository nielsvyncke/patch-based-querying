import numpy as np
import pandas as pd
from skimage import io
import cv2 as cv
import os
import argparse

parser = argparse.ArgumentParser(description='Plot PR curves for different models')
parser.add_argument('--dataset', type=str, help='Dataset to use', required=True)

# dataset is in args
dataset = parser.parse_args().dataset[:-1]

for batch_size in [10,20,30,50,100]:
    structure = 1
    dim = 80
    num_neighbors = 1

    # get number of slices
    dataset_dir = f'/scratch/multimodal/{dataset}/labels'
    num_slices = len(os.listdir(dataset_dir))

    # select middle slice from every batch
    print(f"Batch size: {batch_size}")
    query_idxs = [batch_size * i + batch_size // 2 for i in range(num_slices // batch_size)]
    # take the middle of the remaining slices
    if num_slices % batch_size != 0:
        query_idxs.append(num_slices - (num_slices % batch_size) // 2 - 1)
    print(f"Query slices: {query_idxs}")

    # ae pretrained
    ae_pretrained_file_path = f'/scratch/experiments/{dataset}/ae_False_{batch_size}_{dim}_{structure}_{num_neighbors}/all_patches.csv'
    ae_pretrained = pd.read_csv(ae_pretrained_file_path)
    ae_pretrained = ae_pretrained.sort_values(by='similarity', ascending=False)

    # ae finetuned
    ae_finetuned_file_path = f'/scratch/experiments/{dataset}/ae_True_{batch_size}_{dim}_{structure}_{num_neighbors}/all_patches.csv'
    ae_finetuned = pd.read_csv(ae_finetuned_file_path)
    ae_finetuned = ae_finetuned.sort_values(by='similarity', ascending=False)

    # vae pretrained
    vae_pretrained_file_path = f'/scratch/experiments/{dataset}/vae_False_{batch_size}_{dim}_{structure}_{num_neighbors}/all_patches.csv'
    vae_pretrained = pd.read_csv(vae_pretrained_file_path)
    vae_pretrained = vae_pretrained.sort_values(by='similarity', ascending=False)

    # vae finetuned
    vae_finetuned_file_path = f'/scratch/experiments/{dataset}/vae_True_{batch_size}_{dim}_{structure}_{num_neighbors}/all_patches.csv'
    vae_finetuned = pd.read_csv(vae_finetuned_file_path)
    vae_finetuned = vae_finetuned.sort_values(by='similarity', ascending=False)


    def calculate_precision_recall(df, dataset_dir):
        precisions = []
        recalls = []
        # get total number of structures
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
        # iterate over patches
        for patch in df.iterrows():
            row = patch[1]
            slice = int(row['slice'])
            x = int(row['x'])
            y = int(row['y'])
            dim = int(row['dim'])

            # open slice
            slice_path = os.path.join(dataset_dir, sorted(os.listdir(dataset_dir))[slice])
            img = io.imread(slice_path)
            # detect blobs
            # check if mask exists
            if slice not in masks:
                masks[slice] = img == structure

            mask = masks[slice]
            
            
            num_labels, labels_img = cv.connectedComponents((mask).astype(np.uint8))
            # get image patch
            patch = img[x:x+dim, y:y+dim]
            # compute overlap
            overlap = np.mean(patch == structure)
            if overlap > 0:
                true_positives += 1
            else:
                false_positives += 1
            # iterate over structures
            for label in range(1, num_labels):
                if (labels_img == label)[x:x+dim, y:y+dim].any():
                    structures_detected += np.sum((labels_img == label))
                    masks[slice][labels_img == label] = False
            # calculate precision and recall
            precision = true_positives / (true_positives + false_positives)
            recall = structures_detected / total_structures
            precisions.append(precision)
            recalls.append(recall)
        return precisions, recalls

    print ("Calculating precision and recall for AE Pretrained...")
    precisions_ae_pretrained, recalls_ae_pretrained = calculate_precision_recall(ae_pretrained, dataset_dir)
    print ("Calculating precision and recall for AE Finetuned...")
    precisions_ae_finetuned, recalls_ae_finetuned = calculate_precision_recall(ae_finetuned, dataset_dir)
    print ("Calculating precision and recall for VAE Pretrained...")
    precisions_vae_pretrained, recalls_vae_pretrained = calculate_precision_recall(vae_pretrained, dataset_dir)
    print ("Calculating precision and recall for VAE Finetuned...")
    precisions_vae_finetuned, recalls_vae_finetuned = calculate_precision_recall(vae_finetuned, dataset_dir)

    # check if recall goes up to 1
    print('Recall AE Pretrained:', recalls_ae_pretrained[-1])
    print('Recall AE Finetuned:', recalls_ae_finetuned[-1])
    print('Recall VAE Pretrained:', recalls_vae_pretrained[-1])
    print('Recall VAE Finetuned:', recalls_vae_finetuned[-1])

    print('plotting PR curves')
    # plot PR curves
    import matplotlib.pyplot as plt
    plt.figure()
    # plt.plot(recalls_ae_pretrained, precisions_ae_pretrained, label='AE Pretrained')
    plt.plot(recalls_ae_finetuned, precisions_ae_finetuned)
    # plt.plot(recalls_vae_pretrained, precisions_vae_pretrained, label='VAE Pretrained')
    # plt.plot(recalls_vae_finetuned, precisions_vae_finetuned, label='VAE Finetuned')

    
    points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]
    for point in points:
        idx = int(len(precisions_ae_finetuned) * (1-point))
        plt.scatter(recalls_ae_finetuned[idx], precisions_ae_finetuned[idx], color='red')
        plt.text(recalls_ae_finetuned[idx], precisions_ae_finetuned[idx], f'{int(point*100)}%', fontsize=9, ha='right')

    # set axis range from 0 to 1
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Precision-Recall curves')
    # legend left bottom
    # plt.legend(loc='lower left')
    plt.savefig(f'PR {dataset} {batch_size}.png')

    # save data for four models in a csv
    # df = pd.DataFrame({'Recall AE Pretrained': recalls_ae_pretrained, 'Precision AE Pretrained': precisions_ae_pretrained, 'Recall AE Finetuned': recalls_ae_finetuned, 'Precision AE Finetuned': precisions_ae_finetuned, 'Recall VAE Pretrained': recalls_vae_pretrained, 'Precision VAE Pretrained': precisions_vae_pretrained, 'Recall VAE Finetuned': recalls_vae_finetuned, 'Precision VAE Finetuned': precisions_vae_finetuned})
    # df.to_csv(f'PR {dataset} {batch_size}.csv')

    # plt.show()

    print('compute average precision')

    def compute_average_precision(precisions, recalls):
        average_precision = 0
        for i in range(1, len(precisions)):
            average_precision += (recalls[i] - recalls[i-1]) * precisions[i]
        return average_precision

    print('Average Precision AE Pretrained:', compute_average_precision(precisions_ae_pretrained, recalls_ae_pretrained))
    print('Average Precision AE Finetuned:', compute_average_precision(precisions_ae_finetuned, recalls_ae_finetuned))
    print('Average Precision VAE Pretrained:', compute_average_precision(precisions_vae_pretrained, recalls_vae_pretrained))
    print('Average Precision VAE Finetuned:', compute_average_precision(precisions_vae_finetuned, recalls_vae_finetuned))

    print(f'{compute_average_precision(precisions_ae_finetuned, recalls_ae_finetuned):.5f}')