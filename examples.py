import math
import os
import random
from skimage import io, transform
import numpy as np
import torch
from annoy import AnnoyIndex
from models.ae import AE
from models.vae import BetaVAE
import cv2 as cv
import argparse
import json
import shutil

# define PatchInfoRecord
# this holds the slice, x, y, dim, overlap, and entropy of a patch
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

# define PatchInfoList
# this holds a list of PatchInfoRecord    
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

# define SearchTree
# this is a wrapper for AnnoyIndex
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
            self.tree.build(100)
            self.build = True
        return self.tree.get_nns_by_vector(vector, num, include_distances=True)[1]

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="/scratch/multimodal/2017_04_07_Po93_6Q_46/")
parser.add_argument("--output_dir", type=str, default="/scratch/experiments/")
parser.add_argument("--structure", type=int, default=1)
parser.add_argument("--num_neighbors", type=int, default=3)
parser.add_argument("--dims", nargs='+', type=int, default=[110])
parser.add_argument("--num_query_slices", type=int, default=25)
parser.add_argument("--min_overlap", type=float, default=0.7)
parser.add_argument("--reduction", type=float, default=0.8)
parser.add_argument("--encoder", type=str, default="vae")
parser.add_argument("--retrained", action="store_true")
parser.add_argument("--rotations", action="store_true")
parser.add_argument("--experiment_index", type=int, default=700)
parser.add_argument("--no_eval", action="store_true")

# get arguments
args = parser.parse_args()
structure = args.structure
num_neighbors = args.num_neighbors
rotations = args.rotations
dims = args.dims
print(dims)
num_query_slices = args.num_query_slices
encoder = args.encoder
experiment_index = args.experiment_index

input_dir = args.input_dir
output_dir = args.output_dir

raw_dir = os.path.join(input_dir, "raw")
label_dir = os.path.join(input_dir, "labels")

raw_files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".png")])
label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".png")])

num_slices = len(raw_files)

weights_path = "weights_pub/{}_{}best.pth.tar".format(encoder, "retrained_" if args.retrained else "")
latent_size = 32
activation_str = "relu"

# load model
model = AE(
    latent_size = latent_size,
    activation_str = activation_str
) if encoder == "ae" else BetaVAE(
    latent_size = latent_size,
    activation_str = activation_str,
)

model.load_state_dict(torch.load(weights_path)["state_dict"])
model.eval()

# define function to encode image patch
def encodeImage(patch):
    # resize patch to 64x64
    patch = transform.resize(patch, (64, 64))
    # convert to tensor
    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float()
    # encode patch
    encoding = model.encode(patch) if isinstance(model, AE) else model.encode(patch)[0]
    # return encoding
    return encoding.detach().numpy().flatten()

# encode all patches
queries = PatchInfoList()

if num_query_slices == 1:
    query_idxs = [0]
else:
    offset = (num_slices-1) / (num_query_slices-1)
    query_idxs = [math.ceil(index * offset) for index in range(num_query_slices)]

for slice_idx in query_idxs:
    print(f"Processing slice {slice_idx}...")
    # get label image
    label_img = io.imread(label_files[slice_idx])
    for dim in dims:
        for x in range(0, label_img.shape[0], dim):
            for y in range(0, label_img.shape[1], dim):
                # calculate overlap
                label = label_img[x:x+dim, y:y+dim]
                # if patch is not square, skip
                if label.shape[0] != label.shape[1] or label.shape[0] != dim or label.shape[1] != dim:
                    continue
                overlap = np.mean(label == structure)
                # add to encodings
                record = PatchInfoRecord(slice_idx, x, y, dim, overlap)
                queries.addRecord(record)

# add query patches to search tree
pos_search_tree = SearchTree(dim=latent_size)
neg_search_tree = SearchTree(dim=latent_size)
pos_queries = []
neg_queries = []
for index, overlap in queries:
    if index % 100 == 0:
        print(f"Processing query {index} of {queries.getLength()}...")
    # get patch
    record = queries.getRecord(index)
    (slice_idx, x, y, dim) = record.getLoc()
    patch = io.imread(raw_files[slice_idx])[x:x+dim, y:y+dim]
    if rotations:
        for rot in range(4):
            # encode patch
            patch_encoding = encodeImage(np.rot90(patch, rot))
            # add to search tree
            if overlap > args.min_overlap:
                pos_search_tree.addVector(patch_encoding, record)
            elif overlap == 0:
                neg_search_tree.addVector(patch_encoding, record)
    else:
        # encode patch
        patch_encoding = encodeImage(patch)
        # add to search tree
        if overlap > args.min_overlap:
            pos_queries.append((patch_encoding, record))
            # pos_search_tree.addVector(patch_encoding, record)
        elif overlap == 0:
            neg_queries.append((patch_encoding, record))
            # neg_search_tree.addVector(patch_encoding, record)

# add random query patch to each search tree
# set random seed
random.seed(10)
pos_query = random.choice(pos_queries)
neg_query = random.choice(neg_queries)
pos_search_tree.addVector(pos_query[0], pos_query[1])
neg_search_tree.addVector(neg_query[0], neg_query[1])

# get encodings for all patches
all_patches = PatchInfoList()

for slice_idx in range(0, num_slices, 2):
    if slice_idx in query_idxs:
        continue
    print(f"Processing slice {slice_idx}...")

    data_img = io.imread(raw_files[slice_idx])
    label_img = io.imread(label_files[slice_idx])

    for dim in dims:
        for x in range(0, data_img.shape[0], dim):
            for y in range(0, data_img.shape[1], dim):
                # calculate overlap
                label = label_img[x:x+dim, y:y+dim]
                # if patch is not square, skip
                if label.shape[0] != label.shape[1] or label.shape[0] != dim or label.shape[1] != dim:
                    continue
                overlap = np.mean(label == structure)
                # add to encodings
                record = PatchInfoRecord(slice_idx, x, y, dim, overlap)
                all_patches.addRecord(record)

                # compute similarity
                patch = data_img[x:x+dim, y:y+dim]
                
                patch_encoding = encodeImage(patch)
                # query search tree
                pos_dist = pos_search_tree.queryVector(patch_encoding, num_neighbors)
                neg_dist = neg_search_tree.queryVector(patch_encoding, num_neighbors)
                # compute similarity
                similarity = 1/np.exp(np.mean(pos_dist)) - 1/np.exp(np.mean(neg_dist))
                record.add_similarity(similarity)
print(len(all_patches.recordList))
# get most similar patches
most_similar = all_patches.mostSimilar(100*(1-args.reduction))
# get most similar patches
most_similar = most_similar[:10]

# show pos and neg query and returned patches
import matplotlib.pyplot as plt

# Save the positive and negative query patches
plt.figure(figsize=(20, 40))

# Show the positive query patch in the top-left corner
plt.subplot(2, 1, 1)  # Corrected subplot to include positional arguments
pos_img = io.imread(raw_files[pos_query[1].slice])[pos_query[1].x:pos_query[1].x+pos_query[1].dim, pos_query[1].y:pos_query[1].y+pos_query[1].dim]
plt.imshow(pos_img, cmap="gray")
plt.title("Positive Query", fontsize=120)
plt.axis("off")

# Show the negative query patch below the positive query patch
plt.subplot(2, 1, 2)  # Corrected subplot to include positional arguments
neg_img = io.imread(raw_files[neg_query[1].slice])[neg_query[1].x:neg_query[1].x+neg_query[1].dim, neg_query[1].y:neg_query[1].y+neg_query[1].dim]
plt.imshow(neg_img, cmap="gray")
plt.title("Negative Query", fontsize=120)
plt.axis("off")

# Save the query patch figure
plt.tight_layout()  # Ensures proper spacing between subplots
# increase vertical spacing between subplots
plt.subplots_adjust(hspace=1)
plt.savefig(f"examples/queries.png")
plt.close()

# Save the retrieved patches
plt.figure(figsize=(20, 10))

# Show the retrieved patches in a row
for index, record in enumerate(most_similar):
    plt.subplot(1, len(most_similar), index + 1)  # Dynamically create subplots based on number of results
    slice_idx, x, y, dim = record.getLoc()
    overlap = record.getOverlap()
    patch_img = io.imread(raw_files[slice_idx])[x:x+dim, y:y+dim]
    
    # Convert grayscale image to RGB format
    patch_img_rgb = cv.cvtColor(patch_img, cv.COLOR_GRAY2RGB)
    
    # add green border if overlap > 0 and red border if overlap == 0
    if overlap > 0:
        cv.rectangle(patch_img_rgb, (0, 0), (dim, dim), (0, 255, 0), 5)
    else:
        cv.rectangle(patch_img_rgb, (0, 0), (dim, dim), (255, 0, 0), 5)

    plt.imshow(patch_img_rgb)
    plt.axis("off")

# Save the retrieved patches figure
plt.tight_layout()  # Ensures proper spacing between subplots
plt.savefig(f"examples/{encoder}.png")
# plt.show()
plt.close()