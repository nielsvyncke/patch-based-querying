import math
import os
import random
from skimage import io, transform
import numpy as np
import torch
from annoy import AnnoyIndex
from models.ae import AE
from models.vae import BetaVAE
import argparse
import json
import shutil
import time

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
            self.tree.build(750)
            self.build = True
        return self.tree.get_nns_by_vector(vector, num, include_distances=True)[1]

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="EMBL/")
parser.add_argument("--structure", type=int, default=1)
parser.add_argument("--num_neighbors", type=int, default=1)
parser.add_argument("--dims", nargs='+', type=int, default=[80])
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--min_overlap", type=float, default=0.5)
parser.add_argument("--encoder", type=str, default="ae_finetuned")
parser.add_argument("--no_eval", action="store_true", default=True)

# get arguments
args = parser.parse_args()
structure = args.structure
num_neighbors = args.num_neighbors
dims = args.dims
print(dims)
batch_size = args.batch_size
encoder = args.encoder

input_dir = "images/" + args.input_dir
output_dir = "data/" + args.input_dir

raw_dir = os.path.join(input_dir, "raw")
label_dir = os.path.join(input_dir, "labels")

raw_files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".png")])
label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".png")])

num_slices = len(raw_files)

latent_size = 32
activation_str = "relu"

def loadModel(name="ae"):
    """
        Loads the model with its precomputed parameters.
    """

    weights_dir = 'weights'

    if "ae" in name.lower() and not "vae" in name.lower():
        weights_path = os.path.join(weights_dir, f'{name}.pth.tar')
        model = AE(32)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))['state_dict'])
    
    elif "vae" in name.lower():
        weights_path = os.path.join(weights_dir, f'{name}.pth.tar')
        model = BetaVAE(32)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))['state_dict'])    
    else:
        raise NotImplementedError(("Model '{}' is not a valid model. " +
            "Argument 'name' must be in ['ae', 'vae'].").format(name))

    return model.cuda().eval()

# load model
model = loadModel(encoder)

# define function to encode image patch
def encodeImage(patch):
    # resize patch to 64x64
    patch = transform.resize(patch, (64, 64))
    # convert to tensor
    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().cuda()
    # encode patch
    encoding = model.encode(patch) if isinstance(model, AE) else model.encode(patch)[1]
    # return encoding
    return encoding.detach().cpu().numpy().flatten()

# encode all patches
queries = PatchInfoList()

# select middle slice from every batch
print(f"Batch size: {batch_size}")
query_idxs = [batch_size * i + batch_size // 2 for i in range(num_slices // batch_size)]
# take the middle of the remaining slices
if num_slices % batch_size != 0:
    query_idxs.append(num_slices - (num_slices % batch_size) // 2 - 1)
print(f"Query slices: {query_idxs}")


# start timer
start = time.time()
for slice_idx in query_idxs:
    print(f"Processing slice {slice_idx}...")
    # get label image
    label_img = io.imread(label_files[slice_idx])
    for dim in dims:
        for x in range(0, label_img.shape[0], dim):
            for y in range(0, label_img.shape[1], dim):
                # calculate overlap
                label = label_img[x:x+dim, y:y+dim]
                overlap = np.mean(label == structure)
                # add to encodings
                record = PatchInfoRecord(slice_idx, x, y, dim, overlap)
                queries.addRecord(record)

# add query patches to search tree
pos_search_tree = SearchTree(dim=latent_size)
neg_search_tree = SearchTree(dim=latent_size)
slice_idx_prev = -1
for index, overlap in queries:
    if index % 100 == 0:
        print(f"Processing query {index} of {queries.getLength()}...")
    # get patch
    record = queries.getRecord(index)
    (slice_idx, x, y, dim) = record.getLoc()
    if slice_idx != slice_idx_prev:
        slice = io.imread(raw_files[slice_idx])
    patch = slice[x:x+dim, y:y+dim]

    # encode patch
    patch_encoding = encodeImage(patch)
    # add to search tree
    if overlap > args.min_overlap:
        pos_search_tree.addVector(patch_encoding, record)
    elif overlap == 0:
        neg_search_tree.addVector(patch_encoding, record)
    slice_idx_prev = slice_idx

# get encodings for all patches
all_patches = PatchInfoList()

for slice_idx in range(num_slices):
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

# get most similar patches
# most_similar = all_patches.mostSimilar(100*(1-args.reduction))

# end timer
end = time.time()
print(f"Time elapsed: {end - start}")

os.makedirs(output_dir, exist_ok=True)

# remove output directory if it exists
output_dir = os.path.join(output_dir, f"{args.encoder}_{batch_size}_{dims[0]}_{structure}_{num_neighbors}")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# create output directory
os.makedirs(output_dir, exist_ok=True)

# add time elapsed to args
args.time_elapsed = end - start

# write to file
all_patches.writeToFile(os.path.join(output_dir, "all_patches.csv"))
# write argparser arguments to file
with open(os.path.join(output_dir, "args.json"), "w") as f:
    json.dump(vars(args), f)


# # copy raw files
# raw_output_dir = os.path.join(output_dir, "raw")
# os.makedirs(raw_output_dir, exist_ok=True)
# false_negatives = 0
# for idx in range(num_slices):
#     if idx in query_idxs:
#         continue
#     print(f"Processing slice {idx}...")
    
#     # mark matches on slices
#     data_img = io.imread(raw_files[idx])
#     output_img = io.imread(raw_files[idx])[..., np.newaxis].repeat(3, axis=2)
#     label_img = io.imread(label_files[idx])

#     # detect blobs of structure
#     num_labels, labels_img = cv.connectedComponents((label_img == structure).astype(np.uint8))
#     mask = np.zeros_like(labels_img)

#     for record in most_similar:
#         (slice_idx, x, y, dim) = record.getLoc()
#         if slice_idx != idx:
#             continue
#         mask[x:x+dim, y:y+dim] = 1
#         patch = data_img[x:x+dim, y:y+dim]
#         dimX, dimY = patch.shape
#         overlap = record.getOverlap()
        
#         if overlap > 0:
#             cv.rectangle(output_img, (y, x), (y+dimY, x+dimX), (0, 255, 0), 2)
#         else:
#             cv.rectangle(output_img, (y, x), (y+dimY, x+dimX), (255, 0, 0), 2)

#     for label in range(1, num_labels):
#         if np.sum(mask[labels_img == label]) > 0:
#             output_img[labels_img == label] = [0, 255, 0]
#         else:
#             false_negatives += 1
#             output_img[labels_img == label] = [255, 0, 0]
#     if not args.no_eval:
#         io.imsave(os.path.join(raw_output_dir, f"{idx}.png"), output_img)
# print(f"False negatives: {false_negatives}")
