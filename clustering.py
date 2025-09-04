import sys
import os
import numpy as np
import torch
from models import ae, vae
from skimage.transform import resize
import tqdm
import imageio.v2 as imageio
from sklearn.manifold import TSNE as TS
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import ConnectionPatch


def openImage(name):
    """Open an image file and return a handle to it."""
    try:
        return imageio.imread(name)
    except IOError:
        print("Cannot open", name)
        sys.exit(1)
        
def loadModel(name="ae"):
    """
        Loads the model with its precomputed parameters.
    """

    weights_dir = 'weights'

    if "ae" in name.lower() and not "vae" in name.lower():
        weights_path = os.path.join(weights_dir, f'{name}.pth.tar')
        model = ae.AE(32)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))['state_dict'])
    
    elif "vae" in name.lower():
        weights_path = os.path.join(weights_dir, f'{name}.pth.tar')
        model = vae.BetaVAE(32)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))['state_dict'])    
    else:
        raise NotImplementedError(("Model '{}' is not a valid model. " +
            "Argument 'name' must be in ['ae', 'vae'].").format(name))

    return model.cuda().eval()

def getEncoding(descr, patch):
    """
        Computes the encoding of a given patch with a given descriptor.
    """
    
    if isinstance(descr, ae.AE) or isinstance(descr, vae.BetaVAE):
        patch = np.array(resize(patch, (65, 65)))
        patch = np.expand_dims(np.expand_dims(patch, axis=0), axis=0)
        patch = torch.from_numpy(patch).float().cuda()

        if isinstance(descr, vae.BetaVAE):
            _, patch_encoding, _ = descr.encode(patch)

        else:
            patch_encoding = descr.encode(patch)

        patch_encoding = patch_encoding.detach().cpu().numpy()
        patch_encoding = patch_encoding.reshape(patch_encoding.shape[0], np.prod(patch_encoding.shape[1:]))

    else:
        raise NotImplementedError(("Argument 'descr' is not a valid descriptor" +
                "Argument 'descr' must be of type 'AE', 'BetaVAE' but type {} was given.").format(descr.__class__))

    return patch_encoding[0]

def label_binary(labels):
    if np.mean(labels == 0) == 1:
        return 0
    elif np.mean(labels == 1) > 0.5 and np.mean(labels > 1) == 0:
        return 1
    else:
        return -1

def label_multi(labels):
    if np.mean(labels == 0) == 1:
        return -1
    elif np.mean(labels == 1) > 0.75 and np.mean(labels > 1) == 0:
        return 1 
    elif np.mean(labels == 2) > 0.75 and np.mean((labels == 1) + (labels > 2)) == 0:
        return 2
    elif np.mean(labels == 3) > 0.75 and np.mean((labels == 1) + (labels == 2) + (labels > 3)) == 0:
        return 3
    elif np.mean(labels == 4) > 0.75 and np.mean((labels == 1) + (labels == 2) + (labels == 3) + (labels > 4)) == 0:
        return 4
    elif np.mean(labels == 5) > 0.75 and np.mean((labels == 1) + (labels == 2) + (labels == 3) + (labels == 4) + (labels > 5)) == 0:
        return 5
    else:
        return -1

def encode_dataset(dataset, model='ae', stride=1, dims=[64], start=0, end=10, binary=True):
    path = 'images/' + dataset + '/'
    data_path = 'raw/'
    labels_path = 'labels/'
    modelstr = model
    
    # load model
    model = loadModel(model)
    
    for index, image in enumerate(os.listdir(path + data_path)[start:end]):
        data = []
        if os.path.exists('data/data_{}_{}_{}_{}_{}.csv'.format(dataset, modelstr, '_'.join(map(str, dims)), image.split('.')[0], stride)):
            continue
        print(image)
        # open image data and labels
        image_data = np.asarray(openImage(path + data_path + image))
        image_labels = np.asarray(openImage(path + labels_path + image))
        
        for dim in dims:
            for i in tqdm.tqdm(range(0, image_data.shape[0] - dim, stride), position=0, leave=False):
                for j in tqdm.tqdm(range(0, image_data.shape[1] - dim, stride), position=1, leave=False):
                    new_entry = [index, dim, i, j]
                    encoding = getEncoding(model, image_data[i:i+dim, j:j+dim])
                    
                    labels = image_labels[i:i+dim, j:j+dim]
                    
                    label = label_binary(labels) if binary else label_multi(labels)
                    if label == -1:
                        continue
                    new_entry.append(label)
                    new_entry += list(encoding)
                    data.append(new_entry)
                    
        with open('data/data_{}_{}_{}_{}_{}.csv'.format(dataset, modelstr, '_'.join(map(str, dims)), image.split('.')[0], stride), 'w') as f:
            np.savetxt(f, np.array(data), delimiter=';')

def load_data(dataset, model='ae', stride=1, dims=[64], start=0, end=10):
    path = 'images/' + dataset + '/'
    data_path = 'raw/'
    data = []
    for dim in dims:
        for image in os.listdir(path + data_path)[start:end]:
            data.append(np.loadtxt('data/data_{}_{}_{}_{}_{}.csv'.format(dataset, model, '_'.join(map(str, dims)), image.split('.')[0], stride), delimiter=';'))
    return np.concatenate(data)

def plot(data, dataset, model='ae', start=0, end=10, show_image=True, binary=True):
    path = 'images/' + dataset + '/'
    data_path = 'raw/'

    if not binary:
        N = 7500  # Number of samples per class
        unique_labels = np.unique(data[:, 4])
        print(unique_labels)
        sampled_data = []

        for label in unique_labels:
            label_indices = np.where(data[:, 4] == label)[0]
            print(label, len(label_indices))
            if len(label_indices) > N:
                sampled_indices = np.random.choice(label_indices, N, replace=False)
            else:
                sampled_indices = label_indices
            sampled_data.append(data[sampled_indices])

        data = np.concatenate(sampled_data, axis=0)
    
    y = data[:,4]
    X = data[:,5:]
    ts = TS(n_components=2, learning_rate='auto', n_iter=1000, verbose=3)
    X_ts = ts.fit_transform(X)

    if show_image:
        # select samples
        if binary:
            # red samples
            red = np.argwhere((data[:,4] == 0)*(data[:,0] == 0)).flatten()
            samples_red = np.random.choice(red, 2, replace=False)
            
            # green samples
            green = np.argwhere((data[:,4] == 1)*(data[:,0] == 0)).flatten()
            samples_green = np.random.choice(green, 2, replace=False)
        
            samples = np.concatenate([samples_red, samples_green])
        else:
            samples = []
            labels = np.unique(y)
            for label in labels:
                label_indices = np.where((y == label) & (data[:, 0] == 0))[0]
                if len(label_indices) > 2:
                    samples.extend(np.random.choice(label_indices, 2, replace=False))
                else:
                    samples.extend(label_indices)
        fig = plt.figure()
        sfigs = fig.subfigures(1,2)
        ax1 = sfigs[0].add_subplot()
        plt.xticks([], [])
        plt.yticks([], [])
        ax2 = sfigs[1].add_subplot()
        plt.axis('off')
        image = openImage(path + data_path+os.listdir(path + data_path)[0])
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for index, sample in enumerate(samples):
            if binary:
                sample_data = X_ts[sample]
                ax1.scatter(*sample_data, color='k',s=20)
                sample_loc = data[sample,1:4].flatten().astype('int')
                cv2.rectangle(image, (sample_loc[2], sample_loc[1]), (sample_loc[2]+sample_loc[0], sample_loc[1]+sample_loc[0]), ((index <2)*255,(index >=2)*255,0), 5)
                
                con = ConnectionPatch(xyA=sample_loc[:0:-1], xyB=sample_data, coordsA="data", coordsB="data",
                            axesA=ax2, axesB=ax1, color="black")
                            
            else:
                sample_data = X_ts[sample]
                ax1.scatter(*sample_data, color='k',s=20)
                sample_loc = data[sample,1:4].flatten().astype('int')
                if y[sample] == 1:
                    cv2.rectangle(image, (sample_loc[2], sample_loc[1]), (sample_loc[2]+sample_loc[0], sample_loc[1]+sample_loc[0]), (0,255,0), 5)
                elif y[sample] == 2:
                    cv2.rectangle(image, (sample_loc[2], sample_loc[1]), (sample_loc[2]+sample_loc[0], sample_loc[1]+sample_loc[0]), (255,255,0), 5)
                elif y[sample] == 3:
                    cv2.rectangle(image, (sample_loc[2], sample_loc[1]), (sample_loc[2]+sample_loc[0], sample_loc[1]+sample_loc[0]), (255,0,255), 5)
                elif y[sample] == 4:
                    cv2.rectangle(image, (sample_loc[2], sample_loc[1]), (sample_loc[2]+sample_loc[0], sample_loc[1]+sample_loc[0]), (0,0,0), 5)
                elif y[sample] == 5:
                    cv2.rectangle(image, (sample_loc[2], sample_loc[1]), (sample_loc[2]+sample_loc[0], sample_loc[1]+sample_loc[0]), (0,0,255), 5)
                else:
                    cv2.rectangle(image, (sample_loc[2], sample_loc[1]), (sample_loc[2]+sample_loc[0], sample_loc[1]+sample_loc[0]), (255,0,0), 5)
        
                con = ConnectionPatch(xyA=sample_loc[:0:-1], xyB=sample_data, coordsA="data", coordsB="data",
                            axesA=ax2, axesB=ax1, color="black")
                            
            ax2.add_artist(con)
        ax2.imshow(image)
        
        print(np.unique(np.where(y == 1, 'g', np.where(y == 2, 'y',  np.where(y == 3, 'm',  np.where(y == 4, 'k', np.where(y == 5, 'b', 'r')))))))
        ax1.scatter(X_ts[:, 0], X_ts[:, 1], c=np.where(y == 1, 'g', np.where(y == 2, 'y',  np.where(y == 3, 'm', np.where(y == 4, 'k', np.where(y == 5, 'b', 'r'))))), s=2)
        
        plt.show()
    else:
        plt.scatter(X_ts[:, 0], X_ts[:, 1], c=np.where(y == 1, 'g', np.where(y == 2, 'y',  np.where(y == 3, 'm', np.where(y == 4, 'k', np.where(y == 5, 'b', 'r'))))), s=2)
        plt.xticks([], [])
        plt.yticks([], [])
        if "vae" in model:
            plt.savefig(f'results/EMBL/vae_{start}_{end}.png')
        else:
            plt.savefig(f'results/EMBL/ae_{start}_{end}.png')
        plt.close()

def run_clustering(model='ae', dataset='EMBL', start=0, end=10, dims=[100], stride=2, show_image=True, binary=True):
    np.random.seed(1000)
    encode_dataset(dataset, model, stride=stride, dims=dims, start=start, end=end, binary=binary)
    
    data = load_data(dataset, model, stride=stride, dims=dims, start=start, end=end)
    plot(data, dataset, model, start, end, show_image, binary)

if __name__ == "__main__":
    if sys.argv[1] == "exp1":
        run_clustering(model='ae_pretrained', dataset='EMBL', start=0, end=10)
    elif sys.argv[1] == "exp2":
        run_clustering(model='vae_pretrained', dataset='EMBL', start=0, end=10)
    elif sys.argv[1] == "exp3":
        run_clustering(model='ae_finetuned', dataset='EMBL', start=0, end=10)
    elif sys.argv[1] == "exp4":
        run_clustering(model='vae_finetuned', dataset='EMBL', start=0, end=10)
    elif sys.argv[1] == "exp5":
        for i in range(0, 56, 5):
            run_clustering(model='ae_finetuned', dataset='EMBL', start=i, end=i+10, show_image=False)
        run_clustering(model='ae_finetuned', dataset='EMBL', start=0, end=65, show_image=False)
    elif sys.argv[1] == "exp6":
        for i in range(0, 56, 5):
            run_clustering(model='vae_finetuned', dataset='EMBL', start=i, end=i+10, show_image=False)
        run_clustering(model='vae_finetuned', dataset='EMBL', start=0, end=65, show_image=False)
    elif sys.argv[1] == "exp7":
        run_clustering(model='ae_finetuned', dataset='VIB', start=210, end=215, binary=False, dims=range(40,70,2), stride=4)
    elif sys.argv[1] == "exp8":
        run_clustering(model='vae_finetuned', dataset='VIB', start=210, end=215, binary=False, dims=range(40,70,2), stride=4)
