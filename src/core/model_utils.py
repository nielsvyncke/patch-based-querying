"""Model loading and encoding utilities.

This module provides functions for loading pre-trained models and encoding
image patches using autoencoders.

Author: Niels Vyncke
"""

import os
import torch
from skimage import transform
from src.models.ae import AE
from src.models.vae import BetaVAE


def loadModel(name="ae"):
    """Load pre-trained autoencoder model with weights.
    
    Loads either an Autoencoder (AE) or Variational Autoencoder (VAE) 
    with pre-trained weights from the weights directory.
    
    Args:
        name (str): Model name, should contain 'ae' or 'vae' (default: "ae")
        
    Returns:
        torch.nn.Module: Loaded model in evaluation mode on CUDA device
        
    Raises:
        NotImplementedError: If model name is not recognized
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
    """Encode an image patch using a pre-trained autoencoder.
    
    Resizes the patch to 64x64, converts to tensor format, and computes
    the latent encoding using the provided model.
    
    Args:
        patch (np.ndarray): Input image patch
        model (torch.nn.Module): Pre-trained AE or VAE model
        
    Returns:
        np.ndarray: Flattened latent encoding vector
    """
    # Resize patch to 64x64
    patch = transform.resize(patch, (64, 64))
    # Convert to tensor format (batch_size=1, channels=1, height, width)
    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().cuda()
    # Encode patch (VAE returns mean, logvar, sample - we use mean)
    encoding = model.encode(patch) if isinstance(model, AE) else model.encode(patch)[1]
    # Return flattened encoding as numpy array
    return encoding.detach().cpu().numpy().flatten()
