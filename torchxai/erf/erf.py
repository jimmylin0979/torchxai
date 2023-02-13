"""
Effective Receptive Field (ERF)
NIPS 2016 Understanding the Effective Receptive Field in Deep Convolutional Neural Networks

Reference:
1. 
"""

import torch
import torchvision
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, List, Tuple

def ERF(model: torch.nn.Module, 
        image: np.ndarray, 
        device: torch.device, 
        hook_layers: List[torch.nn.Module]):
    """
    _summary_

    Args:
        model (torch.nn.Module): _description_
        image (np.ndarray): _description_
        device (torch.device): _description_
        hook_layers (List[torch.nn.Module]): _description_

    Returns:
        _type_: _description_
    """
    
    # 
    featureMaps = []
    def hook_fn(module, input, output):
        featureMaps.append(output)

    # 
    image = torchvision.transforms.functional.to_tensor(image)
    input_c, input_h, input_w = image.shape

    heatmap = np.zeros([len(hook_layers), input_h, input_w])
    for heatmap_idx, layer in enumerate(hook_layers):
        
        #    
        input_tensor = image.unsqueeze(0)
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True
        # print(input_tensor.requires_grad, input_tensor.grad)

        #
        featureMaps = list()
        hook = layer.register_forward_hook(hook_fn)
        _ = model(input_tensor)

        #
        featureMap = featureMaps[0].mean(dim=1, keepdim=False).squeeze()
        center_x, center_y = (featureMap.shape[0] // 2 - 1, featureMap.shape[1] // 2 - 1)
        input_tensor.retain_grad()
        featureMap[center_x][center_y].backward(retain_graph=True)

        #
        # print(input_tensor.requires_grad, input_tensor.grad)
        grad = torch.abs(input_tensor.grad)
        grad = grad.mean(dim=1, keepdim=False).squeeze()
        heatmap[heatmap_idx] = heatmap[heatmap_idx] + grad.cpu().numpy()

        hook.remove()

    # 
    return heatmap

def visualize(heatmap: np.ndarray):
    """
    Visualization of ERF heatmap 

    Args:
        heatmap (np.ndarray): _description_
    """

    plt.imshow(heatmap, cmap='viridis')
    plt.show()