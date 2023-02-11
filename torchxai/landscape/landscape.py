"""
Loss Landscape
NIPS 2018 Visualizing the Loss Landscape of Neural Nets

Reference:
1. 
"""

import torch
import numpy as np
import logging

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, List, Tuple

def init_directions(model: torch.nn.Module) -> List[Tuple[float, float]]:
    """
    TODO
    _summary_

    Args:
        model (torch.nn.Module): _description_

    Returns:
        List[Tuple[float, float]]: _description_
    """

    noises = []

    n_params = 0
    for name, param in model.named_parameters():
        delta = torch.normal(0.0, 1.0, size=param.size())
        nu = torch.normal(0.0, 1.0, size=param.size())

        param_norm = torch.norm(param)
        delta_norm = torch.norm(delta)
        nu_norm = torch.norm(nu)

        delta /= delta_norm
        delta *= param_norm

        nu /= nu_norm
        nu *= param_norm

        noises.append((delta, nu))

        n_params += np.prod(param.size())

    return noises


def init_network(model: torch.nn.Module, all_noises: float, alpha: float, beta: float) -> torch.nn.Module:
    """
    TODO
    _summary_

    Args:
        model (torch.nn.Module): _description_
        all_noises (float): _description_
        alpha (float): _description_
        beta (float): _description_

    Returns:
        torch.nn.Module: return a initialized model
    """

    with torch.no_grad():
        for param, noises in zip(model.parameters(), all_noises):
            delta, nu = noises
            new_value = param + alpha * delta + beta * nu
            param.copy_(new_value)

    return model


def loss_landscape(model: torch.nn.Module, 
                   dataloader: torch.utils.data.DataLoader,
                   device: torch.device,
                   criterion: torch.nn.Module = torch.nn.CrossEntropyLoss(), 
                   resolution: int = 25,
                   verbose: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Search loss landscape of model around one local minimal, and construct a loss surface around it,
        the function will search (resolution x resolution) area, the bigger the resolution is, the longer it will take.
    Return x, y, loss_score points of loss surface in the format of np.ndarray 

    Args:
        model (torch.nn.Module): the model inherited from torch.nn.Module class
        dataloader (torch.utils.data.DataLoader): the input images
        device (torch.device): torch.device, cpu or cuda device
        criterion (torch.nn.Module): criterion / loss function. Defaults to torch.nn.CrossEntropyLoss().
        resolution (int, optional): resolution. Defaults to 25.
        verbose (int, optional): Controls the verbosity when fitting and predicting. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: x, y, loss_score points of loss surface
    """

    # # Set default criterion function to cross entropy loss
    # if criterion is None:
    #     criterion = torch.nn.CrossEntropyLoss()
    
    # Declare space for x, y, z 
    X, Y = np.meshgrid(np.linspace(-1, 1, resolution),
                    np.linspace(-1, 1, resolution), indexing='ij')
    loss_surface = np.empty_like(X)

    # Initialize directions of model
    noises = init_directions(model)
    
    # Calculate loss landscape
    pbar = tqdm(total=resolution * resolution)
    for i in range(resolution):
        for j in range(resolution):
            
            total_loss = 0.0
            n_batch = 0
            alpha = X[i, j]
            beta = Y[i, j]
            
            net = init_network(model, noises, alpha, beta).to(device)
            for batch, labels in dataloader:
                batch = batch.to(device)
                labels = labels.to(device)
                batch_size = batch.shape[0]
                with torch.no_grad():
                    preds = net(batch)
                    loss = criterion(preds, labels)
                    total_loss += loss.item()
                    n_batch += 1
            loss_surface[i, j] = total_loss / (n_batch * batch_size)
            del net, batch, labels

            pbar.update(1)
            if verbose == 1:
                print(f"alpha : {alpha:.2f}, beta : {beta:.2f}, loss : {loss_surface[i, j]:.2f}")
            torch.cuda.empty_cache()
    
    # 
    np.save(f'x.npy', X)
    np.save(f'y.npy', Y)
    np.save(f'z.npy', loss_surface)

    return X, Y, loss_surface

def visualize(x: np.ndarray, y: np.ndarray, loss_surface: np.ndarray) -> None:
    """
    Visualization of loss landscape, and save figure locally 

    Args:
        x (np.ndarray): 
        y (np.ndarray): 
        loss_surface (np.ndarray): 
    """
    loss_surface = np.log(loss_surface)
    plt.figure(figsize=(10, 10))
    plt.contour(x, y, loss_surface)
    plt.savefig(f"loss_landscape.png", dpi=100)
    plt.close()

#
if __name__ == "__main__":
    pass