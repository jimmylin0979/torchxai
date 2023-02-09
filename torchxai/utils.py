import torch
import numpy as np
import pickle
from typing import Optional, List, Dict

def load_pickle(filepath: str):
    """
    Return a numpy ndarray instance of pickled file

    Args:
        filepath (str): the location of selected pickle files 

    Returns:
        features (np.ndarray)
    """

    with open(filepath, "rb") as fr:
        features = pickle.load(fr)
    return features

def load_pt(filepath: str):
    """
    Return a list of numpy ndarray instance of a saved pytorch state_dict file 

    Args:
        filepath (str): the location of selected files 

    Returns:
        features (List[np.ndarray]): list of np.ndarray
    """

    features = []
    for i in range(len(filepath)):
        pass
    return features

def extract_features(model: torch.nn.Module, 
                     dataloader: torch.utils.data.DataLoader, 
                     device: torch.device, 
                     exclude_layers: List[str]) -> Dict[str, np.ndarray]:
    """
    Extract and collect the output features of intermidate layers, 
        return the collection in the format of dictionary

    Args:
        model (torch.nn.Module): the model inherited from torch.nn.Module class
        dataloader (torch.utils.data.DataLoader): the input images
        device (torch.device): torch.device, cpu or cuda device
        exclude_layers (List[str]): the layers you want to exclude during the extraction, e.g., you might do not want to intermidate feature of some certain layer 

    Returns:
        features (Dict[str, np.ndarray]): a dictionary with (layer_name, layer_features) as key-value pair
    """

    # Insert hooks to model in order to catch intermidate layer output
    features = {}
    layer2name = {}
    def hook_fn(module, input, output):
        """
        _summary_

        Args:
            module (_type_): _description_
            input (_type_): _description_
            output (_type_): _description_
        """
        out = output.clone().detach().numpy()
        name = layer2name[module]
        if features[name] is None:
            features[name] = out
        else:
            features[name] = np.concatenate((features[name], out), axis=0)
        
    hooks = {}
    for name, module in model.named_modules():
        
        # TODO: Filter
        # # Only insert hook to certain layer, such as convolution layer
        # if not isinstance(module, torch.nn.Conv2d):
        #     continue

        layer2name[module] = name
        features[name] = None
        hooks[name] = module.register_forward_hook(hook_fn)
    
    # Start model inference
    for batch_idx, batch in enumerate(dataloader):
        x, _ = batch
        x = x.to(device)
        _ = model(x)

    # Remove hook to clear memory cost on it
    for name in hooks:
        hooks[name].remove()

    return features

# 
if __name__ == "__main__":
    
    # Test: extract_features
    from torchvision import models, transforms, datasets

    device = torch.device("cpu")
    model = models.resnet18(pretrained=True)
    model = model.to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    # take subset for testing 
    train_indices = [i for i in range(100)]
    dataset = torch.utils.data.Subset(dataset, train_indices)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    features = extract_features(model, dataloader, device, exclude_layers=None)
    # for name in features.keys():
    #     if features[name] is not None:
    #         print(name)

    print(features['conv1'].shape)
    print(features['layer1'].shape)
