# torchxai
Visualzation methods that help developers to realize the deep network   

## Getting Start

```bash
# clone the repo
git clone https://github.com/jimmylin0979/torchxai.git
cd torchxai

# install requirements and CVNets package
pip install -r requirements.txt
pip install --editable .
```

<!-- ```bash
pip install torchxai
``` -->

## Visualization Method

### 1. Centered Kernel Alignement (CKA)

Reference: ICML 2019 Similarity of Neural Network Representations Revisited

CKA can reveal pathology in neural networks representations.  
According to this fetaure, it's recommend to use CKA to analyze whether two networks acts similar, or use to reveal how the layer interacts with each other in one network.


Please have a look on example at [`example/CKA.ipynb`](./example/CKA.ipynb)

### 2. Loss Landscape

Reference: NIPS 2018 Visualizing the Loss Landscape of Neural Nets

Topographically map of the loss function in the parameter space.  
The generated loss landscape can help visualize the optimization process, identify local minima and saddle points, and understand the generalization ability of the model.

Please have a look on example at [`example/Landscape.ipynb`](./example/Landscape.ipynb)

### 3. Receptive Field

Reference: NIPS 2016 Understanding the Effective Receptive Field in Deep Convolutional Neural Networks

Receptive field refers to the portion of an input image that a neuron in a CNN is sensitive to. In other words, it represents the area of the input image that contributes to the activation of a particular neuron.  
The receptive field of a neuron can be visualized as a region in the input image that, when activated, causes the neuron to fire.

Please have a look on example at [`example/ReceptiveField.ipynb`](./example/ReceptiveField.ipynb)

## RoadMap

- [x] Finish setup.py, and register on pypi platform 
- [x] Allow project to install with `-e` flags in `pip install` command 
- [x] Automatic deploy with Github Actions
- [x] Method CKA
    - [x] Linear-based CKA
    - [ ] Kernel-based CKA
- [x] Method Loss Landscape
- [x] Method Receptive Field
- [ ] Method CAM

## Reference

1. [CKA-Centered-Kernel-Alignment](https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment)
2. [loss-landscapes](https://github.com/marcellodebernardi/loss-landscapes)
