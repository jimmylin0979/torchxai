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

<!-- ### 2. Loss Landscape

Reference: NIPS 2018 Visualizing the Loss Landscape of Neural Nets

Loss landscape 

Please have a look on example at [`example/Landscape.ipynb`](./example/Landscape.ipynb) -->


## RoadMap

- [x] Finish setup.py, and register on pypi platform 
- [x] Allow project to install with `-e` flags in `pip install` command 
- [x] Automatic deploy with Github Actions
- [x] Method CKA
    - [x] Linear-based CKA
    - [ ] Kernel-based CKA
- [ ] Method Loss Landscape
- [ ] Method Receptive Field

## Reference

1. [CKA-Centered-Kernel-Alignment](https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment)