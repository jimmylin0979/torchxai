"""
Centered Keneral Alignment (CKA)
ICML 2019 Similarity of Neural Network Representations Revisited

Reference:
1. CKA-Centered-Kernel-Alignment: https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment
"""

import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, List, Dict

def HSIC(X: np.ndarray, Y: np.ndarray, method: str = "linear"):
    """
    Calculate Hilbert-Schmidt Indenoendence Criterion (HSIC) metric of matrix X, Y
    where
        K = XX^T, L = YY^T 
        HSIC(K, L) = \frac1{{(n-1)}^2\;}tr(KHLH)

    Args:
        X (np.ndarray, 2D[N, dim1]): internal representations from one layer
        Y (np.ndarray, 2D[N, dim2]): internal representations from one layer, dim1 not need to equal with dim2
        method (str): 'linear' or 'kernel'  
    """

    n = X.shape[0]

    def centering(X):
        # Calculate centered matrix HX
        # define centering matrix H = I_n - \frac1n11^T 
        H = np.eye(n) - np.ones([n, n]) / n
        return np.dot(H, X)

    if method == "linear":
        # calculate equation M = KHLK
        K = np.dot(X, X.T)
        L = np.dot(Y, Y.T)
        M = centering(K) * centering(L)
    
    elif method == "kernel":
        raise NotImplementedError

    # calculate \frac1{{(n-1)}^2}tr(KHLH)
    M = M * np.eye(n)
    return np.sum(M) / ((n - 1) ** 2)


def CKA(X: np.ndarray, Y: np.ndarray, method: str = "linear"):
    """
    Calculate Centered Kernel Alignment (CKA) metric of matrix X, Y
    X, Y should be internal representation (i.e. activation matrices) from deep neural network, 
        but can come from across two different networks or within the same network

    where 
        CKA(X, Y) = \frac{HSIC(X,\;Y)}{\sqrt{HSIC(X,\;X)HSIC(Y,\;Y)}}

    Args:
        X (np.ndarray, 2D[N, dim1]): internal representations from one layer
        Y (np.ndarray, 2D[N, dim2]): internal representations from one layer, dim1 not need to equal with dim2
        method (str): 'linear' or 'kernel'  
    """

    #
    N = X.shape[0]
    X = X.reshape((N, -1))
    Y = Y.reshape((N, -1))

    # 
    num = HSIC(X, Y, method)
    den = HSIC(X, X, method) * HSIC(Y, Y, method)
    metric = num / den
    return metric

def CKA_matrix(X: Dict[str, np.ndarray], Y: Dict[str, np.ndarray]):
    """
    Calculate the CKA correlation matrix of two lists X, Y, and return 
    X, Y should be list of internal representation (i.e. activation matrices) from deep neural network, 
        but can come from across two different networks or within the same network

    Args:
        X (list[np.ndarray]): _description_
        Y (list[np.ndarray]): _description_
    """
    
    # 
    len_X, len_Y = len(X), len(Y)
    cka_matrix = np.zeros((len_X, len_Y))
    #
    for r, kx in tqdm(enumerate(X.keys())):
        for c, ky in enumerate(Y.keys()):
            cka_score = CKA(X[kx], Y[ky])
            cka_score = abs(cka_score)
            cka_matrix[r][c] = cka_score

    return cka_matrix

def visualize(cka_matrix: np.ndarray):
    """
    Visualization of cka_matrix, and save figure locally 

    Args:
        cka_matrix (np.ndarray): the CKA correlation matrix generated from function CKA_matrix
    """
    # Vosualize the result via heatmap
    sns.set_theme()
    ax = sns.heatmap(cka_matrix, vmax=1.0, vmin=0.0, annot=False)
    ax.invert_yaxis()
    plt.savefig(f"cka.png")
    plt.show()

#  
if __name__ == "__main__":

    # linear based CKA
    # CKA
    X, Y = np.random.randn(100, 64), np.random.randn(100, 64)
    print('Linear CKA, between X and X: {}'.format(CKA(X, X)))
    print('Linear CKA, between Y and Y: {}'.format(CKA(Y, Y)))
    print('Linear CKA, between X and Y: {}'.format(CKA(X, Y)))

    # CKA_matrix
    X, Y = dict(), dict()
    for i in range(3):
        X[str(i)] = np.random.randn(100, 64)
        Y[str(i)] = np.random.randn(100, 64)
    print('Linear CKA_matrix, between X and Y:\n{}'.format(CKA_matrix(X, Y)))

    # TODO: Kernel based CKA