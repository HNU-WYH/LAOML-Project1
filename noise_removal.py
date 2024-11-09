from spyder_kernels.utils.lazymodules import scipy
from data_loader import DataLoader
from data_loader import LogisticClassifier
import scipy.sparse.linalg as la
import numpy as np
import scipy

def noise_removal(X, k, flag = "pca"):
    if flag not in ["pca", "svd"]:
        raise ValueError("flag must be either 'pca' or 'svd')")

    if flag == "pca":
        X -= X.mean(axis=0)

    U, S, Vt = la.svds(X, k=k)
    S = scipy.sparse.diags(S)
    X_approx = U @ S @ Vt

    return X_approx

if __name__ == '__main__':
    my_data = DataLoader(r".\data\data2.csv")
    X = my_data.X.astype('float64')
    X_new = noise_removal(X, 6, "svd")

    import matplotlib.pyplot as plt

    plt.plot([np.linalg.norm(X_new[i] - X[i]) for i in range(X.shape[0])])
    plt.show()