from nltk import accuracy

from data_loader import DataLoader
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy

def data_whitening(X, k):
    # Centering
    M = X - X.mean(axis=0)

    # SVD Decomposition
    U, S, _ = la.svds(M, k=k)

    # Whitening
    Xw = U[:,::-1] @ scipy.sparse.diags(S[::-1])
    Xw = Xw / np.std(Xw, axis=0)[None,:]

    return Xw

def noise_removal(X, k, percentile = 0.9):
    # Whitening
    Xw = data_whitening(X, k)

    norm_list = np.array([np.linalg.norm(Xw[i, :]) for i in range(Xw.shape[0])])
    outlier_mask = norm_list >= np.quantile(norm_list, percentile)
    return outlier_mask


def assess_performance(outlier_mask, vis_flag=False):
    confusion_matrix, (P, R), F1 = F1_score(outlier_mask)
    acc = (confusion_matrix[0] + confusion_matrix[2]) / \
          (confusion_matrix[0] + confusion_matrix[1] + confusion_matrix[2] + confusion_matrix[3])

    if vis_flag:
        visualize_confusion(*confusion_matrix)
        print(f'Accuracy: {acc:.4f}')
        print(f"Precision: {P:.4f}")
        print(f"Recall: {R:.4f}")
        print(f"F1-score: {F1:.4f}")

    return F1, (P, R), acc

def F1_score(outlier_mask, outlier_num = 2000):
    """
    The outlier is in the last #outlier_num rows.
    """

    total_num = len(outlier_mask)

    TP = (outlier_mask[-outlier_num:].sum())
    FP = (outlier_mask[:-outlier_num].sum())

    TN = total_num - outlier_num - FP
    FN = outlier_num - TP

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_score = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0

    return (TP, FP, TN, FN), (Precision, Recall), F1_score


def visualize_confusion(TP, FP, TN, FN):
    confusion_matrix = np.array([[TN, FP],
                                 [FN, TP]])

    categories = ['Normal', 'Outlier']

    # plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_PRC(P_list, R_list):
    # data clean
    valid_indices = (~np.isnan(P_list)) & (~np.isnan(R_list)) & (~np.isinf(P_list)) & (~np.isinf(R_list))
    P_array = P_list[valid_indices]
    R_array = R_list[valid_indices]

    # sort data
    sorted_indices = np.argsort(R_array)
    R_sorted = R_array[sorted_indices]
    P_sorted = P_array[sorted_indices]

    # plot data
    plt.figure(figsize=(8, 6))
    plt.plot(R_sorted, P_sorted, linestyle='-', color='b')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # read data
    my_data = DataLoader(r".\data\data2.csv")
    X = my_data.X.astype('float64')

    # Outlier rate
    q = 1 - 2000 / X.shape[0]

    # data whitening & noise removal
    F1, _, _ = assess_performance(noise_removal(X, 81, q), vis_flag=True)


    # Uncomment below to plot the PR-Curve
    # Plot PR Curve
    P_list = []
    R_list = []
    for q in np.arange(0, 1, 0.01):
        _, (P, R), _ = assess_performance(noise_removal(X, 81, q), vis_flag=False)
        P_list.append(P)
        R_list.append(R)

    # plot PR-Curve
    plot_PRC(np.array(P_list), np.array(R_list))










