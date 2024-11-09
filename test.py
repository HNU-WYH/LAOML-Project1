from data_loader import DataLoader
import scipy.sparse.linalg as la
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

def noise_removal(X, k):
    # Whitening
    Xw = data_whitening(X, k)

    return Xw

if __name__ == '__main__':
    # read data
    my_data = DataLoader(r".\data\data2.csv")
    X = my_data.X.astype('float64')

    # data whitening
    X_new = data_whitening(X, 75)

    # compute norm
    import matplotlib.pyplot as plt
    norm_list = np.array([np.linalg.norm(X_new[i,:]) for i in range(X_new.shape[0])])
    plt.plot(norm_list)
    plt.show()

    outlier_mask = norm_list >= np.quantile(norm_list, 0.9)

    # plt.plot(outlier_mask.cumsum())
    # plt.show()

    TP = (outlier_mask[-2000:].sum())
    FP = (outlier_mask[:-2000].sum())

    TN = len(outlier_mask[:-2000]) - FP
    FN = 2000 - TP

    import seaborn as sns

    # 构建混淆矩阵
    confusion_matrix = np.array([[TN, FP],
                                 [FN, TP]])

    # 定义类别标签
    categories = ['Normal', 'Outlier']

    # 绘制混淆矩阵
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    print(f"Precision（精确率）: {Precision:.4f}")
    print(f"Recall（召回率）: {Recall:.4f}")
    print(f"F1-score: {F1_score:.4f}")


