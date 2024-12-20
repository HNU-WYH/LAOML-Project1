import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import time

class DataLoader:
    def __init__(self, file_path= r".\data\data.csv"):
        self.X = None
        self.y = None
        self.num = 0

        self._read(file_path)

    def _read(self, file_path):
        ori_data = np.loadtxt(file_path, skiprows=1, delimiter=',', dtype=int)
        self.X = ori_data[:,:-1]
        self.y = ori_data[:,-1]
        self.y[self.y == 0] = -1
        self.num = self.X.shape[0]

    def split(self, ratio = 0.7):
        # the number of training data
        train_num = int(self.num * ratio)

        # shuffling the original data
        rand_idx = np.arange(self.num)
        np.random.shuffle(rand_idx)
        X = self.X[rand_idx]
        y = self.y[rand_idx]

        # split the data set
        X_train, X_test = X[:train_num], X[train_num:]
        y_train, y_test = y[:train_num], y[train_num:]

        return X_train, X_test, y_train, y_test

mydata = DataLoader(r".\data\data.csv")
print(f'there are {(mydata.y == 1).sum()} \
malicious data points in the total {mydata.num} data points')

np.random.seed(2000)
X_train, X_test, y_train, y_test = mydata.split(ratio = 0.5)

def evaluator(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    y_pred = X.dot(w)
    y_pred[y_pred < 0] = -1
    y_pred[y_pred >= 0] = 1
    return (y_pred == y).sum()
    
np.random.seed(2000)
w_init = np.random.uniform(low = -1, high = 1, size=(mydata.X.shape[1]))
pred_corr_num = evaluator(mydata.X, mydata.y, w_init)
print(f'{pred_corr_num} data points in total {mydata.num} samples are correct using random generated weights')

class LogisticClassifier:
    def __init__(self, X_train, y_train, w_init):
        self.X_train = X_train
        self.y_train = y_train
        self.w_init = w_init
        self.w = None

    def logistic_regression(self, K, alpha, lambda_reg, flag = 0):
        if flag == 0:
            self.logistic_regression_entrywise(K, alpha, lambda_reg)
        if flag == 1:
            self.logistic_regression_vectorize(K, alpha, lambda_reg)
        if flag == 2:
            self.logistic_regression_sparse(K, alpha, lambda_reg)

    def logistic_regression_entrywise(self, K, alpha, lambda_reg):
        self.w = self.w_init
        for k in range(K):
            w_grad = lambda_reg * self.w
            for i in range(self.X_train.shape[0]):
                # if k == 5:
                #     print(i)
                w_grad -=(self.y_train[i] * self.X_train[i])\
                /(1 + np.exp(self.y_train[i] * self.X_train[i].dot(self.w)))
            self.w -= w_grad * alpha
        return self.w

    def logistic_regression_vectorize(self, K, alpha, lambda_reg):
        self.w = self.w_init
        for _ in range(K):
            s = self.y_train[:,None] * self.X_train @ self.w
            z = self.y_train / (1 + np.exp(s))
            w_grad = - self.X_train.T @ z + lambda_reg * self.w
            self.w -= w_grad * alpha
        return self.w

    def logistic_regression_sparse(self, K, alpha, lambda_reg):
        self.w = self.w_init
        self.X_train = csr_matrix(self.X_train)
        for _ in range(K):
            s_product = self.X_train @ self.w
            s = self.y_train * s_product
            z = self.y_train / (1 + np.exp(s))
            w_grad = - self.X_train.T @ z + lambda_reg * self.w
            self.w -= w_grad * alpha
        return self.w

    def predict(self, X_test, y_test):
        w = self.w
        y_true =  evaluator(X_test, y_test, w)
        acc = y_true / y_test.shape[0]
        return acc

log_reg = LogisticClassifier(X_train, y_train, w_init)

start = time.time()
log_reg.logistic_regression(K=100, alpha=0.01, lambda_reg=0.01, flag= 0)
accuracy = log_reg.predict(X_test, y_test)
end = time.time()
print(f'Entry-Wise Implement:\n Accuracy on the test set: {accuracy * 100:.2f}%, the time cost is {end-start}s\n')

start = time.time()
log_reg.logistic_regression(K=100, alpha=0.01, lambda_reg=0.01, flag=1)
accuracy = log_reg.predict(X_test, y_test)
end = time.time()
print(f'Dense Implement:\nAccuracy on the test set: {accuracy * 100:.2f}%, the time cost is {end-start}s\n')

start = time.time()
log_reg.logistic_regression(K=100, alpha=0.01, lambda_reg=0.01, flag=2)
accuracy = log_reg.predict(X_test, y_test)
end = time.time()
print(f'Sparse Implement:\nAccuracy on the test set: {accuracy * 100:.2f}%, the time cost is {end - start}s\n')

best_alpha = None
best_lambda_reg = None
best_acc = 0
for alpha in [0.1,0.03,0.01,0.003,0.001]:
    for lambda_reg in [0.1,0.01,0.001,0.0001]:
        print(f"Progress: alpha = {alpha}, lambda = {lambda_reg}", end='\r')
        log_reg.logistic_regression(K=1000, alpha=alpha, lambda_reg=lambda_reg, flag=2)
        accuracy = log_reg.predict(X_test, y_test)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_alpha = alpha
            best_lambda_reg = lambda_reg

print(f'best_acc = {best_acc}, best_alpha = {best_alpha}, best_lambda_reg = {best_lambda_reg}\n')

def data_whitening(X, k):
    # Centering
    M = X - X.mean(axis=0)

    # SVD Decomposition
    U, S, _ = la.svds(M, k=k)

    # Whitening
    Xw = U[:,::-1] @ scipy.sparse.diags(S[::-1])
    Xw = Xw / np.std(Xw, axis=0)[None,:]

    return Xw

def noise_removal(X, k, percentile = 0.9, vis_flag = False):
    # Whitening
    Xw = data_whitening(X, k)

    norm_list = np.array([np.linalg.norm(Xw[i, :]) for i in range(Xw.shape[0])])
    outlier_mask = norm_list >= np.quantile(norm_list, percentile)

    return outlier_mask

def assess_performance(outlier_mask, vis_flag = False):
    confusion_matrix, (P, R), F1 = F1_score(outlier_mask)
    acc = (confusion_matrix[0] + confusion_matrix[2]) /\
              (confusion_matrix[0] + confusion_matrix[1] + confusion_matrix[2] + confusion_matrix[3])
    
    if vis_flag:
        visualize_confusion(*confusion_matrix)
        print(f'Accuracy: {acc*100:.2f}%')
        print(f"Precision: {P*100:.2f}%")
        print(f"Recall: {R*100:.2f}%")
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

# read data
my_data = DataLoader(r".\data\data2.csv")
X = my_data.X.astype('float64')

F1_list = []
best_F1 = 0
best_k = None
best_q = None

for q in np.arange(0.90, 0.95, 0.01):  # test on different values of q
    F1_values = []
    for k in np.arange(1, 86):  # test on different values of k
        print(f"Progress: q={q * 100:.1f}%, k={k}", end='\r')
        F1, _, _ = assess_performance(noise_removal(X, k, q), vis_flag=False)
        F1_values.append(F1)

        # Update best results if current F1 is higher
        if F1 >= best_F1:
            best_F1 = F1
            best_k = k
            best_q = q
    
    F1_list.append(F1_values)

print(f"Best performance: k={best_k} and q={best_q * 100:.0f}% achieve the maximal F1-score of {best_F1:.4f}")

F1_values = F1_list[3]
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, 86), F1_values, marker='o')
plt.xlabel("value of k")
plt.ylabel("F1-score")
plt.title("Variation in F1-score with Increasing of k")
plt.grid(True)
plt.show()

# data whitening & noise removal
F1, _, _ = assess_performance(noise_removal(X, 85, 0.93), vis_flag=True)

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

P_list = []
R_list = []
for q in np.arange(0.01, 1.00, 0.01):
    print(f"Progress: {q * 100:.0f}%", end='\r')
    _, (P, R), _ = assess_performance(noise_removal(X, 85, q), vis_flag=False)
    P_list.append(P)
    R_list.append(R)

# plot PR-Curve
print("k = 85:\n")
plot_PRC(np.array(P_list), np.array(R_list))
