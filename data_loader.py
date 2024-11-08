import numpy as np
import scipy
from scipy.sparse import csr_matrix
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
                w_grad -= (self.y_train[i] * self.X_train[i])/(1 + np.exp(self.y_train[i] * self.X_train[i].dot(self.w)))
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
        y_true =  self.evaluator(X_test, y_test, w)
        acc = y_true / y_test.shape[0]
        return acc

    @staticmethod
    def evaluator(X: np.ndarray, y: np.ndarray, w: np.ndarray):
        y_pred = X.dot(w)
        y_pred[y_pred < 0] = -1
        y_pred[y_pred >= 0] = 1
        return (y_pred == y).sum()


if __name__ == "__main__":
    # Load the data
    mydata = DataLoader(r".\data\data.csv")

    # 1.1 How many malicious data points are there?
    print(f'there are {(mydata.y == 1).sum()} malicious data points in the total {mydata.num} data points')

    # 1.2 What can you say about the sparsity of the data?
    # The data is very sparse with lots of entries to be 0

    # 1.3 Do you think it makes sense to use one-hot-coding for some of the columns?
    # It is reasonable to use one-hot-coding for some of the columns due to they are categorical data without orders.

    # 2. Write a function that splits the data into a training and test set according to some fraction
    np.random.seed(1234)
    X_train, X_test, y_train, y_test = mydata.split(ratio = 0.5)

    # 3. Write a function that, given the matrix X, the vector y, and a weight vector w defining a hyperplane,
    # returns the number of correctly classified points

    # generate random weights
    w_init = np.random.uniform(low = -1, high = 1, size=(mydata.X.shape[1]))

    # number of correctly classified points
    pred_corr_num = LogisticClassifier.evaluator(mydata.X, mydata.y, w_init)
    print(f'{pred_corr_num} data points are correct using random generated weights')

    # 5. Logistic Classifier
    log_reg = LogisticClassifier(X_train, y_train, w_init)

    start = time.time()
    log_reg.logistic_regression(K=100, alpha=0.01, lambda_reg=0.1, flag= 0)
    accuracy = log_reg.predict(X_test, y_test)
    end = time.time()
    print(f'Accuracy on the test set: {accuracy * 100:.2f}%, the time cost is {end-start}s')

    start = time.time()
    log_reg.logistic_regression(K=100, alpha=0.01, lambda_reg=0.1, flag=1)
    accuracy = log_reg.predict(X_test, y_test)
    end = time.time()
    print(f'Accuracy on the test set: {accuracy * 100:.2f}%, the time cost is {end-start}s')

    start = time.time()
    log_reg.logistic_regression(K=100, alpha=0.01, lambda_reg=0.1, flag=2)
    accuracy = log_reg.predict(X_test, y_test)
    end = time.time()
    print(f'Accuracy on the test set: {accuracy * 100:.2f}%, the time cost is {end - start}s')




