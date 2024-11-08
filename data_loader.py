import numpy as np
import scipy

class DataLoader:
    def __init__(self, file_path= r".\data\data.csv"):
        self.X = None
        self.y = None
        self.num = 0

        self.__read(file_path)

    def __read(self, file_path):
        ori_data = np.loadtxt(file_path, skiprows=1, delimiter=',', dtype=int)
        self.X = ori_data[:,:-1]
        self.y = ori_data[:,-1]
        self.y[self.y == 0] = -1
        self.num = self.X.shape[0]

    def split(self, ratio = 0.3):
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

class LinearClassifier:
    def __init__(self, X_train, y_train):
        pass

    @staticmethod
    def hyperplane_evaluator(X: np.ndarray, y: np.ndarray, w: np.ndarray):
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

    #
    ratio = np.random.uniform()
    mydata.split(ratio)
    w = np.random.uniform(low = -1, high = 1, size=(mydata.X.shape[1]))
    pred_corr_num = LinearClassifier.hyperplane_evaluator(mydata.X, mydata.y, w)

    print(pred_corr_num)

