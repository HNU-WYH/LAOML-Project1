import numpy as np


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

