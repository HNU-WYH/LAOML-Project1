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
    data_loader = DataLoader(r".\data\data.csv")
    ratio = np.random.uniform()
    data_loader.split(ratio)
    w = np.random.uniform(low = -1, high = 1, size=(data_loader.X.shape[1]))
    pred_corr_num = LinearClassifier.hyperplane_evaluator(data_loader.X, data_loader.y, w)

    print(pred_corr_num)

