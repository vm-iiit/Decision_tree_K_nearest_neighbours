import numpy as np
import math
import time

class KNNClassifier:

    def __init__(self, k_value=5):
        self.k_parameter = k_value
        self.data_arr = None
        self.train_res = None

    def train(self, train_dataframe_path):
        self.data_arr = np.genfromtxt(train_dataframe_path, delimiter = ',')
        self.train_res = self.data_arr[:,0]
        self.data_arr = self.data_arr[:, 1:]

    def predict(self, test_dataframe_path):
        test_egs = np.genfromtxt(test_dataframe_path, delimiter = ',')
        prediction_list = []
        for test_ind in range(test_egs.shape[0]):
            # print("validation sample number "+str(test_ind+1))
            lis = []
            test_row = test_egs[test_ind,:]
            hashmap = [0 for i in range(10)]
            for train_ind in range(self.data_arr.shape[0]):
                train_row = self.data_arr[train_ind,:]
                diff = test_row - train_row
                diff = diff*diff
                dist = (diff.sum())
                lis.append([dist, self.train_res[train_ind]])
                lis = sorted(lis, key=lambda pair:pair[0])
                lis = lis[0:self.k_parameter]
            for pair in lis:
                hashmap[int(pair[1])] += 1
            maxval = hashmap[0]
            ind = 0
            for iter in range(1,10):
                val = hashmap[iter]
                if val > maxval:
                    maxval = val
                    ind = iter
            prediction_list.append(ind)
        return prediction_list