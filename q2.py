# -*- coding: utf-8 -*-
"""q2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Tnp5ir4kFPDvG1qGfpOZD4w5lPTWvEur
"""

# from google.colab import drive
# from google.colab import files
# drive.mount('/content/gdrive')
import numpy as np
import math
import pandas as pd

class KNNClassifier:

    def __init__(self, k_value=5):
        self.k_parameter = k_value
        self.data_arr = None
        self.train_res = None

    def complete_df(self, df):
        chars ={1: {'b', 'c', 'f', 'k', 's', 'x'},
        2: {'f', 'g', 's', 'y'},
        3: {'b', 'c', 'e', 'g', 'n', 'p', 'r', 'u', 'w', 'y', 't', 'f'},
        4: {'f', 't'},
        5: {'a', 'c', 'f', 'l', 'm', 'n', 'p', 's', 'y'},
        6: {'a', 'f', 'd', 'n'},
        7: {'c', 'w', 'd'},
        8: {'b', 'n'},
        9: {'b', 'e', 'g', 'h', 'k', 'n', 'o', 'p', 'r', 'u', 'w', 'y'},
        10: {'e', 't'},
        11: {'b', 'c', 'u', 'e', 'r', 'z'},
        12: {'f', 'k', 's', 'y'},
        13: {'f', 'k', 's', 'y'},
        14: {'b', 'c', 'e', 'g', 'n', 'o', 'p', 'w', 'y'},
        15: {'b', 'c', 'e', 'g', 'n', 'o', 'p', 'w', 'y'},
        16: {'p', 'u'},
        17: {'n', 'o', 'w', 'y'},
        18: {'n', 'o', 't'},
        19: {'c', 'e', 'f', 'l', 'n', 'p', 's', 'z'},
        20: {'b', 'h', 'k', 'n', 'o', 'r', 'u', 'w', 'y'},
        21: {'a', 'c', 'n', 's', 'v', 'y'},
        22: {'d', 'g', 'l', 'm', 'p', 'u', 'w'}}

        extra=0
        last_index = df.shape[0]-1
        for col in range(df.shape[1]):
            value_set = set(df.iloc[:,col])
            diff_set = chars[col+1] - value_set
            if(len(diff_set)):
                extra += len(diff_set)
                for ele in diff_set:
                    temp_df = pd.DataFrame(columns=df.columns)
                    temp_df = temp_df.append(df.iloc[last_index,:])
                    temp_df.iloc[0,col] = ele
                    df = df.append(temp_df)
        df = pd.get_dummies(df)
        df = df.iloc[0:len(df)-extra,:]
        return df

    def preprocess_dataframe(self, df):
        mode_list = list(df.mode().iloc[0].to_numpy())
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            for ind in range(len(df)):
                if df.iloc[ind][col] == '?' or df.iloc[ind][col] == 'NA':
                    df.iloc[ind][col] = mode_list[int(col)-1]
        return self.complete_df(df)

    def train(self, train_dataframe_path):
        train_df = pd.read_csv(train_dataframe_path, header=None)    
        self.train_res = train_df.iloc[:,0]
        n_f = train_df.drop(train_df.columns[0], axis=1, inplace=False)
        train_df = self.preprocess_dataframe(n_f)
        self.data_arr = train_df.to_numpy()

    def predict(self, test_dataframe_path):
        test_df = pd.read_csv(test_dataframe_path, header=None)
        test_df.columns = [ str(i) for i in range(1, test_df.shape[1]+1)]
        test_df = self.preprocess_dataframe(test_df)
        test_egs = test_df.to_numpy()
        prediction_list = []
        for test_ind in range(test_egs.shape[0]):
            lis = []
            test_row = test_egs[test_ind,:]
            hashmap = {'p':0, 'e':0}
            for train_ind in range(self.data_arr.shape[0]):
                train_row = self.data_arr[train_ind,:]
                diff = test_row - train_row
                diff = diff*diff
                dist = (diff.sum())
                lis.append([dist, self.train_res[train_ind]])
                lis = sorted(lis, key=lambda pair:pair[0])
                lis = lis[0:self.k_parameter]
            for pair in lis:
                hashmap[pair[1]] += 1
            if hashmap['p'] > hashmap['e']:
                ind = 'p'
            else:
                ind = 'e'
            prediction_list.append(ind)
        return prediction_list