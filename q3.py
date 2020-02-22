import numpy as np
import math
import pandas as pd

class node:
    def __init__(self, l, r, attr, val):
        self.left = l
        self.right = r
        self.attribute = attr
        self.value = val
        self.answer = 0

class DecisionTree:
    def __init__(self):
        self.root_node = None

    def build_tree(self, x_f, current_depth, maximum_depth=20, threshold_samples=20):
        # print("buildtree")
        col_list = list(x_f.columns)
        col_list.remove('SalePrice')
        # print("depth = "+str(current_depth))
        # print(" got "+str(len(x_f))+" rows", end = ' ')
        # print(" got "+str(len(col_list))+" columns")
        tree_node = node(None, None, None, None)
        if current_depth == maximum_depth or len(x_f) < threshold_samples:
            tree_node.answer = x_f['SalePrice'].mean()
            return tree_node
        best_attr = []
        ser = x_f.dtypes
        # print("series ",ser)
        for attr in col_list:
            # print("attribute ",attr, end=' ')
            if ser[attr] == object:
                # print("object")
                attr_values_set = np.unique(x_f[attr])
                attr_value_list = list(attr_values_set)
                for split_val in attr_value_list:
                    less_frame = x_f[x_f[attr] == split_val].copy()
                    left_error = 0
                    if len(less_frame):
                        less_array = less_frame['SalePrice'].to_numpy()
                        less_array = less_array.astype('float64')
                        less_array -= less_array.mean()
                        less_array = np.square(less_array)
                        left_error = less_array.sum()*(len(less_array)/len(x_f))
                    more_frame = x_f[x_f[attr] != split_val].copy()
                    right_error = 0
                    if len(more_frame):
                        more_array = more_frame['SalePrice'].to_numpy()
                        more_array = more_array.astype('float64')
                        more_array -= more_array.mean()
                        more_array = np.square(more_array)
                        right_error = more_array.sum()*(len(more_array)/len(x_f))
                    mean_sq_error = left_error + right_error
                    if len(best_attr):
                        if best_attr[2] > mean_sq_error:
                            best_attr = [attr, split_val, mean_sq_error]
                    else:
                        best_attr = [attr, split_val, mean_sq_error]                 
            else:
                # print("numerical")
                attr_values_set = np.unique(x_f[attr])
                attr_value_list = list(attr_values_set)
                attr_value_list.sort()
                split_list = [((attr_value_list[iv] + attr_value_list[iv+1])/2) for iv in range(len(attr_value_list)-1)]
                # print(split_list)
                for split_val in split_list:
                    less_frame = x_f[x_f[attr] <= split_val].copy()
                    left_error = 0
                    if len(less_frame):
                        less_array = less_frame['SalePrice'].to_numpy()
                        less_array = less_array.astype('float64')
                        less_array -= less_array.mean()
                        less_array = np.square(less_array)
                        left_error = less_array.sum()*(len(less_array)/len(x_f))
                    more_frame = x_f[x_f[attr] > split_val].copy()
                    right_error = 0
                    if len(more_frame):
                        more_array = more_frame['SalePrice'].to_numpy()
                        more_array = more_array.astype('float64')
                        more_array -= more_array.mean()
                        more_array = np.square(more_array)
                        right_error = more_array.sum()*(len(more_array)/len(x_f))
                    mean_sq_error = left_error + right_error
                    if len(best_attr):
                        if best_attr[2] > mean_sq_error:
                            best_attr = [attr, split_val, mean_sq_error]
                    else:
                        best_attr = [attr, split_val, mean_sq_error]    
    
        tree_node.value = best_attr[1]
        tree_node.attribute = best_attr[0]
        # print("splitting on "+tree_node.attribute+" at value "+str(tree_node.value))
        if isinstance(tree_node.value, str):
            left_x = x_f[x_f[tree_node.attribute] == tree_node.value].copy()
            right_x = x_f[x_f[tree_node.attribute] != tree_node.value].copy()
        else:
            left_x = x_f[x_f[tree_node.attribute] <= tree_node.value].copy()
            right_x = x_f[x_f[tree_node.attribute] > tree_node.value].copy()
        if(len(left_x)==0):
            tree_node.answer = right_x['SalePrice'].mean()
            return tree_node
        if(len(right_x)==0):
            tree_node.answer = left_x['SalePrice'].mean()
            return tree_node
        left_x.drop(columns=[best_attr[0]], inplace=True)
        right_x.drop(columns=[best_attr[0]], inplace=True)
        tree_node.left = self.build_tree(left_x, current_depth+1, maximum_depth, threshold_samples)
        tree_node.right = self.build_tree(right_x, current_depth+1, maximum_depth, threshold_samples)
        return tree_node

    def preprocessing(self, df):
        df.fillna(df.mean(), inplace=True)
        df.fillna(value="others", inplace=True)
        df.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True)
        convert_to_num = {'LotShape':{'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1}, 
                   'LandContour':{'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1},
                   'Utilities':{'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1},
                   'LandSlope':{'Gtl':3, 'Mod':2, 'Sev':1},
                   'ExterQual':{'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1},
                   'ExterCond':{'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1},
                   'BsmtQual':{'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'others':0},
                   'BsmtCond':{'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'others':0},
                   'BsmtExposure':{'Gd':5, 'Av':4, 'Mn':3, 'No':2, 'others':1},
                   'BsmtFinType1':{'GLQ':7, 'ALQ':6, 'BLQ':5, 'Rec':4, 'LwQ':3, 'Unf':2, 'others':1},
                   'BsmtFinType2':{'GLQ':7, 'ALQ':6, 'BLQ':5, 'Rec':4, 'LwQ':3, 'Unf':2, 'others':1},
                   'HeatingQC':{'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1},
                   'KitchenQual':{'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1}, 
                   'Functional':{'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1}, 
                   'FireplaceQu':{'Ex':6, 'Gd':5, 'TA':4, 'Masonry':3, 'Fa':2, 'Po':1, 'others':0}, 
                   'GarageFinish':{'Fin':4, 'RFn':3, 'Unf':2, 'others':1}, 
                   'GarageQual':{'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'others':0}, 
                   'GarageCond':{'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'others':0}, 
                   'PavedDrive':{'Y':3, 'P':2, 'N':1}}
        for attribute in df.columns:
            if attribute in convert_to_num.keys():
                df[attribute] = df[attribute].map(convert_to_num[attribute])
        return df

    def train(self, train_dataframe_path):
        train_df = pd.read_csv(train_dataframe_path, index_col="Id")
        train_df = self.preprocessing(train_df)
        self.root_node = self.build_tree(train_df, 1)

    def predict(self, test_dataframe_path):
        test_df = pd.read_csv(test_dataframe_path, index_col="Id")
        test_df = self.preprocessing(test_df)
        pred_list = []
        ser = test_df.dtypes
        for test_index in range(len(test_df)):
            current_node = self.root_node;
            while current_node.left != None and current_node.right != None:
                if ser[current_node.attribute] == object:
                    if test_df.iloc[test_index][current_node.attribute] == current_node.value:
                        current_node = current_node.left
                    else:
                        current_node = current_node.right
                else:
                    if test_df.iloc[test_index][current_node.attribute] <= current_node.value:
                        current_node = current_node.left
                    else:
                        current_node = current_node.right
            pred_list.append(current_node.answer)
        return pred_list