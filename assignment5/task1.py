# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:15:05 2016

@author: han
"""
import numpy as np


def load_data(file_name, delim = ','):
    fopen = open(file_name)
    lines = fopen.readlines()
    
    first_line = lines[0]
    first_line = first_line.strip().split(delim)
    first_line = np.array(first_line, np.int)
    
    rest_lines = lines[1:]
    temp_lines = [line.strip().split(delim) for line in rest_lines]
    data_mat = np.array(temp_lines, np.float)
    feature_mat = data_mat[:,:-1]
    labels = data_mat[:,-1]
    
    # 把类别改为 -1 和 1
    labels[labels==0] = -1
    
    return first_line, feature_mat, labels
    
    

def train_naivebayes(feature_mat, labels, d):
    
    d = np.ceil(d*len(feature_mat))
                
    row_size = feature_mat.shape[0]
    col_size = feature_mat.shape[1]
    
    #binary classification datasets
    # 将这两个类别分为 min 和 max
    min_label = min(labels)
    max_label = max(labels)
    
    
    minlabel_num = 0
    maxlabel_num = 0
    for i in range(len(labels)):
        if labels[i] == min_label:
            minlabel_num += d[i]
        else:
            maxlabel_num += d[i]
        
      
    # 计算 P(features | label)
    dic_min = {}
    dic_max = {}
    
    # 遍历每一列
    for i in range(col_size):
        col = feature_mat[:,i]
        for j in range(row_size):
            if labels[j] == min_label:
                if col[j] in dic_min:    
                    dic_min[col[j]] = dic_min[col[j]] + d[j]
                else:
                    dic_min[col[j]] = d[j]
            else:
                if col[j] in dic_max:    
                    dic_max[col[j]] = dic_max[col[j]] + d[j]
                else:
                    dic_max[col[j]] = d[j]
  
            
            
        
  
#    print('sum d', sum(d))
#    print('min:', minlabel_num)
#    print('max:', maxlabel_num)
    
    return dic_min, dic_max, minlabel_num, maxlabel_num
    

def naivebayes_classify(features, dic_min, dic_max, minlabel_num, maxlabel_num):
    
    temp1 = []
    temp2 = []
    
    for i in range(len(features)):
        if features[i] in dic_min:
            temp1.append(dic_min[features[i]])
        else:
            temp1.append(1)
            
        if features[i] in dic_max:
            temp2.append(dic_max[features[i]])
        else:
            temp2.append(1)
    
    temp1 = np.array(temp1)
    temp2 = np.array(temp2)
    

    temp1 = np.log(temp1/minlabel_num)
    temp2 = np.log(temp2/maxlabel_num)
    
    p_classmin = sum(temp1) + np.log(minlabel_num/(minlabel_num+maxlabel_num))
    p_classmax = sum(temp2) + np.log(maxlabel_num/(minlabel_num+maxlabel_num))
    
    if p_classmin>p_classmax:
        return 0
    else:
        return 1
 
def classify(test_mat, test_labels, train_mat, train_labels):
    
    d = np.ones(len(train_labels))/len(train_labels)
    
    dic_min, dic_max, minlabel_num, maxlabel_num = train_naivebayes(train_mat, train_labels, d)
    
    min_label = min(train_labels)
    max_label = max(train_labels)
    
    test_size = len(test_labels)
    result_labels = np.zeros(test_size)
    for i in range(test_size):
        if naivebayes_classify(test_mat[i], dic_min, dic_max, minlabel_num, maxlabel_num)==0:
            result_labels[i] = min_label
        else:
            result_labels[i] = max_label
    
    result = result_labels - test_labels
    result[result!=0]=1
    eq = test_size - sum(result)
    accuracy = eq/test_size
    return accuracy
         
# 将一个下标arr分成第i个十等分和其余下标
def ten_fold(arr, i, feature_mat, labels):
    size = len(arr)
    offset = size/10
    start = int(i*offset)
    end = int ((i+1)*offset)
    fold_i = arr[start:end]
    rest = np.concatenate((arr[:start],arr[end:]),axis=0)
    #print(fold_i)
    #print(rest)
    test_mat = feature_mat[fold_i]
    test_labels = labels[fold_i]
    train_mat = feature_mat[rest]
    train_labels = labels[rest]
#    print(10*'-',i)
#    print(start, end)
    
    return test_mat, test_labels, train_mat, train_labels
    
    
def ten_fold_cross_validation(feature_mat, labels):
    data_size = len(labels)
    index = np.arange(0,data_size,1)
    accuracy_list = []
    for i in range(10):
        test_mat, test_labels, train_mat, train_labels = ten_fold(index, i, feature_mat, labels)
        accuracy = classify(test_mat, test_labels, train_mat, train_labels)
        accuracy_list.append(accuracy)
    
    mean_accuracy = sum(accuracy_list)/10
    temp = np.array(accuracy_list)    
    std_accuracy = np.std(temp)
    #print(accuracy_list)
    return mean_accuracy, std_accuracy
  
    
# 1 representing discrete feature and 0 representing numerical feature
# change numerical feature to discrete feature
def to_discrete(first_line, feature_mat):
    
    #将数值矩阵和离散矩阵分割
    numerical_num = len(first_line) - sum(first_line)
    numerical_mat = feature_mat[:, :numerical_num]
    discrete_mat = feature_mat[:,numerical_num:]
#    print(numerical_mat)
    
    # 转离散型数据
    # 数值型属性范围为[0,1]，可分为若干等分离散化
    # 0-1等分
    numerical_mat[numerical_mat<0.5]=0
    numerical_mat[numerical_mat>=0.5]=1
    
     # 0到3 等分  (平均准确率最高)
    #numerical_mat = np.array((numerical_mat*10)/3, np.int)

    # 0到5 等分
    #numerical_mat = np.array(np.ceil(numerical_mat*10/2), np.int)
    
    # 0到10 等分
    #numerical_mat = np.array(numerical_mat*10, np.int)
      
    # 拼接
    temp = np.concatenate((numerical_mat,discrete_mat),axis=1)
    
    return temp
    
    
    
    

if __name__ == '__main__': 
    print('running naivebayes 10 fold cross validation')
    
    first_line, feature_mat, labels = load_data('breast-cancer-assignment5.txt')
    print('breast-cancer-assignment5.txt, data size', len(labels))
    mean_accuracy, std_accuracy = ten_fold_cross_validation(feature_mat, labels)
    print('mean:', mean_accuracy, '\nstandard deviation:',std_accuracy) 
    print(10*'--')    
    
    
    
    first_line, feature_mat, labels = load_data('german-assignment5.txt')
    print('german-assignment5.txt, data size:', len(labels))
    feature_mat = to_discrete(first_line, feature_mat)
    mean_accuracy, std_accuracy = ten_fold_cross_validation(feature_mat, labels)
    print('mean:', mean_accuracy, '\nstandard deviation:',std_accuracy) 
    print(10*'--')
    
    
    
    
    
'''
german-assignment5.txt, data size: 1000
0.7 0.0464758001545
--------------------
breast-cancer-assignment5.txt, data size 277
0.736111111111 0.0708363900456
--------------------
'''   
    
   
    

    
    