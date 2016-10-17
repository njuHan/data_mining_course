# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 10:40:10 2016

@author: han
"""

import numpy as np


#读入数据，返回特征矩阵和标签
def load_data(file_name, delim = ','):
    fopen = open(file_name)
    temp_lines = [line.strip().split(delim) for line in fopen.readlines()]
    data_mat = np.array(temp_lines, float)
    feature_mat = data_mat[:,:-1]
    labels = data_mat[:,-1]
    
    return feature_mat, labels
    
    
def pca_project(train_file, test_file, k):
    feature_mat, train_labels = load_data(train_file)  
    test_feature_mat, test_labels = load_data(test_file)
    
#    step1 去除平均值
    mean_val = np.mean(feature_mat,axis=0)
    mean_removed = feature_mat - mean_val
    
#    step2 计算协方差矩阵，rowvar=0表示每一行为一个样本
    cov_mat = np.cov(mean_removed, rowvar=0)
	
#    step3 计算协方差矩阵的特征值和特征向量
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
	
#    step4 将特征值排序
    eig_val_index = np.argsort(eig_vals) #从小到大
    eig_val_index = eig_val_index[::-1] # 颠倒，从大到小
    
#    step5 保留最大的前K个特征向量
    k_index = eig_val_index[0:k]
    k_vects = eig_vects[: , k_index]

    
#    step6 将数据转换到前K个特征向量构建的新空间中
    train_mat = feature_mat * k_vects
    test_mat =  test_feature_mat * k_vects

    classify(train_mat,train_labels, test_mat, test_labels)
    
    
    
def OneNNclassify(inx, train_mat , train_labels):
    dataset_size = train_mat.shape[0]
    diff_mat = np.tile(inx, (dataset_size,1)) - train_mat
    
    #必须转成 array 才能用单个元素的指数运算**
    #。。。。握了棵草，这个bug搞了半天
    diff_mat = np.asarray(diff_mat)
    sq_diffmat = diff_mat**2
    sq_dist = sq_diffmat.sum(axis=1)
    dist = sq_dist**0.5
    sort_dist_index = dist.argsort()
    #距离最小的向量
    output_label = train_labels[sort_dist_index[0]]
    
    return output_label
    


def classify(train_mat,train_labels, test_mat, test_labels):
    row_size = test_mat.shape[0]
    result_labels = np.zeros(row_size)
    for i in range(row_size):
        result_labels[i] = OneNNclassify(test_mat[i,:], train_mat , train_labels)
        
    #计算正确率
    temp1 = result_labels  - test_labels
    temp2 = temp1==0
    temp3 = np.array(temp2,int)
    
    print('正确个数: ',temp3.sum(),'\t测试集总数: ',temp2.shape[0],'\t正确率:',temp3.sum()/temp2.shape[0])
     
    

    
if __name__ == '__main__':  
    
    train_file = 'sonar-train.txt'
    test_file = 'sonar-test.txt'
    print('-----sonar------')
    print('k=10:')
    k=10
    pca_project(train_file, test_file, k)
    print('k=20:')
    k=20
    pca_project(train_file, test_file, k)
    print('k=30:')
    k=30
    pca_project(train_file, test_file, k)
    
    train_file = 'splice-train.txt'
    test_file = 'splice-test.txt'
    print('-----splice------')
    print('k=10:')
    k=10
    pca_project(train_file, test_file, k)
    print('k=20:')
    k=20
    pca_project(train_file, test_file, k)
    print('k=30:')
    k=30
    pca_project(train_file, test_file, k)
    
'''
-----sonar------
k=10:
正确个数:  60       测试集总数:  103     正确率: 0.582524271845
k=20:
正确个数:  58       测试集总数:  103     正确率: 0.563106796117
k=30:
正确个数:  58       测试集总数:  103     正确率: 0.563106796117
-----splice------
k=10:
正确个数:  1649     测试集总数:  2175    正确率: 0.75816091954
k=20:
正确个数:  1659     测试集总数:  2175    正确率: 0.76275862069
k=30:
正确个数:  1600     测试集总数:  2175    正确率: 0.735632183908
'''
    
   
  

   
    
    
    
	
	