# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 10:40:34 2016

@author: han
"""


import numpy as np

#读入数据，返回特征矩阵和标签向量
def load_data(file_name, delim = ','):
    fopen = open(file_name)
    temp_lines = [line.strip().split(delim) for line in fopen.readlines()]
    data_mat = np.array(temp_lines, np.float)
    feature_mat = data_mat[:,:-1]
    labels = data_mat[:,-1]
    
    return feature_mat, labels
    

    
def svd_project(train_file, test_file, k):
    feature_mat, train_labels = load_data(train_file)
    test_feature_mat, test_labels = load_data(test_file)
    
    #step1 svd 分解
    # mxm , mxn , nxn <- svd mxn
    U, Sigma, VT = np.linalg.svd(feature_mat)
        
    
    #step2 取 VT的前K个行向量（V的列向量），转置组成nxk的矩阵
    V = np.transpose(VT)
    V_k = V[:,0:k]
    
    #step3 映射
    train_mat = feature_mat * np.mat(V_k)
    test_mat =  test_feature_mat * np.mat(V_k)
    
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
    svd_project(train_file, test_file, k)
    print('k=20:')
    k=20
    svd_project(train_file, test_file, k)
    print('k=30:')
    k=30
    svd_project(train_file, test_file, k)
    
    train_file = 'splice-train.txt'
    test_file = 'splice-test.txt'
    print('-----splice------')
    print('k=10:')
    k=10
    svd_project(train_file, test_file, k)
    print('k=20:')
    k=20
    svd_project(train_file, test_file, k)
    print('k=30:')
    k=30
    svd_project(train_file, test_file, k)
  
'''
-----sonar------
k=10:
正确个数:  61       测试集总数:  103     正确率: 0.592233009709
k=20:
正确个数:  60       测试集总数:  103     正确率: 0.582524271845
k=30:
正确个数:  58       测试集总数:  103     正确率: 0.563106796117
-----splice------
k=10:
正确个数:  1650     测试集总数:  2175    正确率: 0.758620689655
k=20:
正确个数:  1662     测试集总数:  2175    正确率: 0.764137931034
k=30:
正确个数:  1627     测试集总数:  2175    正确率: 0.748045977011
'''  
    
    
    
    

  
    
    