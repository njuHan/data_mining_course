# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 13:34:48 2016

@author: han
"""
import numpy as np
import random

def load_data(file_name, delim = ','):
    fopen = open(file_name)
    temp_lines = [line.strip().split(delim) for line in fopen.readlines()]
    data_mat = np.array(temp_lines, np.float)
    feature_mat = data_mat[:,:-1]
    labels = data_mat[:,-1]
    
    return feature_mat, labels
    
#随机选取K个中心    
def k_center(data_size,k):
    
    index_list = list(range(data_size))
    k_index = random.sample(index_list, k)
    return k_index
    
  
    
    
def dist(inx, data_mat):
    dataset_size = data_mat.shape[0]
    diff_mat = np.tile(inx, (dataset_size,1)) - data_mat
    diff_mat = abs(diff_mat)
    
    dist = diff_mat.sum(axis=1)
    
    #字典
    inx_dist = {}
    #inx 与其他点的距离
    for i in range(dataset_size):
        inx_dist[i] = dist[i]
        
#    print(dist)
#    print(inx_dist)
    
    return inx_dist

def get_dist(data_mat):
    row_size = data_mat.shape[0]
    dist_dict = {}     
    for i in range(row_size):
        inx = data_mat[i,:]
        dist_dict[i] = dist(inx, data_mat)
    
    
    return dist_dict
    
    
def get_cost(k_index, dist_dict, feature_mat,labels ):
    
    data_size = feature_mat.shape[0]
    cluster_labels = np.zeros(data_size)
    cost = 0
    #将每个数据点 i 分配到最近的中心点
    for i in range(feature_mat.shape[0]):
        # i 和 第0个中心点 k_index[0]的距离，最小距离初始化，标签初始化
        min_dist = dist_dict[i][k_index[0]]
        cluster_labels[i] = labels[k_index[0]]
        #遍历找到距离最小的中心点
        for j in range(len(k_index)):
            if dist_dict[i][k_index[j]]<min_dist:
                 min_dist = dist_dict[i][k_index[j]]
                 cluster_labels[i] = labels[k_index[j]]
        
        #累加距离， cost
        cost = cost + min_dist
        
    return cost, cluster_labels


# 找到一个代价最小的点替换第i个中心点                 
def replace_i(k_index, k, dist_dict, feature_mat,labels):
    #是否替换的标志
    is_replaced = False
    min_cost, cluster_labels = get_cost(k_index, dist_dict, feature_mat,labels)
    min_index = np.array(k_index)
    min_labels = labels
    # 用非中心点 i 替换第 K 个中心点， 再计算距离
    for i in range(len(labels)):
        if i in k_index:
            continue
        new_kindex = np.array(k_index)
        new_kindex[k] = i
        new_cost, new_labels = get_cost(new_kindex, dist_dict, feature_mat,labels)
#        print(new_kindex)
#        print(new_cost)
        if min_cost > new_cost:
            is_replaced = True
            min_cost = new_cost
            min_index = new_kindex
            min_labels = new_labels
    
    return is_replaced , min_index, min_cost, min_labels
                 
    
def k_medoids(feature_mat, labels,dist_dict, k):
    #随机选择K个点作为初始medoid
    data_size = feature_mat.shape[0]
    k_index =  k_center(data_size,k) 
    print(k_index)
    init_cost, init_labels = get_cost(k_index, dist_dict, feature_mat,labels)
    print(init_cost)
    print('----------')
    is_replaced = True
    min_cost = init_cost
    min_labels = init_labels
    
    while(is_replaced):
        is_replaced = False
        for i in range(k):
            print('-------------in: ', i,'------------')
            is_replaced, new_index, new_cost, new_labels = replace_i(k_index, i, dist_dict, feature_mat,labels)
            if is_replaced == True:
                #更新中心点 和 最小cost
                k_index = new_index
                min_cost = new_cost
                min_labels = new_labels
                print(k_index, min_cost)
                
    print('min index: ', k_index)
    print('min cost:', min_cost)
        
    return min_labels
    
    
def get_purity(true_labels, test_labels,k ):
    n = len(true_labels)
    confusion_mat = np.zeros([k,k],int)
    for i in range(n):
        confusion_mat[true_labels[i]][test_labels[i]] = confusion_mat[true_labels[i]][test_labels[i]] + 1
        
    p = np.zeros(k)
    m = np.zeros(k)
    for j in range(k):
        m[j] = np.sum(confusion_mat[:,j])
        p[j] = np.max(confusion_mat[:,j])
        
    sum_p = np.sum(p)
    sum_m = np.sum(m)
    purity = sum_p/sum_m
    print(sum_p,sum_m,purity)

    # gini index
    g = np.zeros(k)
    for i in range(k):
        gi = 1
        for j in range(k):
            gi = gi - np.square(confusion_mat[j][i] / m[i])
        g[i] = gi
    sum_gm = 0
    for i in range(k):
        sum_gm = sum_gm + g[i] * m[i]
    gini_index = sum_gm / sum_m
    print(gini_index)
        
    

if __name__ == '__main__':  
    feature_mat, labels = load_data('german.txt')
    dist_dict = get_dist(feature_mat)
    k = 2    
    min_labels = k_medoids(feature_mat, labels,dist_dict, k)
    
    result = min_labels - labels
    result = result==0
    result = sum(result)/len(result)
    print(result)
   
    
    print('***********')
    
    #把-1元素变为0
    true_labels = np.array(labels==1,dtype=int)
    test_labels = np.array(min_labels==1,dtype=int)
    get_purity(true_labels, test_labels, k )
#    l=np.array([-1,  1, -1, -1,  1, -1, -1])
#    temp = np.array(l==1,dtype=int)
#    print(l)
#    print(temp)
  
   
#    
#    print(get_cost(np.array([7,2]), dist_dict, feature_mat,labels ))
#    
#    
    '''
    min index:  [7 2]
min cost: 15.0
0.454545454545
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    '''
    