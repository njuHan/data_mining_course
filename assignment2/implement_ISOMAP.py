# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 10:41:36 2016

@author: han
"""
import numpy as np


from heapq import heappush, heappop

def Dijkstra(graph, start):
    A = [None] * len(graph)
    queue = [(0, start)]
    while queue:
        path_len, v = heappop(queue)
        if A[v] is None: # v is unvisited
            A[v] = path_len
            for w, edge_len in graph[v].items():
                if A[w] is None:
                    heappush(queue, (path_len + edge_len, w))

    # -1 表示不连通    
    return [-1 if x is None else x for x in A] 


def load_data(file_name, delim = ','):
    fopen = open(file_name)
    temp_lines = [line.strip().split(delim) for line in fopen.readlines()]
    data_mat = np.array(temp_lines, np.float)
    feature_mat = data_mat[:,:-1]
    labels = data_mat[:,-1]
    
    return feature_mat, labels
    
def merge_data(train_file,test_file):
    feature_mat, train_labels = load_data(train_file)
    test_feature_mat, test_labels = load_data(test_file)    
    
    row_size = feature_mat.shape[0] + test_feature_mat.shape[0]
    col_size = feature_mat.shape[1]
    data_mat = np.append(feature_mat, test_feature_mat)
    data_mat = data_mat.reshape(row_size, col_size)
    
    return data_mat,train_labels,test_labels 
    
    
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
     

    


#计算给定点inx 的最近K个点的距离 
def knn_dist(inx, data_mat, k):
    dataset_size = data_mat.shape[0]
    diff_mat = np.tile(inx, (dataset_size,1)) - data_mat
    
    diff_mat = np.asarray(diff_mat)
    sq_diffmat = diff_mat**2
    sq_dist = sq_diffmat.sum(axis=1)
    dist = sq_dist**0.5
    sort_dist_index = dist.argsort()
    #字典s
    k_dict = {}
    #距离最小的k个向量,加上自己，有k+1个
    for i in range(k+1):
        if dist[sort_dist_index[i]]==0:
            continue
        else:
            k_dict[sort_dist_index[i]] = dist[sort_dist_index[i]]
        
    return k_dict
    

    


# use k-NN to construct a weighted graph
def con_knn_wetgraph(data_mat,knn):
    row_size = data_mat.shape[0]
    wet_graph = {}     
    for i in range(row_size):
        inx = data_mat[i,:]
        wet_graph[i] = knn_dist(inx, data_mat, knn)
    
    #转无向图邻接表
    for i in range(row_size):
        for key, value in wet_graph[i].items():
            wet_graph[key][i] = value
    
    return wet_graph, row_size

# 构建 size x size 的对称距离矩阵
def con_dist_mat(wet_graph, size):
    
    dist_mat = []
    for i in range(size):
        temp = Dijkstra(wet_graph,i)
        dist_mat = np.append(dist_mat, temp)
        
    dist_mat = dist_mat.reshape(size,size)
    
    return dist_mat
    

def MDS(dist_mat, k):
    m = dist_mat.shape[0] 
    disti2 = []
    for i in range(m):
        temp = 0
        for j in range(m):
            temp = temp + dist_mat[i, j]**2
        temp = temp / m
        disti2.append(temp)
    distj2 = []
    for i in range (m):
        temp = 0
        for j in range(m):
            temp = temp + dist_mat[j, i]**2
        temp = temp / m
        distj2.append(temp)
    dist2 = 0
    for i in range(m):
        for j in range(m):
            dist2 = dist2 + dist_mat[i, j]**2
    dist2 = dist2/m/m

    B = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            B[i, j] = -0.5 * (dist_mat[i, j]**2 - disti2[i] - distj2[j] + dist2)

    #对矩阵B做特征值分解
    eig_vals, eig_vects = np.linalg.eig(B)
    indices = np.argsort(eig_vals)  
    indices = indices[::-1] 
    
    #取 k 个最大的特征值构成对角矩阵diag， k_eigvects为对应的特征向量矩阵
    k_eigvals = eig_vals[indices[0: k]] 
    diag = np.diag(k_eigvals) 
    k_eigvects = eig_vects[:, indices[0: k]] 
    k_eigvects = np.mat(k_eigvects)
    
    output = (k_eigvects * diag**0.5)
    return output    

def isomap_project(knn, k, train_file, test_file):
    
    data_mat,train_labels,test_labels = merge_data(train_file, test_file)
    wg,size= con_knn_wetgraph(data_mat,knn)
    dist_mat = con_dist_mat(wg,size)
    
    x = MDS(dist_mat, k)
    
    #classify
    train_mat = x[0:train_labels.shape[0],:]
    test_mat = x[train_labels.shape[0]:, :]
    
    classify(train_mat,train_labels, test_mat, test_labels)

if __name__ == '__main__': 
    train_file = 'sonar-train.txt'
    test_file = 'sonar-test.txt'
    knn=10
    print('-----sonar------')
    print('k=10:')
    isomap_project(knn, 10, train_file, test_file)
    print('k=20:')
    isomap_project(knn, 20, train_file, test_file)
    print('k=30:')
    isomap_project(knn, 30, train_file, test_file)
    
    train_file = 'splice-train.txt'
    test_file = 'splice-test.txt'
    knn=10
    print('-----splice------')
    print('k=10:')
    isomap_project(knn, 10, train_file, test_file)
    print('k=20:')
    isomap_project(knn, 20, train_file, test_file)
    print('k=30:')
    isomap_project(knn, 30, train_file, test_file)
    
    
'''    
-----sonar------
k=10:
正确个数:  54       测试集总数:  103     正确率: 0.52427184466
k=20:
正确个数:  53       测试集总数:  103     正确率: 0.514563106796
k=30:
正确个数:  56       测试集总数:  103     正确率: 0.543689320388
-----splice------
k=10:
正确个数:  1495     测试集总数:  2175    正确率: 0.687356321839
k=20:
正确个数:  1520     测试集总数:  2175    正确率: 0.698850574713
k=30:
正确个数:  1523     测试集总数:  2175    正确率: 0.700229885057
'''    
    
    
    
    
    
   
    
    
    
    
    
   

   