# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 13:35:36 2016

@author: han
"""
import numpy as np
from k_medoids import k_medoids_smalldata 
from k_medoids import k_medoids_bigdata


def load_data(file_name, delim = ','):
    fopen = open(file_name)
    temp_lines = [line.strip().split(delim) for line in fopen.readlines()]
    data_mat = np.array(temp_lines, np.float)
    feature_mat = data_mat[:,:-1]
    labels = data_mat[:,-1]
    
    return feature_mat, labels

#计算给定点inx 的最近n个点的距离 
def knn_dist(inx, data_mat, n):
    dataset_size = data_mat.shape[0]
    diff_mat = np.tile(inx, (dataset_size,1)) - data_mat
      
    diff_mat = np.asarray(diff_mat)
    sq_diffmat = diff_mat**2
    sq_dist = sq_diffmat.sum(axis=1)
    dist = sq_dist**0.5
          
    #距离最小的k个向量,除去第一个自己，
#    sort_dist_index = dist.argsort()  
#    n_neighbor = []
#    for i in range(1,n+1):
#        #n_neighbor[i-1] = int(sort_dist_index[i])
#        n_neighbor.append(sort_dist_index[i])
#            
#    print(n_neighbor)
    
    index = get_min_index(dist, n+1)
    n_neighbor = index[1:]
   
        
    return n_neighbor
    
def con_graph(data_mat, n):
    size = data_mat.shape[0]
    graph = np.zeros([size,size],int)
    for i in range(size):
        n_neighbor = knn_dist(data_mat[i,:], data_mat, n)
        #print('in con graph', i)
        for j in n_neighbor:
            graph[i][j] = 1
            graph[j][i] = 1
        
    return graph
    
    
def spectral_clustering(feature_mat, n, k): 
    
    graph = con_graph(feature_mat, n)   
    
    # D_mat is diagonal weight matrix    
    col_sum = graph.sum(axis=0) 
    #D_mat = np.diag(col_sum)
    size = feature_mat.shape[0]
    D_mat = np.zeros([size,size],int)
    for i in range(size):
        D_mat[i][i] = col_sum[i]
    
    #print('D', D_mat)  
    D_mat = np.mat(D_mat)
    graph = np.mat(graph)
    
    L_mat = D_mat - graph
    X_mat = np.dot(np.linalg.inv(D_mat) , L_mat)
    eig_vals, eig_vects = np.linalg.eig(X_mat)
    sorted_index = np.argsort(eig_vals)
    #print(eig_vals[sorted_index][0:10])
    k_index = sorted_index[1:k+1]
    k_vects = eig_vects[: , k_index]
    
    return k_vects



def spec_smalldata(feature_mat,labels, n):
    k = 2
    Y = spectral_clustering(feature_mat, n, k) 
    k_medoids_smalldata(Y, labels)
    
def spec_bigdata(feature_mat,labels, n):
    k = 10 
    Y = spectral_clustering(feature_mat, n, k)
    k_medoids_bigdata(Y, labels)
    
     
        
def get_min_index(a, n):
    l = a.tolist()
    max_num = max(l)+1
    index = []
    for i in range(n):
        min_val = min(l)
        min_index = l.index(min_val)
        index.append(min_index)
        l[min_index] = max_num
        
    return index
        
    
if __name__ == '__main__':  
    print('-----------dataset1 german.txt:')
    feature_mat, labels = load_data('german.txt')
    for n in [3,6,9]:
        print('n= ',n,':')
        spec_smalldata(feature_mat, labels, n)
    
   
    print('-----------dataset2 mnist.txt:')
    feature_mat, labels = load_data('mnist.txt')
    for n in [3,6,9]:
        print('n= ',n,':')
        spec_bigdata(feature_mat, labels, n)
    
   
    

    
    
    