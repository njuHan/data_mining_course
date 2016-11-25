# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:12:33 2016

@author: han
"""

#Binary classification with the ridge regression 

import numpy as np
import random
from numpy.linalg import norm

def load_data(file_name, delim = ','):
    fopen = open(file_name)
    temp_lines = [line.strip().split(delim) for line in fopen.readlines()]
    data_mat = np.array(temp_lines, np.float)
    feature_mat = data_mat[:,:-1]
    labels = data_mat[:,-1]
    
    return feature_mat, labels
    

def sub_gradient(x):
    x[x>=0]=1
    x[x<0]=-1
    return x
    
def gradient(x,y,beta):
    lamd = 0.1
    temp1 = 2*(y-np.mat(beta)*np.transpose(np.mat(x)))
    temp2 = temp1*(-x)
    temp3 = lamd*2*beta
    
    return temp2 + temp3
    
def obj_func(beta, feature_mat, labels):
    lamd = 0.1
    N = feature_mat.shape[0]
    temp_sum = 0
    for i in range(N):
        x = feature_mat[i]
        y = labels[i]
        temp1 = float(y - np.mat(beta)*np.transpose(np.mat(x)))
        temp1 = temp1**2
        temp_sum = temp_sum + temp1
        
    return temp_sum/N + lamd*norm(beta,2)
    
    
def error_rate(feature_mat, labels, beta):
    m = feature_mat.shape[0]
    result  = np.zeros(m)
    for i in range(m):
        temp = np.mat(beta)*np.transpose(np.mat(feature_mat[i]))
        temp = float(temp)
        if temp>=0:
            result[i] = 1
        else:
            result[i] = -1
    error = labels - result
    error[error!=0] = 1
    return sum(error)/m
    
        

def sgd( train_feature, train_labels, test_feature, test_labels,  k):
    m, d = np.shape(train_feature)
    beta = np.ones(d)/10
    train_func = []
    train_error = []
    test_error = []
    for i in range(m):
        data_index = list(range(m))
        for j in range(m):
            alpha = 4/(1+j+i) + 0.01
            rand_index = int(random.uniform(0,len(data_index)))
           
            beta = beta - alpha * gradient(train_feature[rand_index],train_labels[rand_index],beta)
            del(data_index[rand_index])
           
        train_func.append( str(obj_func(beta,train_feature,train_labels)) +'\n')
        train_error.append( str(error_rate(train_feature, train_labels, beta)) + '\n')
        test_error.append( str(error_rate(test_feature, test_labels, beta)) + '\n' )
                
            
            
        
            
        #print(obj_func(beta,feature_mat,labels))        
        #print(error_rate(feature_mat, labels, beta))
    
    return train_func, train_error, test_error
                
     
    

if __name__ == '__main__': 
    k=100
    
    # dataset1 
    train_feature, train_labels = load_data('dataset1-a9a-training.txt')
    test_feature, test_labels = load_data('dataset1-a9a-testing.txt')
    
    train_func, train_error, test_error = sgd(train_feature, train_labels, test_feature, test_labels,k)
    
    file_object = open('task2_dataset1_train_func.txt', 'w')
    file_object.writelines(train_func)
    file_object.close( )
    
    file_object = open('task2_dataset1_train_error.txt', 'w')
    file_object.writelines(train_error)
    file_object.close()
 
    file_object = open('task2_dataset1_test_error.txt', 'w')
    file_object.writelines(test_error)
    file_object.close()
    