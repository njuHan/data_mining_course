# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 14:10:53 2016

@author: han
"""
import numpy as np
from task1 import naivebayes_classify, train_naivebayes,to_discrete,ten_fold



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
    
    




def evaluate(train_mat, train_labels, d):
    
    dic_min, dic_max, minlabel_num, maxlabel_num = train_naivebayes(train_mat, train_labels,d)
    classify = (dic_min, dic_max, minlabel_num, maxlabel_num)
    
    min_label = min(train_labels)
    max_label = max(train_labels)
    
    size = len(train_labels)
    result_labels = np.zeros(size)
    for i in range(size):
        if naivebayes_classify(train_mat[i], dic_min, dic_max, minlabel_num, maxlabel_num)==0:
            result_labels[i] = min_label
        else:
            result_labels[i] = max_label
    
    result = result_labels - train_labels
    result[result!=0]=1
    eq = size - sum(result)
    accuracy = eq/size
    return accuracy, result_labels, classify
    
    

def adaboost_train(feature_mat, labels, itnum=10):

      
    classify_list = []
    m = feature_mat.shape[0]
    d = np.ones(m)/m
    
    for i in range(itnum): 
        accuracy, result_labels, classify = evaluate(feature_mat, labels, d)
        err_rate = 1 - accuracy
        
        #计算 alpha
        alpha = float(0.5*np.log((1.0-err_rate)/max(err_rate,1e-16)))
        classify = classify + (alpha, )
        
        #更新权重 d
        
        temp = np.multiply(-1*alpha*labels, result_labels)
        temp = np.exp(temp)
        d = np.multiply(d, temp)    
        d = d/sum(d)
#        print(d)
#        print('sum d', sum(d))
    
        classify_list.append(classify)
    
    return classify_list
    
def adaboost_classify(feature, classify_list):
    # classify = (dic_min, dic_max, minlabel_num, maxlabel_num, alpha )
    # naivebayes_classify(features, dic_min, dic_max, minlabel_num, maxlabel_num):
    label_sum = 0
    for i in range(len(classify_list)):
        if naivebayes_classify(feature, classify_list[i][0], classify_list[i][1], classify_list[i][2], classify_list[i][3])==0:
            result_labels = -1
        else:
            result_labels = 1
        
        alpha = classify_list[i][4]
        label_sum += alpha * result_labels
        
        
    if label_sum>=0:
        return 1
    else:
        return -1

def classify(test_mat, test_labels, train_mat, train_labels):
    

    classify_list = adaboost_train(train_mat, train_labels)
    
    test_size = len(test_labels)
    result_labels = np.zeros(test_size)
    for i in range(test_size):
        result_labels[i] = adaboost_classify(test_mat[i], classify_list)
        
    
    result = result_labels - test_labels
    result[result!=0]=1
    eq = test_size - sum(result)
    accuracy = eq/test_size
    return accuracy
    
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
    
if __name__ == '__main__': 
    
    print('running adaboost 10 fold cross validation...')
    first_line, feature_mat, labels = load_data('breast-cancer-assignment5.txt')
    print('breast-cancer-assignment5.txt:', len(labels))
    mean_accuracy, std_accuracy = ten_fold_cross_validation(feature_mat, labels)
    print('mean:', mean_accuracy, 'standard deviation:',std_accuracy)  
    print(10*'--') 

    
    
    first_line, feature_mat, labels = load_data('german-assignment5.txt')
    print('german-assignment5.txt, data size:', len(labels))
    feature_mat = to_discrete(first_line, feature_mat)
    mean_accuracy, std_accuracy = ten_fold_cross_validation(feature_mat, labels)
    print('mean:', mean_accuracy, 'standard deviation:',std_accuracy)  
    
    print(10*'--')    
    
 

    