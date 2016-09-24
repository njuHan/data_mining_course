# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 22:35:15 2016

@author: han
"""
import global_var as glv
import scipy as sp
import numpy as np


# 输入为所有文档的单词数组
#例如3个文档： docs = [["hello", "world", "hello"], 
#            ["goodbye", "cruel", "world","cruel", "world", "world", "world"]]
def calculate_tf_idf(docs):
    
    #construct csr matrix
    indptr = [0]
    indices = []
    data = []
    glv.WORD_LIST = {}
    for d in docs:
        for term in d:
            index = glv.WORD_LIST.setdefault(term, len(glv.WORD_LIST))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
    glv.CSR_MATRIX = sp.sparse.csr_matrix((data, indices, indptr), dtype=np.double)
    glv.CSR_MATRIX.sum_duplicates()
    
    
    #单词计数矩阵，单词在文中出现次数
    count_array =  glv.CSR_MATRIX.toarray()
    #print(count_array)
    # m行 n列
    m,n = count_array.shape
    
    #行求和矩阵， 每篇文章的单词总数
    row_sum_array = glv.CSR_MATRIX.toarray()
    temp2 = np.sum(row_sum_array,axis=1)
    for i in range(m):
        row_sum_array[i,:] = temp2[i]
    print('row sum\n',row_sum_array)
    
    #列逻辑求和， 具有该列单词的文章总数
    col_sum_array = glv.CSR_MATRIX.toarray()
    bool_array = col_sum_array>=1
    temp3 = np.sum(bool_array,axis=0)
    for i in range(n):
        col_sum_array[:,i] = temp3[i]
    
    print('col sum\n',col_sum_array)
    print('log:\n',np.log(col_sum_array))
    
    #文件总数矩阵
    file_num_array = glv.CSR_MATRIX.toarray()
    file_num_array[:] = glv.ALL_FILE_NUM
    
    #tf 
    tf_array = count_array/row_sum_array
    print('tf array\n',tf_array)
    
    #idf
    idf_array = np.log(m/col_sum_array)
    print('idf array\n',idf_array)    
    
    #tf idf 矩阵
    tfidf_array = tf_array*idf_array
    glv.TFIDF_MATRIX = sp.sparse.csr_matrix(tfidf_array)
    print('tf idf: \n',tfidf_array)
    return


if __name__ == '__main__':  
    docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world","cruel", "world", "world", "world"]]
    calculate_tf_idf(docs)
    print(glv.CSR_MATRIX.toarray())
    print(glv.WORD_LIST)
    