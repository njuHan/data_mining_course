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
    glv.CSR_MATRIX = sp.sparse.csr_matrix((data, indices, indptr), dtype=np.float)
    glv.CSR_MATRIX.eliminate_zeros()
    glv.CSR_MATRIX.sum_duplicates()
    glv.CSR_MATRIX.multiply(2)
    
   
    
    #sum col 包含该列单词的文档总数
    logical_mat = glv.CSR_MATRIX.power(0)
    sum_col = logical_mat.sum(axis=0)
    #print('sum col:\n',sum_col)
    
    # sum row 文档单词总数
    sum_row = glv.CSR_MATRIX.sum(axis=1)
    #print('sum row:\n',sum_row)
    
    #tf
    tf = sp.sparse.csr_matrix(glv.CSR_MATRIX.multiply(1/sum_row))
    #print('tf:\n',tf.toarray())    
    
    #tf idf
    #glv.ALL_FILE_NUM = 2
    #print('idf :\n',glv.ALL_FILE_NUM/sum_col)
    
    #print('tf*idf:\n', tf.multiply(glv.ALL_FILE_NUM/sum_col))
    
    #print('idf log:\n',np.log(glv.ALL_FILE_NUM/sum_col))
    tf_idf = tf.multiply(np.log(glv.ALL_FILE_NUM/sum_col))
    #print('tf idf:\n', tf_idf)
    glv.TFIDF_MATRIX = sp.sparse.csr_matrix(tf_idf) 
    #print(glv.TFIDF_MATRIX)
    
    return


if __name__ == '__main__':  
    docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world","cruel", "world", "world", "world"]]
    glv.ALL_FILE_NUM = len(docs)
    calculate_tf_idf(docs)
    print(glv.TFIDF_MATRIX)
    


#  (0, 0)        4.60517018599
#  (0, 1)        1.60943791243
#  (1, 1)        6.43775164974
#  (1, 2)        2.30258509299
#  (1, 3)        4.60517018599

#(0, 0)        0.462098120373
#  (1, 2)        0.0990210257943
#  (1, 3)        0.198042051589