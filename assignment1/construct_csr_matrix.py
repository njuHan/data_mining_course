# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 19:24:33 2016

Compressed Sparse Row matrix
http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

@author: han
"""

import scipy as sp
import global_var as glv  

# 输入为所有文档的单词数组
#例如3个文档： docs = [["hello", "world", "hello"], 
#            ["goodbye", "cruel", "world","cruel", "world", "world", "world"]]
  
def construct_csr_matrix(docs):
    indptr = [0]
    indices = []
    data = []
    for d in docs:
        for term in d:
            index = glv.WORD_LIST.setdefault(term, len(glv.WORD_LIST))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
    mat = sp.sparse.csr_matrix((data, indices, indptr), dtype=int)
    print(mat)
    print(mat.toarray())
   
    print(glv.WORD_LIST.get('world'))
    col_index = glv.WORD_LIST.get('world')
    print(mat.toarray()[1,col_index])
    return

if __name__ == '__main__':  
    docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world","cruel", "world", "world", "world"]]
    construct_csr_matrix(docs)