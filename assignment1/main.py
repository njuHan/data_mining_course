# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:24:36 2016

@author: han
"""
import numpy as np
import scipy as sp
 
from remove_stopword import remove_stopword 
from get_doc_path import get_doc_path
import global_var as glv   
    
def process_doc(doc):
    print('processing doc: ',doc)
    infile = open(doc, 'r', encoding='utf-8')
    i=0
    while i<9:
        #print('in doc\n')
        line = infile.readline().strip()
        print(line,i, line=='')
        i = i+1

    infile.close()
    return
  
if __name__ == '__main__':  
#    get_doc_path('./test') 
#    print(remove_stopword('111'))
#    print(glv.DOC_LIST)
#    with open(glv.DOC_LIST[0], 'r' , encoding='utf-8') as f:
#        print(f.read())
    docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world","cruel", "world", "world", "world"]]
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    for d in docs:
        for term in d:
            index = vocabulary.setdefault(term, len(vocabulary))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
    mat = sp.sparse.csr_matrix((data, indices, indptr), dtype=int)
    print(mat)
    print(mat.toarray())
   
    print(vocabulary.get('world'))
    col_index = vocabulary.get('world')
    print(mat.toarray()[1,col_index])
    

    

    
   
