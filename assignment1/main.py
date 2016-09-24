# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:24:36 2016

@author: han
"""


from get_doc_path import get_doc_path
from process_docs import process_docs
from calculate_tf_idf import calculate_tf_idf
import global_var as glv  
 
    

  
if __name__ == '__main__':  
    get_doc_path('./ICML') 
    docs = process_docs()
    print('out---------------------')
    calculate_tf_idf(docs)
    #print(docs)
    #print(glv.TFIDF_MATRIX.toarray())
    print('文章总数:', glv.ALL_FILE_NUM)
    print(glv.TFIDF_MATRIX.get_shape())
    print(glv.TFIDF_MATRIX.getnnz())


    

    

    
   
