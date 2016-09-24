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
    get_doc_path('./test') 
    print(remove_stopword('111'))
    print(glv.DOC_NAME_LIST)
    with open(glv.DOC_NAME_LIST[0], 'r' , encoding='utf-8') as f:
        print(f.read())


    

    

    
   
