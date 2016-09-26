# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 20:20:22 2016

@author: han
"""

import global_var as glv

def output_result():
    m,n = glv.TFIDF_MATRIX.get_shape()

    # output word dic
    glv.WORD_LIST = [0 for x in range(0, n)]    
    for k, v in glv.WORD_DIC.items():
        glv.WORD_LIST[v] = str(v)+':'+k
    
    result = []
    # 文件夹 7 的结果
    result_7 = []
    result.append(','.join(glv.WORD_LIST)+'\n')
    result_7.append(','.join(glv.WORD_LIST)+'\n')
    for i in range(m):
        temp_array = glv.TFIDF_MATRIX.getrow(i).toarray()[0]
        line = process_array(temp_array)
        result.append(line)
        if i>=420 and i<=445:
            result_7.append(line)
        
    outfile = open('result_all.txt','w+',encoding='utf-8')
    outfile.writelines(result)
    outfile.close()
    
    outfile = open('result_7.txt','w+',encoding='utf-8')
    outfile.writelines(result_7)
    outfile.close()
        

# if you get a legnth 7 feature vector as [1.2, 0.6, 0, 0, 0, 0, 1], 
#it should be transformed into [0:1.2, 1:0.6, 6:1].
def process_array(array):
    line = ''
    for i in range(len(array)):
        if array[i]!=0 :
            line = line + str(i)+':'+str(array[i])+','
            
    line = line[:-1] + '\n'     
    
    return line
        
    
        
    