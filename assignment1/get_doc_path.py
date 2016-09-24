# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:21:41 2016

@author: han
"""
import os
import global_var as glv  


def get_doc_path(path):  
    
    # 所有文件夹
    dir_list = []  
    # 所有文件  
    file_list = []  
    # 返回一个列表，其中包含在目录条目的名称
    files = os.listdir(path)  
    
    for f in files:  
        if(os.path.isdir(path + '/' + f)):  
            # 排除隐藏文件夹。因为隐藏文件夹过多  
            if(f[0] == '.'):  
                pass  
            else:  
                # 添加非隐藏文件夹  
                dir_list.append(f)  
        if(os.path.isfile(path + '/' + f)):  
            # 添加.txt文件
            if (f[-4:]=='.txt'):
                file_list.append(f)      
    
    for dl in dir_list:  
        #print ('dir: ' , dl ) 
        get_doc_path(path + '/' + dl)  
    for fl in file_list:  
        # 打印文件 
        filepath = path + '/' + fl
        #print (filepath) 
        glv.ALL_FILE_NUM = glv.ALL_FILE_NUM + 1  
        glv.DOC_PATH_LIST.append(filepath)
        
        
        