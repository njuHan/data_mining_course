# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 10:20:00 2016

@author: han
"""

import global_var as glv  
from get_doc_path import get_doc_path
from porter_stemmer import porter_stemmer
from itertools import chain
import re

def is_stopword(word):
    return (word in glv.STOP_WORD_LIST)
    
def get_words(line):
    words = []
    
    #使用正则表达式，把单词提出出来
    #长度大于2，由 字母 下划线 中划线组成
    raw_words= re.findall("[\w-]{3,}",line)    

    #to lower and remove stop words
    for w in raw_words:
        w =  w.lower()
        if is_stopword(w)==False:
            words.append(w)
            
    # porter stemmer    
    for i in range(len(words)):
        words[i] = porter_stemmer(words[i])
        
    return words

def process_doc(doc_path):
    print('processing doc: ',doc_path)
    infile = open(doc_path, 'r', encoding='utf-8')
    doc = []
    for line in infile:
        if line.strip() == '':
            continue
        #print('------------------\n',line)
        words = get_words(line)
        #print(words)
        doc = chain(doc,words)
    infile.close()
    return list(doc)
    
 
def process_docs():
    docs = []
    i = 0
    for e in glv.DOC_PATH_LIST:
        doc = process_doc(e)
        print('第',i,'篇文章')
        i = i+1
        print('文章词数:',len(doc))
        #print(doc)
        docs.append(doc)
    return docs

if __name__ == '__main__':  
    get_doc_path('./test') 
    docs = process_docs()
    print(docs)
    print(glv.ALL_FILE_NUM)
    
    