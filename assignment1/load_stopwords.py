# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 10:20:00 2016

@author: han
"""

def load_stopwords(filename):
    infile = open(filename, 'r')
    while 1:
        line = infile.readline()
        if line == ' ':
            break
        print (line)
    infile.close()
    return
    
if __name__ == '__main__':
    load_stopwords('stop_words.txt')