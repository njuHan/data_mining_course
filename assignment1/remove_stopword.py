# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 10:20:00 2016

@author: han
"""

import global_var as glv  


def remove_stopword(line):
   
    return (line in glv.STOP_WORD_LIST)

    