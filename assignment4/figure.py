# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 13:00:57 2016

@author: han
"""

import numpy as np
import pylab as plt

def load_data(file_name):
    fopen = open(file_name)
    lines = [line.strip() for line in fopen.readlines()]
   
    fopen.close()
    return lines
    
def plot_error_rate(train_error, test_error, fig_name):
    x = list(range(len(train_error)))
    plt.plot(x, train_error, 'b--o', label='train error rate')
    plt.plot(x, test_error,'r-*' , label='test error rate')
    plt.legend()
    
    plt.savefig(fig_name, dpi=300)
    plt.show()
    
def plot_obj_func(obj_func, fig_name):
    x = list(range(len(obj_func)))
    
    plt.plot(x, obj_func, 'b--o', label='obj func')
    plt.legend()
    
    plt.savefig(fig_name, dpi=300)
    plt.show()

if __name__ == '__main__': 
    
    # task1
    train_error = load_data('task1_dataset1_train_error.txt')
    test_error = load_data('task1_dataset1_test_error.txt')
    plot_error_rate(train_error, test_error, 'task1_error_rate.png')
    
    obj_func = load_data('task1_dataset1_train_func.txt')
    plot_obj_func(obj_func, 'task1_obj_func.png')
    
    
    #task2
    train_error = load_data('task2_dataset1_train_error.txt')
    test_error = load_data('task2_dataset1_test_error.txt')
    plot_error_rate(train_error, test_error, 'task2_error_rate.png')
    
    obj_func = load_data('task2_dataset1_train_func.txt')
    plot_obj_func(obj_func, 'task2_obj_func.png')
    
    
   









#x = np.linspace(0, 2*np.pi, 50)
#y = np.sin(x)
#y2 = y + 0.1 * np.random.normal(size=x.shape)
#
#fig, ax = plt.subplots()
#ax.plot(x, y, 'k--')
#ax.plot(x, y2, 'ro')
#
## set ticks and tick labels
#ax.set_xlim((0, 2*np.pi))
#ax.set_xticks([0, np.pi, 2*np.pi])
#ax.set_xticklabels(['0', '$\pi$', '2$\pi$'])
#ax.set_ylim((-1.5, 1.5))
#ax.set_yticks([-1, 0, 1])
#
## Only draw spine between the y-ticks
#ax.spines['left'].set_bounds(-1, 1)
## Hide the right and top spines
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
## Only show ticks on the left and bottom spines
#ax.yaxis.set_ticks_position('left')
#ax.xaxis.set_ticks_position('bottom')
#
#plt.show()