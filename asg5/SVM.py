# -*- coding: utf-8 -*-
"""
Adam Sorensen
Machine Learning Fall 2017
University of Utah
SVM algorithm
"""
SIZE = 67693

import math
import random

def simple_main(data, labels, r, tradeOff, e):
    w = initialize_weights()
    # r = 1 or .1, .01
    w, b, c = s_adjust_v(data, labels, w, r, e)
    #a = test_accuracy(data, w, b)
    return w, b, c
    
    
def s_adjust_v(data, labels, r, tradeOff, e):
    w = initialize_weights()
    
    for t in range(0, e):
        i = 0
        
        for dic in data:
            keys = list(dic.keys())
            random.shuffle(keys)
            y = labels[i]
            dot = 0
            for d in keys:
                dot = dot + y*w[int(d)]
                
            
            if dot <= 1:
                for b in range(0, len(w)):
                    if str(b) in dic:
                        w[b] = (1-r)*w[b] + (r*tradeOff*y*1)
                    else:
                        w[b] = (1-r)*w[b]
                        
            else:
                for b in range(0, len(w)):
                    w[b] = (1-r)*w[b]
                
                
            i = i + 1
            
    return w
    
    
def test_accuracy(data, labels, w):
    x = []
    m = 0.0;
    total = float(len(data))
    i = 0
    for dic in data:
        y = labels[i]
        dot = 0
        for d in dic:
            dot = dot + (w[int(d)])
            
        
        if y*dot <= 1:
            # mistake
            m = m + 1
        i = i + 1
            
    c = total - m
    return float(c) / total
    
def test_SVM():
    learning = [10, 1, .1, .01, .001, .0001]
    trade = [10, 1, .1, .01, .001, .0001]
    a = cross_validate(.1, .1)
    print(a)
    for r in learning:
        for t in trade:
            a = cross_validate(r, t)
            print("Learning rate: " + str(r) + " tradeoff: "+ str(t) + " ACCURACY: " + str(a))
        
    
def cross_validate(r, tradeOff):
    cross0 = 'data/CVSplits/training00.data'
    cross1 = 'data/CVSplits/training01.data'
    cross2 = 'data/CVSplits/training02.data'
    cross3 = 'data/CVSplits/training03.data'
    cross4 = 'data/CVSplits/training04.data'
    
    e = 1
    
    r0 = importFile(cross0)
    r1 = importFile(cross1)
    r2 = importFile(cross2)
    r3 = importFile(cross3)
    r4 = importFile(cross4)
    
    train0, labels0 = get_data(r0+r1+r2+r3)
    test0, Tlabels0 = get_data(r4)
    #print(test0)
    
    train1, labels1 = get_data(r0+r1+r2+r4)
    test1, Tlabels1 = get_data(r3)
    
    train2, labels2 = get_data(r0+r1+r3+r4)
    test2, Tlabels2 = get_data(r2)
    
    train3, labels3 = get_data(r0+r2+r3+r4)
    test3, Tlabels3 = get_data(r1)
    
    train4, labels4 = get_data(r1+r2+r3+r4)
    test4, Tlabels4 = get_data(r0)
    
    
       
    
    w = s_adjust_v(train0, labels0, r, tradeOff, e)
    ac0 = test_accuracy(test0, Tlabels0, w)
    
    w = s_adjust_v(train1, labels1, r, tradeOff, e)
    ac1 = test_accuracy(test1, Tlabels1, w)
    
    w = s_adjust_v(train2, labels2, r, tradeOff, e)
    ac2 = test_accuracy(test2, Tlabels2, w)
    
    w = s_adjust_v(train3, labels3, r, tradeOff, e)
    ac3 = test_accuracy(test3, Tlabels3, w)
    
    w = s_adjust_v(train4, labels4, r, tradeOff, e)
    ac4 = test_accuracy(test4, Tlabels4, w)
    
    
    
    
    
    
    final = (ac0 + ac1 + ac2 + ac3 + ac4) / float(5)
    return final
    
def get_data(results):
    data = []
    temp = {}
    labels = []
    
            
    for r in results:
        i = 0
        size = len(r)
        labels.append(int(r[0]))
        for i in range(1, size):
            temp[r[i][:-2]] = 1
        data.append(temp.copy())
        temp.clear()
        
    return data, labels
    
def importFile(file):
    results = []
    with open(file) as inputFile:
        for l in inputFile:
            results.append(l.strip().split(' '))
    return results
    
    
# function initializes weight vector to 0's
def initialize_weights():
    w = []
    for i in range(0, SIZE):
        r = 0
        w.append(r)
    return w
    
def majority(labels):
    p = 0
    n = 0  
    total = len(labels)
    for f in labels:
        if f == '1':
            p = p + 1
        else:
            n = n + 1
    ma = max(n, p)
    f = float(ma)/float(total)
    return f