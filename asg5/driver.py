# -*- coding: utf-8 -*-
"""
Adam Sorensen
Machine Learning Fall 2017
University of Utah
Main driver function for ASG 5
"""

from SVM import *
from bayes import *
from logistic_regression import *
import math
# Max index: 67692

def main():
    trainFile = './data/speeches.train.liblinear'
    testFile = './data/speeches.test.liblinear'
    train = importFile(trainFile)
    test = importFile(testFile)
    
    data, labels = get_data(train)
    testD, Tlabels = get_data(test)
    
    
    
    
    
    # logistic regression
    #regression_main(data, labels)
    
    # BAYES
    #test_H_bayes()
    naive_bayes(data, labels, .5, testD, Tlabels)
    
    # SVM
    #test_SVM()
    #w = s_adjust_v(data, labels, r, tradeOff, e)
    #a = test_accuracy(testD, testLabels, w, b)
    
    
    print("done")
    
    
def importFile(file):
    results = []
    with open(file) as inputFile:
        for l in inputFile:
            results.append(l.strip().split(' '))
    return results
    
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
        
    
main()