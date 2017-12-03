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
from bagged_trees import *

# Max index: 67692
SIZE = 67693

def main():
    trainFile = './data/speeches.train.liblinear'
    testFile = './data/speeches.test.liblinear'
    train = importFile(trainFile)
    test = importFile(testFile)
    
    data, labels = get_data(train)
    testD, Tlabels = get_data(test)
    
    
    
    
    
    # logistic regression
    w = logistic_main(data, labels, 1, .0001, 100)
    a = test_accuracy(testD, Tlabels, w)
    print("Logistic regression test accuracy: " + str(a))
    #test_logistic()
    
    # BAYES
    #test_H_bayes()
    naive_bayes(data, labels, .5, testD, Tlabels)
    
    # SVM
    #test_SVM()
    w = s_adjust_v(data, labels, 10, .0001, 3)
    a = test_accuracy(testD, Tlabels, w)
    print("SVM test accuracy: " + str(a))
    
    # bagged trees
    #bagged_trees(data, labels, testD, Tlabels)
    
    
    
    
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