# -*- coding: utf-8 -*-
"""
Adam Sorensen
Machine Learning Fall 2017
University of Utah
Naive Bayes classifier implementation
"""

SIZE = 67693
import math

def naive_bayes(data, labels, smooth, testData, testLabels):
    pos, neg, p1, p0 = get_counts(labels)
    A, B = construct_AB(data, labels, pos, neg, smooth)
    a = test_bayes_accuracy(testData, testLabels, A, B, p1, p0)
    return a
    #bayes_helper(data, labels, pos, neg)
    #c = get_count_of_word(data, labels, 900, 1)
    #print(c)
    
    
    
def get_counts(labels):
    size = len(labels)
    pos = 0
    neg = 0
    
    for l in labels:
        if l == 1:
            pos = pos + 1
        else:
            neg = neg + 1
    priors1 = float(pos / float(size))
    priors0 = float(neg / float(size))
    return pos, neg, priors1, priors0

    
    
def get_prop(count, c, s):
    t = 2*s
    count = float(count)
    p = float((c + s) / (count + t))
    return p
    
def get_count_of_feature(data, labels, word, label):
    key = str(word)
    count = 0
    i = 0
    for d in data:
        if key in d and labels[i] == label:
            count = count + 1
        i = i + 1
    return count
    
def construct_AB(data, labels, pos, neg, s):
    A = [0]
    B = [0]
    for i in range(1, SIZE):
        c = get_count_of_feature(data, labels, i, 1)
        c = get_prop(pos, c, s)
        A.insert(i, c)
    
    for i in range(1, SIZE):
        c = get_count_of_feature(data, labels, i, -1)
        c = get_prop(neg, c, s)
        B.insert(i, c)
        
    return A, B
    
    
def test_bayes_accuracy(testData, Tlabels, A, B, p1, p0):
    l = 0
    correct = 0
    total = float(len(Tlabels))
    for dic in testData:
        label1 = 0.0
        label0 = 0.0
        
        for i in range(1, SIZE):
            key = str(i)
            if key in dic:
                label1 = label1 + math.log(A[i], 2)
                label0 = label0 + math.log(B[i], 2)
            else:
                label1 = label1 + math.log((1-A[i]), 2)
                label0 = label0 + math.log((1-B[i]), 2)
                
        label1 = label1 + math.log(p1, 2)
        label0 = label0 + math.log(p0, 2)
        # mark label as 1
        if label1 > label0 and Tlabels[l] == 1:
            correct = correct + 1
        elif label0 > label1 and Tlabels[l] == -1:
            correct = correct + 1
        else:
            pass
        l = l + 1
    a = correct / total
            
    return a
    
def test_H_bayes():
    smooth = [2, 1.5, 1, .5]
    for s in smooth:
        a = cross_validate(s)
        print("Smoothing term: " + str(s) + " ACCURACY: " + str(a))
    
def cross_validate(s):
    cross0 = 'data/CVSplits/training00.data'
    cross1 = 'data/CVSplits/training01.data'
    cross2 = 'data/CVSplits/training02.data'
    cross3 = 'data/CVSplits/training03.data'
    cross4 = 'data/CVSplits/training04.data'
    
    r0 = importFile(cross0)
    r1 = importFile(cross1)
    r2 = importFile(cross2)
    r3 = importFile(cross3)
    r4 = importFile(cross4)
    
    train0, labels0 = get_data(r0+r1+r2+r3)
    test0, Tlabels0 = get_data(r4)
    
    train1, labels1 = get_data(r0+r1+r2+r4)
    test1, Tlabels1 = get_data(r3)
    
    train2, labels2 = get_data(r0+r1+r3+r4)
    test2, Tlabels2 = get_data(r2)
    
    train3, labels3 = get_data(r0+r2+r3+r4)
    test3, Tlabels3 = get_data(r1)
    
    train4, labels4 = get_data(r1+r2+r3+r4)
    test4, Tlabels4 = get_data(r0)
    
    
    pos, neg, p1, p0 = get_counts(labels0)
    A, B = construct_AB(train0, labels0, pos, neg, s)
    a0 = test_bayes_accuracy(test0, Tlabels0, A, B, p1, p0)
    
    pos, neg, p1, p0 = get_counts(labels1)
    A, B = construct_AB(train1, labels1, pos, neg, s)
    a1 = test_bayes_accuracy(test1, Tlabels1, A, B, p1, p0)
    
    pos, neg, p1, p0 = get_counts(labels2)
    A, B = construct_AB(train2, labels2, pos, neg, s)
    a2 = test_bayes_accuracy(test2, Tlabels2, A, B, p1, p0)
    
    pos, neg, p1, p0 = get_counts(labels3)
    A, B = construct_AB(train3, labels3, pos, neg, s)
    a3 = test_bayes_accuracy(test3, Tlabels3, A, B, p1, p0)
    
    pos, neg, p1, p0 = get_counts(labels4)
    A, B = construct_AB(train4, labels4, pos, neg, s)
    a4 = test_bayes_accuracy(test4, Tlabels4, A, B, p1, p0)
    
    final = (a0 + a1 + a2 + a3 + a4) / float(5)
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