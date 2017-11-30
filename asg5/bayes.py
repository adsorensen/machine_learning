# -*- coding: utf-8 -*-
"""
Adam Sorensen
Machine Learning Fall 2017
University of Utah
Naive Bayes classifier implementation
"""

def naive_bayes(data, labels):
    pos, neg = get_counts(labels)
    bayes_helper(data, labels, pos, neg)
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
    return pos, neg
    
    
def bayes_helper(data, labels, pos, neg):
    c = get_count_of_word(data, labels, 4, -1)
    p = get_prop(neg, c, 1)
    
    
    
def get_prop(count, c, s):
    t = 2*s
    count = float(count)
    p = float((c + s) / (count + t))
    return p
    
def get_count_of_word(data, labels, word, label):
    key = str(word)
    count = 0
    i = 0
    for d in data:
        if key in d and labels[i] == label:
            count = count + 1
        i = i + 1
    return count
    
    
