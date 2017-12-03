# -*- coding: utf-8 -*-
"""
Adam Sorensen
Machine Learning Fall 2017
University of Utah
Bagged trees implementation
"""

SIZE = 67693
import math
from random import randint

COUNT = 0

class Node:
    def __init__(self, element):
        #self.key = key
        self.element = element
        self.yes = None
        self.no = None

    def getEl(self):
        return self.element

    def getYes():
        return self.yes
        
    def getNo():
        return self.no
        
class dataPoint:
    def __init__(self, data, label):
        self.data = data
        self.label =label
        
    def getData(self):
        return self.data
        
    def getLabel(self):
        return self.label
        
        
def bagged_trees(data, labels, Tdata, Tlabels):
    points = make_data_points(data, labels)
    testPoints = make_data_points(Tdata, Tlabels)
    
    depth = 3
#==============================================================================
#     r = ID3main(points, depth)
#     a, l = testData(testPoints, r, depth)
#     print(a)
#==============================================================================
    
    
    trees = train_trees(1000, points, depth)
    a = test_trees(trees, testPoints)
    print("test accuracy: " + str(a))
    a = test_trees(trees, points)
    print("train accuracy: " + str(a))
    #print(type(points[0].getLabel()))
    
    # train 1000 trees
    
    
def train_trees(amount, points, depth):
    trees = []
    size = len(points)
    randomPoints = []
    for i in range(0, amount):
        if i % 100 == 0:
            print("have created " + str(i) + "trees")
        randomPoints[:] = []
        for n in range(0, 100):
            m = randint(0, size - 1)
            randomPoints.append(points[m])
            
        t = ID3main(randomPoints, depth)
        trees.append(t)
        
        
    return trees
        
        
def test_trees(trees, points):
    size = len(points)
    for p in points:
        p = 0
        n = 0
        correct = 0
        label = p.getLabel()
        for t in trees:
            l = traverse_tree(p, t, 4)
            if l == 1:
                p = p + 1
            else:
                n = n + 1
        if p > n and label == 1:
            correct = correct + 1
        elif n > p and label == -1:
            correct = correct + 1
        else:
            pass
    ac = correct/float(size)
    return ac
        
def traverse_tree(data, r, depth):
    n = r
    for i in range(0, depth):
        e = n.element
        if e == '-1' or e =='1':
            if e == '1':
                return 1
            else:
                return -1
        if str(e) in data.getData():
            n = n.yes
        else:
            n = n.no
        
    
    
    
def make_data_points(data, labels):
    points = []
    i = 0
    for d in data:
        l = str(labels[i])
        x = dataPoint(d, l)
        points.append(x)
        i = i + 1
        
    return points
    
    
        
def getAllPosNeg(points):
    
    pos = []
    neg = []
    for p in points:
        if p.getLabel() == '1':
            pos.append(p)
        else:
            neg.append(p)

    return pos, neg
    
def posNeg(points, A):
    pos = []
    neg = []
    
    i = 0
    for p in points:
        if A[i] == 1:
            pos.append(p)
        else:
            neg.append(p)
        i = i + 1
    
    return pos, neg
    
def getEntropy(pos, neg):
    psize = len(pos)
    nsize = len(neg)
    allsize = psize + nsize
    
    
    pfrac = (psize/float(allsize))
    nfrac = (nsize/float(allsize))
    if pfrac == 0 or nfrac == 0:
        return 0
    
    e = (pfrac * -1) * math.log(pfrac, 2) - nfrac * math.log(nfrac, 2)
    return e
    
def findCommonLabel(points):
    pos = 0
    neg = 0
    for p in points:
        if p.getLabel() == '1':
            pos = pos + 1
        else:
            neg = neg + 1
    if pos >= neg:
        return '1'
    else:
        return '-1'
    

def ID3main(points, maxDepth):
    attributes = []
    for i in range(1, SIZE):
        attributes.append(i)
    global COUNT
    COUNT = 0
    
    root = ID3rec(points, attributes, maxDepth)
    return root
    
        
def ID3rec(points, attributes, maxDepth):
    global COUNT
    #print(COUNT)
    
    commonLabel = findCommonLabel(points)
    x = COUNT
    if x > maxDepth:
        return Node(commonLabel)
    
        
    pos, neg = getAllPosNeg(points)
    first = points[0].getLabel()
    
    flag = False
    for p in points:
        if p.getLabel() == first:
            flag = True
        else:
            flag = False
            break
    
    if(flag):
        COUNT = COUNT - 1
        return Node(first)
    else:
        # make root node
        if len(attributes) == 0:
            COUNT = COUNT - 1
            return Node(commonLabel)
        else:
            entropy = getEntropy(pos, neg)
            e = getBestAttribute(points, attributes, entropy)
            
            #s = attributes.get(e)
            A = []
            for p in points:
                d = p.getData()
                if str(e) in d:
                    A.append(1)
                else:
                    A.append(0)
            
            
            newAttributes = attributes[:]
            
            
            root = Node(e)
            
            # yes and no are sets divided whether A is 1 or 0
            yes, no = posNeg(points, A)
           
            if len(yes) == 0:
                COUNT = COUNT - 1
                root.yes = Node(commonLabel)
            else:
                COUNT = COUNT + 1
                root.yes = ID3rec(yes, newAttributes, maxDepth)
                
            if len(no) == 0:
                COUNT = COUNT - 1
                root.no = Node(commonLabel)
            else:
                COUNT = COUNT + 1
                root.no = ID3rec(no, newAttributes, maxDepth) 
                
            COUNT = COUNT - 1
            return root
        
 
def getBestAttribute(points, attributes, e):
    labels = []
    for p in points:
        labels.append(p.getLabel())
    
    #testAttributes = list(attributes.keys())
    m = -0.2
    element = 1
    for i in range(1, SIZE):
        if i in attributes:
            feature = []
            for p in points:
                d = p.getData()
                if str(i) in d:
                    feature.append(1)
                else:
                    feature.append(0)
                
            
            info = getInfoGain(feature, labels, e)
            
            if m < info:
                m = info
                element = i
        else:
            pass
    return element
        
    
def getInfoGain(feature, labels, eAll):
    size = len(feature)
    size2 = len(labels)
    
    # feature, label
    yy = 0
    yn = 0
    ny = 0
    nn = 0
    
    for i in range(0, size2):
        if (labels[i] == '1'):
            if feature[i] == 1:
                yy = yy + 1
            else:
                ny = ny + 1
        else:
            if feature[i] == 1:
                yn = yn + 1
            else:
                nn = nn + 1
    totalyes = yy + yn
    totalno = ny + nn
    
    if totalyes == 0:
        temp1 = 0
        temp2 = 0
    else:
        temp1 = yy/float(totalyes)
        temp2 = yn/float(totalyes)
    
    #print(yy, yn, ny, nn)

    if temp1 == 0 or temp2 == 0:
        e1 = 0
    else:
        e1 = (temp1 * -1) * math.log(temp1, 2) - (temp2 * math.log(temp2, 2))
    
    if totalno == 0:
        temp1 = 0
        temp2 = 0
    else:
        temp1 = ny/float(totalno)
        temp2 = nn/float(totalno)
    
    if temp1 == 0 or temp2 == 0:
        e2 = 0
    else:
        e2 = (temp1 * -1) * math.log(temp1, 2) - (temp2 * math.log(temp2, 2))
     
    temp1 = totalyes/float(size)
    temp2 = totalno/float(size)
    
    eExpected = (temp1 * e1) + (temp2*e2)
    infoGain = eAll - eExpected
    
    return infoGain
    
def traverseTree(data, r, depth):
    n = r
    for i in range(0, depth):
        e = n.element
        if e == '-1' or e =='1':
            if data.getLabel() == e:
                return 1
            else:
                return 0
        if str(e) in data.getData():
            n = n.yes
        else:
            n = n.no
            
            
def getDepth(root):
    d = height(root)
    return d
    
    
def height(n):
    if (n.yes == None and n.no == None):
        return 0
    else:
        return max(height(n.yes), height(n.no)) + 1
        
        
        
        
def testData(points, r, depth):
    correct = 0
    incorrect = 0
    labels = []
    #size2 = len(features[0])
    size = len(points)
    for p in points:
        
        t = traverseTree(p, r, depth)
        if t == 1:
            labels.append(1)
            correct = correct + 1
        else:
            labels.append(0)
            incorrect = incorrect + 1
        
            
        
#==============================================================================
#     for i in range(0, size):
#         data = []
#         for f in features[0:]:
#             print(f)
#             data.append(f[i])
#             
#         
#         t = traverseTree(data, r, size2)
#         if t == 1:
#             correct = correct + 1
#         else:
#             incorrect = incorrect + 1
#         
#         del data[:]
#==============================================================================
        

    #print('Stats for ' + file)
    #print('correct= ' , correct, '   incorrect= ', incorrect)
    ac = 0.0
    ac = correct/float(size)
    #print('accuracy: ', ac)
    #print('')
    return ac, labels