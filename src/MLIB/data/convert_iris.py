"""
Convert the iris dataset into LibSVM format.
"""
import sys

label = {"Iris-setosa":1, "Iris-versicolor":2, "Iris-virginica":3}

F = open("iris.data")
for line in F:
    #print line
    p = line.strip().split(",")
    lbl = label[p[4]]
    print "%d 1:%s 2:%s 3:%s 4:%s" % (lbl,p[0],p[1],p[2],p[3])
F.close()
    
