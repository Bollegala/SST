"""
This module implements the Pegasos SVM algorithm.


@inproceedings{Shwartz:ICML:2007,
    Author = {Shai Shalev-Shwartz and Yoram Singer and Nathan Srebro},
    Booktitle = {ICML'07},
    Pages = {807 -- 814},
    Title = {Pegasos: Primal Estimated sub-GraAdient SOlver for SVM},
    Year = {2007}}


Danushka Bollegala.
2011/06/04.
"""

import sys

sys.path.append("../..")

from math import sqrt
import random
from numpy import ones, dot, zeros
from time import time
import pickle

from MLIB.utils.data import convertLineToArray, convertSEQFileToArray

class PEGASOS:
    
    def __init__ (self):
        # cost parameter on weight norm.
        self.l = 1
        # total number of epohs.
        self.T = 100
        # learnt weight vector.
        self.lw = None
        # Dimensionality of the feature space.
        self.d = -1
        pass
    
    def train(self, trainFileName):
        """
        This function implements the main training routine
        for Pegasos algorithm.
        """
        # Print ths stats.
        print "L2 parameter =", self.l
        print "Total iterations =", self.T
        # Load the feature vectors.
        (labels, vects) = convertSEQFileToArray(trainFileName)
        # Get the (max) dimensionality d of vectors.
        d = vects[0].size
        self.d = d
        # Get the total no. of training vectors.
        n = len(vects)
        print "Total no. of training instances = %d" % n
        # Construct the initial weight vector.
        factor = 1.0 / sqrt(d * self.l)
        w = factor * ones(d)
        b = 0
        # Iterative procedure.
        for t in xrange(1, (self.T + 1)):
            startTime = time()
            z = zeros(d)
            bSum = 0
            k = 0
            for i in xrange(0,n):
                x = vects[i]
                y = labels[i]
                score =  (y * (dot(w,x) + b))
                #print score
                if score < 0:
                    # Misclassification.
                    z += (y * x)
                    bSum += y
                    k += 1
            eta = 1.0 / (self.l * t)
            whalf = ((1.0 - (eta * self.l)) * w) + ((eta / k) * z)
            scale = sqrt(self.l) * sqrt(dot(whalf,whalf))
            scale = min((1.0, (1.0 / scale)))
            w = scale * whalf
            b = b + ((eta / k) * bSum)
            endTime = time()
            accuracy = 100 * (1.0 - (float(k) / n))
            print "Iteration: %d" % t
            print "Time taken for this iteration %.3fs" % (endTime - startTime)
            print "Weight norm = %f" % (sqrt(dot(w,w))) 
            print "Training accuracy = %f\n" % accuracy  
        # Set the learnt weight vector.
        self.lw = w   
        self.bias = b
        pass
    
    def writeModel(self, modelFileName):
        """
        Write the learnt weight vector to modelFileName.
        """
        # write the meta information to the model.
        modelFile = open(modelFileName, "w")
        modelFile.write("# Algorithm = Pegasos\n")
        modelFile.write("# Lambda = %s\n" % str(self.l))
        modelFile.write("# Dimensionality = %d\n" % len(self.lw))
        modelFile.write("# Epochs = %d\n" % self.T)
        modelFile.write("# Bias = %s\n" % str(self.bias))
        # write the weights to the model file.
        for i in range(0, self.lw.size):
            modelFile.write("%d %f\n" % ((i + 1), self.lw[i]))
        modelFile.close()
        pass
    
    def readModel(self, modelFileName):
        """
        Read the weight vector from the modelFileName.
        """
        modelFile = open(modelFileName, "r")
        for line in modelFile:
            if line.startswith("# Algorithm"):
                continue
            elif line.startswith('# Lambda'):
                self.l = float(line.strip().split('=')[1])
            elif line.startswith('# Dimensionality'):
                self.d = int(line.strip().split('=')[1])
                self.lw = zeros(self.d)
            elif line.startswith('# Epochs'):
                self.T = int(line.strip().split('=')[1])
            elif line.startswith('# Bias'):
                self.bias = float(line.strip().split('=')[1])
            else:
                p = line.strip().split()
                if len(p) != 2:     
                    sys.stderr.write("Invalid feature and weight in model %s\n" % line)
                    raise ValueError
                fid = int(p[0]) - 1
                fval = float(p[1])
                self.lw[fid] = fval
        modelFile.close()
        pass    
    
    def predictInstance(self, fv):
        """
        Predict the class label of the feature vector fv.
        """
        score = self.bias + dot(fv, self.lw) 
        #score = random.random() - 0.5
        if score > 0:
            label = 1
        else:
            label = -1
        return (label, score)
    
    def predictFile(self, testFileName, 
                    outputFileName=None,
                    showPreds=True, showStats=True):
        """
        Predict the class labels for each of the feature vectors
        in the testFileName. Compute accuracy if showStats is
        True and the actual labels are provided.
        """
        truePositives = 0
        falsePositives = 0
        trueNegatives = 0
        falseNegatives = 0
        positives = 0
        negatives = 0
        count = 0
        testFile = open(testFileName, "r")
        if outputFileName:
            outputFile = open(outputFileName, "w")
        for line in testFile:
            count += 1
            (lbl, x) = convertLineToArray(line.strip(), 
                                          self.d, 'int')
            (pred, score) = self.predictInstance(x)
            if showPreds:
                print pred, score
            if outputFileName:
                outputFile.write("%d, %s\n" % (pred, str(score)))
            if lbl == 1:
                positives += 1
            elif lbl == -1:
                negatives += 1  
            if pred == 1 and lbl == 1:
                truePositives += 1
            elif pred == 1 and lbl == -1:
                falsePositives += 1
            elif pred == -1 and lbl == 1:
                falseNegatives += 1
            elif pred == -1 and lbl == -1:
                trueNegatives += 1 
        testFile.close()
        if showStats and positives and negatives:
            corrects = truePositives + trueNegatives
            accuracy = float(100 * corrects) / count
            print "Accuracy = %f (%d/%d)" % (accuracy, corrects, count)
            print "True Positive Rate = %f" % (float(100 * truePositives) / positives)
            print "False Positive Rate = %f" % (float(100 * falsePositives) / negatives)
            print "True Negative Rate = %f" % (float(100 * trueNegatives) / negatives)
            print "False Negative Rate = %f" % (float(100 * falseNegatives) / positives)
            precision = float(100 * truePositives) / (truePositives + falsePositives)
            recall = float(100 * truePositives) / (positives)
            F = (2 * precision * recall) / (precision + recall)
            print "Precision = %f" % precision
            print "Recall = %f" % recall
            print "F-score = %f" % F
            print "Total Positives = %d" % positives
            print "Total Negatives = %d" % negatives
            print "Total instances = %d" % count
        if outputFileName:
            outputFile.close()
        pass
    pass

def trainPegasos(l, T, modelFileName, trainFileName):
    """
    This is a utility function to train with Pegasos.
    """
    Learner = PEGASOS()
    Learner.l = l
    Learner.T = T
    Learner.train(trainFileName)
    Learner.writeModel(modelFileName)
    pass

def testPegasos(modelFileName, testFileName, 
                predFileName=None, 
                showPreds=True, showStats=True):
    """
    This is a utility function to test with Pegasos.
    """
    Learner = PEGASOS()
    Learner.readModel(modelFileName)
    Learner.predictFile(testFileName, 
                        predFileName,
                        showPreds, showStats)
    pass

def debug():
    Learner = PEGASOS()
    Learner.train("../data/a1a.train")
    Learner.writeModel("../work/model")
    Learner.readModel("../work/model")
    #Learner.predictFile("../data/a1a.test")
    Learner.predictFile("../data/a1a.train")

    pass

if __name__ == "__main__":
    debug()

