#! -*- coding: utf-8 -*-

"""
LEARNER is a generic class for classification algorithms.
It supports common functionalities such as reading and writing training
models, performing cross-validation, held out evaluation,
calling various training algorithms with specific parameters.
Exact learning procedures are implemented in separate classes that
inherit LEARNER class.

Danushka Bollegala.
2009/12/31.
"""

import sys

sys.path.append("../..")

from MLIB.utils.data import SEQUENTIAL_FILE_READER
from MLIB.classify.predict import PREDICTOR, EVALUATOR

class LEARNER:

    def __init__(self):
        pass

    def get_performance(self,fvects):
        """
        Compute precision, recall and F-scores with the current
        weight vector for the fvects using the EVALUATOR.. 
        """
        E = EVALUATOR(self.n)
        pred = PREDICTOR()
        pred.loadWeights(self.w,self.bias,self.n)
        for v in fvects:
            (lbl,prob) = pred.predictVect(v)
            E.add(v.label,lbl)
        return E

    def getActiveFeatureCount(self):
        """
        Returns the number of non-zero weights in the weight vector.
        """
        count = 0
        for lbl in self.labels:
            count += len(self.w[lbl])
        return count

    def writeModel(self,no_features,model_fname):
        """
        Write the trained model to a file.
        """
        model = open(model_fname,"w")
        model.write("@classes=%d\n" % self.n)
        model.write("@L2=%s\n" % str(self.lmda))
        model.write("@Iterations=%d\n" % self.total_iterations)
        model.write("@Initial Learning Rate=%s\n" % str(self.eta0))
        model.write("@Total features in training data=%d\n" % (no_features))
        model.write("@Active (non-zero) features=%d\n" % \
                    self.getActiveFeatureCount())
        #write the weights
        model.write("\n@WEIGHTS\n")
        for lbl in self.labels:
            model.write("@CLASS %d\n" % lbl)
            curLabel = -1 if lbl == 0 else lbl
            model.write("bias=%s\n" % str(self.bias[lbl]))
            weights = self.w[lbl].items()
            weights.sort(self.weight_sort)
            for (fid,fval) in weights:
                if fval != 0:
                    model.write("%d,%s\n" % (fid,str(fval)))
        model.close()
        pass    

    def weight_sort(self,A,B):
        """
        Sorts L according to the feature values.
        L is a list of tuples ((lbl,fid),fval)
        """
        if A[1] > B[1]:
            return -1
        return 1

    def train(self,fvects,fids):
        """
        Train using SGD. Supports crossValidation. This function
        is called by outside programs.
        """
        # You cannot set both heldout and cross-validation.
        if self.folds and self.heldoutVects:
            sys.stderr.write(
                "Cannot perform heldout and cross-validation simultaneously\n")
            sys.exit(-1)
        if self.folds == 0:
            return self.train_client(fvects,fids)
        else:
            # store the training data folds.
            print "Performing %d-fold cross-validation" % self.folds
            trainVects = {}
            fold = 0
            count = 0
            for fv in fvects:
                trainVects.setdefault(fold,[]).append(fv)
                fold += 1
                if fold == self.folds:
                    fold = 0
            # train for each fold. Accumulate statistics.
            stats = {}
            statKeys = ["macro","micro"]
            statKeys.extend(self.labels)
            for sk in statKeys:
                stats[sk] = {}
                for metric in ["precision","recall","F","accuracy"]:
                    stats[sk][metric] = 0
            for i in trainVects:
                print "Fold number = %d" % (i+1)
                traindata = []
                for j in trainVects:
                    if j != i:
                        traindata.extend(trainVects[j])
                self.train_client(traindata,fids)
                e = self.get_performance(trainVects[i])
                results = e.getMetrics()
                e.show(results)                
                # add the current metrics to stats.
                for sk in statKeys:
                    for metric in ["precision","recall","F","accuracy"]:
                        stats[sk][metric] += results[sk][metric]
            # print the overall results and averages.
            print "Average Results over %d-fold cross validation" % self.folds
            for sk in statKeys:
                for metric in ["precision","recall","F","accuracy"]:
                    stats[sk][metric] /= float(self.folds)
            E = EVALUATOR(self.n)
            E.show(stats)
        pass

    def display_training_error(self,fvects):
        """
        Display information regarding training error.
        """
        # Compute performance on training data.
        print "\n.......Peformance on Training Data........"
        E = self.get_performance(fvects)
        result = E.getMetrics()
        E.show(result)
        return result

    def display_heldout_error(self,fvects):
        """
        Display information regarding heldout data error.
        """           
        # Compute performance on heldout data.
        print "\n.......Peformance on Heldout Data.........."
        E = self.get_performance(fvects)
        result = E.getMetrics()
        E.show(result)
        return result
    pass

def test_logreg(model_fname,test_fname,output_fname=None, prob=True,acc=True):
    """
    Predict labels for the test instances using the trained
    model. If prob is set to True, then show class probabilities.
    If acc is set to True and if the test instances have labels,
    then we will predict accuracies for the test instances.
    If an output_fname is specified we will write the predictions to
    the file instead of writing to the terminal.
    """
    pred = PREDICTOR()
    pred.loadModel(model_fname)
    testFile = SEQUENTIAL_FILE_READER(test_fname)
    count = 0
    E = EVALUATOR(pred.n)
    if output_fname:
        output = open(output_fname,"w")
    else:
        output = sys.stdout
    for mv in testFile:
        v = mv["vect"]
        (lbl,prob) = pred.predictVect(v)
        output.write("%d\t%s\n" % (lbl,str(prob)))
        if pred.n == 2 and v.label == -1 :
            trueLabel = 0
        else:
            trueLabel = v.label
        if v.label is not None:
            E.add(trueLabel,lbl)
        count += 1
    testFile.close()
    if acc:
        result = E.getMetrics()
        E.show(result)
    pass
