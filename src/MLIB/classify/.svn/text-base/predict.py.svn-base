"""
This module contains various functionalities required by the
training algorithms such as rolling averages, predicion and
evaluation metrics.
"""

import math

class ROLLING_AVERAGER:

    """
    Compute the rolling average over given number of runs.
    If the averages are not changing beyond a certain predefined
    bound, then send a trigger.
    """

    def __init__(self,history,bound):
        """
        history is the number of previous steps over which we
        will compute the average.
        """
        self.history = history
        self.buffer = []
        self.oldAvg = 0
        self.bound = bound
        pass

    def add(self,x):
        """
        Add x to the buffer. Compute average.
        If the average has not changed, return 1.
        Otherwise return 0.
        """
        if len(self.buffer) < self.history:
            self.buffer.append(x)
            return 0
        del(self.buffer[0])
        self.buffer.append(x)
        tot = sum(self.buffer)
        avg = float(tot)/float(self.history)
        if self.oldAvg == 0:
            self.oldAvg = avg
            return 0
        ratio = math.fabs(avg-self.oldAvg)/self.oldAvg
        #print avg,self.oldAvg,ratio
        if ratio < self.bound:
            return 1
        self.oldAvg = avg
        pass
    pass    


class PREDICTOR:

    """
    This class can load a model file and predict the labels (in classification)
    or values (in regression) for test instances read from a file.
    Test instances are read sequentially. Therefore, we do not need to
    load the entire test file into memory. Moreover, we can directly
    read a test instance from the stdin and perform a prediction.
    """

    def __init__(self):
        self.w = {}
        self.bias = {}
        self.n = 0
        self.learner = None
        self.currentLabel = None
        pass

    def loadModel(self,model_fname):
        """
        Read the model file to memory.
        """
        model = open(model_fname)
        line = model.readline()
        p = line.strip().split("=")
        self.learner = p[0].strip('@')
        self.n = int(p[1])
        self.labels =[0,1] if self.n == 2 else range(1,self.n+1)
        while line.strip() != "@WEIGHTS":
            line = model.readline()
        # read the weights.
        line = model.readline()
        while line:
            if line.startswith("@CLASS"):
                lbl = int(line.split()[1])
                if self.n == 2 and lbl == -1:
                    lbl = 0
                bias = float(model.readline().split("=")[1])
                self.bias[lbl] = bias
                self.w[lbl] = {}
                line = model.readline()
                continue
            p = line.strip().split(',')
            fid = int(p[0])
            fval = float(p[1])
            self.w[lbl][fid] = fval
            line = model.readline()
        model.close()
        pass

    def loadWeights(self,w,bias,n):
        """
        Instead of reading the weights and bias terms from a model
        file, we can set them directly using this function.
        """
        self.w = w
        self.bias = bias
        self.n = n
        self.labels =[0,1] if self.n == 2 else range(1,self.n+1)
        pass

    def predictVect(self,v):
        """
        Predict the class of vector v.
        Return the class conditional probability.
        """
        pred = {}
        maxVal = None
        maxLabel = None        
        for lbl in self.labels:
            pred[lbl] = self.bias[lbl]
            pred[lbl] += sum([fval*self.w[lbl].get(fid,0) for (fid,fval) in v])
            if maxLabel is None:
                maxLabel = lbl
                maxVal = pred[lbl]
            elif maxVal < pred[lbl]:
                maxVal = pred[lbl]
                maxLabel = lbl       
        # compute the prediction probability taking care of overflow.
        if maxVal > 100:
            prob = 1.0
        else:
            tot = 0
            for lbl in self.labels:
                x = pred[lbl]
                tot += (math.exp(x)) if (x > -100) else 0
            prob = (math.exp(maxVal) / float(tot)) if tot > 0 else 0
        return (maxLabel,prob)

    def predictLine(self,line):
        pass
    pass

class EVALUATOR:

    """
    Compute micro and macro average precision,recall,F and accuracy
    values for binary and multiclass classification.
    """

    def __init__(self,categories):
        # number of classification categories.
        self.n = categories
        # T is the contingency table for each category.
        # TP: true positives, FP: false positives
        # TN: true negatives, FN: false negatives
        self.T = {}
        self.reset()
        pass

    def reset(self):
        """
        Resets all counts.
        """
        labels =[0,1] if self.n == 2 else range(1,self.n+1)
        for i in labels:
            self.T[i] = {"TP":0,"FP":0,"FN":0,"TN":0}
        pass

    def add(self,trueLabel,predictedLabel):
        """
        Accumulate a label assignment.
        """
        labels =[0,1] if self.n == 2 else range(1,self.n+1)
        if trueLabel == predictedLabel:
            self.T[predictedLabel]["TP"] += 1
            for lbl in labels:
                if lbl is not predictedLabel:
                    self.T[lbl]["TN"] += 1
        else:
            self.T[predictedLabel]["FP"] += 1
            self.T[trueLabel]["FN"] += 1
            for lbl in labels:
                if (lbl is not trueLabel) and (lbl is not predictedLabel):
                    self.T[lbl]["TN"] += 1
        pass

    def getMetrics(self):
        """
        compute evaluation metrics and display.
        """
        micro = {"TP":0, "FP":0, "TN":0, "FN":0}
        macro = {"precision":0, "recall":0, "F":0, "accuracy":0}
        result = {"macro":{}, "micro":{}}
        labels = [0,1] if self.n == 2 else range(1,self.n+1)
        for lbl in labels:
            for k in micro:
                micro[k] += self.T[lbl][k]
            A = self.T[lbl]["TP"]
            B = self.T[lbl]["FP"]
            C = self.T[lbl]["FN"]
            D = self.T[lbl]["TN"]
            precision = (float(A)/float(A+B)) if (A+B) != 0 else 0
            recall = (float(A)/float(A+C)) if (A+C) != 0 else 0
            F = (2*precision*recall)/(precision+recall) \
                if (precision != 0) and (recall !=0) else 0
            accuracy = float(A+D)/float(A+B+C+D) if (A+B+C+D) != 0 else 0
            result[lbl] = {"precision":precision,
                           "recall":recall,
                           "F":F,
                           "accuracy":accuracy}
            # add the metrics to compute macro averages.
            macro["precision"] += precision
            macro["recall"] += recall
            macro["F"] += F
            macro["accuracy"] += accuracy
        result["macro"]["precision"] = (float(macro["precision"])/float(self.n))
        result["macro"]["recall"] = (float(macro["recall"])/float(self.n))
        result["macro"]["F"] = (float(macro["F"])/float(self.n))
        result["macro"]["accuracy"] = (float(macro["accuracy"])/float(self.n))
        # compute micro averages.
        A = micro["TP"]
        B = micro["FP"]
        C = micro["FN"]
        D = micro["TN"]
        precision = (float(A)/float(A+B)) if (A+B) != 0 else 0
        recall = (float(A)/float(A+C)) if (A+C) != 0 else 0
        F = (2*precision*recall)/(precision+recall) \
            if (precision != 0) and (recall !=0) else 0
        accuracy = float(A+D)/float(A+B+C+D) if (A+B+C+D) != 0 else 0
        result["micro"]["precision"] = precision
        result["micro"]["recall"] = recall
        result["micro"]["F"] = F
        result["micro"]["accuracy"] = accuracy
        return result

    def show(self,result):
        """
        Print the results to the terminal.
        """
        labels =[0,1] if self.n == 2 else range(1,self.n+1)
        # show metrics for each class.
        for lbl in labels:
            print "               Class = ",lbl
            print "==========================================="
            print "Precision = %f" % result[lbl]["precision"]
            print "Recall = %f" % result[lbl]["recall"]
            print "F = %f" % result[lbl]["F"]
            print "Accuracy = %f" % result[lbl]["accuracy"]
            print "===========================================\n"
        # print macro averages.
        print "               Macro Averages"
        print "==========================================="
        print "Precision = %f" % result["macro"]["precision"]
        print "Recall = %f" % result["macro"]["recall"]
        print "F = %f" % result["macro"]["F"]
        print "Accuracy = %f" % result["macro"]["accuracy"]
        print "===========================================\n"
        # print micro averages.
        print "              Micro Averages"
        print "==========================================="
        print "Precision = %f" % result["micro"]["precision"]
        print "Recall = %f" % result["micro"]["recall"]
        print "F = %f" % result["micro"]["F"]
        print "Accuracy = %f" % result["micro"]["accuracy"]
        print "===========================================\n"    
        pass
    pass    

