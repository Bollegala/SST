# -*- coding: utf-8 -*-

"""
This module implements maximum entropy classification (logistic regression)
using stocastic gradient decent.

  1. SGD: logistic regression (L2)
  2. SGD: multiclass logistic regression (L2)

Danushka Bollegala.

-- ChangeLog --
13 Dec 2009: Added cumulative L2 penalty.
31 Dec 2009: Inherit LEARNER class and separate generic functionality.
"""

import sys,math,time
sys.path.append("../..")

from MLIB.utils.data import SEQUENTIAL_FILE_READER
from MLIB.utils.ProgBar import TerminalController,ProgressBar
from MLIB.classify.predict import ROLLING_AVERAGER
from MLIB.classify.learner import LEARNER

class SGD(LEARNER):

    def __init__(self,classes):
        # n is the number of classes.
        self.n = classes
        # the list of potential labels.
        self.labels = [0,1] if self.n == 2 else range(1,self.n+1)
        # total number of iterations over the dataset.
        self.total_iterations = 1
        # initial learning rate (eta)
        self.eta0 = 0.1
        # L2 regularization coefficient.
        self.c = 0
        # L2 coefficient per instance (self.c/self.N)
        self.lmda = 0
        # accumulated L2 penalty
        self.s = 0
        # accumulated loss for the current iteration.
        self.loss = 0
        # cross-validation.
        self.folds = 0
        # heldout vectors.
        self.heldoutVects = None
        # if verbose mode is set to False we will not display the progress bar.
        self.verbose = True
        # count the number of updates.
        self.k = 0
        # total number of instances.
        self.N = 0
        # if average L2 norm does not change this much then we will terminate.
        self.L2_bound = 0.001
        # rolling averages for L2 norm are taken over this no. of iterations.
        self.L2_rolling = 5        
        pass

    def initialize_weights(self):
        """
        Initialize weight vectors, bias terms and delayed update
        registers.
        """
        # compute lmda.
        self.lmda = float(self.c)/float(self.N)
        # bias term. This should not get a regularization penalty.
        self.bias = {}
        # weight vector
        self.w = {}
        # lastW[j] = k, indicates that feature j was last updated at time k.
        self.lastW = {}
        for lbl in self.labels:
            self.bias[lbl] = 0
            self.w[lbl] = {}
            self.lastW[lbl] = {}
        pass
    
    def update(self,v,eta):
        """
        Update the weight vector according to the SGD formula.
        w[new] = w[old]-eta*grad*v
        grad = y-lbl
        y = 1/(1+exp(-a))
        a = w.v (inner product)
        """
        for lbl in self.labels:
            # compute the inner product between v and w.
            a = self.bias[lbl]
            # apply the delayed update.
            for (fid,fval) in v:
                if fid in self.lastW[lbl]:
                    alpha = self.s - self.lastW[lbl][fid]
                    self.w[lbl][fid] = self.w[lbl].get(fid, 0) - alpha
                self.lastW[lbl][fid] = self.s
                a += fval*self.w[lbl].get(fid,0)
            # compute the gradient.
            y = 1./(1.0+math.exp(-a)) if a > -100. else 0
            g = (y-1) if (lbl == v.label) else y
            self.loss += g if g >=0 else -g
            # update the weight vector.
            for (fid,fval) in v:
                self.w[lbl][fid] =  self.w[lbl].get(fid, 0) - (eta * g * fval)
            # update the bias term.
            self.bias[lbl] -= (eta * g)
            # accumulate L2 penalty.
            self.s += eta * self.lmda
        pass

    def get_norms(self):
        """
        Computes both L1 and L2 norms and active features in one pass.
        """
        l1_sum = 0
        l2_sum = 0
        actives = 0
        for lbl in self.labels:
            for fid in self.w[lbl]:
                # apply and remaing L1 penalities at the end of training.
                alpha = self.s - self.lastW[lbl].get(fid,0)
                self.w[lbl][fid] = self.w[lbl].get(fid, 0) - alpha
                weight = self.w[lbl][fid]
                l1_sum += weight if weight > 0 else -weight
                l2_sum += weight * weight
                if weight != 0:
                    actives += 1
        l2_sum = math.sqrt(l2_sum)
        return (l1_sum,l2_sum,actives)   

    def train_client(self,fvects,no_features):
        """
        Train using SGD. fvects is the list of feature vectors.
        no_features is the number of unique features in all training
        instances.
        """        
        # print train statistics.           
        self.N = len(fvects)
        self.initialize_weights()
        if self.n == 2:
            print "Binary Logistic Regression"
        else:
            print "Multi-class (%d classes) Logistic Regression" % self.n
        print "L2 regularization coefficient = %f" % self.c
        print "Total iterations = %d" % self.total_iterations
        print "Initial learning rate = %f" % self.eta0
        print "Total number of instances = %d" % self.N
        self.k = 1
        self.s = 0
        RA = ROLLING_AVERAGER(self.L2_rolling, self.L2_bound)
        # Iterate over the training dataset.
        for i in range(1, self.total_iterations+1):
            print "\nIteration #%d" % i,
            startTime = time.time()
            self.loss = 0
            count = 0
            for fv in fvects:
                count += 1
                eta = float(self.eta0) / (1. + (float(count)/self.N))
                self.update(fv,eta)
                self.k += 1
            endTime = time.time()
            print "time taken (sec)=", (endTime-startTime)
            # Show the value of the bias term.
            if self.verbose:
                for lbl in self.bias:
                    print "Bias Term %d = %f" % (lbl,self.bias[lbl])
            (L1_norm, L2_norm, actives) = self.get_norms()
            self.active_features = actives
            print "Active Features = %d/%d" % (actives,no_features)          
            print "L1 norm = %f" % L1_norm
            print "L2 norm = %f" % L2_norm
            if RA.add(L2_norm) == 1:
                print "Terminating...L2 norm does not change"
                break
            if self.verbose:
                self.display_training_error(fvects)
                if self.heldoutVects:
                    self.display_heldout_error(self.heldoutVects)              
        # if not in the verbose mode then print the final results.
        if not self.verbose:
            trainError = self.display_training_error(fvects)
            if self.heldoutVects:
                self.display_heldout_error(self.heldoutVects)    
        pass            
    pass



def train_SGD(classes,train_fname,model_fname,iterations=2,
              L2=0,heldout_fname=None,
              crossValidation=None,
              verbose=False):
    """
    Train using binary maximum entropy model (i.e. logistic regression)
    using stocastic gradient decent method. If heldout_fname is given
    then we will report the accuracy on heldout data after each iteration.
    If cross-validation is set to a number
    (e.g. 5 for five-fold cross-validation)
    then we will perform cross-validation and will report accuracy for each fold
    as well as the average. You cannot specify both cross-validation and holdout
    evaluation at the same time. If you do so then an error will be reported.
    """
    TrainSeqFileReader = SEQUENTIAL_FILE_READER(train_fname)
    trainVects = TrainSeqFileReader.read()
    TrainSeqFileReader.close()
    heldoutVects = None
    if heldout_fname:
        HeldoutSeqFileReader = SEQUENTIAL_FILE_READER(heldout_fname)
        heldoutVects = HeldoutSeqFileReader.read()
        HeldoutSeqFileReader.close()
    Learner = SGD(classes)
    Learner.total_iterations = iterations
    Learner.c = L2
    Learner.verbose = verbose
    if crossValidation:
        Learner.folds = crossValidation
    if heldout_fname:
        Learner.heldoutVects = heldoutVects["vects"]
    no_features = classes*len(trainVects["featIDs"])
    Learner.train(trainVects["vects"],no_features)
    print "Writing the model... %s" % model_fname
    Learner.writeModel(no_features,model_fname)
    pass



def utest_SGD_Test():
    """
    unit test for SGD binary prediction.
    """
    model_fname = "../work/model"
    # test binary classification.
    if False:
        #test_fname = "../work/train.bz2"
        test_fname = "../work/rcv1_test.binary.bz2"
    if True:
        test_fname = "../work/iris_multi.train"
    test_logreg(model_fname,test_fname,prob=True,acc=True)
    pass
    
    
def utest_SGD_Train():
    """
    unit test for SGD binary training.
    """
    # binary training.
    if False:
        classes = 2
        train_fname = "../work/train.bz2"
        heldout_fname = "../work/held"         
    # multiclass training.
    if True:
        classes = 3
        train_fname = "../data/iris_multi.train"
        heldout_fname = "../data/iris_multi.test"
    model_fname = "../work/model"
    iterations = 1000
    L2 = 0.01
    #heldout_fname = None
    crossValidation = 0
    train_SGD(classes,train_fname,model_fname,
              iterations=iterations,
              L2=L2,heldout_fname=heldout_fname,
              crossValidation=crossValidation,
              verbose=False)
    pass

if __name__ == "__main__":
    utest_SGD_Train()
    #utest_SGD_Test()
    
