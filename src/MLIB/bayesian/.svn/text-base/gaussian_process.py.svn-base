# -*- coding: utf-8 -*-

"""
This program implements Gaussian Process Regression (GPR) and
Gaussian Process Classification. Numpy and scipy are used
for matrix calculations. 
"""

import sys,math,time
sys.path.append("../..")

from MLIB.utils.data import SEQUENTIAL_FILE_READER, SEQUENTIAL_FILE_WRITER
from MLIB.utils.ProgBar import TerminalController,ProgressBar

from numpy import *
from scipy.linalg import inv
from math import sqrt, exp, fabs

class GPR:

    """
    Implements Gaussian Process Regression.
    """

    def __init__ (self):
        self.beta = 1
        self.verbose = True
        pass

    def set_kernel(self, kernel):
        """
        specify the kernel function to use.
        """
        self.kernel = kernel
        pass

    def train(self, train_vects):
        # load the train data to a matrix.
        if self.verbose:
            print "Loading the training data to memory..."
        (self.D, self.t) = get_train_data(train_vects)
        # create co-variance matrix C.
        self.n = len(train_vects["vects"])
        self.m = len(train_vects["featIDs"])
        if self.verbose:
            print "Creating the covariance matrix..."
            term = TerminalController()
            progress = ProgressBar(term,
                                   "Train instances = %d" % self.n)        
        C = zeros((self.n,self.n))
        for i in range(0,self.n):
            if self.verbose:
                progress.update(float(i + 1) / self.n,
                                "Processing instance no. %d" % (i + 1))
            for j in range(i,self.n):
                x_i = self.D[i,:]
                x_j = self.D[j,:]
                val = self.kernel.value(x_i, x_j)
                if i == j:
                    val += 1.0 / float(self.beta)
                C[i,j] = val
                C[j,i] = val
        # compute the inverse.
        if self.verbose:
            print "Computing the inverse of the matrix..."
        self.Cinv = inv(C)                
        pass

    def predict(self, v):
        (n,m) = shape(self.D)
        x = zeros((m))
        for (fid,fval) in v:
            x[fid - 1] = fval
        k = zeros((n))
        for i in range(0,n):
            y = self.D[i,:]
            k[i] = self.kernel.value(x.T, y)
        c = self.kernel.value(x, x)
        mean = dot(dot(self.Cinv, self.t), k)
        variance = dot(dot(self.Cinv, k.T), k)
        variance = c - variance
        return (mean, variance)

    def load_model(self, model_fname, train_fname):
        """
        Load the matrix from the model file.
        """
        self.Cinv = genfromtxt("%s.matrix" % model_fname)
        train_file = SEQUENTIAL_FILE_READER(train_fname)
        train_vects = train_file.read()
        train_file.close()
        (self.D, self.t) = get_train_data(train_vects)
        para_file = open(model_fname)
        self.beta = float(para_file.readline().split()[1])
        kernel_type = para_file.readline().strip().split("\t")[1]
        if kernel_type == "GAUSSIAN_QUADRATIC_KERNEL":
            self.kernel = GAUSSIAN_QUADRATIC_KERNEL()
            for i in range(0,4):
                (para,val) = para_file.readline().strip().split()
                if para == "theta_0":
                    self.kernel.theta_0 = float(val)
                elif para == "theta_1":
                    self.kernel.theta_1 = float(val)
                elif para == "theta_2":
                    self.kernel.theta_2 = float(val)
                elif para == "theta_3":
                    self.kernel.theta_3 = float(val)
        para_file.close()
        pass

    def save_model(self, model_fname):
        """
        Dumps the matrix inverse to the model_file.
        """
        if self.verbose:
            print "Saving the model file, %s" % model_fname
        para_file = open(model_fname, "w")
        para_file.write("beta\t%f\n" % self.beta)
        if self.kernel.name == "GAUSSIAN_QUADRATIC_KERNEL":
            para_file.write("KERNEL\tGAUSSIAN_QUADRATIC_KERNEL\n")
            para_file.write("theta_0\t%f\n" % self.kernel.theta_0)
            para_file.write("theta_1\t%f\n" % self.kernel.theta_1)
            para_file.write("theta_2\t%f\n" % self.kernel.theta_2)
            para_file.write("theta_3\t%f\n" % self.kernel.theta_3)
        para_file.close()
        savetxt("%s.matrix" % model_fname, self.Cinv)
        pass
    pass


class GAUSSIAN_QUADRATIC_KERNEL:

    def __init__(self, _theta_0=1.0,
                 _theta_1=4.0,
                 _theta_2=0,
                 _theta_3=0):
        self.theta_0 = _theta_0
        self.theta_1 = _theta_1
        self.theta_2 = _theta_2
        self.theta_3 = _theta_3
        self.name = "GAUSSIAN_QUADRATIC_KERNEL"
        pass
        
    def value(self, x,  y):
        s = norm(x - y)
        s = self.theta_0 * exp(-0.5 * self.theta_1 * s * s)
        s = s + self.theta_2 + (self.theta_3 * dot(x, y))
        return s
    pass


class LINEAR_KERNEL:

    def __init__(self):
        pass

    def value(self, x, y):
        res = dot(x, y)
        return res
    pass


def get_train_data(train_vects):
    """
    Convert each vector to a numpy vector and return in
    a data matrix and a target vector.
    """
    n = len(train_vects["vects"])
    m = len(train_vects["featIDs"])
    D = zeros((n,m))
    t = zeros((n))
    for i in range(0,n):
        v = train_vects["vects"][i]
        t[i] = v.label
        for (fid,fval) in v:
            D[i, fid - 1] = fval
    return (D, t)

def norm(x):
    """
    L2 norm of x.
    """
    s = dot(x, x)
    return sqrt(s)

def utest_train_GPR():
    train_fname = "../work/winequality-red"
    model_fname = "../work/model"
    train_GPR(train_fname, model_fname)
    pass

def utest_predict_GPR():
    test_fname = "../work/winequality-red"
    train_fname = "../work/winequality-red"
    model_fname = "../work/model"
    predict_GPR(test_fname, train_fname,
                model_fname, accuracy=True)
    pass

def train_GPR(train_fname, model_fname,
              verbose=True,
              beta=1,
              theta_0=None, theta_1=None,
              theta_2=None, theta_3=None):
    """
    This is the utility function used to train a regression
    model using Gaussian Process.
    """
    train_file = SEQUENTIAL_FILE_READER(train_fname)
    train_vects = train_file.read()
    train_file.close()
    learner = GPR()
    learner.verbose = verbose
    learner.beta = beta
    kernel = GAUSSIAN_QUADRATIC_KERNEL()
    if theta_0:
        kernel.theta_0 = theta_0
    if theta_1:
        kernel.theta_1 = theta_1
    if theta_2:
        kernel.theta_2 = theta_2
    if theta_3:
        kernel.theta_3 = theta_3
    learner.set_kernel(kernel)
    learner.train(train_vects)
    learner.save_model(model_fname)
    pass

def predict_GPR(test_fname,
                train_fname,
                model_fname,
                output_fname=None,
                accuracy=False):
    """
    Predict the outputs for the test instances.
    If the output is not specified, then
    write to the standard output.
    """
    test_file = SEQUENTIAL_FILE_READER(test_fname)
    test_vects = test_file.read()
    test_file.close()
    learner = GPR()
    learner.load_model(model_fname, train_fname)
    count = 0
    error = 0
    if output_fname:
        output_file = SEQUENTIAL_FILE_WRITER(output_fname)
    else:
        output_file = SEQUENTIAL_FILE_WRITER(None, "STDOUT")
    for v in test_vects["vects"]:
        (mean, variance) = learner.predict(v)
        output_file.writeLine("%f\t%f\n" % (mean, variance))
        if accuracy:
            error += (v.label - mean) ** 2
        count += 1
    error = sqrt(error) / float(count)
    if accuracy:
        output_file.writeLine("RMSE = %f\n" % error)
    output_file.close()
    pass   

if __name__ == "__main__":
    utest_train_GPR()
    utest_predict_GPR()
    pass
    
    
