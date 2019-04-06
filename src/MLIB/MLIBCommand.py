#! /usr/bin/python
# coding: utf-8

"""
This is the command line interface for the various machine learning
algorithms implemented in MLIB (Machine Learning Library)

Danushka Bollegala.
6 June 2011.
--------------------------------------------------------------------------
Usage: python MLIB.py -a ALGORITHM [parameners]

 1. CLUSTERING 

  sequential co-clustering -a cluster.seqcoclust
                           -t theta
                           -p phi
                           -i input_file (data matrix)
                           -o output_file (clusters)
                         
  sequential clustering   -a cluster.seqclust
                          -t theta
                          -i input_file (data matrix)
                          -o output_file (clusters)
                          [use -u to cluster rows instead of columns]
                          -v verbose mode
                          (displays some info regarding the clusters)           

                          Note: setting phi = 1 in seqcoclust is NOT
                          equvalent to seqclust!

  estimate thresholds    -a cluster.estimate
                         -u By default we estimate phi (the row clustering
                         threshold). If you want to estimate theta (the
                         column clustering threshold use -u option)
                         -i input_file (data matrix)

                         We create three files with the same name as the
                         input file but with extensions .dist, .hist and
                         .transpose (in case of estimating theta) that
                         contain respectively the similarity distribution,
                         similarity histogram, and the transposed matrix.

                         Note: distribution files can be very large!
                         Manually delete them after you have estimated
                         thresholds to save disk space.

 2. CLASSIFICATION

  binary and multiclass logistic regression with stocastic gradient decent
  (L2 regularization is supported).
                        -a classify.logreg_sgd
                        -n number of classes (2 for binary logistic regression)
                        -i train file
                        -m model file
                        -t number of iterations (default 10)
                        --l2 L2 regularization coefficient (default 1)
                        -d held out instance file (if provided we will compute
                        metrics on this dataset as well)
                        -g g-fold cross validation
                        (default 0, i.e. no cross validation)
                        -v verbose mode (default false)

  binary and multiclass logistic regression with truncated gradient algorithm
  (L1 regularization is supported).
                        -a classify.logreg_tg
                        -n number of classes (2 for binary logistic regression)
                        -i train file
                        -m model file
                        -t number of iterations (default 10)
                        --l1 L1 regularization coefficient (default 1)
                        -d held out instance file (if provided we will compute
                        metrics on this dataset as well)
                        -g g-fold cross validation
                        (default 0, i.e. no cross validation)
                        -v verbose mode (default false)
                        
 binary classification using L2 Support Vector Machines
 (Pegasos with hinge loss).
                        -a classify.pegasos
                        -i train file
                        -m model file
                        --itr number of iterations (default 10)
                        --l2 cost parameter lambda. (default 1)
                        

 3. REGRESSION

  Gaussian Process Regression with Gaussian Quadratic Kernel.
  t0*exp(-0.5*t1*||x-y||^2)+t2+t3*x.y 
                        -a train.gpr
                        -i train file
                        -m model file
                        --beta precision (beta) of the Gaussian noise (default 1)
                        --t0 kernel parameter theta zero (default 1.0)
                        --t1 kernel parameter theta one (default 4.0)
                        --t2 kernel parameter theta two (default 0)
                        --t3 kernel parameter thera three (default 0)
                        -v verbose mode (default false)

 4. PREDICTION

  predict labels and probabilities (for probabilistic models) using a trained
  model file.
                        -a predict.logreg (logistic regression)
                        -i input (test) file
                        -m model file
                        -o output file (if not specified we will write to the
                        terminal instead to the output file)
                        -w if set we will compute probabilities
                        -v if set we will print an accuracy report
                        (requires test instances to have gold labels)
                        
                        -a predict.pegasos
                        -i input (test) file
                        -m model file
                        -o output file (if not specified we will write to the
                        terminal instead to the output file)
                        -v is set we will print an accuracy report
                        (requires test instances to have gold labels)
                        

  predict outputs (mean and variance) for Gaussain Process Regression.
                        -a predict.gpr
                        -i test file
                        -m model file
                        -d train file
                        -o output file
                        -v if set we will print root mean square error (RMSE)

 5. DATA PROCESSING

  select held-out data -a data.heldout
                       -n number of instances to hold out
                       -i input train instances file
                       -d held out instances file 
                       -o remaining train instances file 
                       We select n instances uniformly at random from
                       the train file and write them into held out file.
                       This function processes the input file sequentially
                       (never loads all train instances to memory), therefore
                       can process very large data sets.

  scale train data    -a data.scale-train
                      -i train file to be scaled
                      -o write the scaled instances to the output file
                      -r range file (stores the values of the scaling
                         parameters.)
                      -x [0|1] set to 0 to disable x scaling (default 1)
                      -y [0|1] set to 1 to enable y scaling (defaulr 0)
                      --upper upper range (default 1)
                      --lower lower range (default 0)
                      This command scales train data instances in the
                      range [lower,upper]. If x scaling is set, we scale
                      each feature to range [lower,upper]. If y scaling
                      is set we normalize each feature vector such that
                      the length (L2 norm) is 1.

  scale test data     -a data.scale-test
                      -i test file to be scaled
                      -o write the scaled instances to the output file
                      (if omitted we will write to the stdout)
                      -r range file (reads the scaling parameters from file)
                      
  one-sided under sampling for binary classification
                      -a data.sample
                      -i train file name to be read
                      -o sample file name to be written

  analyze features    -a data.anafeats
                      -i feature vector file to be analyzed
                      -o file to write the sorted frequencies


 -h This help message
"""

#################### TODO #################################################


import sys,getopt
sys.path.append("..")

from MLIB.cluster import seqcoclust,seqclust,estimate

from MLIB.utils.data import select_held_out_data
from MLIB.utils.data import  scale_train_data
from MLIB.utils.data import  scale_test_data
from MLIB.utils.data import  analyze_feature_file
from MLIB.utils.data import oneSidedUnderSampling

from MLIB.classify.logreg_sgd import train_SGD
from MLIB.classify.truncated_gradient import train_TG
from MLIB.classify.learner import test_logreg

from MLIB.bayesian.gaussian_process import train_GPR
from MLIB.bayesian.gaussian_process import predict_GPR

from MLIB.classify.pegasos import trainPegasos
from MLIB.classify.pegasos import testPegasos


def help_message():
    """
    Display the help message.
    """
    print __doc__
    pass
    

def command_line_processor():
    """
    Process command line arguments.
    """
    try:
        opts,args = getopt.getopt(sys.argv[1:],
                                  "a:hi:o:vun:x:y:r:m:d:g:w:",
                                  ["t0=", "t1=", "t2=", "t3=", "beta=",
                                   "l1=", "l2=", "itr=","phi=", "theta=",
                                   "upper=", "lower="])
    except getopt.GetoptError, err:
        print err
        help_message()
        sys.exit(-1)

    # parameters.    
    algo = None
    input_fname = None
    output_fname = None
    # clustering thresholds.
    theta = None
    phi = None
    transpose = False
    # number of classes for classification. (n=2 for binary)
    # OR the number of instances to set aside as heldout data.
    n = None
    # feature scaling parameters.
    x_scale = 1
    y_scale = 0
    range_fname = None
    upper = 1
    lower = 0
    verbose = False
    model_fname = None
    heldout_fname = None
    # set -g N to enable N-fold cross validation.
    g = 0
    # L1 and L2 regularization coefficients.
    l1 = 1
    l2 = 1
    # The number of epohs for online algorithms.
    iterations = 10
    # Whether to show prediction probabilities or not.
    show_probs = False
    # Kernel parameters for Gaussian Process.
    t0 = t1 = t2 = t3 = None
    # Noise accuracy for Gaussian noise.
    beta = 1

    # read in the parameters.
    for opt, val in opts:
        if opt == "-h":
            usage()
            sys.exit(1)
        if opt == "-i":
            input_fname = val
        if opt == "-o":
            output_fname = val
        if opt == "--theta":
            theta = float(val)
        if opt == "--phi":
            phi = float(val)
        if opt == "-u":
            transpose = True
        if opt == "-a":
            algo = val.strip().lower()
        if opt == "-n":
            n = int(val)
        if opt == "--itr":
            iterations = int(val)
        if opt == "-x":
            x_scale = int(val)
        if opt == "-y":
            y_scale = int(val)
        if opt == "-r":
            range_fname = val.strip()
        if opt == "--lower":
            lower = float(val)
        if opt == "--upper":
            upper = float(val)
        if opt == "-v":
            verbose = True
        if opt == "-d":
            heldout_fname = val.strip()
        if opt == "-m":
            model_fname = val.strip()
        if opt == "--l1":
            l1 = float(val)
        if opt == "--l2":
            l2 = float(val)
        if opt == "-g":
            g = int(val)
        if opt == "-w":
            show_probs = True
        if opt == "--beta":
            beta = float(val)
        if opt == "--t0":
            t0 = float(val)
        if opt == "--t1":
            t1 = float(val)
        if opt == "--t2":
            t2 = float(val)
        if opt == "--t3":
            t3 = float(val)
            pass
        pass
            
    # cluster.seqcoclust
    if algo == "cluster.seqcoclust" and input_fname and \
           output_fname and (theta >= 0) and (theta <=1) and\
           (phi >= 0) and (phi <= 1):
        seqcoclust.perform_sequential_coclustering(input_fname,
                                                   output_fname,
                                                   theta,
                                                   phi,
                                                   verbose)        
    # cluster.seqclust
    elif algo == "cluster.seqclust" and input_fname and \
           output_fname and (theta >= 0):
        seqclust.perform_sequential_clustering(input_fname,
                                               output_fname,
                                               theta,
                                               transpose)

    # estimate thresholds (cluster.estimate)
    elif algo == "cluster.estimate" and input_fname:
        dist_fname = "%s.dist" % input_fname
        hist_fname = "%s.hist" % input_fname
        estimate.process(input_fname, dist_fname,
                         hist_fname, transpose)

    # select held out instances (data.heldout)
    elif algo == "data.heldout" and input_fname and output_fname and \
             heldout_fname and n:
        select_held_out_data(input_fname, heldout_fname,
                             output_fname, n)

    # perform scaling on train data (data.scale-train)
    elif algo == "data.scale-train" and input_fname \
         and output_fname  and range_fname:
        if theta is None:
            theta = 1.
        if phi is None:
            phi = 0.
        scale_train_data(input_fname, output_fname, range_fname, 
                         x_scale, y_scale,
                         upper, lower)

    # perform scaling on test data (data.scale-test)
    elif algo == "data.scale-test" and input_fname and range_fname:
        scale_test_data(input_fname, range_fname, output_fname)
        
    # perform one-sided undersampling (data.sample).
    elif algo == "data.sample" and input_fname and output_fname:
        oneSidedUnderSampling(input_fname, output_fname)    

    # count the frequency of features in a feature vector file.
    elif algo == "data.anafeats" and input_fname and output_fname:
        analyze_feature_file(input_fname, output_fname)

    # train a logistic regression classifier using stocastic gradient decent.
    elif algo == "classify.logreg_sgd" and input_fname and model_fname and n:
        train_SGD(n, input_fname, model_fname,
                  iterations=iterations,
                  L2=l2,
                  heldout_fname=heldout_fname,
                  crossValidation=g,
                  verbose=verbose)
        
    # train a binary L2 SVM using Pegasos with hinge loss.
    elif algo == "classify.pegasos" and input_fname and model_fname:
        trainPegasos(l2, iterations, model_fname, input_fname)
    
    # predict using the trained pegasos svm.
    elif algo == "predict.pegasos" and input_fname and model_fname:
        showPreds = False if output_fname else True
        testPegasos(model_fname, input_fname,
                    predFileName=output_fname, 
                    showPreds=showPreds, 
                    showStats=verbose)

    # train a logistic regression classifier using truncated gradient.
    elif algo == "classify.logreg_tg" and input_fname and model_fname and n:
        train_TG(n, input_fname, model_fname,
                  iterations=iterations,
                  L1=l1,
                  heldout_fname=heldout_fname,
                  crossValidation=g,
                  verbose=verbose)
        
    # predict labels and probabilities using a model.
    elif algo == "predict.logreg" and model_fname and input_fname:
        show_accuracy = verbose
        test_logreg(model_fname,
                 input_fname,
                 output_fname=output_fname,
                 prob=show_probs,
                 acc=show_accuracy)

    # train a Gaussian process regression model.
    elif algo == "train.gpr" and model_fname and input_fname:
        train_GPR(input_fname, model_fname,
                  verbose, beta, t0, t1, t2, t3)

    # predict using Gaussian process regression model.
    elif algo == "predict.gpr" and model_fname and input_fname and heldout_fname:
        predict_GPR(input_fname, heldout_fname, model_fname,
                    output_fname, verbose)    
        
    # if not all parameters are specified, display the help message.    
    else:
        help_message()
    pass

if __name__ == "__main__":
    command_line_processor()

