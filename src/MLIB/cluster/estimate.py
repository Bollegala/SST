#! /usr/bin/python

"""
This module has the functionality to estimate clustering thresholds
used by sequntial clustering and sequential co-clustering algorithms.

2010/6/4: Extend to use multiprocessing module to speed up computations.
2010/6/16: Measure relateness using conditional probability
"""

import sys
sys.path.append("../..")

from multiprocessing import Process, Queue, Lock
from Queue import Empty

from MLIB.utils.dmatrix import DMATRIX
from MLIB.utils.ProgBar import TerminalController, ProgressBar

NO_OF_PROCESSORS = 32


def cosine(x, y):
    """
    For efficiency, set x to the smaller vector.
    """
    prod = sum([(x[i] * y.get(i,0)) for i in x])
    sim = float(prod)/(x.L2() * y.L2())
    return sim

def cond_prob(x, y):
    """
    Compute the probability p(x|y) using MLE as follows
    \frac{\sum_{w \in x} f(w,y)} {\sum_{w in y} f(w,y)}.
    Here, f(w,x) denotes the frequency of feature w in the
    feature vector x.
    """
    top = sum([y.get(w,0) for w in x])
    bottom = sum([y.get(w,0) for w in y])
    p = float(top) / float(bottom)
    return p
    

def load_matrix(matrix_fname):
    """
    Read the data matrix.
    """
    global M
    print "Loading matrix: %s" % matrix_fname
    M = DMATRIX()
    M.read_matrix(matrix_fname)
    return M
    
def get_candidates(M, i):
    """
    In the row with id i, find the columns that have any value.
    The keys of those column vectors are candidates.
    """
    cands = set()
    a = M.get_row(i)
    for j in a:
        b = M.get_column(j)
        cands.update(b.keys())
    return cands

def write_distribution(M, result_fname):
    """
    Compute the row similarity distribution.
    To compute column similarity distribution, transpose
    the matrix first.
    """
    work_queue = Queue()
    lock = Lock()
    distFile = open(result_fname,"w")
    row_ids = M.get_row_id_list()
    no_rows,no_cols = M.shape()
    for (counter, i) in enumerate(row_ids):
        work_queue.put(i)
    term = TerminalController()
    progress = ProgressBar(term,"Total rows = %d, columns = %d"\
                           % (no_rows,no_cols))
    count = 0
    # compute similarity.
    procs = [Process(target=do_work, args=(work_queue,
                                           lock, M, no_rows,
                                           distFile,
                                           progress)) 
             for i in range(NO_OF_PROCESSORS)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    distFile.close()
    pass


def do_work(q, lock, M, no_rows, distFile, progress):
    """
    Performs the acutual similarity computation and
    writes to the distribution file.
    Use conditional probability.
    """
    while True:
        try:
            lock.acquire()
            i = q.get(block=False)
            row_i = M.get_row(i)
            cands = get_candidates(M, i)
            count = int(q.qsize())
            progress.update(1 - (float(count) / no_rows),
                            "%d completed row = %d candidates = %d" % (
                                (no_rows - count), i,len(cands)))
            lock.release()
            for j in cands:
                if j > i:
                    row_j = M.get_row(j)
                    pij = cond_prob(row_i,row_j)
                    pji = cond_prob(row_j,row_i)
                    if pij > 0 or pji > 0:
                        lock.acquire()
                        if pij > 0:
                            distFile.write("%d,%d,%f\n" % (i,j,pij))
                        if pji > 0:
                            distFile.write("%d,%d,%f\n" % (j,i,pji))
                        distFile.flush()
                        lock.release()
        except Empty:
            break
    pass

def write_histogram(hist_file,dist_file):
    """
    Read the distribution file and compute the histogram.
    Write it to the hist_file.
    """
    bins = 1000
    h = {}
    dist = open(dist_file)
    step = 1.0/bins
    for line in dist:
        sim = float(line.strip().split(",")[2])
        bid = int(sim/step)
        h[bid] = 1+h.get(bid,0)
    dist.close()
    hist = open(hist_file,"w")
    for bid in sorted(h.keys()):
        hist.write("%f,%f,%d\n" % (bid*step,(bid+1)*step,h[bid]))
    hist.close()   
    pass


def compute_threshold_dist(dist_fname):
    """
    Compute the clustering threshold from the distribution.
    Expected value is computed according to the distribution
    g(x) = a*x^(-k)
    E[x] = (a*(1-delta*(2-k)))/(2-k)
    k = delta*g0+1
    a = g(delta)*delta^k    
    """
    # parameters.
    zero_threshold = 0.05
    one_threshold = 0.9
    delta = 0.05
    # compute zeros and ones.
    zeros = 0
    ones = 0
    count = 0
    simsum = 0
    dist_file = open(dist_fname)
    for line in dist_file:
        p = line.strip().split(",")
        count += 1
        sim = float(p[2])
        simsum += sim
        if sim < zero_threshold:
            zeros += 1
        elif sim > one_threshold:
            ones += 1
    dist_file.close()
    print "zeros =", zeros
    print "ones =", ones
    g1 = float(ones)/float(count)
    g0 = float(zeros)/float(count)
    # compute expectation.
    k = (delta*g0)+1
    a = g0*(delta**k)
    Ex = 1-(delta**(2-k))
    Ex = (a*Ex)/(2-k)
    print "Expectation of similarity =", Ex
    avg = simsum/float(count)
    print "Sample mean =", avg
    pass


def compute_threshold_from_histogram(histogram_fname):
    """
    Read in the histogram. Count the no. of lines.
    Add the last bin to the penultimate bin and remove
    the last bin. Now group into 10 bins and compute
    the actual distribution. Compute the predicion and
    the prediction error.
    """
    print "Histogram = ", histogram_fname
    values = []
    histFile = open(histogram_fname)
    for line in histFile:
        p = line.strip().split(",")
        values.append(float(p[2]))
    histFile.close()
    N = len(values)
    values[N-2] = values[N-2]+values[N-1]
    del values[N-1]
    nBins = 100
    mergeStep = len(values)/nBins #should be an integer.
    dist = []
    for i in range(0,nBins):
        total = 0
        for j in range(i*mergeStep,(i+1)*mergeStep):
            total += values[j]
        dist.append(total)
    #normalize the distribution.
    distTotal = sum(dist)
    normDist = []
    for value in dist:
        normDist.append(float(value)/float(distTotal))
    #print normDist
    #predict parameters of the power-law.
    delta = 1.0/nBins
    gDelta = normDist[0]
    k = delta*gDelta+1
    a = gDelta*(delta**k)
    theta = a*(1-(delta**(2-k)))
    theta = theta/(2-k)
    # compute the prediction error.
    error = 0
    for i in range(0,nBins):
        x = (i+1)*delta
        gx = a*(x**(-k))
        error += (gx-normDist[i])**2
    return {'theta':theta,
            'error':error,
            'a':a,
            'k':k}                               

def save_transpose(mat_fname, trans_fname):
    """
    Read, take the transpose the matrix and save it.
    """
    M = DMATRIX()
    M.read_matrix(mat_fname)
    M.transpose()
    M.write_matrix(trans_fname)
    pass    

def process(mat_fname,dist_fname,hist_fname,transpose):
    """
    Read the matrix from mat_fname. By default we will compute
    the row clustering threshold. If you want to compute the
    column clustering threshold set transpose to True. We will
    then save the transposed matrix as mat_fname.transposed and
    then will compute threshold using this new matrix instead
    of the original matrix. Computed similarity distribution is
    written to dist_fname. We will then compute a histogram
    from this distribution. Threshold will be computed using the
    histogram file.
    """
    matrix = mat_fname
    if transpose:
        matrix = "%s.transposed" % mat_fname
        save_transpose(mat_fname,matrix)
    M = load_matrix(matrix)
    write_distribution(M, dist_fname)
    write_histogram(hist_fname,dist_fname)
    res = compute_threshold_from_histogram(hist_fname)
    if transpose:
        print "Estimated column clustering threshold (theta) = %f" %\
              res['theta']
    else:
        print "Estimated row clustering threshold (phi) = %f" % res['theta']
    print "Prediction Error = %f" % res['error']
    print "Model parameters: a = %f\tk=%f" % (res['a'],res['k'])
    pass

def unit_test():
    """
    Unit test.
    """
    load_matrix("../work/matrix")
    write_distribution("../work/dist")
    pass

if __name__ == "__main__":
    unit_test()
    pass
    
