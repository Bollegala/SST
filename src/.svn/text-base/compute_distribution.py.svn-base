#! /usr/bin/python

"""
Given an element co-occurrence matrix, this program computes the
similarity between rows (lexical elements)
and save the results in a distribution file. 
"""


import sys
import getopt

from multiprocessing import Process, Queue, Lock
from Queue import Empty

from MLIB.utils.dmatrix import DMATRIX
from MLIB.utils.ProgBar import TerminalController, ProgressBar


NO_OF_PROCESSORS = 32


def similarity_function(u, v):
    """
    Call different similarity functions.
    u: base entry.
    v: neighbor.
    """
    return tau_paper_ACL(v, u)
    #return tau_ACL(v, u)
    #return Lin(u, v)
    #return cosine(u, v)


def cosine(x, y):
    """
    For efficiency, set x to the smaller vector.
    """
    prod = sum([(x[i] * y.get(i,0)) for i in x])
    sim = float(prod)/(x.L2() * y.L2())
    return sim

def tau_paper_ACL(v, u):
    """
    Computes the ACL2011 version of relatedness.
    """
    up = 0
    down = 0
    for w in v:
	if v[w] > 0:
	    up += u.get(w, 0)
    for w in u:
	if u[w] > 0:
	    down += u[w]
    relatedness = float(up) / float(down)
    return relatedness

def tau_code_ACL(v, u):
    """
    Computes p(v|u) version of relatedness
    considering second order co-occurrences.
    """
    up = 0
    down = 0
    for w in u:
	#up += min(u[w], v.get(w, 0))
	val = v.get(w, 0)
	if val > 0:
	    up += val
	if u[w] > 0:
	    down += u[w]
    relatedness = float(up) / float(down)
    return relatedness
    pass

def Lin(x, y):
    """
    Dekang Lin's similarity measure.
    """
    tot_x = 0
    tot_y = 0
    tot_overlap = 0
    for i in x:
        if x[i] > 0:
            tot_x += x[i]
    for j in y:
        if y[j] > 0:
            tot_y += y[j]
        if j in x and x[j] > 0:
            tot_overlap += (y[j] + x[j])
    if tot_overlap == 0:
        return 0.0
    else:
        sim = float(2* tot_overlap) / float(tot_x + tot_y)
        return sim
    pass
  

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
    In the row with id i (say a), find the columns that have any value (say b).
    For keys in b, get the row vectors corresponding to those keys.
    Sort those row vectors according to the number of elements (non-zero)
    and select the top N as the candidates.
    """
    h = {} # id vs the no. of non-zero elements
    a = M.get_row(i)
    for j in a:
        b = M.get_column(j)
        for k in b.keys():
            if (k not in h) and M.row_exists(k):
                c = M.get_row(k)
                elements = len(c)
                h[k] = elements
    l = h.items()
    l.sort(sort_tuple_list)
    cands = [ele[0] for ele in l]
    return cands

def sort_tuple_list(A, B):
    """
    This function is used as a comparator to sort a list of tuple of the form
    (id, no_of_items) in the descending order of the no_of_items.
    """
    if A[1] > B[1]:
        return -1
    return 1

def write_distribution(M, result_fname, DIelements):
    """
    Compute the row similarity distribution.
    To compute column similarity distribution, transpose
    the matrix first. if Domain independent row elements are
    given (DIelements), then compute the similarity between
    those elements and all the row elements.
    """
    work_queue = Queue()
    lock = Lock()
    distFile = open(result_fname,"w")
    row_ids = []
    for rowid in DIelements:
        if M.row_exists(rowid):
            row_ids.append(rowid)
    (no_rows, no_cols) = M.shape()
    for (counter, i) in enumerate(row_ids):
        work_queue.put(i)
    term = TerminalController()
    progress = ProgressBar(term,"Total rows = %d, columns = %d"\
                           % (no_rows,no_cols))
    count = 0
    # compute similarity.
    procs = [Process(target=do_work, args=(work_queue,
                                           lock, M, len(row_ids),
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
    """
    # How many candidates with which we must compare when
    # measuring similarity. The more the slower but accurate.
    topCandidates = 1000
    while True:
        try:
            i = q.get(block=False)
            row_i = M.get_row(i)
            cands = get_candidates(M, i)
            count = int(q.qsize())
            progress.update(1 - (float(count) / no_rows),
                            "row = %d all candidates = %d (remaining rows = %d)" % (i, len(cands), count))
            for j in cands[:topCandidates]:
                row_j = M.get_row(j)
                sim = similarity_function(row_i,row_j)
                if sim > 0:
                    lock.acquire()
                    distFile.write("%d,%d,%f\n" % (i,j,sim))
                    distFile.flush()
                    lock.release()
        except Empty:
            break
    pass


def process(mat_fname,dist_fname):
    """
    Read the matrix from mat_fname. Compute the similarity distribution
    between row elements and write the values to the dist_fname.
    """
    matrix = mat_fname
    M = load_matrix(matrix)
    write_distribution(M, dist_fname)
    pass


def help_message():
    print "python compute_distribution.py -i matrix_file -o distribution_file"
    pass


def command_line():
    """
    Get the file names from the command line and process.
    """
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:o:h")
    except getopt.GetoptError, err:
        print err
        help_message()
        sys.exit(-1)
        
    matrix_fname = None
    dist_fname = None
    
    for opt, val in opts:
        if opt == "-h":
            help_message()
            sys.exit(-1)
        if opt == "-i":
            matrix_fname = val.strip()
        if opt == "-o":
            dist_fname = val.strip()
    if matrix_fname and dist_fname:
        process(matrix_fname, dist_fname)
    else:
        help_message()
    pass    


if __name__ == "__main__":
    command_line()
    pass
    

