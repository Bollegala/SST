#! /usr/bin/python
#! coding: utf-8
"""
Performs sequential coclustering.
Given a matrix,theta (column threshold), and phi (row threshold),
produces a set of clusterings (row clusters and column clusters)
"""

import sys, math, re, getopt
sys.path.append("../..")

from MLIB.cluster.matrix import MATRIX
from MLIB.utils.ProgBar import TerminalController,ProgressBar

class SEQCLUST:

    def __init__(self):
        pass

    def patsort(self, A, B):
        if A[1] > B[1]:
            return(-1)
        return(1)

    def sim(self, c, v):
        sim = 0
        for wpair in v:
            if wpair in c.wpairs:
                sim += float(v[wpair]*c.wpairs[wpair])
        return(sim)            

    def cluster(self, m, theta):
        #first sort patterns according to the total frequency
        #of all word-pairs in which they appear.
        pats = [] # (pat_id, total_frequency_in_wpairs)
        for pat in m.get_row_id_list():
            row = m.get_row(pat)
            total = 0
            for k in row:
                total += row[k]
            pats.append((pat, total))
        N = len(pats)
        pats.sort(self.patsort)
        #initialize clusters.
        clusts = []
        count = 0
        m.L2_normalize_rows()
        term = TerminalController()
        progress = ProgressBar(term, "Clustering total rows = %d" %N)
        for (pat, total) in pats:
            maxsim = 0
            maxclust = None
            count += 1
            for c in clusts:
                v = m.get_row(pat)
                s = self.sim(c, v)
                if s > maxsim:
                    maxsim = s
                    maxclust = c
            if maxsim > theta:
                progress.update(float(count)/N,
                                "MERGED %d: row = %d freq = %d clusts = %d" \
                                % (count, pat, total, len(clusts)))
                maxclust.merge(pat, m.get_row(pat))
            else:
                progress.update(float(count)/N,
                                "   NEW %d: %s freq = %d clusts = %d" \
                                % (count, pat, total, len(clusts)))
                clusts.append(SEQ_CLUST_DATA(pat, m.get_row(pat)))
        return(clusts)

    def write_clusters(self, clusts, theta, fname):
        """
        format.
        total_no_clusts sparsity singletons theta comma_sep_lists
        sparsity = singletons/total_no_clusts
        """
        F = open(fname, "w")
        singletons = 0
        for c in clusts:
            if len(c.pats) == 1:
                singletons += 1
        sparsity = float(singletons)/float(len(clusts))
        print "Total Clusters =", len(clusts)
        print "singletons =", singletons
        print "sparsity =", sparsity
        print "theta =", theta
        F.write("TOTAL_CLUSTERS=%d SINGLETONS=%d SPARSITY=%f THETA=%f "\
                % (len(clusts), singletons, sparsity, theta))
        for c in clusts:
            F.write("%s " % ",".join([str(x) for x in c.pats]))
        F.close()       
    pass

class SEQ_CLUST_DATA:

    def __init__(self, pat, v):
        self.pats = [pat]
        self.wpairs = {}
        for k in v:
            if v[k] != 0:
                self.wpairs[k] = v[k]
        pass

    def normalize(self):
        sqd = 0
        for k in self.wpairs:
            sqd += self.wpairs[k]**2
        sqd = math.sqrt(float(sqd))
        for k in self.wpairs:
            self.wpairs[k] = float(self.wpairs[k])/float(sqd)
        pass
    
    def merge(self, pat, v):
        self.pats.append(pat)
        for k in v:
            if v[k] != 0:
                self.wpairs[k] = self.wpairs.get(k, 0)+v[k]
        self.normalize()
    pass

def usage():
    sys.stderr.write("""python seqclust.py -i <input_matrix_file>
                     -o <output_clusters_file>
                     -t <threshold>\n""")
    pass

def process_command_line():
    """
    Get the command line arguments and validate.
    """
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:t:u",\
                                   ["help", "input=","output=",
                                    "theta=", "transpose"])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)
    # parameter values.
    matrix_fname = None
    clust_fname = None
    theta = 0
    transpose = True
    for opt, val in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(1)
        if opt in ("-i", "--input"):
            matrix_fname = val
        if opt in ("-o", "--output"):
            clust_fname = val
        if opt in ("-t", "--theta"):
            theta = float(val)
        if opt in ("-u", "--transpose"):
            transpose = False
    if matrix_fname and clust_fname and (theta >= 0) and (theta <=1):
        perform_sequential_clustering(matrix_fname, clust_fname,
                                      theta, transpose)
    pass

def perform_sequential_clustering(matrix_fname, clust_fname, theta, transpose):
    """
    Perform sequenctial clustering.
    """
    M = MATRIX(True)
    M.read_matrix(matrix_fname)
    if transpose:
        MT = M.transpose()
    else:
        MT = M    
    clustAlgo = SEQCLUST()
    clusts = clustAlgo.cluster(MT, theta)
    clustAlgo.write_clusters(clusts, theta, clust_fname)
    sys.stderr.write("Clustering Finished....Terminating\n")
    pass

if __name__ == "__main__":
    process_command_line()

