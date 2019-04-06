#! /usr/bin/python
#! coding: utf-8
"""
Performs sequential coclustering.
Given a matrix,theta (column threshold), and phi (row threshold),
produces a set of clusterings (row clusters and column clusters)
"""

import sys,math,re,getopt
sys.path.append("../..")

from MLIB.utils.dmatrix import DMATRIX
from MLIB.utils.ProgBar import TerminalController,ProgressBar

class SEQCLUST:

    def __init__(self):
        #If you only want the progress bar, then set VERBOSE to False
        self.VERBOSE = False
        pass

    def patsort(self, A, B):
        if A[1] > B[1]:
            return(-1)
        return(1)
   
    def cosine(self, x, y):
        """
        For efficiency, set x to the smaller vector.
        """
        prod = 0
        for i in x:
            prod += x[i]*y.get(i,0)
        sim = float(prod)/(x.L2()*y.L2())
        return sim   

    def coclustering(self, M, theta, phi):
        """
        Implements sequential co-clustering.
        (alternation variant)
        """
        # Initialization. sorting row counts.
        cols = []
        rows = []
        columnIndex = {}
        rowIndex = {}
        for rowid in M.get_row_id_list():
            rows.append((rowid,M.get_row_sum(rowid)))
        rows.sort(self.patsort)
        no_rows = len(rows)
        # sorting column counts.
        for colid in M.get_column_id_list():
            cols.append((colid, M.get_column_sum(colid)))
        cols.sort(self.patsort)
        no_cols = len(cols)
        colclusts = {}
        rowclusts = {}
        theta_max = -1
        phi_max = -1
        if not self.VERBOSE:
            term = TerminalController()
            progress = ProgressBar(term,
                                   "Clustering rows = %d, columns = %d" % \
                                   (no_rows, no_cols))
        total = no_rows + no_cols
        count = 0
        # start alternative clustering.
        while(cols or rows):
            if cols:
                # column clustering.
                count += 1
                current_column = cols[0][0]
                del cols[0]
                theta_max = 0
                max_col_clust = -1
                validClusts = self.get_clusters(rowIndex,
                                                M.get_column(current_column))
                for c in validClusts:
                    s = self.cosine(M.get_column(current_column),
                                    M.get_column(c))
                    if s > theta_max:
                        theta_max = s
                        max_col_clust = c
                if theta_max > theta:
                    colclusts[max_col_clust].append(current_column)
                    self.update_index(rowIndex, M.get_column(current_column),
                                      max_col_clust)
                    M.merge("COLUMNS",max_col_clust,current_column)
                    if self.VERBOSE:
                        print "COL\t%d\tMRG\tSIM=%f\tTotal=(%d,%d) [%d/%d]" % \
                              (current_column,theta_max,
                               len(rowclusts), len(colclusts),
                               count, total)
                        M.full_print()
                    else:
                        progress.update(float(count)/total,\
                                        "COL %d MRG SIM=%f Total=(%d,%d) [%d/%d]" %\
                                        (current_column,theta_max,\
                                         len(rowclusts), len(colclusts),\
                                         count, total))
                        pass                                        
                else:
                    colclusts[current_column] = [current_column]
                    self.update_index(rowIndex, M.get_column(current_column),
                                      current_column)
                    if self.VERBOSE:
                        print "COL\t%d\tNEW\tSIM=%f\tTotal=(%d,%d) [%d/%d]" % \
                              (current_column,theta_max,
                               len(rowclusts), len(colclusts),
                               count, total)
                        M.full_print()
                    else:
                        progress.update(float(count)/total,\
                                        "COL %d NEW SIM=%f Total=(%d,%d) [%d/%d]" % \
                                        (current_column,theta_max,\
                                         len(rowclusts), len(colclusts),\
                                         count, total))
                        pass
            if rows:
                # row clustering.
                count += 1
                current_row = rows[0][0]
                del rows[0]
                phi_max = 0
                max_row_clust = -1
                validClusts = self.get_clusters(columnIndex,
                                                M.get_row(current_row))
                for c in validClusts:
                    s = self.cosine(M.get_row(current_row),
                                    M.get_row(c))
                    if s > phi_max:
                        phi_max = s
                        max_row_clust = c
                if phi_max > phi:
                    rowclusts[max_row_clust].append(current_row)
                    self.update_index(columnIndex, M.get_row(current_row),
                                      max_row_clust)
                    M.merge("ROWS",max_row_clust,current_row)
                    if self.VERBOSE:
                        print "ROW\t%d\tMRG\tSIM=%f\tTotal=(%d,%d) [%d/%d]" % \
                              (current_row,phi_max,
                               len(rowclusts), len(colclusts),
                               count, total)
                        M.full_print()
                    else:
                        progress.update(float(count)/total,\
                                        "ROW %d MRG SIM=%f Total=(%d,%d) [%d/%d]" % \
                                        (current_row,phi_max,
                                         len(rowclusts), len(colclusts),
                                         count, total))
                        pass                                        
                else:
                    rowclusts[current_row] = [current_row]
                    self.update_index(columnIndex, M.get_row(current_row),
                                      current_row)
                    if self.VERBOSE:
                        print "ROW\t%d\tNEW\tSIM=%f\tTotal=(%d,%d) [%d,%d]" % \
                              (current_row,phi_max,
                               len(rowclusts), len(colclusts),
                               count, total)
                        M.full_print()
                    else:
                        progress.update(float(count)/total,\
                                        "ROW %d NEW SIM=%f Total=(%d,%d) [%d/%d]" % \
                                        (current_row,phi_max,
                                         len(rowclusts), len(colclusts),
                                         count, total))
                        pass                 
        # Final steps.
        return (rowclusts,colclusts)

    def update_index(self, index, v, clusterId):
        """
        Add the keys of the vector v to index (as keys) and
        append clusterId to the corresponding posting lists.
        """
        for key in v:
            if key in index:
                if clusterId not in index[key]:
                    index[key].append(clusterId)
            else:
                index[key] = [clusterId]
        pass

    def get_clusters(self, index, v):
        """
        Find the clusters from the index that contain keys in v,
        and return the list of such clusters.
        """
        clusts = []
        for key in v:
            L = index.get(key,[])
            for clust in L:
                if clust not in clusts:
                    clusts.append(clust)
        return clusts
 

    def write_coclusters(self, rowclusts, colclusts, theta, phi, fname):
        """
        Two files will be written filename.row and filename.column,
        respectively for row and column clustering results.
        format.
        total_no_clusts sparsity singletons theta comma_sep_lists
        sparsity = singletons/total_no_clusts
        """
        # write column clusters.
        print "################## COLUMNS ##################"
        self.write_clusters(colclusts, theta, "theta", "%s.column" % fname)
        # write row clusters.
        print "################### ROWS ####################"    
        self.write_clusters(rowclusts, phi, "phi", "%s.row" % fname)
        pass

    def write_clusters(self, clusts, theta, param_name, fname):
        F = open(fname, "w")
        singletons = 0
        for c in clusts:
            if len(clusts[c]) == 1:
                singletons += 1
        sparsity = float(singletons)/float(len(clusts))
        print "Total Clusters =", len(clusts)
        print "singletons =", singletons
        print "sparsity =", sparsity
        print param_name, "=", theta
        F.write("TOTAL_CLUSTERS=%d SINGLETONS=%d SPARSITY=%f %s=%f "\
                % (len(clusts), singletons, sparsity, param_name, theta))
        for c in clusts:
            F.write("%s " % ",".join([str(x) for x in clusts[c]]))
        F.close()
        pass        
    pass

def load_clusters(clust_fname, IGNORE_SINGLETONS = False):
    """
    load clusters from file.
    clust[id] = [pattern_ids]
    """
    import re
    F = open(clust_fname, "r")
    p = re.compile("\s+").split(F.readline().strip())
    F.close()
    no_clusters = int(p[0].split("=")[1])
    singletons = int(p[1].split("=")[1])
    sparsity = float(p[2].split("=")[1])
    theta = float(p[3].split("=")[1])
    clust = {}
    pat2clustid = {}
    clustid = 0
    for cstr in p[4:]:
        patterns = cstr.split(",")
        clustid += 1
        if IGNORE_SINGLETONS and (len(patterns)==1):
            continue
        for x in patterns:
            patid = int(x)
            clust.setdefault(clustid, []).append(patid)
            pat2clustid[patid] = clustid
            pass
    return {"no_clusters":no_clusters,
            "singletons":singletons,
            "sparsity":sparsity,
            "theta":theta,
            "clust":clust,
            "pat2clustid":pat2clustid}
   

def usage():
    sys.stderr.write("""python seqclust.py -i <input_matrix_file>
                     -o <output_clusters_file>
                     -t <threshold_theta>
                     -p <threshold_phi>\n""")
    pass

def process_command_line():
    """
    Get the command line arguments and validate.
    """
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:t:p:",\
                                   ["help", "input=","output=", "theta=", "phi="])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)
    # parameter values.
    matrix_fname = None
    clust_fname = None
    theta = 0
    phi = 0
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
        if opt in ("-p", "--phi"):
            phi = float(val)
    if matrix_fname and clust_fname and (theta >= 0) and (theta <=1):
        perform_sequential_coclustering(matrix_fname, clust_fname, theta, phi)
    else:
        usage()
    pass

def perform_sequential_coclustering(matrix_fname, clust_fname, theta, phi,
                                    verbose = True):
    """
    Perform sequenctial clustering.
    """
    M = DMATRIX(True)
    print "Reading data matrix...",
    M.read_matrix(matrix_fname)
    print "Done."
    M.full_print()
    clustAlgo = SEQCLUST()
    clustAlgo.VERBOSE = verbose
    (rowclusts,colclusts) = clustAlgo.coclustering(M, theta, phi)
    clustAlgo.write_coclusters(rowclusts, colclusts, theta, phi, clust_fname)
    sys.stderr.write("Clustering Finished....Terminating\n")
    pass

if __name__ == "__main__":
    process_command_line()

