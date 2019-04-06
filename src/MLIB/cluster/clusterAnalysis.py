#! -*- coding: utf-8 -*-

"""
This program implements the functionality necessary to do the following,
  1. Given a matrix, a row index and a column index files, find the total
     frequency of each row or column and sort the rows or columns in the
     descending order of the frequency.
     Write the sorted row and column ids to an external file.

  2. Given a file that contain frequency of each row or column ids,
     and a threshold, select the row or column ids that have frequency
     greater than this threshold. Write the selected ids to a file.

  3. Given a matrix, a list of row ids and column ids,
     remove and rows or columns
     not in the lists and save the reduced matrix to a file.
 
  4. Given a set of row or column clusters, compute the centroid of the
     clusters and sort the elements in the cluster according to the distance
     from the corresponding cluster centroid. This function is useful when
     generating a ranked thesaurus from a set of clusters.

Danushka Bollegala.
2010/01/28: Initial implementation.
"""

import sys, re
sys.path.append("../..")

from MLIB.utils.dmatrix import DMATRIX
from MLIB.utils.vector import VECTOR, cosine_similarity

class CLUSTERS:

    """
    Given a file of clusters, this class can load all the
    clusters to memory.
    """

    def __init__(self):
        pass

    def loadClusters(self, clustFname, IGNORE_SINGLETONS = False):
        """
        load clusters from file.
        clust[id] = [pattern_ids]
        pat2clustid[patid] = clustid
        """
        F = open(clustFname, "r")
        p = re.compile("\s+").split(F.readline().strip())
        F.close()
        self.no_clusters = int(p[0].split("=")[1])
        self.singletons = int(p[1].split("=")[1])
        self.sparsity = float(p[2].split("=")[1])
        self.theta = float(p[3].split("=")[1])
        self.clust = {}
        self.pat2clustid = {}
        clustid = 1
        for cstr in p[4:]:
            patterns = cstr.split(",")
            if IGNORE_SINGLETONS and (len(patterns)==1):
                continue
            for x in patterns:
                patid = int(x)
                self.clust.setdefault(clustid, []).append(patid)
                self.pat2clustid[patid] = clustid
            clustid += 1
        pass

    def showInfo(self):
        """
        Display various information regarding the clusters.
        """
        print "no. of clusters =", self.no_clusters
        print "singletons =", self.singletons
        print "sparsity =", self.sparsity
        print "threshold =", self.theta
        print "non-singletons =", self.no_clusters - self.singletons
        pass

    def getElementsByClusterID(self, clustID):
        """
        Returns the list of elements for the cluster with ID clustID.
        """
        return self.clust[clustID]

    def getClustID(self, element):
        """
        Returns the ID of the cluster which the element belongs to.
        """
        return self.pat2clustid[element]

    def getClustIDList(self):
        """
        Returns the list of cluster IDs.
        """
        return self.clust.keys()    
    pass

def loadIds(fname, delimiter='\t'):
    """
    loads row or column ids to a dictionary.
    """
    h = {}
    F = open(fname)
    for line in F:
        p = line.strip().split(delimiter)
        idval = int(p[1])
        idstr = p[0].strip()
        h[idval] = {'idstr':idstr,'freq':0}
    F.close()
    return h

def sort_ids(A, B):
    """
    A and B are elements of the list [(rowid,{'rowstr','freq'})]
    """
    if A[1]['freq'] > B[1]['freq']:
        return -1
    return 1

def saveIds(fname, sortedList, delimiter='\t'):
    """
    Write the sorted ids to file.
    """
    F = open(fname, 'w')
    for e in sortedList:
        F.write("%s%s%d%s%f\n" % (e[1]['idstr'],
                                  delimiter,
                                  e[0],
                                  delimiter,
                                  e[1]['freq']))
    F.close()
    pass


def countFrequency(matrixFname, rowFname, colFname):
    """
    Read the matrix and compute row and column totals.
    If row and col files are given then we will read the ids from those files.
    We will then write the sorted frequencies to files with extensions .total.
    """
    D = DMATRIX()
    print "Loading matrix,", matrixFname, "...",
    D.read_matrix(matrixFname)
    print "done"
    print "Loading row ids,", rowFname, "...",
    rows = loadIds(rowFname)
    print "done"
    print "Loading col ids,", colFname, "...",
    cols = loadIds(colFname)
    print "done"
    print "Getting row sums..."
    for rowid in rows:
        rows[rowid]['freq'] = D.get_row_sum(rowid)
    print "Getting column sums..."
    for colid in cols:
        cols[colid]['freq'] = D.get_col_sum(colid)
    rowItems = rows.items()
    colItems = cols.items()
    print "sorting rows...",
    rowItems.sort(sort_ids)
    print "done"
    print "sorting columns...",
    colItems.sort(sort_ids)
    print "done"
    saveIds("%s.total" % rowFname, rowItems)
    saveIds("%s.total" % colFname, colItems)
    pass

def sortEelements(clustFname, matrixFname, idFname, thesaurusFname, rows=False):
    """
    Read the matrix, clusters and ids. Now find the centroid in each cluster and
    compute the similarity from this centroid to each element in the cluster.
    Sort the elements in a cluster in the descending order of their
    similarity to the cluster centroid. Write the sorted elements to
    thesaurusFile. If rows is set to True then we will assume that clusters
    are row clusters. By default we assume the clusters to be column clusters.
    """
    pass

def reduceMatrix(clustFname, reducedFname, reducedRows, reducedColumns):
    """
    Remove any row or column ids that are not in reduced lists.
    Save the reduced matrix to a file.
    """
    rows = getIdList(reducedRows)
    cols = getIdList(reducedColumns)
    F = open(clustFname)
    G = open(reducedFname, 'w')
    for line in F:
        p = line.strip().split()
        if int(p[0]) not in rows:
            continue
        G.write('%d ' % int(p[0]))
        for e in p[1:]:
            t = e.split(':')
            colid = int(t[0])
            val = float(t[1])
            if colid in cols:
                G.write('%d:%f ' % (colid,val))
        G.write('\n')
    F.close()
    G.close()
    pass

def getIdList(fname, delimiter='\t'):
    """
    Given a file with row or column ids (feature\tid) we will load the
    ids to a list and return it.
    """
    L = []
    F = open(fname)
    for line in F:
        p = line.strip().split(delimiter)
        L.append(int(p[1]))
    F.close()
    return L

def selectElements(idFname, threshold, selectedFname, delimiter='\t'):
    """
    selects elements with frequency greater than the specified threshold.
    The file must be sorted first using,
    sort -k 3 -n -r rowids > rowids.sorted
    """
    F = open(idFname)
    G = open(selectedFname, "w")
    count = 0
    for line in F:
        p = line.strip().split(delimiter)
        freq = float(p[2])
        if freq <= threshold:
            break
        count += 1
        G.write("%s%s%d%s%d\n" % (p[0].strip(),
                                  delimiter,
                                  int(p[1]),
                                  delimiter,
                                  freq))
    G.close()
    F.close()
    print "Total number of elements selected =", count
    pass

def centroidRank(clustFname,
                 matrixFname,
                 thesaurusFname,
                 idFname=None,
                 rows=False):
    """
    Given a set of clusters (by default we assume column clusters),
    we will compute the cluster centroid of each cluster and
    rank the elements in a cluster in the descending order of their
    cosine similarity to the cluster centroid. We will write the sorted
    clusters to file thesaurusFname. If we are given names of each
    cluster in an idFname file, then instead of writing the cluster
    numbers we will write the names of columns to the thresholdFname.
    If rows is set to True we assume that clustFname represents
    row clusters. The matrix is given by the matrixFname.
    """
    C = CLUSTERS()
    C.loadClusters(clustFname, IGNORE_SINGLETONS=True)
    C.showInfo()
    # compute the centroid for each cluster.
    M = DMATRIX()
    M.read_matrix(matrixFname)
    clusters = []
    for clustID in C.getClustIDList():
        centroid = VECTOR()
        n = 0
        for element in C.getElementsByClusterID(clustID):
            if rows:
                v = M.get_row(element)
            else:
                v = M.get_column(element)
            centroid.add(v)
            n += 1
        # take the mean.
        for k in v:
            centroid[k] = centroid[k]/float(n)
        # compute the similarity to the centroid.
        simScores = []
        for element in C.getElementsByClusterID(clustID):
            if rows:
                v = M.get_row(element)
            else:
                v = M.get_column(element)
            sim = cosine_similarity(v, centroid)
            simScores.append((sim, element))
        # sort the elements in a cluster.
        simScores.sort(elementSorter)
        clusters.append(simScores)
        pass
    # if the id files is given then write the ids.
    # otherwise just write the elements.
    thesaurus = open(thesaurusFname, "w")
    if idFname:
        names = loadElementNames(idFname)
    for clust in clusters:
        for (sim,ele) in clust:
            if idFname:
                name = names[ele]
                thesaurus.write("%s:%f\t" % (name,sim))
            else:
                thesaurus.write("%s:%f\t" % (ele,sim))
        thesaurus.write("\n")
    thesaurus.close()
    pass

def loadElementNames(idFname):
    """
    Given a file where each line contains the name of an element
    separated by a tab with the ID assigned to theD element,
    we read this file and return a dictionary where the keys are
    element IDs and values are element names.
    """
    F = open(idFname)
    h = {}
    for line in F:
        p = line.strip().split('\t')
        name = p[0].strip()
        elementID = int(p[1])
        h[elementID] = name
    F.close()
    return h

def elementSorter(A, B):
    """
    This is the comparator used to sort elements in a cluster
    according to their similarity to the cluster centroid.
    """
    if A[0] > B[0]:
        return -1
    else:
        return 1
    pass

def utility_centroidRank():
    """
    This is a utility function that calls centroidRank function.
    """
    # select a type of thesaurus.
    
    #THESAURUS_TYPE = "ROW"
    THESAURUS_TYPE = "COLUMN"

    # Generate the COLUMN thesaurus.
    if THESAURUS_TYPE == "COLUMN":
        basePath = "../../../work"
        clustFname = "%s/cluster.column" % basePath
        matrixFname = "%s/matrix" % basePath
        thesaurusFname = "%s/thesaurus.column" % basePath
        idFname = "%s/colids" % basePath
        rows = False
        centroidRank(clustFname,matrixFname,thesaurusFname,idFname,rows)
        
    # Generate the ROW thesaurus.
    if THESAURUS_TYPE == "ROW":
        basePath = "../../../work"
        clustFname = "%s/cluster.row" % basePath
        matrixFname = "%s/matrix" % basePath
        thesaurusFname = "%s/thesaurus.row" % basePath
        idFname = "%s/rowids" % basePath
        rows = True
        centroidRank(clustFname,matrixFname,thesaurusFname,idFname,rows)
    pass

def main():
    #countFrequency("../../../work/matrix", "../../../work/colids","../../../work/rowids")
    #selectElements("../../../work/rowids.sorted", 10, "../../../work/rowids.selected")
    #selectElements("../../../work/colids.sorted", 10, "../../../work/colids.selected")
    #reduceMatrix("../../../work/matrix", "../../../work/matrix.reduced", "../../../work/rowids.selected", "../../../work/colids.selected")
    utility_centroidRank()
    pass

if __name__ == "__main__":
    main()
    


