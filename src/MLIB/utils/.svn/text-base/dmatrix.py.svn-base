#! -*- coding: utf-8 -*-

"""
MATRIX, implements a sparse 2D matrix with following functions.
This implementation is more efficient than the matrix.py version
because it can directly access elements using both row and column
indices.

read_matrix(fname) #Read a matrix from a file (row_id col_id:val ...)
get_row_id_list() # Returns a sorted list of all row ids.
row_exists(row_id) # if there is a row with this id returns True.
get_row(row_id) # returns a hash of column_ids for the row.
get_column(column_id) # returns a hash of row_ids for the column.
set_matrix(MATRIX) # copies the given matrix to this.
get_elememnt(row_id, colum_id)
set_element(row_id, column_id, val)
write_matrix(fname) #writes the matrix to a text file.
"""

######## CHANGE LOG #############################################################
# 2010/11/29: Implemented pointwise mutual information to weight co-occurrences.
#
#################################################################################

import sys, math
from vector import VECTOR

class DMATRIX:
    """
    Implements a sparse double array matrix to store data.
    """
    def __init__ (self, SPARSE=True):
        """
        matrix is internally represented as a hash of hashes where each hash
        is a row of the matrix.
        {row_id:{col_id:val}}
        If SPARSE is set to True then we will create only sparse matrices and
        sparse vectors.
        """
        self.rvects = {}
        self.cvects = {}
        self.sparse = SPARSE
        pass

    def transpose(self):
        """
        Exchange the rows and columns of the matrix.
        """
        rows = self.rvects
        self.rvects = self.cvects
        self.cvects = rows
        pass

    def normalize_column_max(self):
        """
        Divides each column by its maximum value. This is useful when training
        using logistic regression. Lower bound must also be normalized to zero
        if the feature vectors are not sparsed.
        """
        for cid in self.cvects:
            maxVal = float(max(self.cvects[cid].values()))
            #print cid, maxVal
            if maxVal != 0:
                for rid in self.cvects[cid]:
                    val = self.get_element(rid, cid)
                    val = val / maxVal
                    self.set_element(rid, cid, val)                
        pass        

    def read_matrix(self, fname):
        """
        Read the matrix from file fname.
        Each line in fname represents a row of the matrix.
        First element in the line is the row id.
        row_id col_id:val col_id:val
        """
        nonzeros = 0
        F = open(fname, "r")
        for line in F:
            p = line.strip().split()
            rid = int(p[0])
            for ele in p[1:]:
                col_id, val = ele.split(":")
                cid = int(col_id)
                fval = float(val)
                if (not self.sparse) or (self.sparse and fval != 0):
                    self.rvects.setdefault(rid,VECTOR())[cid] = fval
                    self.cvects.setdefault(cid,VECTOR())[rid] = fval
                    nonzeros += 1
        F.close()
        return nonzeros

    def get_row_id_list(self):
        """
        Returns a list of row ids sorted ascendingly.
        """
        L = self.rvects.keys()
        L.sort()
        return L

    def get_column_id_list(self):
        """
        Returns a list of column ids, sorted ascendingly.
        """
        L = self.cvects.keys()
        L.sort()
        return L
  
        
    def get_row_sum(self, row_id):
        """
        Returns the absolute sum of elements in the row with row_id.
        """
        return self.rvects[row_id].L1()

    def get_column_sum(self, col_id):
        """
        Returns the absolute sum of elements in the column with col_id.
        """
        return self.cvects[col_id].L1()
    

    def row_exists(self, row_id):
        """
        If a row with the given row_id exists, returns True.
        Otherwise returns False.
        """
        return(row_id in self.rvects) 

    def get_row(self, row_id):
        """
        Returns the row (a vector) with the row_id.
        """
        return(self.rvects[row_id])

    def get_column(self, colum_id):
        """
        Returns the colum (a vector) with the colum_id.
        """
        return(self.cvects[colum_id])

    def increment_element(self, row_id, col_id, val):
        """
        Increments the specified element by val.
        If the element does not exist, then it is considered as zero.
        """
        try:
            curVal = self.get_element(row_id, col_id)
        except KeyError:
            curVal = 0
        self.set_element(row_id, col_id, (curVal + val))
        pass   

    
    def get_element(self, row_id, colum_id):
        """
        Returns the value at position.
        """
        return(self.rvects[row_id][colum_id])

    def set_element(self, row_id, colum_id, value):
        """
        set [row_id][colum_id] to value.
        """
        self.rvects.setdefault(row_id, VECTOR())[colum_id] = value
        self.cvects.setdefault(colum_id, VECTOR())[row_id] = value
        pass

    def merge(self, rowsORcols, maxInd, curInd):
        """
        Merge the vector corresponding to curInd to the
        vector corresponding to maxInd. Update the rest
        of the elements in the matrix accordingly.
        """
        if rowsORcols == "ROWS":
            # Merging row with id curInd to maxInd.
            self.rvects[maxInd].add(self.rvects[curInd])
            # for the columns that had values in the row that we deleted above,
            # delete their row values, and update the columns that has values
            # in the rows with id maxInd.
            for cid in self.rvects[curInd]:
                del self.cvects[cid][curInd]
                self.cvects[cid][maxInd] = self.rvects[maxInd][cid]
            # delete the row with id curInd.
            del self.rvects[curInd]
        if rowsORcols == "COLUMNS":
            # Merging column with id curInd to maxInd.
            self.cvects[maxInd].add(self.cvects[curInd])
            # for the rows that had values in the column that we deleted above,
            # delete their column values, and update the rows that has values
            # in the column with id maxInd.
            for rid in self.cvects[curInd]:
                del self.rvects[rid][curInd]
                self.rvects[rid][maxInd] = self.cvects[maxInd][rid]
            # delete the column with id curInd.
            del self.cvects[curInd]            
        pass    

    def get_entropies(self, columns=True):
        """
        If columns is True then we compute the entropy of each column
        id over the distribution of row ids and return a dictionary
        in which keys are column ids and values are their corresponding entropies.
        If columns is set to False then we compute entropies for
        row ids over the distribution of column ids.
        """
        entropies = {}
        if columns:
            dist = self.cvects
        else:
            dist = self.rvects
        for element in dist:
            val = 0
            for index in dist[element]:
                prob = float(dist[element][index]) / float(dist[element].L1())
                val += (prob * math.log(prob, 2))
            entropies[element] = -val
        return entropies

    def get_PMI(self):
        """
        Compute the pointwise mutual information between each rowid and
        each columnid and return the values in a dmatrix of the format
        PMI[rowid][colid]. We use the discounting formula proposed by
        Pantenl & Ravichandran (ACL 2004) to correct the pmi values
        for rare events. Note that PMI can be negative if the two events
        x and y do not co-occur a lot but occur individually in a corpus.
	Negative PMI values will be set to zero.
        """
        PMI = DMATRIX()
        N = sum([self.get_row(x).L1() for x in self.get_row_id_list()])
        #N = sum([self.get_column(x).L1() for x in self.get_column_id_list()])
        for rid in self.get_row_id_list():
            for cid in self.get_row(rid):
                cef = self.get_element(rid, cid)
                if cef == 0:
                    PMI[rid][cid] = 0
                    continue
                pef = float(cef) / float(N)
                ce = self.get_row(rid).L1()
                cf = self.get_column(cid).L1()
                pe = float(ce) / float(N)
                pf = float(cf) / float(N)
                df1 = float(cef) / float(1.0 + cef)
                minVal = min(ce, cf)
                df2 = float(minVal) / float(1.0 + minVal)
                df = df1 * df2
                mi = math.log((pef / (pe * pf)), 2.0)
                val = mi * df
		#if val > 0:
		PMI.set_element(rid, cid, val)
        return PMI  

    def shape(self):
        """
        Returns the number of rows and columns of the matrix.
        """
        return (len(self.rvects), len(self.cvects))

    def sanity(self):
        """
        This sanity check will confirm that row values and
        column values are consistent.
        """
        for rid in self.rvects:
            for cid in self.cvects:
                aval = self.rvects[rid].get(cid,0)
                bval = self.cvects[cid].get(rid,0)
                if aval != bval:
                    raise "Inconsistent at %d %d %f %f" % (rid, cid, aval, bval)
        pass               
            

    def __str__(self):
        """
        Can be used to print the matrix to terminal.
        """
        s = ""
        for rid in self.rvects:
            s += "%d " % rid
            for cid in self.rvects[rid]:
                s += "%d:%f " % (cid, self.rvects[rid][cid])
            s += "\n"
        return s

    def full_print(self):
        """
        Displays the entire matrix including the sparse elements.
        """
        print self.__str__()
        return
        (n, m) = self.shape()
        print '(%d,%d)' % (n,m)
        for i in range(0,n):
            for j in range(0,m):
                if (i+1) in self.rvects:
                    row = self.rvects[i+1]
                    if (j+1) in row:
                        print int(row[j+1]), '\t',
                    else:
                        print '0\t',
                else:
                    print '0\t',
            print
        print
        pass
                

    def write_matrix(self, fname):
        """
        Writes to a file.
        """
        F = open(fname, "w")
        for rid in self.rvects:
            F.write("%d " % rid)
            for cid in self.rvects[rid]:
                F.write("%d:%f " % (cid, self.rvects[rid][cid]))
            F.write("\n")
        F.close()
        pass    
    pass

def convertToPySparseMatrix(dmat):
    """
    Given a dmatrix format matrix, we will construct a pysparse.spmatrix.ll_mat
    and return it. This is useful when performing further computations such
    as eigenvalue, eigenvector computations.
    """
    pass

def debug_DMATRIX():
    M = DMATRIX()
    M.read_matrix("../work/clusttest.data")
    print M
    M.sanity()
    M.merge("COLUMNS",2,1)
    M.sanity()
    print M
    pass

def debug_VECTOR():
    V = VECTOR()
    V[1] = 1
    V[2] = 2
    W = VECTOR()
    W[3] = 3
    W[4] = 4
    print V
    print W
    V.add(W)
    print V
    pass
    

if __name__ == "__main__":
    debug_VECTOR()
    debug_DMATRIX()
    pass
