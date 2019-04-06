"""
MATRIX, implements a sparse 2D matrix with following functions.

read_matrix(fname) #Read a matrix from a file (row_id col_id:val ...)
get_row_id_list() # Returns a sorted list of all row ids.
row_exists(row_id) # if there is a row with this id returns True.
get_row(row_id) # returns a hash of column_ids for the row.
get_column(column_id) # returns a hash of row_ids for the column.
set_matrix(MATRIX) # copies the given matrix to this.
get_elememnt(row_id, colum_id)
set_element(row_id, column_id, val)
transpose() #transposes this.
write_matrix(fname) #writes the matrix to a text file.
"""

import sys, math

class MATRIX:
    """
    Implements a sparse matrix to store data.
    """
    def __init__ (self, SPARSE = True):
        """
        matrix is internally represented as a hash of hashes where each hash
        is a row of the matrix.
        {row_id:{col_id:val}}
        If SPARSE is set to True then we will create only sparse matrices and
        sparse vectors.
        """
        self.mdata = {}
        self.sparse =SPARSE
        pass

    def read_matrix(self, fname):
        """
        Read the matrix from file fname.
        Each line in fname represents a row of the matrix.
        First element in the line is the row id.
        row_id col_id:val col_id:val
        """
        F = open(fname, "r")
        self.colids = []
        for line in F:
            p = line.strip().split()
            row_id = int(p[0])
            h = {}
            for ele in p[1:]:
                col_id, val = ele.split(":")
                if (not self.sparse) or (self.sparse and float(val) != 0):
                    h[int(col_id)] = float(val)
                    self.colids.append(int(col_id))
            self.mdata[row_id] = h
        F.close()
        pass

    def get_row_id_list(self):
        """
        Returns a list of row ids sorted ascendingly.
        """
        L = self.mdata.keys()
        L.sort()
        return(L)

    def get_column_id_list(self):
        """
        Returns a list of column ids, sorted ascendingly.
        """
        self.colids = list(set(self.colids))
        self.colids.sort()
        return(self.colids)

    def L2_normalize_rows(self):
        """
        L2 normalize rows.
        """
        for row_id in self.mdata:
            sqdsum = 0
            for col_id in self.mdata[row_id]:
                sqdsum += self.mdata[row_id][col_id]**2
            sqdsum = math.sqrt(sqdsum)
            if sqdsum != 0:
                for col_id in self.mdata[row_id]:
                    self.mdata[row_id][col_id] = float(
                        self.mdata[row_id][col_id])/float(sqdsum)
        pass

    def L1_normalize_rows(self):
        """
        L1 normalize rows.
        """
        for row_id in self.mdata:
            total = 0
            for col_id in self.mdata[row_id]:
                total += math.fabs(self.mdata[row_id][col_id])
            if total != 0:
                for col_id in self.mdata[row_id]:
                    self.mdata[row_id][col_id] = float(
                        self.mdata[row_id][col_id])/float(total)
        pass

    def get_row_sum(self, row_id):
        """
        Returns the sum of elements in the row with row_id.
        """
        total = 0
        row = self.get_row(row_id)
        for colid in row:
            total += row[colid]
        return(total)

    def row_exists(self, row_id):
        """
        If a row with the given row_id exists, returns True.
        Otherwise returns False.
        """
        return(row_id in self.mdata) 

    def get_row(self, row_id):
        """
        Returns the row (a vector) with the row_id.
        """
        return(self.mdata[row_id])

    def get_column(self, colum_id):
        """
        Returns the colum (a vector) with the colum_id.
        """
        v = {}
        for row_id in self.mdata:
            if coulm_id in self.mdata[row_id]:
                v[row_id] = self.mdata[row_id][coumn_id]
        return(v)            

    def transpose(self):
        """
        Returns the transporse of the matrix.
        """
        M = MATRIX(self.sparse)
        for row_id in self.mdata:
            for col_id in self.mdata[row_id]:
                M.mdata.setdefault(col_id, {})[row_id] = self.mdata[row_id][col_id]
        return(M)

    def set_matrix(self, M):
        """
        Copies the data of M to this matrix.
        """
        from copy import copy
        self.mdata = copy(M.mdata)
        pass

    def get_element(self, row_id, colum_id):
        """
        Returns the value at position.
        """
        return(self.mdata[row_id][colum_id])

    def set_element(self, row_id, colum_id, value):
        """
        set [row_id][colum_id] to value.
        """
        self.mdata.setdefault(row_id, {})[colum_id] = value
        pass

    def set_row(self, row_id, vect):
        """
        vect is a dictionary of {col_id:val}. We set the values
        of the elements in the row specified by the row_id to
        these values. This function is more efficient if you want
        to set multiple values of a row to some values.
        """
        row = self.mdata[row_id]
        for colid in vect:
            val = vect[colid]
            if val == 0:
                if colid in row:
                    del(row[colid])
            else:
                row[colid] = vect[colid]
        pass

    def matrix_string (self):
        """
        produce a space separated string.
        This can be readily write to a file or terminal.
        """
        str = ""
        for row_id in self.mdata:
            str += "%d " % row_id
            for col_id in self.get_row(row_id):
                str += "%d:%f " % (col_id, self.get_element(row_id, col_id))
            str += "\n"
        return(str)

    def write_matrix(self, fname):
        """
        Write the matrix that is stored in self.mdata to the file named fname.
        Use this function if the matrix is large and do not use matrix_string to
        first get a one long string and then write to a file.
        row_id col_id:val ...
        """
        F = open(fname, "w")
        for row_id in self.mdata:
            F.write("%d " % row_id)
            for col_id in self.get_row(row_id):
                F.write("%d:%f " % (col_id, self.get_element(row_id, col_id)))
            F.write("\n")
            F.flush()
        F.close()
        pass

    def __str__ (self):
        """
        Displayed with print MATRIX.
        """
        return(self.matrix_string())    
    pass
