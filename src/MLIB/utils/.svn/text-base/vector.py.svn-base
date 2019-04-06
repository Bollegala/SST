# coding: utf-8

import math

class FAST_VECTOR:

    """
    Use numpy arrays to create an efficient vector implementation.
    """

    def __init__(self):
        """
        label is the vector label, fids and fvals are respectively
        feature ids (int) and feature values (float).
        """
        self.label = None
        self.fvals = {}
        pass

    def add(self,fid,fval):
        """
        Add the feature with id fid and value fval to the vector.
        """
        self.fvals[fid] = fval
        pass

    def size(self):
        """
        Returns the number of elements in the vector.
        """
        return len(self.fvals)

    def get(self,fid):
        """
        Return the fval of fid.
        """
        return self.fvals.get(fid,0)

    def __iter__(self):
        return self.fvals.iteritems()

    def __getitem__(self,fid):
        """
        Return the value at index fid.
        """
        return self.fvals.get(fid)

    def __setitem__(self,fid,fval):
        """
        Set the value of fid to fval.
        """
        self.fvals[fid] = fval
        pass
    
    def createFromArray(self, label, A):
        """
        Given a numpy array A and a label, convert this
        into a vector instance and store it.
        """
        self.label = label
        for (fid, fval) in enumerate(A):
            if fval != 0:
                # We must add one to the index to compute the
                # actual feature id!
                self.__setitem__((fid + 1), fval)
        pass 
    pass


class VECTOR:
    """
    Implements a sparse vector. Stores the squared sum and the
    sum of elements for efficient L1 and L2 normalization.
    """
    def __init__ (self):
        self.element = {}
        self.L1_norm = 0
        self.L2_norm = 0
        self.label = None
        self.comment = None
        pass

    def __getitem__(self, index):
        return self.element[index]

    def __setitem__(self, index, value):
        curVal = self.get(index,0)
        self.L1_norm = self.L1_norm-math.fabs(curVal)+math.fabs(value)
        self.L2_norm = self.L2_norm-(curVal**2)+(value**2)
        self.element[index] = value
        pass

    def get(self, i, default):
        return self.element.get(i, default)

    def copy(self):
        if self.__class__ is VECTOR:
            newVect = VECTOR()
            newVect.element = self.element.copy()
            newVect.L1_norm = self.L1_norm
            newVect.L2_norm = self.L2_norm
            return newVect
        import copy
        return copy.copy(self)
            

    def clear(self):
        self.element.clear()
        self.L1_norm = 0
        self.L2_norm = 0
        pass

    def __len__(self):
        return len(self.element)

    def __delitem__(self, index):
        curVal = self.element[index]
        del self.element[index]
        self.L1_norm = self.L1_norm-math.fabs(curVal)
        self.L2_norm = self.L2_norm-(curVal**2)
        pass

    def __str__(self):
        return "VALUES = %s L1 = %f L2 = %f" % \
               (str(self.element), self.L1_norm, self.L2_norm)

    def __iter__(self):
        return self.element.__iter__()

    def __contains__(self, index):
        return self.element.__contains__(index)

    def keys(self):
        return self.element.keys()

    def items(self):
        return self.element.items()

    def values(self):
        return self.element.values()

    def L1(self):
        return self.L1_norm

    def L2(self):
        return math.sqrt(self.L2_norm)

    def add(self, A):
        """
        Adds the vector A to self.
        """
        for i in A:
            self[i] = self.get(i,0)+A[i]
        pass
    pass

def inner_product(A, B):
    """
    Returns the inner product between the two vectors.
    """
    return sum([(B[i] * A.get(i, 0)) for i in B])
    
def cosine_similarity(A,B):
    """
    Returns the cosine similarity between two vectors.
    """
    tot = 0
    if A.L2() == 0 or B.L2() == 0:
        return 0
    for k in A.keys():
        if k in B:
            tot += A[k] * B[k]
    if tot == 0:
        return 0
    res = float(tot)/(float(A.L2()) * float(B.L2()))
    return res

def utest_efficient_vector():
    """
    Unit test for EFFICIENT_VECTOR class.
    """
    v = EFFICIENT_VECTOR()
    v.add(1,10)
    v.add(2,15)
    for e in v:
        print e
    pass

if __name__ == "__main__":
    utest_efficient_vector()



