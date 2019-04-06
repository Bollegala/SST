#! -*- coding: utf-8 -*-

"""
This program does the follwing.
   Read the similarity distribution file and find the top k similar neighbours
   or elements with similarity greater than a threshold theta. Which mode to
   operate in can be specified.
   Find the names of those features and write a file in the form:
   MAIN_ENTRY TOP1,similarity TOP2,similarity ...
   This will be our sentiment sensitive distributional thesaurus.

"""

import sys
import os
import getopt

class STACK_THETA:

    """
    Use this stack to select the elements with similarity greater
    than a predefined threshold theta.
    """

    def __init__(self, theta):
        self.min_sim = theta
        self.L = []
        # prevent out of memory errors by specifying the
        # maximum size of the stack
        self.max_stack_size = 1000
        self.min_ID = -1
        self.min_val = float("infinity")
        pass

    def add(self, feat, val):
        """
        Add the feature to the list. If val is equal or greater
        than theta (min_sim), then add feature to the list.
        However, if the stack has already reached its maximum
        allowed size, then replace the element with minimum similarity
        in the stack with the current element if the current element
        has similarity greater than the minimum element.
        If val is smaller than theta (min_sim), then do not add
        feat to the list.
        """
        if val >= self.min_sim:
            if len(self.L) < self.max_stack_size:
                self.L.append((feat,val))
                if val < self.min_val:
                    self.min_val = val
                    self.minID = len(self.L) - 1
            else:
                # find the minimum element and replace it with the
                # current element if the current element has similarity
                # greater than the minimum element.
                if val > self.min_val:
                    self.L[self.min_ID] = (feat,val)
                    self.update_minimum_value()
        pass

    def update_minimum_value(self):
        """
        Search the list and find the minimum value and its index.
        Update minVal and minID.
        """
        min_val = float("infinity")
        min_ID = -1
        for (i, (feat,val)) in enumerate(self.L):
            if val < min_val:
                min_val = val
                min_ID = i
        self.min_val = min_val
        self.min_ID = min_ID            
        pass

    def show(self):
        print self.L
        pass
    
    pass


def select_top_theta(dist_fname, select_fname, theta):
    """
    selects the top k most similar features for each feature from the
    similarity distribution. Write the selected features to select_fname.
    """
    distFile = open(dist_fname)
    topFile = open(select_fname, "w")
    h = {}
    count = 0
    for line in distFile:
        p = line.strip().split(",")
        first = int(p[0])
        second = int(p[1])
        val = float(p[2])
        count += 1
        if count % 1000000 == 0:
            print count, count/1000000
        if first not in h:
            h[first] = STACK_THETA(theta)
        h[first].add(second,val)
    # write the topks to topFile.
    for first in h:
        topFile.write("%d\t" % first)
        h[first].L.sort(sort_comparator)
        for (feat,val) in h[first].L:
            topFile.write("%d,%f\t" % (feat,val))
        topFile.write("\n")
    topFile.close()
    distFile.close()
    pass

def sort_comparator(A, B):
    """
    This function is used to sort the list with tuples (feat,val).
    """
    if A[1] > B[1]:
        return -1
    return 1

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

def generate_thesaurus(topk_fname, thesaurus_fname, fid_fname):
    """
    Given a file that contains the feature ids and similarity scores for the
    top k most similar features (topk_fname), we will replace all feature
    ids by feature names.
    """
    feats = loadIds(fid_fname)
    F = open(topk_fname)
    G = open(thesaurus_fname, "w")
    for line in F:
        p = line.strip().split()
        first = int(p[0].strip())
        first_id = feats[first]['idstr']
        G.write("%s\t" % first_id)
        for e in p[1:]:
            (fid, val) = e.split(",")
            G.write("%s,%f\t" % (feats[int(fid)]['idstr'], float(val)))
        G.write("\n")
    F.close()
    G.close()
    pass

def create_thesaurus(dist_fname, element_id_fname, thesaurus_fname):
    """
    Selects the top k most similar elements from the similarity distribution file (dist_fname)
    and write the ids of those elements to a file with .id extension.
    Next, we will read in this elements idthes file and the ids assigned to the elements
    (element_id_fname) and replace the element ids with element names to create the
    thesaurus which we will write to a file (thesaurus_fname).
    """
    thesaurus_id_fname = "%s.id" % thesaurus_fname
    if os.path.exists(thesaurus_id_fname):
        os.remove(thesaurus_id_fname)
    select_top_theta(dist_fname,  thesaurus_id_fname, 0)
    generate_thesaurus(thesaurus_id_fname, thesaurus_fname, element_id_fname)
    pass

def help_message():
    """
    Default message.
    """
    print """python -d similarity distribution file
                    -e element id file
                    -t thesaurus file"""
    pass

def command_line():
    """
    Get the file names from the command line and process.
    """
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:e:t:h")
    except getopt.GetoptError, err:
        print err
        help_message()
        sys.exit(-1)
    dist_fname = None
    element_id_fname = None
    thesaurus_fname = None
    for opt, val in opts:
        if opt == "-h":
            help_message()
            sys.exit(-1)
        if opt == "-d":
            dist_fname = val.strip()
        if opt == "-e":
            element_id_fname = val.strip()
        if opt == "-t":
            thesaurus_fname = val.strip()
    if dist_fname and element_id_fname and thesaurus_fname:
        create_thesaurus(dist_fname, element_id_fname, thesaurus_fname)
    else:
        help_message()
    pass        
    
if __name__ == "__main__":
    command_line()
    pass
