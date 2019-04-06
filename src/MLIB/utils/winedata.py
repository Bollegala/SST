"""
This scripts reads in the wine quality regression dataset,
normalizes the values into range [0,1] and produces data files
in the sparse format.

Danushka Bollegala.
12-July-2010.
"""

import sys

def convert_to_sparse_format(fin, fout):
    """
    Converts all values including the outputs to features.
    """
    features = 12
    F = open(fin)
    G = open(fout, "w")
    for line in F:
        p = line.strip().split(';')
        vals = [float(x) for x in p]
        G.write("%d " % vals[features-1])
        for i in range(1,features+1):
            G.write("%d:%f " % (i, vals[i-1]))
        G.write("\n")
    F.close()
    G.close()
    pass

def save_regression_data(fin, fout):
    """
    Remove the last feature (which is the output and is normalized)
    and put it as the regression output. Save to a file.
    """
    F = open(fin)
    G = open(fout, "w")
    for line in F:
        h = {}
        p = line.strip().split()[1:]
        for e in p:
            t = e.split(':')
            h[int(t[0])] = float(t[1])
        G.write("%f " % h[12])
        for i in range(1,12):
            G.write("%d:%f " % (i, h[i]))
        G.write("\n")
    F.close()
    G.close()
    pass

def convert_to_matlab_format(fin, fout):
    """
    Convert the sparse format back to semicolon separated
    MATLAB format. Output label is assigned as the last feature.
    """
    F = open(fin)
    G = open(fout, "w")
    for line in F:
        h = {}
        p = line.strip().split()[1:]
        vals = [(x.split(':')[1]) for x in p] 
        G.write("%s\n" % ";".join(vals))
    F.close()
    G.close()
    pass
    

if __name__ == "__main__":
    #convert_to_sparse_format(sys.argv[1], sys.argv[2])
    #save_regression_data(sys.argv[1], sys.argv[2])
    convert_to_matlab_format(sys.argv[1], sys.argv[2])
    pass
        
