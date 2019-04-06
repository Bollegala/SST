# coding: utf-8

"""
This program contains the functionality for reading input files (instance files)
in numerous formats. It also has the capability to scale train and test vectors.

Change Log:
2011/09/04: Added support for non-numeric label types.
"""

import sys
import math
import random
import numpy

from config import COMMENT_DELIMITTER, ATTRIBUTE_DELIMITTER
from config import VALUE_DELIMITTER, DONT_WRITE_ZEROS
from vector import FAST_VECTOR 

    
class SEQUENTIAL_FILE_READER:
    
    """
    Implements a sequential file reader. Given the name of a file
    we open this file and provide sequential access to the opened file.
    The format of the file can be plain text, bunzip (.bz2) or gzip (.gz).
    The file can be read one line at a time using readLine() function,
    to get a vector with a label. Alternatively, you can call the
    read() method to read all lines to a list of vectors with their
    corresponding labels. However, this might be problematic with large
    feature files and is not required by one-pass online training
    algorithms. Moreover, during prediction time we only need to read
    the feature file one line at a time. The function getLineCount()
    reads the file to the end and counts the number of lines. It is used
    for held-out data splitting.    
    """
    
    def __init__(self,fname):
        """
        Opens the file fname for sequential reading.
        If INTEGER_IDS is set to True (default value) we will assume the
        feature ids to be integers. Otherwise we will consider feature
        ids to be strings.
        """
        self.fname = fname
        self.fileReader = self.openFile(fname)
        pass

    def countLines(self):
        """
        Returns the number of lines in the file.
        """
        count = 0
        for line in self.fileReader:
            count += 1
        self.reset()
        return count

    def readTextLine(self):
        """
        Read the next line from the file and return it as it is.
        Do not convert it to a vector. This is efficient compared
        to the readline() method that also converts to a vector.
        This method is useful when we do not want to convert the
        read line to a vectors in cases such as selecting training
        examples for held out evaluation.
        """
        return self.fileReader.readline()
        pass
    
    def readline(self):
        """
        Reads the next line from the file and return a feature vector.
        """
        labels = {}
        fids = {}
        line = self.fileReader.readline()
        if line == '':
            return None
        fvect = self.get_vector(line,labels,fids)
        return {"vect":fvect,
                "label":labels.keys()[0],
                "fids":fids}
    
    def read(self):
        """
        Reads a feature vector file and returns a feature id vs
        feature string hash and a list of feature vectors.
        """
        labels = {}
        fids = {}
        # read lines from fileReader and create vectors.
        fvects = [self.get_vector(line,labels,fids) for line in self.fileReader]
        return {"vects":fvects,
                "labels":labels,
                "featIDs":fids}
    
    def next(self):
        """
        Returns the next line in the file.
        """
        v = self.readline()
        if v is None:
            raise StopIteration
        else:
            return v

    def __iter__(self):
        """
        Iterator over lines in a file.
        """
        return self
    
    def close(self):
        """
        Close the file.
        """
        self.fileReader.close()
        pass

    def get_vector(self,line,labels,fids):
        """
        Create a vector from the line.
        """
        v = FAST_VECTOR()
        p = line.strip().split(ATTRIBUTE_DELIMITTER)
        # first element can be the label.
        startPos = 0
        if p[0].find(VALUE_DELIMITTER) == -1:
        	
            # This is the label.
            # In binary classification it has to be either 0 or 1.
            # In regression it has to be a float.
            # In multi class classification it has to be an integer
            # from 1 to N, where N is the number of classes.
            # It can also be a string for muti-class classification
            # indicating the class label.
            
            try:
            	# if the following line is successful, then we have
            	# a numeric label.
            	lbl = float(p[0])
            	# if the following condition is true, then that numeric
            	# value is an integer.
            	if int(lbl) == lbl:
            		lbl = int(lbl)
            except ValueError:
            	# The label is not numeric. It is a charachter string.
            	lbl = p[0].strip()
            
            # Set the value of the label.
            v.label = lbl
            labels[v.label] = 1
            startPos = 1
        for ele in p[startPos:]:
            fele = ele.split(VALUE_DELIMITTER)
            featvalStr = fele[-1]
            featidStr = VALUE_DELIMITTER.join(fele[:-1])
            if featidStr.isdigit():
                fid = int(featidStr)
            else:
                fid = featidStr.strip()
            fval = float(featvalStr)
            fids[fid] = 1
            v.add(fid,fval)
        return v
    
    def reset(self):
        """
        Move the cursor to the begining of the file.
        """
        self.fileReader.seek(0)
        pass

    def openFile(self,fname):
        """ 
        Detects the format of the file and opens it.
        Returns a handle to the file.
        """
        #detect the format of the file using the extension and open it.
        fileExtension = fname.strip().lower().split('.')[-1]
        if fileExtension == "gz":
            from gzip import GzipFile
            fileReader = GzipFile(fname,'r')
        elif fileExtension == "bz2":
            from bz2 import BZ2File
            fileReader = BZ2File(fname,'r')
        else:
            fileReader = open(fname,'r')
        return fileReader
    pass


class SEQUENTIAL_FILE_WRITER:

    """
    Opens a file to write with a specific format.
    This class has the functionality to write a feature
    vector to a file with labels.
    """

    def __init__(self,fname,format=None):
        self.format = format
        if format == "gz":
            from gzip import GzipFile
            self.file = GzipFile(fname,'w')
        elif format == "bz2":
            from bz2 import BZ2File
            self.file = BZ2File(fname,'w')
        elif format == "STDOUT":
            self.file = sys.stdout
        else:
            self.file = open(fname,'w')        
        pass

    def writeVector(self,v,WriteLabel=True):
        """
        Writes a vector with label (if specified) to the file.
        v = {'label':int,'attrib':{}}
        """
        label = v.label
        if (label is not None) and WriteLabel:
            self.file.write('%s ' % str(label))
        for fid in sorted(v.fvals.keys()):
            fval = v.fvals[fid]
            if fval == 0:
                if not DONT_WRITE_ZEROS:
                    self.file.write('%d:%s ' % (fid,str(fval)))
            else:
                self.file.write('%d:%s ' % (fid,str(fval)))
                
        self.file.write('\n')
        pass

    def writeLine(self,line):
        """
        Writes a line of text into file.
        """
        self.file.write(line)
        pass

    def close(self):
        """
        Closes the file.
        """
        # Although we can call close() on sys.stdout as well
        # this does not make sense!
        if self.format is not "STDOUT":
            self.file.close()
        pass
    pass        

class TEST_VECT_SCALE:

    """
    This class loads the range file produced by the scale_train_data
    method and use it to scale test data. We will load the range file
    to memory so that we do not read the range file each time when
    we encounter a new test instance.
    """

    def __init__(self,range_fname):
        """
        Open the range file and read scaling parameters.
        """
        rangeFile = open(range_fname)
        self.xscale = int(rangeFile.readline().strip().split('=')[1])
        self.yscale = int(rangeFile.readline().strip().split('=')[1])
        self.upper = float(rangeFile.readline().strip().split('=')[1])
        self.lower = float(rangeFile.readline().strip().split('=')[1])
        self.fmin = {}
        self.fmax = {}
        line = rangeFile.readline()
        while(line):
            if not line.startswith('#'):
                p = line.strip().split(',')
                fid = int(p[0])
                minVal = float(p[1])
                maxVal = float(p[2])
                self.fmin[fid] = minVal
                self.fmax[fid] = maxVal
            line = rangeFile.readline()
        rangeFile.close()
        pass

    def scale(self,v):
        """
        Scales the vector v using the range information.
        """
        # perform xscaling first.
        if self.xscale:
            # sys.stderr.write("Performing xscaling\n")
            for (fid,fval) in v:
                if (fid not in self.fmax) or (fid not in self.fmin):
                    v[fid] = 0
                else:
                    width = float(self.fmax[fid]-self.fmin[fid])
                    if width == 0.:
                        scaledWidth = 1.
                    else:
                        scaledWidth = float(v[fid]-self.fmin[fid])/width
                    v[fid] = scaledWidth*(self.upper-self.lower)+self.lower
        # perform yscaling if specified.
        if self.yscale:
            # sys.stderr.write("Perform yscaling\n")
            sqdTot = 0
            for (fid,fval) in v:
                sqdTot += fval*fval
            L2 = math.sqrt(sqdTot)
            for (fid,fval) in v:
                v[fid] = fval/L2
        pass
    pass


class DISPLAY:

    """
    Contains functionality to write both to the terminal and to a log file
    simultaneously.
    """

    def __init__(self,_LOG_FNAME=None,silent=False):
        self.LOG_FNAME = _LOG_FNAME
        self.silent = silent
        if _LOG_FNAME:
            self.LOG_FILE = open(_LOG_FNAME,'w')
        pass

    def write(self,txt=''):
        """
        Write text to terminal and log file.
        """
        if not self.silent:
            print txt
        if self.LOG_FNAME:
            self.LOG_FILE.write('%s\n' % unicode(txt,'utf-8','ignore'))        
        pass

    def close(self):
        """
        close the log file.
        """
        if self.LOG_FNAME:
            self.LOG_FILE.close()
        pass
    pass


def scale_train_data(train_fname,
                     scale_fname,
                     range_fname,
                     x_scale=True,
                     y_scale=False,
                     upper=1.,
                     lower=0.):
    """
    Scale the training instances in train_fname and write to
    scale_fname. x_scale will scale each feature into range [lower,upper].
    y_scale will normalize each feature vector such that the length
    (L2 norm) is one. It is possible to set x_scale only, y_scale only
    or both x and y scales. Usually it is better not to perform
    y_scaling because it will reduce the difference between sparse and
    dense instances. x_scaling is oftern desirable because features
    with high absolute values will not get correctly tuned during
    logistic regression because we truncate the exponentials to
    prevent overflow. Usually [0,1] range is sufficient. Therefore,
    it is recommended that you use this function with the default
    values unless in cases where you really want to perform a different
    kind of scaling for your particular application.
    range_file saves the minimum and maximum of each feature seen
    during training so that we can use that information to normalize
    the test data. Moreover, it retains information regarding what
    normalizations and ranges were used when scaling training data
    so that we can perform the exact scaling for the test data.
    For features that do not appear in train data but only in test_data
    we cannot scale them set their values to zero.
    (Note that this can be problematic when performing y_scaling
    because the precense of a large feature value that only appear
    in test data can severly affect the values of other features
    in the test instance that are x_scaled properly. This is another
    reason for not selecting y_scaling. A workaround this problem of
    y_scaling you can append all your test instances without their labels
    to the train_file and then perform scaling on this extended train file
    to obtain a range file that can be used during testing.)
    The scaling is performed in such a way that we only process one
    line of a file at a time. This enables us to process feature vector
    files with large numbers of training instances.
    """
    if upper <= lower:
        raise "Invalid upper and lower ranges specified!"
    fmin = {} # store the minimum value for each feature.
    fmax = {} # store the maximum value for each feature.
    F = SEQUENTIAL_FILE_READER(train_fname)
    for mv in F:
        v = mv['vect']
        for (fid,fval) in v:
            fmin.setdefault(fid,fval)
            fmax.setdefault(fid,fval)
            if fval > fmax[fid]:
                fmax[fid] = fval
            if fval < fmin[fid]:
                fmin[fid] = fval
                
    # write the ranges to range file.
    rangeFile = open(range_fname,'w')
    rangeFile.write("x_scale=%d\n" % x_scale)
    rangeFile.write("y_scale=%d\n" % y_scale)
    rangeFile.write("upper=%f\n" % upper)
    rangeFile.write("lower=%f\n" % lower)
    rangeFile.write("#featureID,min,max\n")
    for fid in fmin:
        rangeFile.write("%d,%s,%s\n" % (fid,str(fmin[fid]),str(fmax[fid])))
    rangeFile.close()    
    # perform scaling on train file.
    F.reset() # move the cursor to the beginning of the file.
    # guess the scale file format.
    format = get_file_format(scale_fname)
    scaleFile = SEQUENTIAL_FILE_WRITER(scale_fname,format)
    S = TEST_VECT_SCALE(range_fname)
    for mv in F:
        v = mv['vect']
        S.scale(v)
        label = mv['label']
        scaleFile.writeVector(v, WriteLabel=True)
    scaleFile.close()                             
    F.close()
    pass


def scale_test_data(test_fname, range_fname, output_fname=None):
    """
    Scale the test instances using the scaling parameners specified
    in the range file. If output file is not specified we will write
    to stdout.
    """
    # guess the output format. If no output fname is given write to stdout.
    if output_fname:
        format = get_file_format(output_fname)
    else:
        format = "STDOUT"
    output_file = SEQUENTIAL_FILE_WRITER(output_fname,format)
    S = TEST_VECT_SCALE(range_fname)
    test_file = SEQUENTIAL_FILE_READER(test_fname)
    for mv in test_file:
        v = mv['vect']
        label = v.label
        S.scale(v)
        if label is not None:
            output_file.writeVector(v, WriteLabel=True)
        else:
            v.label = None
            output_file.writeVector(v, WriteLabel=False)            
    test_file.close()
    output_file.close()
    pass


def get_file_format(fname):
    """
    Guesses the format of the file from its name.
    """
    fileExtension = fname.strip().lower().split('.')[-1]
    if fileExtension == 'gz':
        format = 'gz'
    elif fileExtension == 'bz2':
        format = 'bz2'
    else:
        format = None
    return format


def select_held_out_data(input_fname,held_fname,train_fname,N):
    """
    Select N number of instances uniformly at random from the
    input_fname and write them to held_fname. Remainder of the
    instances (not selected) are written to train_fname.
    The format of the held_fname and train_fname
    will be gussed by their file extensions.
    Supported types plain text (default), gz and bz2.
    """
    input_file = SEQUENTIAL_FILE_READER(input_fname)    
    # randomly select lines from the train_fname.
    total = input_file.countLines()
    print "Total lines in %s = %d" % (input_fname,total)
    selectedLines = []
    while True:
        n = random.randint(0,total-1)
        if n not in selectedLines:
            selectedLines.append(n)
        if len(selectedLines) == N:
            break
    # write the lines to the held out file.
    held_file = SEQUENTIAL_FILE_WRITER(held_fname,
                                       get_file_format(held_fname))
    train_file = SEQUENTIAL_FILE_WRITER(train_fname,
                                        get_file_format(train_fname))
    lineCount = 0
    line = input_file.readTextLine()
    trainInstances = 0
    while line:
        if lineCount in selectedLines:
            held_file.writeLine(line)
        else:
            train_file.writeLine(line)
            trainInstances += 1
        lineCount += 1
        line = input_file.readTextLine()
    # close all opened files.
    input_file.close()
    train_file.close()
    held_file.close()
    print "Selected %d instances and wrote to %s" % \
          (len(selectedLines),held_fname)
    print "Remaining %d instances were written to %s" % \
          (trainInstances, train_fname)
    pass

def feature_value_comparator(A,B):
    """
    A and B are tuples of the form (featid,featval).
    """
    if A[1] > B[1]:
        return -1
    return 1

def analyze_feature_file(feat_fname, output_fname):
    """
    Count the frequency of each feature in the feat_fname and
    write the total number of features and the sorted count
    to file output_fname.
    """
    feat_file = SEQUENTIAL_FILE_READER(feat_fname)
    feats = {}
    for fv in feat_file:
        for fid in fv["fids"]:
            feats[fid] = feats.get(fid, 0) + 1
    L = feats.items()
    L.sort(feature_value_comparator)
    outfile = open(output_fname, "w")
    outfile.write("Total_Features\t%d\n" % len(feats))
    for (fid,freq) in L:
        outfile.write("%s\t%d\n" % (str(fid),freq))
    feat_file.close()
    outfile.close()
    pass

def oneSidedUnderSampling(trainFileName, sampleFileName):
    """
    Performs one-sided undersampling for binary valued data in the trainFileName.
    Writes the sampled instances to the sampleFileName. We will first count the
    number of positive and negative instances and then determine the minority
    class. We will then select all instances from the minority class and one
    randomly selected instance from the majority class. Next, we will use 1-NN
    rule to classify the majority class. All misclassified instances will be 
    appeneded to the sample. The sample will then be written to the sampleFileName.
    """
    (labels, vects) = convertToArray(trainFileName)  
    assert(len(labels) == len(vects))
    n = len(labels)
    d = vects[0].size
    print "Dimensionality of feature vectors = %d" % d
    # determining the minority class.
    posVects = []
    negVects = []
    for i in range(0,n):
        label = labels[i]
        vect = vects[i]
        if label == 1:
            posVects.append((label, vect))
        else:
            negVects.append((label, vect))
    print "No. of positive instances = %d" % len(posVects)
    print "No. of negative instances = %d" % len(negVects)
    majorityClass = minorityClass = 0
    if len(posVects) > len(negVects):
        majorityClass = 1
        minorityClass = -1
        majorityInstances = posVects
        minorityInstances = negVects
    else:
        majorityClass = -1
        minorityClass = 1
        majorityInstances = negVects
        minorityInstances = posVects
    print "Majority class is %d" % majorityClass
    print "Minority class is %d" % minorityClass 
    # sample is appened to the list L of tuples of the format (label, vect).
    L = []
    
    # add all minority class instances to L and one instance from the majority class.
    L.extend(minorityInstances)
    centIndex = getCenter(majorityInstances)
    L.append(majorityInstances[centIndex])
    del majorityInstances[centIndex]
    print "Index of the majority class selected for the initial sample = %d" % centIndex
    #L.append(majorityInstances.pop())
   
    # Use 1-NN classification to classify majority class instances using L.
    count = 0
    stepSize = len(majorityInstances) / 10
    for (label, vect) in majorityInstances:
        count += 1
        if (count % stepSize) == 0:
            print "%d percent completed..." % int((10 * count) / stepSize )
        minDist = float('infinity')
        minLabel = 0
        for (lbl, v) in L:
            dist = numpy.linalg.norm(v - vect)
            if dist < minDist:
                minDist = dist
                minLabel = lbl
        #print minLabel, label
        if minLabel != label:
            L.append((label, vect))    
    # write the selected sample to file.
    print "No. of instances in the sample = %d" % len(L)  
    sampleFile = SEQUENTIAL_FILE_WRITER(sampleFileName)
    for (lbl, v) in L:
        vx = FAST_VECTOR()
        vx.createFromArray(lbl, v)
        sampleFile.writeVector(vx, WriteLabel=True)
    sampleFile.close()
    pass

def getCenter(L):
    """
    L is a list of tuples of the form (label, numpy.array).
    We will find the instance in this set which has the minimum
    Eucledian distance to the centroid of the set of vectors
    in L. This will be the instance selected from the majority
    class in the one-sided undersampling routine. We will return
    the index of this central instance in L.
    """
    # Compute the centroid of L.
    d = L[0][0].size
    n = len(L)
    cent = sum([v for (l, v) in L])
    cent = cent / float(n)
    # Find the closest instance in L to the centroid.
    minDist = float('infinity')
    minIndex = -1
    for (i, (l, v)) in enumerate(L):
        dist = numpy.linalg.norm(v - cent)
        if dist < minDist:
            minDist = dist
            minIndex = i
    return minIndex   

def convertLineToArray(line, d, lblType):
    """
    Creates an numpy array with the dimensionality  d from
    the feature vector specified by the text line.
    If there is a label or a regression value at the start,
    the return this value separately. 
    lblType is the data type of the labels.
    Feature ids must be integers because they are used as
    zero-based indexes in the array. 
    """
    x = numpy.zeros(d)
    p = line.strip().split(ATTRIBUTE_DELIMITTER)
    elementStart = 1
    label = None
    if p[0].find(VALUE_DELIMITTER) == -1:
        # There is a label in the line.
        if lblType == 'int':
            label = int(p[0])
        elif lblType == 'float':
            label = float(p[0])
        else:
            label = p[0].strip()
        elementStart = 2
    # Process the remainder of the line.
    for e in p[elementStart:]:
        fp = e.split(VALUE_DELIMITTER)
        if len(fp) != 2:
            sys.stderr.write("Invalid feature in line\n")
            sys.stderr.write("%s\n" % line)
            raise ValueError
        fid = int(fp[0]) - 1
        fval = float(fp[1])
        if fid < d:
            x[fid] = fval
    return (label, x)


def convertFileToArray(instanceFileName, lblType):
    """
    Call convertLineToArray function for each line in the
    instanceFileName. This method is much faster than the
    convertSEQFileToArray because this function does not
    need first load data into a SEQUENTIAL_FILE_READER.
    However, this function cannot handle compressed files.
    """
    labels = []
    vects = []
    dataFile = open(instanceFileName, "r")
    for line in dataFile:
        (l, x) = convertLineToArray(line.strip(), lblType)
        labels.append(l)
        vects.append(x)
    dataFile.close()
    return (labels, vects)


def convertSEQFileToArray(instanceFileName):
    """
    Given an instance file in the libsvm format, we will convert it into
    numpy arrays. We will first read the file using the SEQUENTIAL_FILE_READER
    and then perform the conversion. Note that after converting the arrays
    will not be sparse. Moreover, the feature id 1 will be mapped to the
    zeroth index in the array and all other features will be mapped to one
    less index in the array.
    """
    seqReader = SEQUENTIAL_FILE_READER(instanceFileName)
    instances = seqReader.read()
    # set the dimensionality d to the max feature id.
    d = max(instances["featIDs"].keys())
    n = len(instances["vects"])
    vectList = []
    labels = numpy.zeros(n, dtype=int)
    for (i, v) in enumerate(instances["vects"]):
        labels[i] = v.label
        x = numpy.zeros(d, dtype=float)
        for (fid, fval) in v:
            try:
                x[(fid - 1)] = fval
            except IndexError:
                sys.stderr.write("%d is not a valid feature index\n" % fid)
                sys.stderr.write("Total number of features allowed is %d\n" % d)
                raise IndexError 
        vectList.append(x)
    assert(len(vectList) == len(instances["vects"]))
    assert(len(labels) == len(instances["vects"]))
    return (labels, vectList)
        

def unit_test_analyze_feature_file():
    """
    unit test for analysing a feature file.
    """
    analyze_feature_file("../work/train_vects.all",
                         "../work/analyzed")
    pass

def unit_test_file_reader():
    """
    unit test for the sequential file reader.
    """
    #fname = "../work/rcv1_train.binary.bz2"
    #fname = "../work/rcv1_test.binary.bz2"
    fname = "../work/small.train.gz"
    # checking the read()method
    F = SEQUENTIAL_FILE_READER(fname)
    F.read()
    F.close()
    # checking the readLine() method.
    F = SEQUENTIAL_FILE_READER(fname)
    for v in F:
        print v['label']
    F.close()
    pass

def unit_test_file_writer():
    """
    unit test for writing files.
    """
    fname = "../work/small.train.gz"
    wfname = "../work/testwrite.bz2"
    F = SEQUENTIAL_FILE_READER(fname)
    W = SEQUENTIAL_FILE_WRITER(wfname,'bz2')
    for v in F:
        W.writeVector(v['vect'])
    F.close()
    W.close()
    pass

def unit_test_line_counter():
    """
    Unit test for counting the number of lines in a file.
    """
    fname = "../work/small.train.gz"
    F = SEQUENTIAL_FILE_READER(fname)
    print F.countLines()
    F.close()
    pass

def unit_test_held_out():
    """
    Unit test for held out data selection.
    """
    N = 20
    train_fname = "../work/rcv1_test.binary.bz2"
    held_fname = "../work/held.bz2"
    select_held_out_data(train_fname,held_fname,N)
    pass

def unit_test_vect_scale():
    """
    Unit test for scaling vectors.
    """
    train_fname = "../work/scaletest"
    scale_fname = "../work/scaled"
    range_fname = "../work/range"
    scale_train_data(train_fname,scale_fname,range_fname,
                     x_scale=True,y_scale=False,upper=1,lower=0)
    pass

def unit_test_convert_to_array():
    """
    Unit test for converting to numpy arrays.
    """
    trainFileName = "../work/small.vects"
    convertToArray(trainFileName)
    pass

def unit_test_oneSidedUnderSampling():
    """
    Unit test for oneSidedUnderSampling routine.
    """
    trainFileName = "../work/small.vects"
    sampleFileName = "../work/sample.vects"
    oneSidedUnderSampling(trainFileName, sampleFileName)
    pass


if __name__ == "__main__":
    #unit_test_file_reader()
    #unit_test_file_writer()
    #unit_test_line_counter()
    #unit_test_held_out()
    #unit_test_vect_scale()
    #unit_test_analyze_feature_file()
    #unit_test_convert_to_array()
    unit_test_oneSidedUnderSampling()
    pass
