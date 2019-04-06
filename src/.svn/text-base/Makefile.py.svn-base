#! /usr/bin/python
#! -*- coding: utf-8 -*-

"""
This is the overall make file for the senticross project.
It can do the following:
 1) select row elements.
 2) select column elements. 
 3) create the frequency matrix.
 4) select domain independent row elements.
 5) weight the elements in the co-occurrence matrix using PMI.
 6) compute the similarity between domain independent row elements and all row elements.
 7) create a thesaurus between domain independent row elements and all row elements.
 8) Use the thesaurus to compute projected feature vectors.
 9) Append original feature vectors with the projected feature vectors.
 10) Tune the combination parameter lambda.
 11) train on multiple/single source domains and test on a single target domain.
"""

import create_matrix
import MLIB.utils.dmatrix
import compute_distribution
import create_thesaurus
import classify

import sys
import getopt

def getUnlabaledDataSize(domain, ratio):
    """
    Given the domain name and the ratio for the unlabaled data,
    returns the no. of unlabaled instances.
    """
    dataSize = {'books':5947,
                'electronics':34377,
                'dvd':34377,
                'kitchen':16746}
    assert((0 <= ratio) and (ratio <= 1))
    n = int(dataSize[domain] * ratio)
    return n

def createCoocMatrix(matrixFileName, trainDataPath,
                     sourceDomains, targetDomain,
                     rowidFileName, colidFileName,
                     rowidTh, colidTh, matrixElementFreqTh,
                     DIelementsFileName, DIcount,
                     SOURCE_LABELED_INSTANCES,
                     SOURCE_UNLABELED_INSTANCES,
                     TARGET_UNLABELED_INSTANCES):
    # generating the candidate rows and columns.
    MGen = create_matrix.MATRIX_BUILDER(False,
                                        rowidFileName,
                                        colidFileName,
                                        matrixFileName)
    allDomains = [targetDomain]
    allDomains.extend(sourceDomains)
    for domain in allDomains:
        print domain
        for label in ["positive", "negative", "unlabeled"]:
            fname = "%s/%s/%s.tagged" % (trainDataPath,domain,label)
            print fname,
            if domain == targetDomain:
                if TARGET_UNLABELED_INSTANCES >= 0:
                    MGen.UNLABELED_SIZE = getUnlabaledDataSize(domain,
                                                               TARGET_UNLABELED_INSTANCES)
                print MGen.process_file(fname, "unlabeled", domain)
            else:
                if SOURCE_UNLABELED_INSTANCES >= 0:
                    MGen.UNLABELED_SIZE = getUnlabaledDataSize(domain,
                                                               SOURCE_UNLABELED_INSTANCES)
                if SOURCE_LABELED_INSTANCES >= 0:
                    MGen.NO_TRAIN_INSTANCES = SOURCE_LABELED_INSTANCES
                print MGen.process_file(fname, label, domain)
    MGen.write()
    # selecting rowids.
    create_matrix.select_features_count(rowidFileName, rowidTh)
    # selecting colids.
    create_matrix.select_features_count(colidFileName, colidTh)
    # create the matrix.
    MBuild = create_matrix.MATRIX_BUILDER(True,
                                          "%s.selected" % rowidFileName,
                                          "%s.selected" % colidFileName,
                                          matrixFileName)
    MBuild.MIN_FREQ = matrixElementFreqTh
    for domain in allDomains:
        print domain
        for label in ["positive", "negative", "unlabeled"]:
            fname = "%s/%s/%s.tagged" % (trainDataPath,domain,label)
            print fname,
            if domain == targetDomain:
                if TARGET_UNLABELED_INSTANCES >= 0:
                    MGen.UNLABELED_SIZE = getUnlabaledDataSize(domain,
                                                               TARGET_UNLABELED_INSTANCES)
                print MBuild.process_file(fname, "unlabeled", domain)
            else:
                if SOURCE_UNLABELED_INSTANCES >= 0:
                    MGen.UNLABELED_SIZE = getUnlabaledDataSize(domain,
                                                               SOURCE_UNLABELED_INSTANCES)
                if SOURCE_LABELED_INSTANCES >= 0:
                    MGen.NO_TRAIN_INSTANCES = SOURCE_LABELED_INSTANCES
                print MBuild.process_file(fname, label, domain)
    MBuild.write()
    MBuild.writeDI(allDomains, DIelementsFileName, DIcount)
    pass

def loadDIElements(DIelementsFileName):
    """
    Load the list of domain independent row elements.
    """
    print "Loading domain independent elements from %s" % DIelementsFileName,
    F = open(DIelementsFileName)
    DIelements = []
    for line in F:
        ele = int(line.strip().split('\t')[1])
        DIelements.append(ele)
    print "...Done"
    return DIelements


def debug_TKDE():
	targetDomain = "kitchen"
	domains = ["books", "electronics", "dvd", "kitchen"]
	sourceDomains = domains[:]
	sourceDomains.remove(targetDomain)
	acc = processAll(sourceDomains, targetDomain,
			 20000, 80000, 1000,
	                 SOURCE_LABELED_INSTANCES=800)
	pass


def batch_thesaurusSize():
    """
    Increase the number of base entries in the thesaurus and
    evaluate the classification accuracy over the four domains.
    """
    sizes = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    colidTh = 80000
    resFile = open("../work/batchThesaurusSize.csv", "w")
    domains = ["books", "electronics", "dvd", "kitchen"]
    resFile.write("DIsize, target, accuracy\n")
    for rowidTh in sizes:
	for target in domains:
	    sources = domains[:]
	    sources.remove(target)
	    res = processAll(sources, target, rowidTh, 80000, rowidTh,
			     SOURCE_LABELED_INSTANCES=800)
	    resFile.write("%d, %s, %f\n" % (rowidTh, target, res))
	    resFile.flush()
    resFile.close()			  
    pass                 
	  

def processAll(sourceDomains, targetDomain, rowidTh, colidTh, DIcount,
               SOURCE_LABELED_INSTANCES=None, SOURCE_UNLABELED_INSTANCES=None,
               TARGET_UNLABELED_INSTANCES=None):
    """
    The Main processing routine.
    """
    ##################### Files and Directories. ####################
    trainDataPath = "../data/train_data"
    matrixFileName = "../work/matrix"
    rowidFileName = "../work/rowids"
    colidFileName = "../work/colids"
    DIelementsFileName = "../work/DIrowids"
    distFileName = "../work/simdist"
    thesaurusFileName = "../work/thesaurus"
    ##################################################################

    ################### Parameters. ##################################
    # Only select row elements with this minimum frequency.
    #rowidTh = 20000

    # Only select column elements with this minimum frequency.
    #colidTh = 80000

    # When writing the matrix select elements with this co-occurrences.
    matrixElementFreqTh = 2

    # How manny training instances should we use?
    #TRAIN_INSTANCES = 800

    # The no. of domain independent rows selected.
    #DIcount = 15000
    ###################################################################
    
    
    # Generating all row and column elements.
    if True:
        createCoocMatrix(matrixFileName, trainDataPath,
                         sourceDomains, targetDomain,
                         rowidFileName, colidFileName,
                         rowidTh, colidTh, matrixElementFreqTh,
                         DIelementsFileName, DIcount,
                         SOURCE_LABELED_INSTANCES,
                         SOURCE_UNLABELED_INSTANCES,
                         TARGET_UNLABELED_INSTANCES)
        pass

    DIelements = loadDIElements(DIelementsFileName)
    # Compute pairwise similarity and construct the thesaurus.
    if True:
        M = MLIB.utils.dmatrix.DMATRIX(SPARSE=True)
        M.read_matrix(matrixFileName)
        pmiMatrix = M.get_PMI()
        compute_distribution.write_distribution(pmiMatrix,
                                                distFileName,
                                                DIelements)
        print "Generating the thesaurus..."
        create_thesaurus.create_thesaurus(distFileName,
                                          rowidFileName,
                                          thesaurusFileName)
        
        pass

    # Train and Test.
    if True:
        results = classify.single_round([targetDomain], sourceDomains,
                                        thesaurusFileName, 1000, 1000,
                                        0, SOURCE_LABELED_INSTANCES)
        print results
        return results["overall"]
        pass        
    
    pass

def commandLine():
    """
    Take user input from the terminal and return the results.
    """
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hs:t:")
    except getopt.GetoptError, err:
        print err
        usage()
        sys.exit(-1)
    # parsing arguments.
    sources = None
    target = None
    for opt, val in opts:
        if opt == "-s":
            sources = val.strip().split(',')
        if opt == "-t":
            target = val.strip()
    if sources and target:
        processAll(sources, target, SOURCE_LABELED_INSTANCES=800)
    else:
        usage()
    pass

def usage():
    """
    Help message.
    """
    print "Usage: python Makefile.py -s <sources> -t <target>"
    pass
    
def pairwiseEvaluation():
    """
    Compute all pairwise adaptations.
    """
    resfname = "../work/pairwise.csv"
    domains = ["books", "electronics", "dvd", "kitchen"]
    F = open(resfname, "w")
    F.write("source, target, accuracy\n")
    for targetDomain in domains:
        for sourceDomain in domains:
            if targetDomain == sourceDomain:
                continue
            acc = processAll([sourceDomain], targetDomain,
                             SOURCE_LABELED_INSTANCES=800)
            F.write("%s, %s, %f\n" % (sourceDomain, targetDomain, acc))
            F.flush()
    F.close()          
    pass

def threeOneEvaluation():
    """
    Use three source domains and one target domain.
    Does not balance the training dataset size.
    """
    resfname = "../work/paper_ACL.xls"
    domains = ["books", "electronics", "dvd", "kitchen"]
    F = open(resfname, "w")
    F.write("source, target, accuracy\n")
    for targetDomain in domains:
        sourceDomains = domains[:]
        sourceDomains.remove(targetDomain)
        acc = processAll(sourceDomains, targetDomain,
			 20000, 80000, 15000,
                         SOURCE_LABELED_INSTANCES=800)
        F.write("%s, %s, %f\n" % ("_".join(sourceDomains), targetDomain, acc))
        F.flush()
    F.close()
    pass

def twoOneEvaluation():
    """
    Use two source domains and one target domain.
    Does not balance the training dataset size.
    """
    resfname = "../work/2_1_unbalanced.xls"
    F = open(resfname, "w")
    F.write("source, target, accuracy\n")
    domains = ["books", "electronics", "dvd", "kitchen"]
    S = [["books", "electronics"], ["books", "dvd"], ["books", "kitchen"],
         ["electronics", "dvd"], ["electronics", "kitchen"], ["dvd", "kitchen"]]
    for sourceDomains in S:
        targetDomains = domains[:]
        for d in sourceDomains:
            targetDomains.remove(d)
        for targetDomain in targetDomains:
            acc = processAll(sourceDomains, targetDomain,
                             SOURCE_LABELED_INSTANCES=800)
            F.write("%s, %s, %f\n" % ("_".join(sourceDomains), targetDomain, acc))
            F.flush()
    F.close()            
    pass

def threeOneEvaluation_balanced():
    """
    Use three source domains and one target domain.
    Balances the training dataset size.
    """
    resfname = "../work/3_1_balanced.xls"
    domains = ["books", "electronics", "dvd", "kitchen"]
    F = open(resfname, "w")
    F.write("source, target, accuracy\n")
    for targetDomain in domains:
        sourceDomains = domains[:]
        sourceDomains.remove(targetDomain)
        acc = processAll(sourceDomains, targetDomain,
                         SOURCE_LABELED_INSTANCES=267)
        F.write("%s, %s, %f\n" % ("_".join(sourceDomains), targetDomain, acc))
        F.flush()
    F.close()
    pass

def twoOneEvaluation_balanced():
    """
    Use two source domains and one target domain.
    Balances the training dataset size.
    """
    resfname = "../work/2_1_balanced.xls"
    F = open(resfname, "w")
    F.write("source, target, accuracy\n")
    domains = ["books", "electronics", "dvd", "kitchen"]
    S = [["books", "electronics"], ["books", "dvd"], ["books", "kitchen"],
         ["electronics", "dvd"], ["electronics", "kitchen"], ["dvd", "kitchen"]]
    for sourceDomains in S:
        targetDomains = domains[:]
        for d in sourceDomains:
            targetDomains.remove(d)
        for targetDomain in targetDomains:
            acc = processAll(sourceDomains, targetDomain,
                             SOURCE_LABELED_INSTANCES=400)
            F.write("%s, %s, %f\n" % ("_".join(sourceDomains), targetDomain, acc))
            F.flush()
    F.close()            
    pass

def unlabeledDatasetSize_source():
    """
    Use different proportions of source/target domain unlabaled data
    and evaluate the overall classification accuracy.
    To experiment with,
    source domain unlabaled dataset size SET SOURCE_UNLABELED_INSTANCES = t
    target domain unlabaled dataset size SET TARGET_UNLABELED_INSTANCES = t
    both source and target SET t to above both.
    Unlabeled dataset sizes for the remaining domains are set to 1.
    """
    targetDomain = "dvd"
    domains = ["books", "electronics", "dvd", "kitchen"]
    fname = "../work/%s_src_unlabeled.csv" % targetDomain
    F = open(fname, "w")
    ratios = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    lblSize = 50
    for t in ratios:
        # 1 source.
        for d in domains:
            if d != targetDomain:
                acc = processAll([d], targetDomain,
                                 SOURCE_LABELED_INSTANCES=lblSize,
                                 SOURCE_UNLABELED_INSTANCES=t,
                                 TARGET_UNLABELED_INSTANCES=1)
                F.write("%f,%s,%f\n" % (t, d, acc))
                F.flush()
        # 2 sources.
        for d in domains:
            if d == targetDomain:
                continue
            c = domains[:]
            c.remove(targetDomain)
            c.remove(d)
            acc = processAll(c, targetDomain,
                             SOURCE_LABELED_INSTANCES=lblSize/2,
                             SOURCE_UNLABELED_INSTANCES=t,
                             TARGET_UNLABELED_INSTANCES=1)
            F.write("%f,%s,%f\n" % (t, "_".join(c), acc))
            F.flush()
        # 3 sources.
        c = domains[:]
        c.remove(targetDomain)
        acc = processAll(c, targetDomain,
                         SOURCE_LABELED_INSTANCES=lblSize/3,
                         SOURCE_UNLABELED_INSTANCES=t,
                         TARGET_UNLABELED_INSTANCES=1)
        F.write("%f,%s,%f\n" % (t, "_".join(c), acc))    
        F.flush()
    F.close()
    pass

def unlabeledDatasetSize_target():
    """
    Use different proportions of source/target domain unlabaled data
    and evaluate the overall classification accuracy.
    To experiment with,
    source domain unlabaled dataset size SET SOURCE_UNLABELED_INSTANCES = t
    target domain unlabaled dataset size SET TARGET_UNLABELED_INSTANCES = t
    both source and target SET t to above both.
    Unlabeled dataset sizes for the remaining domains are set to 1.
    """
    targetDomain = "dvd"
    domains = ["books", "electronics", "dvd", "kitchen"]
    fname = "../work/%s_tgt_unlabeled.csv" % targetDomain
    F = open(fname, "w")
    ratios = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    lblSize = 50
    for t in ratios:
        print "ratio =", t
        # 1 source.
        for d in domains:
            if d != targetDomain:
                acc = processAll([d], targetDomain,
                                 SOURCE_LABELED_INSTANCES=lblSize,
                                 SOURCE_UNLABELED_INSTANCES=1,
                                 TARGET_UNLABELED_INSTANCES=t)
                F.write("%f,%s,%f\n" % (t, d, acc))
                F.flush()
        # 2 sources.
        for d in domains:
            if d == targetDomain:
                continue
            c = domains[:]
            c.remove(targetDomain)
            c.remove(d)
            acc = processAll(c, targetDomain,
                             SOURCE_LABELED_INSTANCES=lblSize/2,
                             SOURCE_UNLABELED_INSTANCES=1,
                             TARGET_UNLABELED_INSTANCES=t)
            F.write("%f,%s,%f\n" % (t, "_".join(c), acc))
            F.flush()
        # 3 sources.
        c = domains[:]
        c.remove(targetDomain)
        acc = processAll(c, targetDomain,
                         SOURCE_LABELED_INSTANCES=lblSize/3,
                         SOURCE_UNLABELED_INSTANCES=1,
                         TARGET_UNLABELED_INSTANCES=t)
        F.write("%f,%s,%f\n" % (t, "_".join(c), acc))    
        F.flush()
    F.close()
    pass


def labeledDatasetSize():
    """
    Use different proportions of the source domain labeled data
    and evaluate the overall classification accuracy.
    """
    targetDomain = "dvd"
    domains = ["books", "electronics", "dvd", "kitchen"]
    fname = "../work/%s_src_labeled.csv" % targetDomain
    F = open(fname, "w")
    sizes = [0, 160, 320, 480, 640, 800]
    for n in sizes:
        # 1 source.
        for d in domains:
            if d != targetDomain:
                acc = processAll([d], targetDomain,
                                 SOURCE_LABELED_INSTANCES=n,
                                 SOURCE_UNLABELED_INSTANCES=1,
                                 TARGET_UNLABELED_INSTANCES=1)
                F.write("%d,%s,%f\n" % (n, d, acc))
                F.flush()
        # 2 sources.
        for d in domains:
            if d == targetDomain:
                continue
            c = domains[:]
            c.remove(targetDomain)
            c.remove(d)
            acc = processAll(c, targetDomain,
                             SOURCE_LABELED_INSTANCES=n,
                             SOURCE_UNLABELED_INSTANCES=1,
                             TARGET_UNLABELED_INSTANCES=1)
            F.write("%d,%s,%f\n" % (n, "_".join(c), acc))
            F.flush()
        # 3 sources.
        c = domains[:]
        c.remove(targetDomain)
        acc = processAll(c, targetDomain,
                         SOURCE_LABELED_INSTANCES=n,
                         SOURCE_UNLABELED_INSTANCES=1,
                         TARGET_UNLABELED_INSTANCES=1)
        F.write("%d,%s,%f\n" % (n, "_".join(c), acc))    
        F.flush()
    F.close()
    pass


if __name__ == "__main__":
	#commandLine()
	#debug_TKDE()
	threeOneEvaluation()
        #batch_thesaurusSize()
	pass
