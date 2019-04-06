#! /usr/bin/python
#! -*- coding: utf-8 -*-

"""
Copied from Makefile.py to support validation domains.
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
                'kitchen':16746,
		'music':15000,      #validation domain.
		'video':15000}    #validation domain.
    assert((0 <= ratio) and (ratio <= 1))
    n = int(dataSize[domain] * ratio)
    return n

def createCoocMatrix(matrixFileName, trainDataPath,
                     sourceDomains, targetDomains,
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
    allDomains = targetDomains[:]
    allDomains.extend(sourceDomains)
    for domain in allDomains:
        print domain
        for label in ["positive", "negative", "unlabeled"]:
            fname = "%s/%s/%s.tagged" % (trainDataPath,domain,label)
            print fname,
            if domain in targetDomains:
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
            if domain in targetDomains:
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



def batch_K():
    """
    Experiment with different numbers of expansion candidates
    during train and test times using two validation domains.
    """
    kValues = [200, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    resFile = open("../work/batcj_k.csv", "w")
    trainDomains =  ["books", "electronics", "dvd", "kitchen"]
    validationDomains = ["music", "video"]
    resFile.write("TRAIN_K, TEST_K, music, video, overall\n")
    totalRows = 2000
    totalColumns = 40000
    thesaurusFileName = "../work/thesaurus"
    #processAll(trainDomains, validationDomains, totalRows, totalColumns, totalRows)
    for trainK in kValues:
	for testK in kValues:
	    print "TRAIN_K = %d, TEST_K = %d" % (trainK, testK)
	    results = classify.single_round(validationDomains, trainDomains,
					    thesaurusFileName, trainK, testK, 0, 800)
	    resFile.write("%d, %d, %f, %f, %f\n" % (trainK, testK, results["music"],
						    results["video"], results["overall"]))
	    resFile.flush()
	    print results
    resFile.close()    
    pass        
	  

def processAll(sourceDomains, targetDomains, rowidTh, colidTh, DIcount,
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
                         sourceDomains, targetDomains,
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
    pass



if __name__ == "__main__":
    batch_K()
    pass
