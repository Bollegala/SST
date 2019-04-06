"""
This program create an element co-occurrence matrix for a given source
domain. It will generate both lexical elements and sentiment features
(for both unigrams and bigrams). 

The following steps are conducted by this program.
----------------------------------------------------

1. Select lexical elements and sentiment features that occur more than a given
   threshold to put in the co-occurrence matrix.

2. Compute the co-occurrence matrix for the selected rows (lexical elements)
   and columns (lexical+sentiment elements).

"""

import re
import sys
import os
import subprocess
import math

def get_tokens(line):
    """
    Return a list of dictionaries, where each dictionary has keys
    lemma (lemmatized word), infl (inflections if any), and pos (the POS tag).
    The elements in the list are ordered according to their
    appearance in the sentence.
    """
    elements = line.strip().split()
    # first token is the face.
    tokens = []
    for e in elements:
        if e == "^_^":
            continue
        p = e.split("_")
        if len(p) != 2:
            continue
        word = p[0]
        if word.find("+") > 0:
            # inflection info available.
            g = word.split("+")
            lemma = g[0]
            infl = g[1]
        else:
            lemma = word
            infl = None
        pos = p[1]
        tokens.append({'lemma':lemma,
                  'infl':infl,
                  'pos':pos})
        pass
    return tokens


class MATRIX_BUILDER:

    def __init__(self,
                 BUILD_MODE=False,
                 rowid_fname=None,
                 colid_fname=None,
                 matrix_fname=None):
        # How many instances should be taken as training data.
        self.NO_TRAIN_INSTANCES = 800
        # If you want to only select the top few unlabeled reviews
        # then set the following parameter. If you want to use
        # all unlabeled data available set this to None
        self.UNLABELED_SIZE = None
        # Do not write elements with frequency less than this value.
        self.MIN_FREQ = 2
        self.matrix = {}
        self.rowids = {} # {id:, freq:}
        self.colids = {} # {id:, freq:}
        self.row_count = 1
        self.col_count = 1
        self.BUILD_MODE = BUILD_MODE
        self.rowid_fname = rowid_fname
        self.colid_fname = colid_fname
        self.matrix_fname = matrix_fname
        if self.BUILD_MODE:
            self.load_selected_features()
            # Record the frequency of each row element in documents.
            # This is used to compute domain independent features.
            self.rowDist = {}
            # A list of filenames in each domain.
            self.docs = {}
            pass
        self.load_stop_words("./stopWords.txt")
        pass

    def load_stop_words(self, stopwords_fname):
        """
        Read the list of stop words and store in a dictionary.
        """
        self.stopWords = []
        F = open(stopwords_fname)
        for line in F:
            self.stopWords.append(line.strip())
        F.close()
        pass

    def get_rating(self, label):
        """
        Instead of assigning an ordinal rating, we will assign a binary
        rating depending of whether the review is positive or negative.
        If the review is unlabeled, then we will return None.
        We will not generate any rating related features from unlabeled
        reviews.
        """
        if label is "positive":
            return "POS"
        elif label is "negative":
            return "NEG"
        elif label is "unlabeled":
            return None
        else:
            raise "Invalid label provided!", label
        pass

    def load_selected_features(self):
        """
        When BUILD_MODE is True we will read the selected features
        from the files and then create a matrix with those features.
        """
        # load the rowids to dictionary.
        rowFile = open(self.rowid_fname)
        for line in rowFile:
            p = line.strip().split('\t')
            self.rowids[p[0].strip()] = {'id':int(p[1]),
                                         'freq':int(p[2])}
        rowFile.close()
        # load the colids to dictionary.
        countRatingFeatures = 0
        countPOSFeatures = 0
        colFile = open(self.colid_fname)
        for line in colFile:
            p = line.strip().split('\t')
            featname = p[0].strip()
            self.colids[featname] = {'id':int(p[1]),
                                         'freq':int(p[2])}
            if self.is_rating_feature(featname):
                countRatingFeatures += 1
                #print "Rating Feature Found: ", featname
            if self.is_POS_feature(featname):
                #print "POS Feature Found: ", featname
                countPOSFeatures += 1
        colFile.close()
        print "Total no. of rating features found =", countRatingFeatures
        print "Total no. of POS features found =", countPOSFeatures
        pass

    def process_file(self, fname, label=None, category=None):
        """
        If BUILD_MODE is False then we will count the frequency of
        each feature and store the details in a dictionary. Later
        we will write this dictionary to a file and sort it.
        We will select the top frequent features. If BUILD_MODE is True
        then we will check whether a particular feature is in the
        set of selected features. If so we will append that feature
        to the matrix. The matrix will then be written to a disk file.
        """
        rating = self.get_rating(label)
        F = open(fname)
        line = F.readline()
        inReview = False
        rating = None
        count = 0
        while line:
            if line.startswith('^^ <?xml version="1.0"?>'):
                line = F.readline()
                continue
            if line.startswith("<review>"):
                inReview = True
                line = F.readline()
                continue
            if inReview and line.startswith("<rating>"):
                # uncomment to use ordinal ratings.
                #rating = float(F.readline().strip())
                rating = self.get_rating(label)
                line = F.readline() #skipping the </rating>
                continue
            if inReview and line.startswith("<Text>"):
                while line:
                    if line.startswith("</Text>"):
                        break
                    if len(line) > 1 and not line.startswith("<Text>"):
                        #print line
                        tokens = get_tokens(line.strip())
                        if tokens:
                            # generate features.
                            rating = self.get_rating(label)
                            selected_lemmas = self.generate_elements(tokens, rating, None)
                            if self.BUILD_MODE:
                                self.process_DIrows(selected_lemmas,
                                                    fname, count, category)
                    line = F.readline()                    
            if inReview and line.startswith("</review>"):
                inReview = False
                count += 1
            #print count, self.UNLABELED_SIZE
            if label == "unlabeled" and \
                   self.UNLABELED_SIZE is not None and count >= self.UNLABELED_SIZE:
                F.close()
                return count
            if label != "unlabeled" and count >= self.NO_TRAIN_INSTANCES:
                F.close()
                return count            
            line = F.readline()
        # write the final lines if we have not seen </review> at the end.
        if inReview:
            count += 1
        F.close()
        return count
        pass

    def process_DIrows(self, L, fname, count, domain):
        """
        For each row element, record the frequency of that element in each
        document. L is a list of row elements that occur in file fid which
        belongs to the domain. We use this dictionary to find domain independent
        row elements later. We append each document to the list of documents
        for that particular domain.
        """
        # Add the document to its domain.
        doc = "%s_%d" % (fname, count)
        self.docs[doc] = domain
        # Add the row elements.
        for e in L:
            if e not in self.rowids:
                continue
            if e not in self.rowDist:
                self.rowDist[e] = {}
            self.rowDist[e][doc] = 1 + self.rowDist[e].get(doc, 0)        
        pass

    def is_rating_feature(self, feature_name):
        """
        If there is any rating related information encoded in the
        features (i.e. POS or NEG) then return True.
        Otherwise return False.
        """
        if feature_name.find("POS") >= 0:
            return True
        elif feature_name.find("NEG") >= 0:
            return True
        return False

    def is_POS_feature(self, feature_name):
        """
        If any of the POS tags exist in the feature name then it
        considered to be a POS feature.
        """
        posFilter = re.compile("(JA)|(JB)|(JBR)|(JBT)|(JJ)|(JJR)|(JJT)|(JK)|(NN)|(NN1)|(NN1\$)|(NN2)|(NNJ)|(NNJ1)|(NNJ2)|(NNL)|(NNL1)|(NNL2)|(NNO)|(NNO1)|(NNO2)|(NNS)|(NNS1)|(NNS2)|(NNSA1)|(NNSA2)|(NNSB)|(NNSB1)|(NNSB2)|(NNT)|(NNT1)|(NNT2)|(NNU)|(NNU1)|(NNU2)|(NP)|(NP1)|(NP2)|(NPD1)|(NPD2)|(NPM1)|(NPM2)")
        return posFilter.search(feature_name)
        

    def generate_elements(self, tokens, rating, category=None):
        """
        Generate the rows and the columns of the matrix.
        """
        # To generate pos unigrams, pos bigrams, pos rating features set this to True.
        GENERATE_POS_FEATURES = False
        # only the words with following pos tags are considered.
        posFilter = re.compile("(JA)|(JB)|(JBR)|(JBT)|(JJ)|(JJR)|(JJT)|(JK)|(NN)|(NN1)|(NN1\$)|(NN2)|(NNJ)|(NNJ1)|(NNJ2)|(NNL)|(NNL1)|(NNL2)|(NNO)|(NNO1)|(NNO2)|(NNS)|(NNS1)|(NNS2)|(NNSA1)|(NNSA2)|(NNSB)|(NNSB1)|(NNSB2)|(NNT)|(NNT1)|(NNT2)|(NNU)|(NNU1)|(NNU2)|(NP)|(NP1)|(NP2)|(NPD1)|(NPD2)|(NPM1)|(NPM2)")
        selected_postags = []
        selected_lemmas = []
        # Adding unigrams.
        for token in tokens:
             pos = token["pos"]
             word = token["lemma"]
             word = word.lower()
             if posFilter.match(pos):
                 if word not in selected_lemmas:
                     selected_lemmas.append(word)
                     selected_postags.append(pos)
        # Adding bigrams.
        bigrams_lemmas = []
        bigrams_postags = []
        for i in range(len(selected_lemmas) - 1):
            bigram_lemma = "%s__%s" % (selected_lemmas[i], selected_lemmas[i+1])
            bigram_postag = "%s__%s" % (selected_postags[i], selected_postags[i+1])
            if bigram_lemma not in bigrams_lemmas:
                bigrams_lemmas.append(bigram_lemma)
                bigrams_postags.append(bigram_postag)
        selected_lemmas.extend(bigrams_lemmas)
        selected_postags.extend(bigrams_postags)
        # create row and column pairs.
        if self.BUILD_MODE:
            for (indFirst, first) in enumerate(selected_lemmas):
                if first in self.rowids:
                    for (indSecond, second) in enumerate(selected_lemmas):
                        if first != second:
                            # Add the lexical feature.
                            first_id = self.rowids[first]['id']
                            feat = second
                            if feat in self.colids:
                                feat_id = self.colids[feat]['id']
                                self.matrix.setdefault(first_id, {})
                                self.matrix[first_id][feat_id] = self.matrix[first_id].get(feat_id, 0) + 1
                            if GENERATE_POS_FEATURES:
                                # Add part-of-speech feature.
                                feat = selected_postags[indSecond]
                                if feat in self.colids:
                                    feat_id = self.colids[feat]['id']
                                    self.matrix.setdefault(first_id, {})
                                    self.matrix[first_id][feat_id] = self.matrix[first_id].get(feat_id, 0) + 1
                            # Add the rating feature.
                            if rating == "POS" or rating == "NEG":
                                feat = "%s__%s" % (second,rating)
                                if feat in self.colids:
                                    feat_id = self.colids[feat]['id']
                                    self.matrix.setdefault(first_id, {})
                                    self.matrix[first_id][feat_id] = self.matrix[first_id].get(feat_id, 0) + 1
                                if GENERATE_POS_FEATURES:
                                    # Add part-of-speech and rating feature.
                                    feat = "%s__%s" % (selected_postags[indSecond], rating)
                                    if feat in self.colids:
                                        feat_id = self.colids[feat]['id']
                                        self.matrix.setdefault(first_id, {})
                                        self.matrix[first_id][feat_id] = self.matrix[first_id].get(feat_id, 0) + 1
                            # Add domain as a feature.
                            if category:
                                feat = "CATEGORY__%s" % category
                                if feat in self.colids:
                                    feat_id = self.colids[feat]['id']
                                    self.matrix.setdefault(first_id, {})
                                    self.matrix[first_id][feat_id] = self.matrix[first_id].get(feat_id, 0) + 1
            pass
        else:
            for (indFirst, first) in enumerate(selected_lemmas):
                # if the row does not exist then add it.
                if first in self.rowids:
                    self.rowids[first]['freq'] += 1
                else:
                    self.rowids[first] = {'id':self.row_count,
                                          'freq':1}
                    self.row_count += 1
                # create columns.
                for (indSecond, second) in enumerate(selected_lemmas):
                    if first != second:
                        feat = second
                        # Add lexical feature.
                        if feat in self.colids:
                            self.colids[feat]['freq'] += 1
                        else:
                            self.colids[feat] = {'id':self.col_count,
                                                 'freq':1}
                            self.col_count += 1
                        # Add part-of-speech feature.
                        if GENERATE_POS_FEATURES:
                            feat = selected_postags[indSecond]
                            if feat in self.colids:
                                self.colids[feat]['freq'] += 1
                            else:
                                self.colids[feat] = {'id':self.col_count,
                                                     'freq':1}
                            self.col_count += 1
                        # Add rating feature.
                        if rating == "POS" or rating == "NEG":
                            feat = "%s__%s" % (second,rating)
                            if feat in self.colids:
                                self.colids[feat]['freq'] += 1
                            else:
                                self.colids[feat] = {'id':self.col_count,
                                                     'freq':1}
                                self.col_count += 1
                            if GENERATE_POS_FEATURES:
                                # Add part-of-speech and rating feature.
                                feat = "%s__%s" % (selected_postags[indSecond], rating)
                                if feat in self.colids:
                                    self.colids[feat]['freq'] += 1
                                else:
                                    self.colids[feat] = {'id':self.col_count,
                                                         'freq':1}
                                self.col_count += 1
                        # Add domain as a feature.
                        if category:
                            feat = "CATEGORY_%s" % category
                            if feat in self.colids:
                                self.colids[feat]['freq'] += 1
                            else:
                                self.colids[feat] = {'id':self.col_count,
                                                     'freq':1}
                                self.col_count += 1
                            
        return selected_lemmas

    def write(self):
        """
        If this is the BUILD_MODE then we will write the matrix.
        Otherwise we will write the row and column ids.
        """
        if self.BUILD_MODE:
            self.write_matrix(self.matrix_fname)
        else:
            self.write_feature_index(self.rowid_fname,
                                     self.colid_fname)
        pass

    def writeDI(self, domains, DIfname, count):
        """
        Write the top ranked domain independent row elements
        to the file DIfname.
        """
        print "Selecting domain independent features..."
        # mutual information between a domain and a feature.
        MI = {}
        for domain in domains:
            MI[domain] = {}
        # Total frequency of each feature.
        featTot = {}
        # Total frequency of each document.
        docTot = {}
        # sum freq(x,d).
        N = 0
        # compute totals.
        for feat in self.rowDist:
            for doc in self.rowDist[feat]:
                val =  self.rowDist[feat][doc]
                featTot[feat] = featTot.get(feat, 0) + val
                docTot[doc] = docTot.get(doc, 0) + val
                N += val
        # compute MI.
        for feat in self.rowDist:
            for doc in self.rowDist[feat]:
                domain = self.docs[doc]
                pxd = float(self.rowDist[feat][doc]) / float(N)
                px = float(featTot[feat]) / float(N)
                pd = float(docTot[doc]) / float(N)
                val = pxd * math.log((pxd / (px * pd)), 2)
                MI[domain][feat] = MI[domain].get(feat, 0) + val
        # sort each feature according to the total MI with all domains.
        totMI = {}
        for domain in domains:
            for feat in MI[domain]:
                totMI[feat] = totMI.get(feat, 0) + MI[domain][feat]
        L = totMI.items()
        L.sort(self.comparator)
        # write the smallest I(x,Dall) features.
        F = open(DIfname, "w")
        for (feat, score) in L[:count]:
            F.write("%s\t%d\t%f\n" % (feat, self.rowids[feat]['id'], score))
        F.close()        
        pass

    def comparator(self, A, B):
        if A[1] > B[1]:
            return -1
        return 1            
    
    def write_matrix(self, fname):
        """
        Write the matrix to fname.
        rowid colid:freq colid:freq ...
        """
        F = open(fname, "w")
        n = 0
        for rowid in sorted(self.matrix.iterkeys()):
            F.write("%d " % rowid)
            for colid in sorted(self.matrix[rowid].iterkeys()):
                val = self.matrix[rowid][colid]
                if val >= self.MIN_FREQ:
                    F.write("%d:%d " % (colid, val))
                    n += 1
            F.write("\n")
        F.close()
        m = len(self.rowids) * len(self.colids)
        density = (100 * float(n)) / float(m)
        print "Density (percentage)= %f" % density
        print "No. of rows =", len(self.rowids)
        print "No. of columns =", len(self.colids)
        print "Potential elements in the matrix (rows * colds) =", m
        print "Actual non-zero elements in the matrix =", n
        pass
    
    def write_feature_index(self, row_fname, column_fname):
        """
        Write the feature stats to fname.
        featureName\tid\tfreq\n
        """
        F = open(row_fname, "w")
        for row in self.rowids:
            F.write("%s\t%d\t%d\n" % (row,
                                      self.rowids[row]['id'],
                                      self.rowids[row]['freq']))
        F.close()
        G = open(column_fname, "w")
        for col in self.colids:
            G.write("%s\t%d\t%d\n" % (col,
                                      self.colids[col]['id'],
                                      self.colids[col]['freq']))
        G.close()
        print "Total no. of row elements before selection:", len(self.rowids)
        print "Total no. of column elements before selection:", len(self.colids)
        pass

    pass

def select_features_count(fname, count):
    """
    Read the fname, sort it and check the frequency of each feature.
    We will select the most frequent count no. of features 
    and will write to a file, fname.selected.
    """
    print "Count-based feature selection..."
    print fname, count,
    subprocess.call("sort -n -r -k 3 %s > %s.sorted" % (fname, fname), shell=True)
    F = open("%s.sorted" % fname)
    G = open("%s.selected" % fname, "w")
    n = 0
    cutoff = -1
    for line in F:
        p = line.strip().split()
        if len(p) != 3:
            print line
            continue
        freq = int(p[2])
        if n < count:
            G.write("%s" % line)
            n += 1
        if n == count and cutoff == -1:
            cutoff = freq
    F.close()
    G.close()
    print "Selected features =", n
    print "Cut off frequency =", cutoff
    pass


def select_features_threshold(fname, threshold):
    """
    Read the fname, sort and check the frequency of each feature.
    We will select the features with frequency
    greater than the threshold and will write to a file, fname.selected
    """
    print "Frequency threshold-based feature selection..."
    print fname, threshold,
    subprocess.call("sort -n -r -k 3 %s > %s.sorted" % (fname, fname), shell=True)
    F = open("%s.sorted" % fname)
    G = open("%s.selected" % fname, "w")
    n = 0
    for line in F:
        p = line.strip().split()
        if len(p) != 3:
            print line
            continue
        freq = int(p[2])
        if freq >= threshold:
            G.write("%s" % line)
            n += 1
    F.close()
    G.close()
    print n
    pass


def process_all(BUILD_MODE):
    base_path = "../data/train_data"
    matrix_fname = "../work/matrix"
    if BUILD_MODE:
        rowid_fname = "../work/rowids.selected"
        colid_fname = "../work/colids.selected"
    else:
        rowid_fname = "../work/rowids"
        colid_fname = "../work/colids"        

    test_categories = ["kitchen","dvd","electronics","books"]
    validation_categories = ["music", "video"]
    all_categories = ["apparel", "sports_and_outdoors","magazines","baby",
                      "toys_and_games", "camera_and_photo",
                      "health_and_personal_care",
                      "outdoor_living","software"]
    unbalanced = ["computer_and_video_games", "beauty", "gourmet_food",
                  "grocery","jewelry_and_watches", "automotive",
                  "cell_phones_and_service"]

    # consider all train data from this category as unlabeled!.
    selected_category = None

    M = MATRIX_BUILDER(BUILD_MODE,
                       rowid_fname,
                       colid_fname,
                       matrix_fname)
    
    for category in test_categories:
        print category
        for label in ["positive", "negative", "unlabeled"]:
            fname = "%s/%s/%s.tagged" % (base_path,category,label)
            print fname,
            if category == selected_category:
                print M.process_file(fname, "unlabeled", category)
            else:
                print M.process_file(fname, label, category)
    M.write()
    pass


if __name__ == "__main__":
    process_all(False) # selecting elements.
    select_features_threshold("../work/rowids", 5)
    select_features_threshold("../work/colids", 20)
    process_all(True) # building the matrix.
    pass
            
