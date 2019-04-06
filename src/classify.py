"""
This program reads in the created thesaurus and performs cross-domain
sentiment classification using it. 

"""

import sys
import re
import math
import subprocess
import string

class FEATURE_GENERATOR:

    def __init__(self):
        self.path = "../data/"
        # How many instances should be taken as training data.
        self.NO_TRAIN_INSTANCES = 800
        self.thesaurus = None # featstr -> [(featstr,sim),...]
        self.inverted_thesaurus = None # related -> [base,sim),...]
        # how many expansions were potentially possible?
        self.candidate_expansions = 0
        # how many expansions did we actually perform?
        self.actual_expansions = 0
        self.clear_stats()
        self.alpha = None
        self.TRAIN_K = None
        self.TEST_K = None
        self.sim = None
        pass

    def clear_stats(self):
        """
        Clear all feature expansion related statistics.
        """
        self.attempts = 0
        self.found_candidates = 0
        pass

    def get_stats(self):
        """
        Returns the feature expansion related statistics.
        """
        if self.attempts == 0:
            ratio = 0
        else:
            ratio = float(self.actual_expansions) / float(self.candidate_expansions)
        return {'Candidates':self.candidate_expansions,
                'Actuals':self.actual_expansions,
                'Ratio':ratio}

    def load_thesaurus(self, thesaurus_fname):
        """
        Load the thesaurus to the dictionary self.thesaurus.
        self.thesaurus[entry_feature_name] = [(featName,sim), ... ]
        We will keep the original names of the features and will not
        replace them with their ids.
        """
        if self.sim is None:
            print "Similarity threshold must be specified before loading the thesaurus!"
            sys.exit(-1)
        print "Loading the thesaurus ", thesaurus_fname, "...",
        self.thesaurus = {}
        F = open(thesaurus_fname)
        totalElements = 0
        for line in F:
            p = line.strip().split('\t')
            first = p[0].strip()
            first = first.lower()
            self.thesaurus[first] = []
            for e in p[1:]:
                vals = e.split(',')
                if len(vals) == 2:
                    feat = vals[0].strip()
                    feat = feat.lower()
                    sim = float(vals[1].strip())
                    if sim >= self.sim:
                        self.thesaurus[first].append((feat,sim))
                        totalElements += 1
        F.close()
        print "done."
        pass

    def generate_vectors(self, feat_fname, categories, MODE):
        """
        For the set of categories specified in the list categories,
        we will read the corresponding reviews and generate feature
        vectors. The feature vectors will then be written to the file
        feat_fname. If the MODE is TRAIN, then we will generate feature
        vectors from training data. If the MODE is set to TEST, then
        we will generate feature vectors from test data.
        Depending on the mode we will be reading data from different
        directories (train_data, test_data). Return the number of
        positive, negative and ignored instances.
        """
        total_vectors = 0
        count_positives = 0
        count_negatives = 0
        count_ignored = 0
        # Decide the directory that we should read data from.
        if MODE == "TRAIN":
            baseDir = "%strain_data" % self.path
            labels = ["positive", "negative"]
        elif MODE == "TEST":
            baseDir = "%stest_data" % self.path
            labels = ["positive", "negative"]
        else:
            raise "Invalid MODE@FEATURE_GENERATOR.generate_vectors ", MODE
        feat_file = open(feat_fname, "w")
        for category in categories:
            for label in labels:
                fname = "%s/%s/%s.tagged" % (baseDir,category,label)
                print fname
                if MODE == "TEST":
                    (pos,neg,ignored,feature_vectors) = self.process_file(fname,
                                                                          "TEST",
                                                                          category,
                                                                          label)
                else:
                    (pos,neg,ignored,feature_vectors) = self.process_file(fname,
                                                                          "TRAIN",
                                                                          category,
                                                                          label)
                    
                count = len(feature_vectors)
                print "positives = %d, negatives = %d, ignored = %d" % (pos,neg,ignored)
                total_vectors += count
                count_positives += pos
                count_negatives += neg
                count_ignored += ignored
                for (rating,fv) in feature_vectors:
                    ext_fv = self.extend_features(fv, MODE)
                    self.write_feature_vector(rating, ext_fv, feat_file)
        feat_file.close()
        return (count_positives, count_negatives, count_ignored)

    def get_tokens(self, line):
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

    def get_rating_from_label(self, label):
        """
        Set the rating using the label.
        """
        if label == "positive":
            return "positive"
        elif label == "negative":
            return "negative"
        elif label == "unlabeled":
            return None
        pass

   
    def get_rating_from_score(self, rateStr):
        """
        Compute the rating from the score.
        If score > positive, score < 3 negative, score = 3 ignore.
        """
        score = float(rateStr.strip())
        if score > 3:
            return "positive"
        elif score < 3:
            return "negative"
        else:
            return None
        pass        

    def process_file(self, fname, MODE, category=None, label=None):
        """
        Open the file fname, generate all the features and return
        as a list of feature vectors.
        """
        feature_vectors = [] #List of feature vectors.
        F = open(fname)
        line = F.readline()
        inReview = False
        rating = None
        count = 0
        tokens = []
        count_positives = 0
        count_negatives = 0
        count_ignored = 0
        while line:
            if line.startswith('^^ <?xml version="1.0"?>'):
                line = F.readline()
                continue
            if line.startswith("<review>"):
                inReview = True
                tokens = []
                line = F.readline()
                continue
            if inReview and line.startswith("<rating>"):
                # Do not uncomment the following line even if you are not
                # using get_rating_from_score because we must skip the
                # rating line.
                ratingStr = F.readline()
                # uncomment to use ordinal ratings.
                #rating = self.get_rating_from_score(ratingStr)
                rating = self.get_rating_from_label(label)
                line = F.readline() #skipping the </rating>
                continue
            if inReview and line.startswith("<Text>"):
                while line:
                    if line.startswith("</Text>"):
                        break
                    if len(line) > 1 and not line.startswith("<Text>"):
                        curTokens = self.get_tokens(line.strip())
                        if curTokens:
                            tokens.extend(curTokens)
                    line = F.readline()                    
            if inReview and line.startswith("</review>"):
                inReview = False
                # generate feature vector from tokens.
                # Do not use rating related features to avoid overfitting.
                fv = self.get_features(tokens, rating=None)
                if MODE == "TRAIN" and count > self.NO_TRAIN_INSTANCES:
                    break
                if rating == "positive":
                    feature_vectors.append(("positive", fv))
                    count_positives += 1
                elif rating == "negative":
                    feature_vectors.append(("negative", fv))
                    count_negatives += 1
                else:
                    count_ignored += 1
                tokens = []
                count += 1
            line = F.readline()
        # write the final lines if we have not seen </review> at the end.
        if inReview:
            count += 1
        F.close()
        return (count_positives, count_negatives, count_ignored, feature_vectors)

    def get_features(self, tokens, rating=None):
        return self.get_features_bigram(tokens, rating)
        #return self.get_features_unigram(tokens, rating)

    def get_features_unigram(self, tokens, rating=None):
        """
        Create a feature vector from the tokens and return it.
        """
        # only the words with following pos tags are considered.
        posFilter = re.compile("(JA)|(JB)|(JBR)|(JBT)|(JJ)|(JJR)|(JJT)|(JK)|(NN)|(NN1)|(NN1\$)|(NN2)|(NNJ)|(NNJ1)|(NNJ2)|(NNL)|(NNL1)|(NNL2)|(NNO)|(NNO1)|(NNO2)|(NNS)|(NNS1)|(NNS2)|(NNSA1)|(NNSA2)|(NNSB)|(NNSB1)|(NNSB2)|(NNT)|(NNT1)|(NNT2)|(NNU)|(NNU1)|(NNU2)|(NP)|(NP1)|(NP2)|(NPD1)|(NPD2)|(NPM1)|(NPM2)")
        fv = {}
        for token in tokens:
            pos = token["pos"]
            word = token["lemma"]
            word = word.lower()
            if posFilter.match(pos):
                fv[word] = fv.get(word, 0) + 1
                #fv[word] = 1.0 
                # Add the rating feature.
                if rating == "POS" or rating == "NEG":
                    feat = "%s__%s" % (word, rating)
                    fv[feat] = fv.get(word, 0) + 1
        return fv

    def get_features_bigram(self, tokens, rating=None):
        """
        Create a feature vector from the tokens and return it.
        """
        # only the words with following pos tags are considered.
        #posFilter = re.compile("(JA)|(JB)|(JBR)|(JBT)|(JJ)|(JJR)|(JJT)|(JK)|(NN)|(NN1)|(NN1\$)|(NN2)|(NNJ)|(NNJ1)|(NNJ2)|(NNL)|(NNL1)|(NNL2)|(NNO)|(NNO1)|(NNO2)|(NNS)|(NNS1)|(NNS2)|(NNSA1)|(NNSA2)|(NNSB)|(NNSB1)|(NNSB2)|(NNT)|(NNT1)|(NNT2)|(NNU)|(NNU1)|(NNU2)|(NP)|(NP1)|(NP2)|(NPD1)|(NPD2)|(NPM1)|(NPM2)")
        fv = {}
        posFilter = re.compile('.*')
        selected_lemmas = []
        selected_postags = []
        for token in tokens:
            pos = token["pos"]
            word = token["lemma"]
            word = word.lower()
            if posFilter.match(pos):
                if word not in selected_lemmas:
                    selected_lemmas.append(word)
                    selected_postags.append(pos)
        # generate bigram features.
        bigrams_lemmas = []
        bigrams_postags = []
        for i in range(len(selected_lemmas) - 1):
            bigram_lemma = "%s__%s" % (selected_lemmas[i], selected_lemmas[i+1])
            bigram_postag = "%s__%s" % (selected_postags[i], selected_postags[i+1])
            if bigram_lemma not in bigrams_lemmas:
                bigrams_lemmas.append(bigram_lemma)
                bigrams_postags.append(bigram_postag)        
        # add features to the feature vectors.
        featList = []
        featList.extend(selected_lemmas) # lexical unigrams.
        featList.extend(bigrams_lemmas) # lexical bigrams.
        #featList.extend(selected_postags) # POS unigrams.
        #featList.extend(bigrams_postags) # POS bigrams.
        for feat in featList:
            #fv[feat] = fv.get(feat, 0) + 1.0
            fv[feat] = 1.0
        return fv
                                
    def extend_features(self, fv, MODE):
        """
        Merge the different features that belong to the same cluster.
        MODE can be either TRAIN or TEST. Depending on the MODE we can
        apply different feature merging methods.
        """
        # if we are not using the thesaurus then uncomment the next line.
        #return self.NO_Thesaurus(fv)
        # add similar features from the thesaurus.
        #return self.extend_with_similar(fv, MODE)
        #return self.extend_combo(fv, MODE)
        return self.collapse_thesaurus(fv, MODE)
        pass
    
    def NO_Thesaurus(self, fv):
        """
        We do not use the thesaurus to merge features. This is the
        baseline feature merging method.
        """
        return fv

    def extend_with_similar(self, fv, MODE):
        """
        Find the most similar features for each feature in the
        original feature vector. Multiply the similarity score and
        the frequency and accumulate the feature values.
        """
        if MODE == "TRAIN":
            k = self.TRAIN_K
        elif MODE == "TEST":
            k = self.TEST_K
        else:
            raise "Invalid MODE\n"
        extd_fv = {}
        for fname in fv:
            extd_fv[fname] = extd_fv.get(fname,0) + fv[fname]
            related_feats = self.thesaurus.get(fname, [])
            self.attempts += 1
            if related_feats:
                self.found_candidates += 1
            for (rank,(rstr,sim)) in enumerate(related_feats[:k]):
                self.candidate_expansions += 1
                if sim > self.sim:
                    self.actual_expansions += 1
                    discount = sim 
                    extd_fv[rstr] = extd_fv.get(rstr,0) + (discount * fv[fname])
        return extd_fv

    def extend_combo(self, fv, MODE):
        """
        Find the most similar features for each feature in the
        original feature vector. Introduce new features of the format
        EXTD_ORIGINALWORD_EXTENDEDWORD. Set the value of the feature to 1.
        """
        if MODE == "TRAIN":
            k = self.TRAIN_K
        elif MODE == "TEST":
            k = self.TEST_K
        else:
            raise "Invalid MODE\n"
        extd_fv = {}
        for fname in fv:
            extd_fv[fname] = extd_fv.get(fname,0) + fv[fname]
            related_feats = self.thesaurus.get(fname, [])
            self.attempts += 1
            if related_feats:
                self.found_candidates += 1
                for (rank,(rstr,sim)) in enumerate(related_feats[:k]):
                    self.candidate_expansions += 1
                    if sim > self.sim:
                        self.actual_expansions += 1
                        feat = self.get_combo_featname(fname, rstr)
                        #extd_fv[feat] = extd_fv.get(feat, 0) + sim
                        extd_fv[feat] = sim
        return extd_fv

    def get_combo_featname(self, original, related):
        """
        Create the feature name for the extended features.
        Use a single canonical feature by sorting original
        and the related words alphabetically.
        """
        L = [original, related]
        L.sort(key=string.lower)
        # alphabetically sort original and related.
        feat = "EXTD=%s=%s" % (L[0], L[1])
        # generated EXTD_original_related.
        #feat = "EXTD=%s=%s" % (original, related)
        return feat

    def invert_thesaurus(self):
        """
        Reads in the loaded thesaurus and inverts it such that we can
        easily find the base entries of a particular word. This inverted
        thesaurus is used to find the lower dimensional base-entry
        representation of features for the sentiment classifier.
        """
        print "Inverting the thesaurus...",
        self.inverted_thesaurus = {}
        for base_entry in self.thesaurus:
            for (word,sim) in self.thesaurus[base_entry]:
                self.inverted_thesaurus.setdefault(word,{})[base_entry] = sim
        print "done."
        pass

    ## def collapse_thesaurus(self, fv, MODE):
    ##     """
    ##     (original code)
    ##     For all the features in the feature vector fv find the base entries
    ##     that correspond to the features. View the similarity score as a
    ##     recommendation score where each related word recommends its base
    ##     entries as features for the classifier.
    ##     We aggregate all the recommendation
    ##     scores of a base entry and then select the top K ranked base entries
    ##     (greater than a score beta) as the candidate features
    ##     for the classifier.
    ##     """
    ##     if self.inverted_thesaurus is None:
    ##         self.invert_thesaurus()
    ##     if MODE == "TRAIN":
    ##         k = self.TRAIN_K
    ##     elif MODE == "TEST":
    ##         k = self.TEST_K
    ##     else:
    ##         raise "Invalid MODE\n"
    ##     extd_fv = {}
    ##     bases = {}
    ##     for fname in fv:
    ##         extd_fv[fname] = extd_fv.get(fname,0) + fv[fname]
    ##         for base in self.inverted_thesaurus.get(fname,{}):
    ##             value = self.inverted_thesaurus[fname][base]
    ##             bases[base] = bases.get(base,0) + math.log(value)
    ##     base_list = bases.items()
    ##     base_list.sort(self.value_comparator)
    ##     # select base entries for expansion.
    ##     rank = 1
    ##     for (base,score)  in base_list[:k]:
    ##         feat ="BASE=%s" % base
    ##         extd_fv[feat] = 1.0 / float(rank)
    ##         #extd_fv[feat] = math.exp(score)
    ##         rank += 1
    ##     return extd_fv

    def collapse_thesaurus(self, fv, MODE):
        """
        For all the features in the feature vector fv find the base entries
        that correspond to the features. View the similarity score as a
        recommendation score where each related word recommends its base
        entries as features for the classifier.
        We aggregate all the recommendation
        scores of a base entry and then select the top K ranked base entries
        (greater than a score beta) as the candidate features
        for the classifier.
        """
        if self.inverted_thesaurus is None:
            self.invert_thesaurus()
        if MODE == "TRAIN":
            k = self.TRAIN_K
        elif MODE == "TEST":
            k = self.TEST_K
        else:
            raise "Invalid MODE\n"
        extd_fv = {}
        bases = {}
        for fname in fv:
            extd_fv[fname] = extd_fv.get(fname,0) + fv[fname]
            for base in self.inverted_thesaurus.get(fname,{}):
                value = self.inverted_thesaurus[fname][base] * fv[fname]
                bases[base] = bases.get(base,0) + value
        base_list = bases.items()
        base_list.sort(self.value_comparator)
        # select base entries for expansion.
        rank = 1
        #factor = self.projectionFactor(fv, bases, 1)
        for (base,score)  in base_list[:k]:
            feat ="BASE=%s" % base
            extd_fv[feat] =  (1.0 / float(rank))
            #extd_fv[feat] = score
            rank += 1
        return extd_fv

    def projectionFactor(self, fv, bases, alpha):
        """
        prjFact = (alpha * L1(fv)) / L1(base)
        """
        L1_fv = len(fv) # fv is binary.
        v = 0
        for i in range(0, len(bases)):
            v += 1.0 / float(i + 1)
        #print len(bases)
        if v == 0:
            return 1
        factor = (alpha * L1_fv) / v
        return factor

    def value_comparator(self, A, B):
        """
        Check the second item in tuples A and B.
        Sort in the descending order.
        """
        if A[1] > B[1]:
            return -1
        return 1
   
    def is_rating_feature(self, feature_name):
        """
        Given the name of a feature check whether it is a feature
        generated using rating information. Specifically such features
        have POS or NEG as substrings. These features must not be
        used for training or expansion during training because they
        introduce a set of highly confident features that do not
        occur during testing.
        """
        if feature_name.find("POS") != -1:
            return True
        elif feature_name.find("NEG") != -1:
            return True
        return False

    def write_feature_vector(self, label, fv, feat_file):
        """
        Write the feature vector to the feature file.
        Use feature names instead of feature ids.
        Classias can work on names as well as ids.
        Therefore, it is easy to interpret the features in the model.
        """
        if label == "positive":
            feat_file.write("1 ")
        elif label == "negative":
            feat_file.write("-1 ")
        else:
            print "Invalid Label", label
            sys.exit(-1)
        for feat in fv:
            # if the feat_name has colon then replace it with *
            #feat = re.sub(":", "*", feat)
            if not self.is_rating_feature(feat):
                feat_file.write("%s:%f " % (feat, fv[feat]))            
        feat_file.write("\n")
        pass
    pass


def main():
    cross_domain_evaluation()
    #analyze_model("../work/model")
    #show_expansions()
    #count_selected_rating_features("../work/colids.selected")
    #compute_averages()
    #batch_sim()
    #batch_k()
    #batch_train_size()
    #batch_instances()
    pass

def cross_domain_evaluation():
    """
    Create a large training vector file and a model using what ever possible.
    This model will be used during the test phase to compute cross domain
    sentiment classification.
    """        
    test_categories = ["kitchen","dvd","electronics","books"]
    #test_categories = ["kitchen"]
    validation_categories = ["music", "video"]
    all_categories = ["apparel", "sports_and_outdoors","magazines","baby",
                      "toys_and_games", "camera_and_photo",
                      "health_and_personal_care",
                      "outdoor_living","software"]
    unbalanced = ["computer_and_video_games", "beauty", "gourmet_food",
                  "grocery","jewelry_and_watches", "automotive",
                  "cell_phones_and_service"]
                  

    # parameters for generating vectors for training.
    TRAIN_K = 400
    TEST_K = 800
    sim = 0.45

    results = {'overall':0}
    #thesaurus_fname = "../work/kitchen.bigram.thes"
    for category in ["books"]:
        print category
        thesaurus_fname = "../work/%s.thes" % category
        print thesaurus_fname
        train_categories = []
        #train_categories.extend(all_categories)
        train_categories.extend(test_categories)
        #train_categories.extend(unbalanced)
        train_categories.remove(category)
        accuracy = single_round([category], train_categories,
                                thesaurus_fname,
                                TRAIN_K, TEST_K, sim)
        print category, accuracy['overall']
        results['overall'] += accuracy['overall']
        results[category] = accuracy[category]
    cross_domain_average = results['overall'] / float(len(test_categories))
    print "\n"
    for category in test_categories:
        print "%s\t%f" % (category, results[category])
    print "Overall\t", cross_domain_average
    pass

def show_expansions():
    """
    Find the top ranked expansion candidates for a given feature.
    """
    category = "books"
    thesaurus_fname = "../work/%s.thes" % category
    F = FEATURE_GENERATOR()
    # parameters for generating vectors for training.
    F.TRAIN_K = 400
    F.TEST_K = 800
    F.sim = 0
    F.load_thesaurus(thesaurus_fname)
    F.invert_thesaurus()
    while 1:
        print "Enter feature name:"
        fname = sys.stdin.readline().strip()
        bases = {}
        for base in F.inverted_thesaurus.get(fname,{}):
            value = F.inverted_thesaurus[fname][base]
            bases[base] = bases.get(base,0) + math.log(value)
        base_list = bases.items()
        base_list.sort(F.value_comparator)
        # select base entries for expansion.
        rank = 1
        print "Expansion Candidates for: %s (%d)" % (fname, len(bases)) 
        for (base,score)  in base_list[:10]:
            print "rank = %d\t score = %f\t %s" % (rank, score, base) 
            rank += 1
            pass
        pass
    pass

def batch_round(TRAIN_K, TEST_K, sim):
    """
    Peform a single round on using the parameters and evaluate on validation
    and test datasets. Return the accuracies in a hash.
    """
    validation_thesaurus = "../work/kitchen.bigram.thes"
    test_categories = ["kitchen","dvd","electronics","books"]
    validation_categories = ["music", "video"]
    all_categories = ["apparel", "sports_and_outdoors", "magazines","baby",
                      "toys_and_games", "camera_and_photo",
                      "health_and_personal_care",
                      "outdoor_living","software"]
    # validation.
    validation_accuracy = 0
    for category in validation_categories:
        train_categories = []
        train_categories.extend(all_categories)
        train_categories.extend(validation_categories)
        train_categories.remove(category)
        accuracy = single_round([category], train_categories,
                                validation_thesaurus,
                                TRAIN_K, TEST_K, sim)
        validation_accuracy += accuracy['overall']
    validation_accuracy = validation_accuracy / float(len(validation_categories))
    # test.
    test_accuracy = 0
    for category in test_categories:
        thesaurus_fname = "../work/%s.thes" % category
        train_categories = []
        train_categories.extend(all_categories)
        train_categories.extend(test_categories)
        train_categories.remove(category)
        accuracy = single_round([category], train_categories,
                                thesaurus_fname,
                                TRAIN_K, TEST_K, sim)
        test_accuracy += accuracy['overall']
    test_accuracy = test_accuracy / float(len(test_categories))
    return {'validation':validation_accuracy,
            'test':test_accuracy}                                                    
        

def single_round(test_categories, train_categories,
                 thesaurus_fname, TRAIN_K, TEST_K, sim, TRAIN_INSTANCES):
    """
    We will use all labeled data from the train_categories and the thesaurus
    to train a model with the specified parameter values. Then we will
    evaluate the performance on each of the test categories and will return
    the classification accuracy. 
    """
    train_fname = "../work/train_vects"
    test_fname = "../work/test_vects"
    model_fname = "../work/model"
    
    F = FEATURE_GENERATOR()
    # parameters for generating vectors for training.
    F.TRAIN_K = TRAIN_K
    F.TEST_K = TEST_K
    F.sim = sim
    F.NO_TRAIN_INSTANCES = TRAIN_INSTANCES
    F.load_thesaurus(thesaurus_fname)
    
    # Training.
    (pos,neg,ignored) = F.generate_vectors(train_fname,
                                           train_categories,
                                           "TRAIN")
    train_lbfgs(train_fname, model_fname)

    # Testing.
    results = {'overall':0}
    for category in test_categories:
        (pos,neg,ignored) = F.generate_vectors(test_fname, [category], "TEST")
        accuracy = test_lbfgs(test_fname, model_fname)
        results[category] = accuracy
        results["overall"] += accuracy
    results["overall"] = results["overall"] / float(len(test_categories))
    return results

def batch_instances():
    """
    Vary the number of training instances and measure the performance on
    train and test data.
    """
    train_fname = "../work/train_vects"
    test_fname = "../work/test_vects"
    model_fname = "../work/model"

    # parameters for generating vectors for training.
    TRAIN_K = 400
    TEST_K = 800
    sim = 0.45
    
    test_categories = ["kitchen","dvd","electronics","books"]

    res_file = open("../work/batch_instances_www.csv", "w")
    res_file.write("#positives, negatives, total, ")
    for category in test_categories:
        res_file.write("%s[TEST], %s[TRAIN], " % (category, category))
    res_file.write("\n")
    res_file.flush()
    for n in [0, 100, 200, 300, 400, 500, 600, 700, 800]:
        total_positives = 0
        total_negatives = 0
        trainAccuracies = []
        testAccuracies = []
        for category in test_categories:
            thesaurus_fname = "../work/%s.bigram.thes" % category
            F = FEATURE_GENERATOR()
            # parameters for generating vectors for training.
            F.TRAIN_K = TRAIN_K
            F.TEST_K = TEST_K
            F.sim = sim
            F.NO_TRAIN_INSTANCES = n
            F.load_thesaurus(thesaurus_fname)

            # Training.
            train_categories = []
            train_categories.extend(test_categories)
            train_categories.remove(category)
            (train_pos,train_neg,train_ignored) = F.generate_vectors(train_fname,
                                                   train_categories,
                                                   "TRAIN")
            train_lbfgs(train_fname, model_fname)
            total_positives += train_pos
            total_negatives += train_neg

            # Evaluating on training data.
            F.NO_TRAIN_INSTANCES = 800
            (train_eval_pos,train_eval_neg,train_eval_ignored) = F.generate_vectors(test_fname,
                                                                                    train_categories,
                                                                                    "TEST")
            train_accuracy = test_lbfgs(test_fname, model_fname)
            trainAccuracies.append(train_accuracy)

            # Evaluating on test data.
            (test_pos,test_neg,test_ignored) = F.generate_vectors(test_fname,
                                                                  [category],
                                                                  "TEST")
            test_accuracy = test_lbfgs(test_fname, model_fname)
            testAccuracies.append(test_accuracy)
            pass
        total_train = total_positives + total_negatives
        res_file.write("%d, %d, %d, " % (total_positives, total_negatives, total_train))
        for i in range(len(test_categories)):
            res_file.write("%f, %f, " % (testAccuracies[i], trainAccuracies[i]))
        res_file.write("\n")                                      
        res_file.flush()
        pass
    res_file.close()            
    pass

def batch_train_size():
    """
    Evaluate the performance when increasing the training dataset size.
    """
    train_fname = "../work/train_vects"
    test_fname = "../work/test_vects"
    model_fname = "../work/model"

    # parameters for generating vectors for training.
    TRAIN_K = 400
    TEST_K = 800
    sim = 0.45
        
    test_categories = ["kitchen","dvd","electronics","books"]
    validation_categories = ["music", "video"]
    all_categories = ["apparel", "sports_and_outdoors", "magazines","baby",
                      "toys_and_games", "camera_and_photo",
                      "health_and_personal_care",
                      "outdoor_living","software"]

    res_file = open("../work/train_size_results.txt", "w")
    res_file.write("#domains, positives, negatives, total, train, test\n")

    for N in range(1, len(all_categories) + 1):
        # peformance on test data.
        test_accuracy = 0
        train_accuracy = 0
        for category in test_categories:
            thesaurus_fname = "../work/%s.bigram.thes" % category
            F = FEATURE_GENERATOR()
            # parameters for generating vectors for training.
            F.TRAIN_K = TRAIN_K
            F.TEST_K = TEST_K
            F.sim = sim
            F.load_thesaurus(thesaurus_fname)
            
            # Training.
            train_categories = []
            train_categories.extend(test_categories)
            train_categories.extend(all_categories[1:N])
            train_categories.remove(category)
            (train_pos,train_neg,train_ignored) = F.generate_vectors(train_fname,
                                                   train_categories,
                                                   "TRAIN")
            train_lbfgs(train_fname, model_fname)
    
            # Evaluating on training data.
            train_accuracy += test_lbfgs(train_fname, model_fname)
        
            # Evaliating on test data.
            (test_pos,test_neg,test_ignored) = F.generate_vectors(test_fname,
                                                              test_categories,
                                                                  "TEST")
            test_accuracy += test_lbfgs(test_fname, model_fname)
            pass
        train_accuracy = train_accuracy / len(test_categories)
        test_accuracy = test_accuracy / len(test_categories)
        train_total = train_pos + train_neg
        res_file.write("%d, %d, %d, %d, %f, %f\n" % (N, train_pos, train_neg, train_total,
                                                     train_accuracy,
                                                     test_accuracy))
        res_file.flush()
        pass
    res_file.close()
    pass


def batch_k():
    """
    Performs batch evaluation on k.
    """
    res_file = open("../work/batch_k.csv", "w")
    res_file.write("#TRAIN_K, TEST_K, Validation, Test\n")
    kvalues = [0,200,400,600,800,1000]
    for TRAIN_K in kvalues:
        for TEST_K in kvalues:
            result = batch_round(TRAIN_K, TEST_K, sim)
            res_file.write("%d, %d, %f, %f\n" % (TRAIN_K, TEST_K,
                                                 result["validation"],
                                                 result["test"]))
            res_file.flush()
    res_file.close()
    pass

def batch_sim():
    """
    Performs batch evaluation on sim.
    """
    res_file = open("../work/batch_sim.csv", "w")
    res_file.write("#sim, Validation, Test\n")
    sim_values_h = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    sim_values_l = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    sim_values = []
    sim_values.extend(sim_values_l)
    sim_values.extend(sim_values_h)
    TRAIN_K = 1000
    TEST_K = 1000
    best_sim = 0
    best_acc = 0
    print sim_values
    for sim in sim_values:
        result = batch_round(TRAIN_K, TEST_K, sim)
        res_file.write("%f, %f, %f\n" % (sim,
                                         result["validation"],
                                         result["test"]))
        if result['test'] > best_acc:
            best_acc = result['test']
            best_sim = sim
        res_file.flush()
    res_file.close()
    return {'best_acc':best_acc, 'best_sim':best_sim}

def train_lbfgs(train_file, model_file):
    """
    Train lbfgs on train file. and evaluate on test file.
    Read the output file and return the classification accuracy.
    """
    retcode = subprocess.call(
        "classias-train -tb -a lbfgs.logistic -pc1=0 -pc2=1 -m %s %s" %\
        (model_file, train_file), shell=True)
    return retcode

def test_lbfgs(test_file, model_file):
    """
    Evaluate on the est file.
    Read the output file and return the classification accuracy.
    """
    output = "../work/output"
    retcode = subprocess.call("cat %s | classias-tag -m %s -t > %s" %\
                              (test_file, model_file, output), shell=True)
    F = open(output)
    accuracy = 0
    correct = 0
    total = 0
    for line in F:
        if line.startswith("Accuracy"):
            p = line.strip().split()
            accuracy = float(p[1])
    F.close()
    return accuracy


def feature_vector_comparator(A, B):
    """
    We use this to sort features in a feature vector in the ascending
    order of the feature ids. We have a list of tuples of the form
    (featid, featval).
    """
    if A[0] > B[0]:
        return 1
    return -1
    
def analyze_model(model_fname):
    """
    Read the model file and print the largest positive and negative
    weighting features.
    """
    model_file = open(model_fname)
    L = []
    for line in model_file:
        if not line.startswith("@"):
            p = line.strip().split()
            weight = float(p[0])
            feature = p[1].strip()
            L.append((weight, feature))
    model_file.close()
    L.sort(feature_vector_comparator)
    # print the top N positive and negative features.
    N = 10
    print "\nTop %d most positive features:" % N
    L.reverse()
    for (weight,feature) in L[:N]:
        print "%s = %f" % (feature, weight)
    print "\nTop %d most negative features:" % N
    L.reverse()
    for (weight,feature) in L[:N]:
        print "%s = %f" % (feature, weight)
    print "\nTotal number of features =", len(L)
    pass

def compute_averages():
    """
    Computes the cross domain averages for various previously
    proposed methods. Because we train using multiple domains
    and test on separate domains, we must also coniser the
    average effect when previously proposed methods would obtain
    if they also consider multiple domains for training.
    """
    # compute the average scores for multiple methods for the same target.
    F = open("../work/BlitzerResults.txt")
    labels = F.readline().split()
    h = {}
    domains = ["B","D","K","E"]
    methods = labels[2:]
    for m in methods:
        h[m] = {}
        for d in domains:
            h[m][d] = 0
    line = F.readline()
    while line:
        p = line.strip().split()
        target = p[1]
        for (i,method) in enumerate(methods):
            h[method][target] += float(p[i+2])
        line = F.readline()
    # compute averages.
    for method in methods:
        print method
        for target in domains:
            print target, h[method][target] / (3 * 100)
        print
    F.close()
    pass

def count_selected_rating_features(fname):
    """
    Read the colids.selected file for the sentiment sensitive thesaurus
    and count the number of rating related features.
    """
    F = open(fname)
    pos_feats = {}
    neg_feats = {}
    for line in F:
        p = line.strip().split('\t')
        feat = p[0].strip()
        fid = int(p[1])
        freq = int(p[2])
        if feat.find('__POS') != -1:
            pos_feats[feat] = {'fid':fid, 'freq':freq}
        elif feat.find('__NEG') != -1:
            neg_feats[feat] = {'fid':fid, 'freq':freq}
    F.close()
    print "Total positive features =", len(pos_feats)
    print "Total negative features =", len(neg_feats)
    print "Total rating (POS+NEG) features =", len(pos_feats) + len(neg_feats)
    pass


def debug():
    train_lbfgs("../work/train_vects", "../work/model")
    print test_lbfgs("../work/test_vects", "../work/model")
    pass

if __name__ == "__main__":
    main()
    #debug()
    pass
