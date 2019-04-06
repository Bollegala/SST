This package provides functionality to adapt a sentiment classifier
trained on one or more source domains (Amazon product category) to a different
(target) domain. 

Requirements
--------------

This software was developed on Ubtunu Linux and has not been tested in other platforms.
However, the code is in Python (ver 2.7) and is not OS dependent. Therefore, it should
be able run in other platforms as well.

1) Python ver 2.7 is required.

2) The code uses the python machine learning library (MLIB).
    MLIB can be downloaded here.
    http://www.iba.t.u-tokyo.ac.jp/~danushka/mlib.html
    There is already a downloaded version of MLIB in the bin directory and you need not download MLIB.

3) classias: a classification library.
    You must download and install classias from the following url.
    http://www.chokkan.org/software/classias/
    We use the L1 regularized logistic regression implementation of classes.

The software was developed and evaluated on a machine that has 24GB RAM, 16 cores (2 xeon 8 core 
processors), 64bit Ubuntu Linux. The software implements parallel processing techniques to
speed up the matrix and similarity computations assuming the above architecture.
If is strongly recommended that  your system meets the above requirements.


Quick introduction
-------------------

   Open a terminal and CD to the bin directory. Type the following command to train from
   the kitchen source domain and test on the electronics target domain.
   
   python Makefile.py -s kitchen -t electronics

   You can specify multiple source domains by separating them with a comma.
   python Makefile.py -s kitchen,dvd -t electronics

The domain names that can be used are kitchen, books, electronics and dvd.
(All in lowercase)

Detailed introduction
----------------------

1. Dataset.
   ---------
   In the data directory there are two directories names test_data and train_data.
   For each Amazon product category in the original multi-domain
   sentiment classification dataset, we have POS tag using RASP. In both test and train
   data directories there are sub-directories named after each Amazon product category.

   positive.review is the file with positive (rating > 3) reviews.
   negative.review is the file with negative (rating < 3) reviews.
   unlabeled.review is the file with unlabeled (rating is arbitrary) reviews.
   Additionally, there are .tagged files in each directory that contains the POS tagged
   reviews (positive.tagged, negative.tagged, and unlabeled.tagged).
   The test_data directory contains six sub-directories. dvd, books, electronics, and
   kitchen are the four evaluation domains used by previous work on cross-domain
   sentiment classification. music and video domains are selected as development
   domains that can be used to tune any parameters using for example cross-validation
   if necessary. Note that there are no unlabeled reviews (consequently tagged unlabeled
   reviews) in test directories. There are approximately 200 positive and 200 negative
   instances for each category in the four evaluation categories. For the two development
   domains we have 100 positive and 100 negative instances in the test directories.
   Similarly, in train_data directory we have training data (positive, negative, and
   unlabeled) for each category. 

2. Create element co-occurrence matrix.
   ------------------------------------
   
   Use create_matrix.py for this purpose.

   Generate a list of row and column elements (lexical, lexical and sentiment) and 
   sort the list according to the total frequency of an element in the corpus.
   Select the top ranked elements. Next, create an element co-occurrence matrix
   for those selected elements in the corpus. (Although we compute the pmi between a 
   feature and domain label to identify domain independent features, we do not use 
   this in our experiments. Instead we use all lexical elements.)
  

3. Compute the similarity distribution for elements.
   --------------------------------------------------
   Use compute_distribution.py to compute the similarity distribution from the 
   co-occurrence matrix. This computation is done in parallel using 32 processors.


4. Create the distributional sentiment-sensitive thesaurus.
   ---------------------------------------------------------
   Use crate_thesaurus.py


3. Train a classifier using the thesaurus. Test on different domains.
   -------------------------------------------------------------------
   Use classify.py

4. Overall functionality
   -----------------------
    Use Makefile.py
    
    Note: There are lots of functionality that can be enabled and tested only by directly 
           editing the source code. Makefile.py is simply a collection of the different
          functionalities that are implemented in the other modules. It has a demo command line
          tool (see the Quick Introduction above for this.)


All author names have been removed from the source code to support anonymous reviewing.

---------------------------------------------------------------------------------------------- 
	


 
