#! /bin/sh

CLUSTERING=1
HELDOUT=1
SCALING=1
CLASSIFY=1

SILENT=1

if [ "$SILENT" = 1 ]; then
    terminal="/dev/null"
else
    terminal="/dev/stdout"
fi


# Clustering.
if [ "$CLUSTERING" = 1 ]; then
    echo "SEQUENTIAL CO-CLUSTERING"
    ./MLIBCommand.py -a cluster.seqcoclust -i ./work/clusttest.data -o ./work/clusters -t 0.1 -p 0.2 

    echo "SEQUENTIAL CLUSTERING"
    ./MLIBCommand.py -a cluster.seqclust -i ./work/clusttest.data -o ./work/clusters -t 0.8 
fi

# Heldout data selection.
if [ "$HELDOUT" = 1 ]; then
 echo "SELECT HELDOUT DATA"
    ./MLIBCommand.py -a data.heldout -n 100 -i work/rcv1_train.binary.bz2 -d work/utests/heldout -o work/utests/develop > $terminal
fi

# Data selection and scaling.
if [ "$SCALING" = 1 ]; then
    echo "SCALE TRAIN DATA"
    ./MLIBCommand.py -a data.scale-train -i work/utests/develop -o work/utests/scaled.train -r work/utests/range -x 1 -y 1 -t 1 -p 0 > $terminal

    echo "SCALE TEST DATA"
    ./MLIBCommand.py -a data.scale-test -i work/utests/heldout -o work/utests/heldout.scaled -r work/utests/range > $terminal
fi

# Classification Algorithms.
if [ "$CLASSIFY" = 1 ]; then
    echo "CLASSIFY SGD LOGISTIC REGRESSION"
    ./MLIBCommand.py -a classify.logreg_sgd -n 3 -i work/iris_multi.train -m work/utests/model -t 10 -q 1 -g 5 > $terminal

    echo "PREDICTING SGD LOGISTIC REGRESSION"
    ./MLIBCommand.py -a predict.logreg -i work/iris_multi.test -m work/utests/model -o work/utests/output -u -v > $terminal

    echo "CLASSIFY TRUNCATED GRADIENT LOGISTIC REGRESSION"
    ./MLIBCommand.py -a classify.logreg_tg -n 3 -i work/iris_multi.train -m work/utests/model -t 10 -q 1 -g 5 > $terminal

    echo "PREDICTING TRUNCATED GRADIENT LOGISTIC REGRESSION"
    ./MLIBCommand.py -a predict.logreg -i work/iris_multi.test -m work/utests/model -o work/utests/output -u -v > $terminal
fi