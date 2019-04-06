
#! /bin/sh

# This script will generate HTML and PDF versions of the documentation
# in doc directory.
url="http://www.miv.t.u-tokyo.ac.jp/danushka/software.html"
epydoc -v --debug --html --url $url --graph all -o ./doc  --name=MLIB ./
epydoc -v --debug --pdf  -o ./doc  --name=MLIB ./