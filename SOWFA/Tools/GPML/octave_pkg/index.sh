#!/bin/bash

DIR=$1
PACKAGE=$(sed -n -e 's/^Title: *\(\w\+\)/\1/p' $DIR/DESCRIPTION)
COV=$(ls $DIR/cov | sed -n -e 's/\(\w\+\).m$/ \1/p')
MEAN=$(ls $DIR/mean | sed -n -e 's/\(\w\+\).m$/ \1/p')
LIK=$(ls $DIR/lik | sed -n -e 's/\(\w\+\).m$/ \1/p')
INF=$(ls $DIR/inf | sed -n -e 's/\(\w\+\).m$/ \1/p')
UTIL=$(ls $DIR/util | sed -n -e 's/\(\w\+\).m$/ \1/p')
PRIOR=$(ls $DIR/prior | sed -n -e 's/\(\w\+\).m$/ \1/p')
DEMO=$(ls $DIR/demo | sed -n -e 's/\(\w\+\).m$/ \1/p')
HELP=$(ls $DIR/help | sed -n -e 's/\(\w\+\).m$/ \1/p')

echo "gpml >> " "$PACKAGE"
echo "Main function"
echo " gp"
echo "Covariance Functions"
echo "$COV"
echo "Mean Functions"
echo "$MEAN"
echo "Likelihood Functions"
echo "$LIK"
echo "Prior Distributions"
echo "$PRIOR"
echo "Inference methods"
echo "$INF"
echo "Utility functions"
echo "$UTIL"
echo "Help functions"
echo "$HELP"
echo "Demo functions"
echo "$DEMO"

