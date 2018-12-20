#!/bin/bash

cd "${1}" && \
( \
  echo "  Move demo functions to folder demo"
  mkdir -p demo
  for f in $(find doc/ -name 'demo*.m' -exec basename {} \;)
  do
    mv {doc/,demo/gpml_}${f}
  done
  mv doc/gpml_randn.m demo/
  
  echo "  Prefix all documentation files with gpml_ and move to help/"
  mkdir -p "help"
  for f in $(find doc/ -name 'usage*.m' -exec basename {} \;)
  do
    mv {doc/,help/gpml_}${f}
  done

  echo "  Move documentation mfiles in root to help/"
  for f in $(ls -1 *Functions.m) infMethods.m priorDistributions.m
  do
    mv {,help/gpml_}${f}
  done
) && \
cd -
