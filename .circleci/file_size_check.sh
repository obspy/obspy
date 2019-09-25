#!/bin/bash
EXIT_CODE=0
for SIZE in $(git log --pretty='format:%H' master.. | git cat-file --batch-check='%(objectsize)')
 do
  if (( $SIZE > 235 ))
   then 
    echo "File you added exceeds our limit of 500k"
    EXIT_CODE=1
   fi
 done
(exit $EXIT_CODE) 
