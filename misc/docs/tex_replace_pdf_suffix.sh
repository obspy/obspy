#!/bin/bash
DIR=$1
for FILE in `ls -1 $DIR/*.tex`
do
    sed '/includegraphics/ s/\.pdf/.png/' ${FILE} > ${FILE}.tmp && mv ${FILE}.tmp ${FILE}
done
