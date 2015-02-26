#!/bin/bash

set -e

if [ $# -ne 2 ];
then
	echo "Usage: $0 <input dir> <output dir>" >&2
	exit 1
fi

for f in `ls $1/*.txt`;
do
	fn=$(basename $f)
	sed 's/[ 	]\+$//g' $f | sed '1!{$!N;s/\n/ /}' > $2/$fn
done
