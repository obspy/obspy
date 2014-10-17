#!/bin/bash
# Script for creating Java tauP output for the purpose of comparing it to Taupy's output
file='data/java_tauptime_testoutput'
# If file already exists, delete it.
[[ -f "$file" ]] && rm "$file"
for degree in 0 45 90 180 360 560
do
    for depth in 0 100 1000 2889
    do
	../../../tauP-2.1.2_java_from_their_website/bin/taup_time -mod iasp91 -h $depth -ph ttall -deg $degree >> $file
    done
done
