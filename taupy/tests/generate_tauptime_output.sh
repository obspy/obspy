#!/bin/bash
# Script for creating Java tauP output for the purpose of comparing it to Taupy's output

touch java_tauptime_testoutput
for degree in 0 45 90 180 360 560
do
    for depth in 0 100 1000 2889
    do
	../../../tauP-2.1.2_java_from_their_website/bin/taup_time -mod iasp91 -h $depth -ph P,S,PcP,ScS,SKS,sS,SS,PKKP,PKiKP -deg $degree >> data/java_tauptime_testoutput
    done
done
