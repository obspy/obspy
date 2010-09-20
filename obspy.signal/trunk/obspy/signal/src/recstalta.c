/*--------------------------------------------------------------------
# Filename: recstalta.c
#  Purpose: Recursive STA/LTA
#   Author: Moritz Beyreuther
# Copyright (C) 2009 M. Beyreuther
#---------------------------------------------------------------------*/
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void recstalta(double *a, double *charfct, int ndat, int nsta, int nlta) {
    int i;
    double csta = 1./((double)nsta);
    double clta = 1./((double)nlta);
    double sta = 0.0;
    double lta = 0.0;

    for (i=1;i<ndat;i++) {
        sta = csta * pow(a[i],2) + (1-csta)*sta;
        lta = clta * pow(a[i],2) + (1-clta)*lta;
        charfct[i] = sta/lta;
    }

    if (nlta < ndat) for (i=0;i<nlta;i++) charfct[i] = 0.0;

    return;
}
