/*--------------------------------------------------------------------
# Filename: stalta.c
#  Purpose: Classic STA/LTA trigger, optimized C version
#   Author: Moritz Beyreuther
# Copyright (C) 2012 ObsPy-Developer-Team
#---------------------------------------------------------------------*/

#include <math.h>

#define OPTIMIZED_VERSION

typedef struct _headS {
    int N;
    int Nsta;
    int Nlta;
} headS;

int stalta(const headS *head, const double *data, double *charfct)
{
    int i;
    double sta = 0.;
    double lta;
#ifdef OPTIMIZED_VERSION
    double buf;
    const double frac = (double) head->Nlta / (double) head->Nsta;
#else
    int j;
#endif

    if (head->N < head->Nlta) {
        return 1;
    }

#ifdef OPTIMIZED_VERSION
    for (i = 0; i < head->Nsta; ++i) {
        sta += pow(data[i], 2);
        charfct[i] = 0.;
    }
    lta = sta;
    for (i = head->Nsta; i < head->Nlta; ++i) {
        buf = pow(data[i], 2);
        lta += buf;
        sta += buf - pow(data[i - head->Nsta], 2);
        charfct[i] = 0.;
    }
    charfct[head->Nlta - 1] = sta / lta * frac;
    for (i = head->Nlta; i < head->N; ++i) {
        buf = pow(data[i], 2);
        sta += buf - pow(data[i - head->Nsta], 2);
        lta += buf - pow(data[i - head->Nlta], 2);
        charfct[i] = sta / lta * frac;
    }
#else
    for (i = 0; i < head->Nlta - 1; ++i) {
        charfct[i] = 0.0;
    }
    for (i = head->Nlta - 1; i < head->N; ++i) {
        sta = 0.0;

        for (j = 0u; j < head->Nsta; j++) {
            sta += pow(data[i - j], 2);
        }
        lta = sta;
        for (j = head->Nsta; j < head->Nlta; j++) {
            lta += pow(data[i - j], 2);
        }
        sta /= (double) head->Nsta;
        lta /= (double) head->Nlta;
        charfct[i] = sta / lta;
    }
#endif

    return 0;
}
