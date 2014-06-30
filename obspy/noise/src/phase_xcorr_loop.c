/*--------------------------------------------------------------------
# Filename: phase_xcorr_loop.c
#  Purpose: Loop for the phase cross correlation.
#   Author: Lion Krischer
# Copyright (C) 2014 L. Krischer
#---------------------------------------------------------------------*/

/* Optimized loop for the phase cross correlation.
 *
 * Parameters:
 *
 *    data_1: Normalized analytic signal 1.
 *    data_2: Normalized analytic signal 2.
 *    npts: Length of data_1 and data_2.
 *    pxc_out: Array to which the CC will be written to.
 *    nu: Exponent for phase cross correlation.
 *
 * Output will be written to pxc_out.
 */

#include <complex.h>
#include <math.h>
#include <stdio.h>

void phase_xcorr_loop(double complex *data_1, double complex *data_2, int npts,
                      double *pxc_out, double nu, int max_lag, int min_lag) {
    int i, k;
    double sum1, sum2, sum3, sum4, factor;

    for (k=0; k <= max_lag; k++) {
        if (min_lag > k) {
            continue;
        }
        sum1 = sum2 = sum3 = sum4 = 0.0;
        factor = 1.0 / (2.0 * npts - k);

        if (nu == 1.0) {
            for (i=0; i < (npts - k); i++) {
                sum1 += cabs(data_1[i] + data_2[k + i]);
                sum2 += cabs(data_1[i] - data_2[k + i]);
                sum3 += cabs(data_1[k + i] + data_2[i]);
                sum4 += cabs(data_1[k + i] - data_2[i]);
            }
        }
        else {
            for (i=0; i < (npts - k); i++) {
                sum1 += pow(cabs(data_1[i] + data_2[k + i]), nu);
                sum2 += pow(cabs(data_1[i] - data_2[k + i]), nu);
                sum3 += pow(cabs(data_1[k + i] + data_2[i]), nu);
                sum4 += pow(cabs(data_1[k + i] - data_2[i]), nu);
            }
        }

        pxc_out[max_lag + k] = factor * (sum1 - sum2);
        pxc_out[max_lag - k] = factor * (sum3 - sum4);
    }
}


