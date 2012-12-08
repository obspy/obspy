/*--------------------------------------------------------------------
# Filename: bbfk.c
#  Purpose: FK Array Analysis module (Seismology)
#   Author: Matthias Ohrnberger, Joachim Wassermann, Moritz Beyreuther
#    Email: beyreuth@geophysik.uni-muenchen.de
#  Changes: 8.2010 now using numpy fftpack instead of realft (Moritz)
#           8.2010 changed window and trace to doubles due to fftpack (Moritz)
#           8.2010 removed group_index_list (Moritz)
#           8.2010 passing rffti from command line for speed (Moritz)
# Copyright (C) 2010 M. Beyreuther, J. Wassermann, M. Ohrnberger
#---------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "platform.h"

#define STEER(I, J, K, L) steer[(I)*nstat*nf*grdpts_y + (J)*nstat*nf + (K)*nstat + (L)]
#define RPTR(I, J, K) Rptr[(I)*nstat*nstat + (J)*nstat + (K)]
#define P(I, J, K) p[(I)*nf*grdpts_y + (J)*nf + (K)]
#define RELPOW(I, J) relpow[(I)*grdpts_y + (J)]
#define ABSPOW(I, J) abspow[(I)*grdpts_y + (J)]
#define STAT_TSHIFT_TABLE(I, J, K) stat_tshift_table[(I) * grdpts_x * grdpts_y + (J) * grdpts_y + K]

typedef struct cplxS {
    double re;
    double im;
} cplx;

typedef enum _methodE
{
    BF     = 0,
    CAPON  = 1,
} methodE;



void calcSteer(const int nstat, const int grdpts_x, const int grdpts_y,
        const int nf, const int nlow, const float deltaf,
        const float * const stat_tshift_table, cplx * const steer) {
    int i;
    int x;
    int y;
    int n;
    double wtau;
    for (i=0; i < nstat; i++) {
        for (x=0; x < grdpts_x; x++) {
            for (y=0; y < grdpts_y; y++) {
                for (n=0; n < nf; n++) {
                    wtau = 2.*M_PI*(float)(nlow+n)*deltaf*STAT_TSHIFT_TABLE(i, x, y);
                    STEER(x,y,n,i).re = cos(wtau);
                    STEER(x,y,n,i).im = -sin(wtau);
                }
            }
        }
    }
}


int generalizedBeamformer(double *relpow, double *abspow, const cplx * const steer,
        const cplx * const Rptr,
        const int nsamp, const int nstat, const int prewhiten, const int grdpts_x,
        const int grdpts_y, const int nfft, const int nf, double dpow,
        const methodE method) {
    /* method: 0 == "bf, 1 == "capon"
     * start the code -------------------------------------------------
     * This assumes that all stations and components have the same number of
     * time samples, nt */

    int x, y, i, j, n;
    const cplx cplx_zero = {0., 0.};
    double *p;
    double *white;
    double gen_power[2];

    if (method >= sizeof(gen_power)) {
        fprintf(stderr, "\nUnkown method!\n");
        exit(EXIT_FAILURE);
    }

    /* we allocate the taper buffer, size nsamp! */
    p = (double *) calloc((size_t) (grdpts_x * grdpts_y * nf), sizeof(double));
    if (p == NULL ) {
        fprintf(stderr, "\nMemory allocation error (p)!\n");
        exit(EXIT_FAILURE);
    }
    white = (double *) calloc((size_t) nf, sizeof(double));
    if (white == NULL ) {
        fprintf(stderr, "\nMemory allocation error (white)!\n");
        exit(EXIT_FAILURE);
    }

    if ((method == CAPON) || (prewhiten == 1)) {
        /* optimized way of abspow normalization */
        dpow = 1.0;
    }
    for (x = 0; x < grdpts_x; ++x) {
        for (y = 0; y < grdpts_y; ++y) {
            /* in general, beamforming is done by simply computing the
             * covariances of the signal at different receivers and than steer
             * the matrix R with "weights" which are the trial-DOAs e.g.,
             * Kirlin & Done, 1999:
             * a) bf: P(f) = e.H R(f) e
             * b) capon: P(f) = 1/(e.H R(f)^-1 e) */
            ABSPOW(x, y) = 0.;
            for (n = 0; n < nf; ++n) {
                double power;
                cplx eHR_ne = cplx_zero;
                for (i = 0; i < nstat; ++i) {
                    register cplx R_ne = cplx_zero;
                    for (j = 0; j < nstat; ++j) {
                        register const cplx s = STEER(x,y,n,j);
                        register const cplx r = RPTR(n,i,j);
                        R_ne.re += r.re * s.re - r.im * s.im;
                        R_ne.im += r.re * s.im + r.im * s.re;
                    }
                    eHR_ne.re += STEER(x,y,n,i).re * R_ne.re + STEER(x,y,n,i).im * R_ne.im; /* eH, conjugate */
                    eHR_ne.im += STEER(x,y,n,i).re * R_ne.im - STEER(x,y,n,i).im * R_ne.re; /* eH, conjugate */
                }
                /* optimization: avoid if condition on BW / CAPON through array access of gen_power */
                gen_power[BF] = sqrt(eHR_ne.re * eHR_ne.re + eHR_ne.im * eHR_ne.im);
                gen_power[CAPON] = 1. / gen_power[BF];
                power = gen_power[method];
                ABSPOW(x,y) += power;
                white[n] = fmax(power, white[n]);
                P(x,y,n) = power;
            }
            RELPOW(x,y) = ABSPOW(x,y)/dpow;
        }
    }

    if (prewhiten == 1) {
        for (x = 0; x < grdpts_x; ++x) {
            for (y = 0; y < grdpts_y; ++y) {
                RELPOW(x,y)= 0.;
                for (n = 0; n < nf; ++n) {
                    RELPOW(x,y) += P(x,y,n)/(white[n]*nf*nstat);
                }
            }
        }
    }

    free((void *) p);
    free((void *) white);

    return 0;
}
