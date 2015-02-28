/*--------------------------------------------------------------------
# Filename: bbfk.c
#  Purpose: FK Array Analysis module (Seismology)
#   Author: Matthias Ohrnberger, Joachim Wassermann, Moritz Beyreuther
#    Email: beyreuth@geophysik.uni-muenchen.de
#  Changes: 8.2010 now using NumPy fftpack instead of realft (Moritz)
#           8.2010 changed window and trace to doubles due to fftpack (Moritz)
#           8.2010 removed group_index_list (Moritz)
#           8.2010 passing rffti from command line for speed (Moritz)
# Copyright (C) 2010 M. Beyreuther, J. Wassermann, M. Ohrnberger
#---------------------------------------------------------------------*/
#include <stdlib.h>
#define _USE_MATH_DEFINES  // for Visual Studio
#include <math.h>
#include "platform.h"

#define STEER(I, J, K, L) steer[(I)*nstat*grdpts_y*grdpts_x + (J)*nstat*grdpts_y + (K)*nstat + (L)]
#define RPTR(I, J, K) Rptr[(I)*nstat*nstat + (J)*nstat + (K)]
#define P_N(I, J) p_n[(I)*grdpts_y + (J)]
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
                    STEER(n,x,y,i).re = cos(wtau);
                    STEER(n,x,y,i).im = -sin(wtau);
                }
            }
        }
    }
}


int generalizedBeamformer(double *relpow, double *abspow, const cplx * const steer,
        const cplx * const Rptr,
        const int nstat, const int prewhiten, const int grdpts_x,
        const int grdpts_y, const int nf, double dpow,
        const methodE method) {
    /* method: 0 == "bf, 1 == "capon"
     * start the code -------------------------------------------------
     * This assumes that all stations and components have the same number of
     * time samples, nt */

    int x, y, i, j, n;
    const cplx cplx_zero = {0., 0.};
    double *p_n;

    /* we allocate the taper buffer, size nsamp! */
    p_n = (double *) calloc(grdpts_x * grdpts_y, sizeof(double));
    if (p_n == NULL ) {
        return 1;
    }

    if (method == CAPON) {
        /* optimized way of abspow normalization */
        dpow = 1.0;
    }
    /* in general, beamforming is done by simply computing the covariances of
     * the signal at different receivers and than steer the matrix R with
     * "weights" which are the trial-DOAs e.g., Kirlin & Done, 1999:
     * BF: P(f) = e.H R(f) e
     * CAPON: P(f) = 1/(e.H R(f)^-1 e) */
    for (n = 0; n < nf; ++n) {
        double inv_fac;
        double white = 0.;
        for (x = 0; x < grdpts_x; ++x) {
            for (y = 0; y < grdpts_y; ++y) {
                double pow;
                cplx eHR_ne = cplx_zero;
                for (i = 0; i < nstat; ++i) {
                    cplx R_ne = cplx_zero;
                    for (j = 0; j < nstat; ++j) {
                        const cplx s = STEER(n,x,y,j);
                        const cplx r = RPTR(n,i,j);
                        R_ne.re += r.re * s.re - r.im * s.im;
                        R_ne.im += r.re * s.im + r.im * s.re;
                    }
                    eHR_ne.re += STEER(n,x,y,i).re * R_ne.re + STEER(n,x,y,i).im * R_ne.im ; /* eH, conjugate */
                    eHR_ne.im += STEER(n,x,y,i).re * R_ne.im - STEER(n,x,y,i).im * R_ne.re; /* eH, conjugate */
                }
                pow = sqrt(eHR_ne.re * eHR_ne.re + eHR_ne.im * eHR_ne.im);
                pow = (method == CAPON) ? 1. / pow : pow;
                white = fmax(pow, white);
                P_N(x,y)= pow;
                ABSPOW(x,y) += pow;
            }
        }
        /* scale for each frequency individually */
        if (prewhiten == 1) {
            inv_fac = 1. / (white * nf * nstat);
        }
        else {
            inv_fac = 1. / dpow;
        }
        for (x = 0; x < grdpts_x; ++x) {
            for (y = 0; y < grdpts_y; ++y) {
                RELPOW(x,y) += P_N(x,y) * inv_fac;
            }
        }
    }


    free(p_n);

    return 0;
}
