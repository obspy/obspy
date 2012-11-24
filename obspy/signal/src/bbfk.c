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

#define TRUE 1
#define FALSE 0
#define STEER(I, J, K, L) steer[(I)*nstat*nf*grdpts_y + (J)*nstat*nf + (K)*nstat + (L)]
#define RPTR(I, J, K) Rptr[(I)*nstat*nstat + (J)*nstat + (K)]
#define P(I, J, K) p[(I)*nf*grdpts_y + (J)*nf + (K)]
#define RELPOW(I, J) relpow[(I)*grdpts_y + (J)]
#define ABSPOW(I, J) abspow[(I)*grdpts_y + (J)]

#define USE_SINE_TABLE

#ifdef USE_SINE_TABLE
#define SINE_REF_LEN 1000
#define SINE_REF_LEN_4 (SINE_REF_LEN / 4)
#define SINE_TABLE_LEN (SINE_REF_LEN + SINE_REF_LEN / 4 + 1)
#endif

typedef struct cplxS {
    double re;
    double im;
} cplx;



void calcSteer(const int nstat, const int grdpts_x, const int grdpts_y,
        const int nf, const int nlow, const float deltaf, float
        ***stat_tshift_table, cplx * const steer) {
    int i;
    int x;
    int y;
    int n;
    double wtau;
    for (i=0; i < nstat; i++) {
        for (x=0; x < grdpts_x; x++) {
            for (y=0; y < grdpts_y; y++) {
                for (n=0; n < nf; n++) {
                    wtau = 2.*M_PI*(float)(nlow+n)*deltaf*stat_tshift_table[i][x][y];
                    STEER(x,y,n,i).re = cos(wtau);
                    STEER(x,y,n,i).im = sin(wtau);
                }
            }
        }
    }
}


int bbfk(double **window, int *spoint, int offset,
         float ***stat_tshift_table, double *abs, double *rel, int *ix,
         int *iy, float flow, float fhigh, float digfreq, int nsamp,
         int nstat, int prewhiten, int grdpts_x, int grdpts_y, int nfft) {
    int		j,k,l,w;
    int		wlow,whigh;
    int nf;
    float	df;
    float	denom = 0;
    float	dpow;
    double	re;
    double	im;
    float	***pow;
    float	**nomin;
    float	*maxpow;
    double	absval;
    float maxinmap = 0.;
    float sumre;
    float sumim;
    float wtau;
    float cos_wtau;
    float sin_wtau;
#ifdef USE_SINE_TABLE
    float fidx;
    int idx;
    float frac;
    float sine_step = 2 * M_PI / SINE_REF_LEN;
    float sine_step_inv = 1. / sine_step;
    float *sine_table;
#endif

    /* mtrace(); */

    df = digfreq/(float)nfft;
    wlow = (int)(flow/df+0.5);
    if (wlow < 1) {
        /*******************************************************/
        /* never use spectral value at 0 -> this is the offset */
        /*******************************************************/
        wlow = 1;
    }
    whigh = (int)(fhigh/df+0.5);
    if (whigh>(nfft/2-1)) {
        /***************************************************/
        /* we avoid using values next to nyquist frequency */
        /***************************************************/
        whigh = nfft/2-1;
    }
    nf = whigh - wlow;

#ifdef USE_SINE_TABLE
    /********************************************************************/
    /* create sine table for fast execution                             */
    /********************************************************************/
    sine_table = (float *)calloc((size_t) SINE_TABLE_LEN, sizeof(float));
    if (sine_table == NULL) {
        fprintf(stderr,"\nMemory allocation error (sine_table)!\n");
        exit(EXIT_FAILURE);
    }
    for (j = 0; j < SINE_TABLE_LEN; ++j) {
        sine_table[j] = sin(j / (float) (SINE_TABLE_LEN - 1) * (M_PI * 2. + M_PI / 2.));
    }
#endif


    /***********************************************************************/
    /* we calculate the scaling factor or denominator, if not prewhitening */
    /***********************************************************************/
    if (prewhiten!=TRUE) {
        denom = 0.;
        for(w=0;w<=nf;w++) {
            dpow = 0;
            for (j=0;j<nstat;j++) {
                /* mimic realft, imaginary part is negativ in realft */
                re = window[j][2*w];
                im = -window[j][2*w+1];
                dpow += (float) (re*re+im*im);
            }
            denom += dpow;
        }
    }
    denom *= (float)nstat;

    /****************************************************/
    /* allocate w-maps, maxpow values and nominator-map */
    /****************************************************/
    nomin = (float **)calloc((size_t) grdpts_x, sizeof(float *));
    if (nomin == NULL) {
        fprintf(stderr,"\nMemory allocation error (nomin)!\n");
        exit(EXIT_FAILURE);
    }
    for (k=0;k<grdpts_x;k++) {
        nomin[k]  = (float *)calloc((size_t) grdpts_y, sizeof(float));
        if (nomin[k] == NULL) {
            fprintf(stderr,"\nMemory allocation error (nomin[k])!\n");
            exit(EXIT_FAILURE);
        }
    }
    maxpow = (float *)calloc((size_t) (nf+1), sizeof(float));
    if (maxpow == NULL) {
        fprintf(stderr,"\nMemory allocation error (maxpow)!\n");
        exit(EXIT_FAILURE);
    }
    pow = (float ***)calloc((size_t) (nf+1), sizeof(float **));
    if (pow == NULL) {
        fprintf(stderr,"\nMemory allocation error (pow)!\n");
        exit(EXIT_FAILURE);
    }
    for (w=0;w<=nf;w++) {
        pow[w] = (float **)calloc((size_t) (grdpts_x), sizeof(float *));
        if (pow[w] == NULL) {
            fprintf(stderr,"\nMemory allocation error (pow[w])!\n");
            exit(EXIT_FAILURE);
        }
        for (k=0;k<grdpts_x;k++) {
            pow[w][k] = (float *)calloc((size_t) grdpts_y, sizeof(float));
            if (pow[w][k] == NULL) {
                fprintf(stderr,"\nMemory allocation error (pow[w][k])!\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    /*************************************************************/
    /* we start with loop over angular frequency, this allows us */
    /* to prewhiten the fk map, if we want to			 */
    /*************************************************************/
    for (w=0;w<=nf;w++) {
    	float PI_2_df_w = 2.*M_PI*df*(float)(w + wlow);
        /***********************************/
        /* now we loop over x index (east) */
        /***********************************/
        for (k=0;k<grdpts_x;k++) {
            /************************************/
            /* now we loop over y index (north) */
            /************************************/
            for (l=0;l<grdpts_y;l++) {
                /********************************************/
                /* this is the loop over the stations group */
                /********************************************/
                sumre = 0.f;
                sumim = 0.f;
                for (j = 0; j < nstat; j++) {
                    wtau =
                            (float) (PI_2_df_w * stat_tshift_table[j][k][l]);
#ifdef USE_SINE_TABLE
                    /* calculate index in sine table */
                    while (wtau > 2.f * M_PI) {
                        wtau -= 2.f * M_PI;
                    }
                    while (wtau < 0.) {
                        wtau += 2.f * M_PI;
                    }
                    fidx = wtau * sine_step_inv;
                    idx = (int) fidx;
                    frac = fidx - idx;
                    sin_wtau = sine_table[idx] * (1. - frac) + sine_table[idx + 1] * frac;
                    cos_wtau = sine_table[idx + SINE_REF_LEN_4] * (1. - frac) + sine_table[idx + 1 + SINE_REF_LEN_4] * frac;
#else
                    sin_wtau = sin(wtau);
                    cos_wtau = cos(wtau);
#endif
                    /* here the real stuff happens */
                    re = window[j][2 * w];
                    im = window[j][2 * w + 1];
                    sumre += (float) (re * cos_wtau - im * sin_wtau);
                    sumim += (float) (im * cos_wtau + re * sin_wtau);
                }
                pow[w][k][l] = (sumre * sumre + sumim * sumim);
                if (pow[w][k][l] >= maxpow[w]) {
                    maxpow[w] = pow[w][k][l];
                }
            }
        }
    }

    /**********************************************/
    /* now we finally calculate the nominator map */
    /**********************************************/
    for (k=0;k<grdpts_x;k++) {
        for (l=0;l<grdpts_y;l++) {
            for (w=0;w<=nf;w++) {
                if (prewhiten==TRUE) {
                    nomin[k][l] += pow[w][k][l] / maxpow[w];
                }
                else {
                    nomin[k][l] += pow[w][k][l] / denom;
                }
#if 0
                map[k][l] = nomin[k][l];
#endif
            }
#if 0
            if (prewhiten!=TRUE) {
                nomin[k][l] /= denom;
            }

#endif
            /*****************************************/
            /* we get the maximum in map and indices */
            /*****************************************/
            if (nomin[k][l] > maxinmap) {
                maxinmap = nomin[k][l];
                *ix = k;
                *iy = l;
            }
        }
    }
#if 0
    maxinmap /= (float)(nstat);
#endif
    *rel = maxinmap;
    if (prewhiten==TRUE) {
        *rel /= (float)((whigh-wlow+1)*nfft)*digfreq;
    }
    else {
        absval = maxinmap*denom/(float)(whigh-wlow+1);
        absval /= (double)(nstat*nstat);
        absval /= (double)nfft;
        absval /= (double)digfreq;
        *abs = (float) absval;
    }

    /* now we free everything */
    for (k=0;k<grdpts_x;k++) 
        free((void *)nomin[k]);
    for (w=0;w<=nf;w++) {
        for (k=0;k<grdpts_x;k++) {
            free((void *)pow[w][k]);
        }
        free((void *)pow[w]);
    }
    free((void *)pow);
    free((void *)maxpow);
    free((void *)nomin);
#ifdef USE_SINE_TABLE
    free((void *)sine_table);
#endif
    return 0;
}


int generalizedBeamformer(const cplx * const steer, const cplx * const Rptr,
        const double flow, const double fhigh, const double digfreq,
        const int nsamp, const int nstat, const int prewhiten, const int grdpts_x,
        const int grdpts_y, const int nfft, const int nf, double dpow,
        int *ix, int *iy, double *absmax, double *relmax, const int method) {
    /* method: 1 == "bf, 2 == "capon"
     * start the code -------------------------------------------------
     * This assumes that all stations and components have the same number of
     * time samples, nt */

    int x, y, i, j, n;
    cplx bufi;
    cplx bufj;
    cplx cplx_zero = {0., 0.};
    double *p;
    double *abspow;
    double *relpow;
    double *white;
    double power;

    /* we allocate the taper buffer, size nsamp! */
    p = (double *) calloc((size_t) (grdpts_x * grdpts_y * nf), sizeof(double));
    if (p == NULL ) {
        fprintf(stderr, "\nMemory allocation error (taper)!\n");
        exit(EXIT_FAILURE);
    }
    abspow = (double *) calloc((size_t) (grdpts_x * grdpts_y), sizeof(double));
    if (abspow == NULL ) {
        fprintf(stderr, "\nMemory allocation error (taper)!\n");
        exit(EXIT_FAILURE);
    }
    relpow = (double *) calloc((size_t) (grdpts_x * grdpts_y), sizeof(double));
    if (relpow == NULL ) {
        fprintf(stderr, "\nMemory allocation error (taper)!\n");
        exit(EXIT_FAILURE);
    }
    white = (double *) calloc((size_t) nf, sizeof(double));
    if (white == NULL ) {
        fprintf(stderr, "\nMemory allocation error (taper)!\n");
        exit(EXIT_FAILURE);
    }

    if (method == 2) {
        /* P(f) = 1/(e.H R(f)^-1 e) */
        dpow = 1.0;  // needed for general way of abspow normalization
    }
    /* if "bf"
     *   P(f) = e.H R(f) e */
    for (x = 0; x < grdpts_x; ++x) {
        for (y = 0; y < grdpts_y; ++y) {
            ABSPOW(x, y) = 0.;
            for (n = 0; n < nf; ++n) {
                bufi = cplx_zero;
                for (i = 0; i < nstat; ++i) {
                    bufj = cplx_zero;
                    for (j = 0; j < nstat; ++j) {
                        bufj.re += RPTR(n,i,j).re * STEER(x,y,n,j).re - RPTR(n,i,j).im * (-STEER(x,y,n,j).im);
                        bufj.im += RPTR(n,i,j).re * (-STEER(x,y,n,j).im) + RPTR(n,i,j).im * STEER(x,y,n,j).re;
                    }
                    bufi.re += STEER(x,y,n,i).re * bufj.re - STEER(x,y,n,i).im * bufj.im;
                    bufi.im += STEER(x,y,n,i).re * bufj.im + STEER(x,y,n,i).im * bufj.re;
                }

                power = sqrt(bufi.re * bufi.re + bufi.im * bufi.im);
                if (method == 2) {
                    power = 1. / power;
                }
                if (prewhiten == 0) {
                    ABSPOW(x,y) += power;
                }
                if (prewhiten == 1) {
                    P(x,y,n) = power;
                }
            }
            if (prewhiten == 0) {
                RELPOW(x,y) = ABSPOW(x,y)/dpow;
            }
            if (prewhiten == 1) {
                for (n = 0; n < nf; ++n) {
                    if (P(x,y,n) > white[n]) {
                        white[n] = P(x,y,n);
                    }
                }
            }
        }
    }

    if (prewhiten == 1) {
        for (x = 0; x < grdpts_x; ++x) {
            for (y = 0; y < grdpts_y; ++y) {
                RELPOW(x,y)= 0.;
                for (n = 0; n < nf; ++n) {
                    RELPOW(x,y) += P(x,y,n)/(white[n]*nf*nstat);
                }
                if (method == 1) {
                    ABSPOW(x,y) = 0.;
                    for (n = 0; n < nf; ++n) {
                        ABSPOW(x,y) += P(x,y,n);
                    }
                }
            }
        }
    }

    *relmax = 0.;
    *absmax = 0.;
    for (x = 0; x < grdpts_x; ++x) {
        for (y = 0; y < grdpts_y; ++y) {
            if (RELPOW(x,y) > *relmax) {
                *relmax = RELPOW(x,y);
                *ix = x;
                *iy = y;
            }
            if (ABSPOW(x,y) > *absmax) {
                *absmax = ABSPOW(x,y);
            }
        }
    }

    free((void *) p);
    free((void *) relpow);
    free((void *) abspow);
    free((void *) white);

    return 0;
}
