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

#define STEER(I, J, K, L, M) steer[(I)*2*nf*grdpts_y*grdpts_x + (J)*2*nf*grdpts_y + (K)*2*nf + (L)*2 + (M)]

#define USE_SINE_TABLE

#ifdef USE_SINE_TABLE
#define SINE_REF_LEN 1000
#define SINE_REF_LEN_4 (SINE_REF_LEN / 4)
#define SINE_TABLE_LEN (SINE_REF_LEN + SINE_REF_LEN / 4 + 1)
#endif


/************************************************************************/
/* numpy fftpack_lite definitions, must be linked with fftpack_lite.so  */
/* e.g. link with -L/path/to/fftpack_lite -l:fftpack_lite.so            */       
/* numpy has no floating point transfrom compiled, therefore we need to */
/* cast the result to double                                            */    
/************************************************************************/
extern void rfftf(int N, double* data, double* wrk);
extern void rffti(int N, double* wrk);


/* splint: cannot be static, is exported and used from python */
int cosine_taper(double *taper, int ndat, double fraction)
{
    int	i1,i2,i3,i4,k;
    double fact,temp;

    /* get i1-4 out of ndat and fraction */
    i1 = 0;
    i4 = ndat-1;
    i2 = (int)(fraction*(double)ndat +0.5);
    if ( (double)i2 > (double)(ndat-1)/2. ) {
        i2 = (int)((double)(ndat-1)/2.);
    }
    i3 = ndat - 1 - (int)(fraction*(double)ndat +0.5);
    if ( (double)i3 < (double)(ndat-1)/2. ) {
        i3 = (int)((double)(ndat-1)/2.+1.);
    }

    for (k=0;k<ndat;k++) {
        if ((k <= i1) || (k >= i4)) {
            if ((k == i2) || (k == i3)) {
                taper[k] = 1.0;
            } else {
                taper[k] = 0.0;
            }
        } else if ((k > i1) && (k <= i2)) {
            temp =  M_PI * (double)(k-i1)/((double)(i2-i1+1));
            fact = 0.5 - 0.5*cos(temp);
            taper[k] = (double) fabs(fact);
        } else if ((k >= i3) && (k < i4)) {
            temp = M_PI * (double)(i4-k)/((double)(i4-i3+1));
            fact = 0.5 - 0.5*cos(temp);
            taper[k] = fabs(fact);
        } else
            taper[k] = 1.0;
    }
    return 0;
}


void calcSteer(int nstat, int grdpts_x, int grdpts_y, int nf, int nlow,
               float deltaf, float ***stat_tshift_table, double *steer) {
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
                    STEER(i,x,y,n,0) = cos(wtau);
                    STEER(i,x,y,n,1) = sin(wtau);
                }
            }
        }
    }
}


int bbfk(int *spoint, int offset, double **trace, int *ntrace, 
         float ***stat_tshift_table, float *abs, float *rel, int *ix, 
         int *iy, float flow, float fhigh, float digfreq, int nsamp,
         int nstat, int prewhiten, int grdpts_x, int grdpts_y, int nfft) {
    int		j,k,l,w;
    int		n;
    int		wlow,whigh;
    int     errcode;
    float	df;
    double	**window;
    double  *taper;
    double	mean;
    float	denom = 0;
    float	dpow;
    double	re;
    double	im;
    float	***pow;
    float	**nomin;
    float	*maxpow;
    double	absval;
    float maxinmap = 0.;
    double *fftpack_work = 0;
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
    /***********************************************************************/
    /* do not remove this comment, get next power of two for window length */
    /***********************************************************************
    while (nfft<nsamp) {
        nfft = nfft << 1;
    }*/

    /********************************************************************/
    /* do not remove this comment, it shows how to allocate the plan of */
    /* fftpack with magic number the fftt                               */
    /********************************************************************/
    /* Magic size needed by rffti (see also
     * http://projects.scipy.org/numpy/browser/trunk/
     * +numpy/fft/fftpack_litemodule.c#L277)*/
    fftpack_work = (double *)calloc((size_t) (2*nfft+15), sizeof(double));
    if (fftpack_work == NULL) {
        fprintf(stderr,"\nMemory allocation error (fftpack_work)!\n");
        exit(EXIT_FAILURE);
    }
    rffti(nfft, fftpack_work);

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

    /****************************************************************************/
    /* first we need the fft'ed window of traces for the stations of this group */
    /****************************************************************************/
    window = (double **)calloc((size_t) nstat, sizeof(double *));
    if (window == NULL) {
        fprintf(stderr,"\nMemory allocation error (window)!\n");
        exit(EXIT_FAILURE);
    }
    /* we allocate the taper buffer, size nsamp! */
    taper = (double *)calloc((size_t) nsamp, sizeof(double));
    if (taper == NULL) {
        fprintf(stderr,"\nMemory allocation error (taper)!\n");
        exit(EXIT_FAILURE);
    }
    errcode = cosine_taper(taper, nsamp, 0.1);
    if (errcode != 0) {
        fprintf(stderr,"\nError during cosine tapering!\n");
    }
    for (j=0;j<nstat;j++) {
        /* be sure we are inside our memory */
        if ((spoint[j] + offset + nsamp) > ntrace[j]) {
            free((void *)window);
            free((void *)taper);
            free((void *)fftpack_work);
            free ((void *)sine_table);
            return 1;
        }
        /* doing calloc is automatically zero-padding, too */
        /* mimic realft, allocate an extra word/index in the array dimensions */
        window[j] = (double *)calloc((size_t) (nfft+1), sizeof(double));
        if (window[j] == NULL) {
            fprintf(stderr,"\nMemory allocation error (window[j])!\n");
            exit(EXIT_FAILURE);
        }
        memcpy((void *)(window[j]+1),(void *)(trace[j]+spoint[j]+offset),nsamp*sizeof(double));
        /*************************************************/
        /* 4.6.98, we insert offset removal and tapering */
        /*************************************************/
        mean = 0.;
        for (n=0;n<nsamp;n++) {
            mean += window[j][n];
        }
        mean /= (double)nsamp;
        for (n=0;n<nsamp;n++) {
            window[j][n] -= mean;
            window[j][n] *= taper[n];
        }
        /* mimic realft */
        rfftf(nfft, (double *)(window[j]+1), fftpack_work);
        window[j][0] = window[j][1]; /* mean of signal */
        window[j][1] = window[j][nfft]; /* real part of nyquist frequency */
    }
    /* we free the taper buffer and the fft plan already! */
    free((void *)taper);
    free((void *)fftpack_work);

    /***********************************************************************/
    /* we calculate the scaling factor or denominator, if not prewhitening */
    /***********************************************************************/
    if (prewhiten!=TRUE) {
        denom = 0.;
        for(w=wlow;w<=whigh;w++) {
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
    maxpow = (float *)calloc((size_t) (nfft/2), sizeof(float));
    if (maxpow == NULL) {
        fprintf(stderr,"\nMemory allocation error (maxpow)!\n");
        exit(EXIT_FAILURE);
    }
    pow = (float ***)calloc((size_t) (nfft/2), sizeof(float **));
    if (pow == NULL) {
        fprintf(stderr,"\nMemory allocation error (pow)!\n");
        exit(EXIT_FAILURE);
    }
    for (w=0;w<nfft/2;w++) {
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
    for (w=wlow;w<=whigh;w++) {
    	float PI_2_df_w = 2.*M_PI*df*(float)w;
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
            for (w=wlow;w<=whigh;w++) {
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
    for (w=0;w<nfft/2;w++) {
        for (k=0;k<grdpts_x;k++) {
            free((void *)pow[w][k]);
        }
        free((void *)pow[w]);
    }
    free((void *)pow);
    free((void *)maxpow);
    free((void *)nomin);
    for (j=0;j<nstat;j++) {
        free((void *)window[j]);
    }
    free((void *)window);
#ifdef USE_SINE_TABLE
    free((void *)sine_table);
#endif
    return 0;
}
