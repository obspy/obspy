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


/************************************************************************/
/* numpy fftpack_lite definitions, must be linked with fftpack_lite.so  */
/* e.g. link with -L/path/to/fftpack_lite -l:fftpack_lite.so            */       
/* numpy has no floating point transfrom compiled, therefore we need to */
/* cast the result to double                                            */    
/************************************************************************/
extern void rfftf(int N, double* data, double* wrk);
extern void rffti(int N, double* wrk);


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


int bbfk(int *spoint, int offset, double **trace, int *ntrace, 
         float ***stat_tshift_table, float *abs, float *rel, int *ix, 
         int *iy, float flow, float fhigh, float digfreq, int nsamp,
         int nstat, int prewhiten, int grdpts_x, int grdpts_y, int nfft) {
    int		j,k,l,w;
    int		n;
    int		wlow,whigh;
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
    float	sumre;
    float	sumim;
    float	wtau;
    double	absval;
    float	maxinmap = 0.;
    double  *fftpack_work = 0;

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
    fftpack_work = (double *)calloc((2*nfft+15), sizeof(double));
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

    /****************************************************************************/
    /* first we need the fft'ed window of traces for the stations of this group */
    /****************************************************************************/
    window = (double **)calloc(nstat, sizeof(double *));
    /* we allocate the taper buffer, size nsamp! */
    taper = (double *)calloc(nsamp, sizeof(double));
    cosine_taper(taper, nsamp, 0.1); 
    for (j=0;j<nstat;j++) {
        /* be sure we are inside our memory */
        if ((spoint[j] + offset + nsamp) > ntrace[j]) {
            free((void *)window);
            free((void *)taper);
            free((void *)fftpack_work);
            return 1;
        }
        /* doing calloc is automatically zero-padding, too */
        /* mimic realft, allocate an extra word/index in the array dimensions */
        window[j] = (double *)calloc(nfft+1, sizeof(double));
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
    nomin = (float **)calloc(grdpts_x, sizeof(float *));
    for (k=0;k<grdpts_x;k++) {
        nomin[k]  = (float *)calloc(grdpts_y, sizeof(float));
    }
    maxpow = (float *)calloc(nfft/2, sizeof(float));
    pow = (float ***)calloc(nfft/2, sizeof(float **));
    for (w=0;w<nfft/2;w++) {
        pow[w] = (float **)calloc(grdpts_x, sizeof(float *));
        for (k=0;k<grdpts_x;k++) {
            pow[w][k] = (float *)calloc(grdpts_y, sizeof(float));
        }
    }

    /*************************************************************/
    /* we start with loop over angular frequency, this allows us */
    /* to prewhiten the fk map, if we want to			 */
    /*************************************************************/
    for (w=wlow;w<=whigh;w++) {
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
                sumre = sumim = 0.;
                for (j=0;j<nstat;j++) {
                    wtau = (float) (2.*M_PI*df*(float)w*stat_tshift_table[j][k][l]);
                    re = window[j][2*w];
                    im = window[j][2*w+1];
                    sumre += (float) (re*cos(wtau)-im*sin(wtau));
                    sumim += (float) (im*cos(wtau)+re*sin(wtau));
                }
                pow[w][k][l] = (sumre*sumre+sumim*sumim);
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
    return 0;
}
