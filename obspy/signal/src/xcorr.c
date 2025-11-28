/*--------------------------------------------------------------------
# Filename: xcorr.c
#  Purpose: Cross correlation in time domain
#   Author: Hansruedi Maurer
#  Changes: Joachim Wassermann
# Copyright (C) 2010 H. Maurer, J. Wassermann
#---------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <memory.h>
#include <float.h>

int X_corr(float *tr1, float *tr2, double *corp, int param, int ndat1, int ndat2, int *shift, double* coe_p)
{
    int a, b;
    int len;
    float *tra1;
    float *tra2;
    double sum1;
    double sum2;
    int lmax=0;
    int imax=0;
    int flag=0;
    double cmax;
    double sum;
    int max=0;
    int eff_lag;

    tra1 = (float *)calloc(ndat1, sizeof(float));
    if (tra1 == NULL) 
    {
        return 1;
    }
    tra2 = (float *)calloc(ndat2, sizeof(float));
    if (tra2 == NULL) 
    {
        free(tra1);
        return 2;
    }

    /* Determing the maximum 'usable' window */
    if (ndat1 > ndat2)
    {
        len = ndat2 - 2*param;
    }
    else
    {
        len = ndat1 - 2*param;
    }
    eff_lag = param;
    if (len <= 0)
    {
        eff_lag /= 2;
        len = ndat2 - 2*eff_lag;
    }	 
    if (len <= (eff_lag/2))
    {
        printf("Warning!  window is too small! \n");
    }
    else
    {
        /* Normalizing the traces  (max amp = 1, zero offset) */
        cmax = 0;
        sum = 0;
        for (a=0;a<ndat1;a++)
        {
            sum += tr1[a];
        }
        sum /= ndat1; /*---- here I have made a doubtfull change */
        for (a=0;a<ndat1;a++)
        {
            tra1[a] = tr1[a] - (float)sum;
        }
        flag = 0;
        if(fabs(sum) < DBL_EPSILON) /* == 0.0 */
            flag = 1;
        for (a=0;a<ndat1;a++)
        {
            if (fabs(tra1[a]) > cmax)
            {
                cmax = fabs(tra1[a]);
            }
        }
        for (a=0;a<ndat1;a++)
        {
            tra1[a] = tra1[a]/(float)cmax;
        }
        cmax = 0;
        sum = 0;
        for (a=0;a<ndat2;a++)
        {
            sum += tr2[a];
        }
        sum /= ndat2;
        for (a=0;a<ndat2;a++)
        {
            tra2[a] = tr2[a] - (float)sum;
        }
        for (a=0;a<ndat2;a++)
        {
            if (fabs(tra2[a]) > cmax)
            {
                cmax = fabs(tra2[a]);
            }
        }
        for (a=0;a<ndat2;a++)
        {
            tra2[a] = tra2[a]/(float)cmax;
        }
        if(fabs(sum) < DBL_EPSILON) /* == 0.0 */
            flag += 1;

        /* xcorr ... */
        cmax = 0;
        /*                                                                         */
        /*I made some changes to ensure the correct values if the time lag is large*/
        /* !!!!!!!! These changes are only valid if ndat1 = ndat2 !!!!!!           */
        /*                                                                         */

        if(flag == 0)
        {
            for (a=0;a<(2*eff_lag+1);a++) /* <------------- */
            {
                corp[a]= 0;
                if((eff_lag-a)>= 0)
                {
                    for (b=0;b<(ndat1 - (eff_lag-a));b++)
                    {
                        corp[a] += tra2[b+eff_lag-a] * tra1[b];
                    }  /* for b to .. */
                }else{
                    for (b=0;b<(ndat1 - (a-eff_lag));b++)
                    {
                        corp[a] += tra1[b+a-eff_lag] * tra2[b];
                    }  /* for b to .. */
                }
                if (fabs(corp[a]) > cmax)
                {
                    cmax = fabs(corp[a]);
                    imax = a-eff_lag;
                    max = a;
                }
                lmax = imax;
            } /* for a to .. */
            sum1 = sum2 = 0.0;

            /* normalize xcorr function */
            for(a=0; a<ndat1;a++)
            {
                sum1 += (*(tra1+a))*(*(tra1+a));
                sum2 += (*(tra2+a))*(*(tra2+a));
            }
            sum1 = sqrt(sum1);
            sum2 = sqrt(sum2);
            cmax = 1/(sum1*sum2);
            for (a=0;a<(2*eff_lag+1);a++)
            {
                corp[a] *= cmax;
            }
            *shift = lmax;
            *coe_p = corp[max];
        }
        else
        {
            *shift = 0;
            *coe_p = 0.0;
        }
    }  /* else */
    free(tra1);
    free(tra2);
    return 0;
}
