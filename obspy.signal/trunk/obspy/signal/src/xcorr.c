#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <memory.h>

void X_corr(float *tr1, float *tr2, int param, int ndat1, int ndat2, int *shift, double* coe_p)
{
    int a, b;
    int len;
    float *tra1;
    float *tra2;
    double *corp;
    double sum1;
    double sum2;
    int lmax=0;
    int imax=0;
    int flag=0;
    double cmax;
    double sum;
    int max=0;
    int eff_lag;

    corp = (double *)calloc((2*param+1), sizeof(double));
    if (corp == NULL) 
    {
        fprintf(stderr,"\nMemory allocation error!\n");
        exit(0);
    }
    tra1 = (float *)calloc(ndat1, sizeof(float));
    if (tra1 == NULL) 
    {
        fprintf(stderr,"\nMemory allocation error!\n");
        exit(0);
    }
    tra2 = (float *)calloc(ndat2, sizeof(float));
    if (tra2 == NULL) 
    {
        fprintf(stderr,"\nMemory allocation error!\n");
        exit(0);
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
            tra1[a] = tr1[a] - sum;
        }
        flag = 0;
        if(sum == 0.0)
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
            tra1[a] = tra1[a]/cmax;
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
            tra2[a] = tr2[a] - sum;
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
            tra2[a] = tra2[a]/cmax;
        }
        if(sum == 0.0)
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
    free((char *)corp);
    free((char *)tra1);
    free((char *)tra2);
}

void X_corr_3C(float *tr1, float *tr2, float *tr3, float *trA, float *trB,
               float *trC, int param, int ndat1, int ndat2, int ndat3,
               int ndatA, int ndatB, int ndatC, int *shift, double* coe_p)
{
    /* Correlation is done in the following way:
     * tr1 is cc-ed with trA, tr2 with trB and tr3 with trC.
     * The normal scenario would be a three component dataset e.g. seismogram
     * on Z, N, E components. The three separately computed correlations are
     * then stacked at the end such that a cc value of 1.0 means perfect
     * correlation on all three components simultaneously.
     * At the moment all 6 traces have to be of the same length. */

    int a, b;
    int len;
    float *tra1;
    float *traA;
    double *corp;
    double *corp_tmp;
    double sum1;
    double sum2;
    int lmax=0;
    int imax=0;
    double cmax;
    double sum;
    int max=0;
    int eff_lag;

    /* Forbid differing trace lengths */
    if ((ndat1 != ndat2) || (ndat1 != ndat3) || (ndat1 != ndatA) ||
        (ndat1 != ndatB) || (ndat1 != ndatC))
    {
        fprintf(stderr,"Error: All traces have to be the same length!\n");
        exit(0);
    }

    corp = (double *)calloc((2*param+1), sizeof(double));
    if (corp == NULL) 
    {
        fprintf(stderr,"Memory allocation error! (*corp)\n");
        exit(0);
    }
    /* Add 2 more places to store cc on components 2&3 
     * This could be done a lot nicer, I'm sure... */
    corp_tmp = (double *)calloc((2*param+1), sizeof(double));
    if (corp_tmp == NULL) 
    {
        fprintf(stderr,"Memory allocation error! (*corp_tmp)\n");
        exit(0);
    }
    tra1 = (float *)calloc(ndat1, sizeof(float));
    if (tra1 == NULL) 
    {
        fprintf(stderr,"Memory allocation error! (*tra1)\n");
        exit(0);
    }
    traA = (float *)calloc(ndatA, sizeof(float));
    if (traA == NULL) 
    {
        fprintf(stderr,"Memory allocation error! (*traA)\n");
        exit(0);
    }

    /* Determing the maximum 'usable' window */
    if (ndat1 > ndatA)
    {
        len = ndatA - 2*param;
    }
    else
    {
        len = ndat1 - 2*param;
    }
    eff_lag = param;
    if (len <= 0)
    {
        eff_lag /= 2;
        len = ndatB - 2*eff_lag;
    }	 
    if (len <= (eff_lag/2))
    {
        printf("Warning!  window is too small! \n");
    }
    else
    {   
        /* WORK ON COMPONENT 1 */
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
            tra1[a] = tr1[a] - sum;
        }
        for (a=0;a<ndat1;a++)
        {
            if (fabs(tra1[a]) > cmax)
            {
                cmax = fabs(tra1[a]);
            }
        }
        for (a=0;a<ndat1;a++)
        {
            tra1[a] /= cmax;
        }
        cmax = 0;
        sum = 0;
        for (a=0;a<ndatA;a++)
        {
            sum += trA[a];
        }
        sum /= ndatA;
        for (a=0;a<ndatA;a++)
        {
            traA[a] = trA[a] - sum;
        }
        for (a=0;a<ndatA;a++)
        {
            if (fabs(traA[a]) > cmax)
            {
                cmax = fabs(traA[a]);
            }
        }
        for (a=0;a<ndatA;a++)
        {
            traA[a] = traA[a]/cmax;
        }
        
        /* xcorr ... */
        cmax = 0;
        /*                                                                         */
        /*I made some changes to ensure the correct values if the time lag is large*/
        /* !!!!!!!! These changes are only valid if ndat1 = ndat2 !!!!!!           */
        /*                                                                         */

        for (a=0;a<(2*eff_lag+1);a++) /* <------------- */
        {
            corp[a]= 0;
            if((eff_lag-a)>= 0)
            {
                for (b=0;b<(ndat1 - (eff_lag-a));b++)
                {
                    corp[a] += traA[b+eff_lag-a] * tra1[b];
                }  /* for b to .. */
            }else{
                for (b=0;b<(ndat1 - (a-eff_lag));b++)
                {
                    corp[a] += tra1[b+a-eff_lag] * traA[b];
                }  /* for b to .. */
            }
        } /* for a to .. */
        sum1 = sum2 = 0.0;

        /* normalize xcorr function */
        for(a=0; a<ndat1;a++)
        {
            sum1 += (*(tra1+a))*(*(tra1+a));
            sum2 += (*(traA+a))*(*(traA+a));
        }
        sum1 = sqrt(sum1);
        sum2 = sqrt(sum2);
        cmax = 1/(sum1*sum2);
        for (a=0;a<(2*eff_lag+1);a++)
        {
            corp[a] *= cmax;
        }

        /* WORK ON COMPONENT 2 */
        /* Next trace */
        cmax = 0;
        sum = 0;
        for (a=0;a<ndat2;a++)
        {
            sum += tr2[a];
        }
        sum /= ndat2; /*---- here I have made a doubtfull change */
        for (a=0;a<ndat2;a++)
        {
            tra1[a] = tr2[a] - sum;
        }
        for (a=0;a<ndat2;a++)
        {
            if (fabs(tra1[a]) > cmax)
            {
                cmax = fabs(tra1[a]);
            }
        }
        for (a=0;a<ndat2;a++)
        {
            tra1[a] /= cmax;
        }
        cmax = 0;
        sum = 0;
        for (a=0;a<ndatB;a++)
        {
            sum += trB[a];
        }
        sum /= ndatB;
        for (a=0;a<ndatB;a++)
        {
            traA[a] = trB[a] - sum;
        }
        for (a=0;a<ndatB;a++)
        {
            if (fabs(traA[a]) > cmax)
            {
                cmax = fabs(traA[a]);
            }
        }
        for (a=0;a<ndatB;a++)
        {
            traA[a] = traA[a]/cmax;
        }
        
        /* xcorr ... */
        cmax = 0;
        /*                                                                         */
        /*I made some changes to ensure the correct values if the time lag is large*/
        /* !!!!!!!! These changes are only valid if ndat1 = ndat2 !!!!!!           */
        /*                                                                         */

        for (a=0;a<(2*eff_lag+1);a++) /* <------------- */
        {
            corp_tmp[a]= 0;
            if((eff_lag-a)>= 0)
            {
                for (b=0;b<(ndat2 - (eff_lag-a));b++)
                {
                    corp_tmp[a] += traA[b+eff_lag-a] * tra1[b];
                }  /* for b to .. */
            }else{
                for (b=0;b<(ndat2 - (a-eff_lag));b++)
                {
                    corp_tmp[a] += tra1[b+a-eff_lag] * traA[b];
                }  /* for b to .. */
            }
        } /* for a to .. */
        sum1 = sum2 = 0.0;

        /* normalize xcorr function */
        for(a=0; a<ndat2;a++)
        {
            sum1 += (*(tra1+a))*(*(tra1+a));
            sum2 += (*(traA+a))*(*(traA+a));
        }
        sum1 = sqrt(sum1);
        sum2 = sqrt(sum2);
        cmax = 1/(sum1*sum2);
        for (a=0;a<(2*eff_lag+1);a++)
        {
            corp_tmp[a] *= cmax;
            corp[a] += corp_tmp[a]; // stack cc of component 2 on top of 1
        }

        /* WORK ON COMPONENT 3 */
        /* Next trace */
        cmax = 0;
        sum = 0;
        for (a=0;a<ndat3;a++)
        {
            sum += tr3[a];
        }
        sum /= ndat3; /*---- here I have made a doubtfull change */
        for (a=0;a<ndat3;a++)
        {
            tra1[a] = tr3[a] - sum;
        }
        for (a=0;a<ndat3;a++)
        {
            if (fabs(tra1[a]) > cmax)
            {
                cmax = fabs(tra1[a]);
            }
        }
        for (a=0;a<ndat3;a++)
        {
            tra1[a] /= cmax;
        }
        cmax = 0;
        sum = 0;
        for (a=0;a<ndatC;a++)
        {
            sum += trC[a];
        }
        sum /= ndatC;
        for (a=0;a<ndatC;a++)
        {
            traA[a] = trC[a] - sum;
        }
        for (a=0;a<ndatC;a++)
        {
            if (fabs(traA[a]) > cmax)
            {
                cmax = fabs(traA[a]);
            }
        }
        for (a=0;a<ndatC;a++)
        {
            traA[a] = traA[a]/cmax;
        }
        
        /* xcorr ... */
        cmax = 0;
        /*                                                                         */
        /*I made some changes to ensure the correct values if the time lag is large*/
        /* !!!!!!!! These changes are only valid if ndat1 = ndat2 !!!!!!           */
        /*                                                                         */

        for (a=0;a<(2*eff_lag+1);a++) /* <------------- */
        {
            corp_tmp[a]= 0;
            if((eff_lag-a)>= 0)
            {
                for (b=0;b<(ndat3 - (eff_lag-a));b++)
                {
                    corp_tmp[a] += traA[b+eff_lag-a] * tra1[b];
                }  /* for b to .. */
            }else{
                for (b=0;b<(ndat3 - (a-eff_lag));b++)
                {
                    corp_tmp[a] += tra1[b+a-eff_lag] * traA[b];
                }  /* for b to .. */
            }
        } /* for a to .. */
        sum1 = sum2 = 0.0;

        /* normalize xcorr function */
        for(a=0; a<ndat3;a++)
        {
            sum1 += (*(tra1+a))*(*(tra1+a));
            sum2 += (*(traA+a))*(*(traA+a));
        }
        sum1 = sqrt(sum1);
        sum2 = sqrt(sum2);
        cmax = 1/(sum1*sum2);
        for (a=0;a<(2*eff_lag+1);a++)
        {
            corp_tmp[a] *= cmax;
            corp[a] += corp_tmp[a]; // stack cc of component 3 on top of 1+2
        }


        /* LAST STEPS */
        /* finally, estimate maximum and position of maximum */
        cmax = 0.0;
        for (a=0;a<(2*eff_lag+1);a++)
        {
            /* in the last step also divide by three to get the scaling back
             * to 0-1 */
            corp[a] /= 3.0;
            if (fabs(corp[a]) > cmax)
            {
                cmax = fabs(corp[a]);
                imax = a-eff_lag;
                max = a;
            }
        }
        *shift = imax;
        *coe_p = corp[max];
    }
    free((char *)corp);
    free((char *)corp_tmp);
    free((char *)tra1);
    free((char *)traA);
}
