
/****************************************************************************************
*       RCS Standard Header
*
*
*$Author: mao $
*
*$State: Exp $
*
*Revision 1.1  2000/04/25 16:11:03  mao
*Initial revision
*
*
*$Locker:  $
*
*
*****************************************************************************************/

#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _USE_MATH_DEFINES  // for Visual Studio
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#define MAX_SEC 10
#define TRUE 1
#define FALSE 0

/**
   NAME: spr_bp_fast_bworth
   SYNOPSIS:
   float flo;          low cut corner frequency
   float fhi;          high cut corner frequency
   int ns;            number of filter sections
   int zph;          TRUE -> zero phase filter
   spr_bp_bworth(header1,header2,flo,fhi,ns,zph);
   DESCRIPTION: Butterworth bandpass filter.
**/
void spr_bp_fast_bworth(float *tr, int ndat, float tsa, float flo, float fhi, int ns, int zph)
{
    int k;                   /* index */
    int n,m,mm;
    static double a[MAX_SEC+1];
    static double b[MAX_SEC+1];
    static double c[MAX_SEC+1];
    static double d[MAX_SEC+1];
    static double e[MAX_SEC+1];
    static double f[MAX_SEC+1][6];

    double temp;
    double c1,c2,c3;
    double w1,w2,wc,q,p,r,s,cs,x;
 
    /* Applying Butterworth bandpass: flo <-> fhi, sections: ns, ZP: zph */
    /* design filter weights */
    /* bandpass */
    w1=sin(flo*M_PI*tsa)/cos(flo*M_PI*tsa);
    w2=sin(fhi*M_PI*tsa)/cos(fhi*M_PI*tsa);
    wc=w2-w1;
    q=wc*wc +2.0*w1*w2;
    s=w1*w1*w2*w2;
    for (k=1;k<=ns;k++)
    {
            c1=(double)(k+ns);
            c2=(double)(4*ns);
            c3=(2.0*c1-1.0)*M_PI/c2;
            cs=cos(c3);
            p = -2.0*wc*cs;
            r=p*w1*w2;
            x=1.0+p+q+r+s;
            a[k]=wc*wc/x;
            b[k]=(-4.0 -2.0*p+ 2.0*r+4.0*s)/x;
            c[k]=(6.0 - 2.0*q +6.0*s)/x;
            d[k]=(-4.0 +2.0*p -2.0*r +4.0*s)/x;
            e[k]=(1.0 - p +q-r +s)/x;
    }
 
    /* set initial values to 0 */
    for(n=0;n<=MAX_SEC;n++)
    {
            for(m=0;m<=5;m++)
            {
                    f[n][m]=0.0;
            }
    }
    /* filtering */
    for (m=1;m<=ndat;m++)
    {
            f[1][5]= *(tr + m-1);
            /* go thru ns filter sections */
            for(n=1;n<=ns;n++)
            {
                    temp=a[n]*(f[n][5]-2.0*f[n][3] +f[n][1]);
                    temp=temp-b[n]*f[n+1][4]-c[n]*f[n+1][3];
                    f[n+1][5]=temp-d[n]*f[n+1][2]-e[n]*f[n+1][1];
            }
            /* update past values */
            for(n=1;n<=ns+1;n++)
            {
                    for(mm=1;mm<=4;mm++)
                    {
                            f[n][mm]=f[n][mm+1];
                    }
            }
            /* set present data value and continue */
            *(tr+m-1) = (float) f[ns+1][5];
    }
    if (zph == TRUE)
    {
        /* filtering reverse signal*/
        for (m=ndat;m>=1;m--)
        {
                f[1][5]= *(tr+m-1);
                /* go thru ns filter sections */
                for(n=1;n<=ns;n++)
                {
                        temp=a[n]*(f[n][5]-2.0*f[n][3] +f[n][1]);
                        temp=temp-b[n]*f[n+1][4]-c[n]*f[n+1][3];
                        f[n+1][5]=temp-d[n]*f[n+1][2]-e[n]*f[n+1][1];
                }
                /* update past values */
                for(n=1;n<=ns+1;n++)
                {
                        for(mm=1;mm<=4;mm++)
                        {
                                f[n][mm]=f[n][mm+1];
                        }
                }
                /* set present data value and continue */
                *(tr+m-1)= (float) f[ns+1][5];
        }
    }
    return;
}

/**
   NAME: spr_hp_fast_bworth
   SYNOPSIS:
   float fc;          corner frequency
   int ns;            number of filter sections
   int zph;          TRUE -> zero phase filter
   spr_hp_bworth(header1,header2,fc,ns,zph);
   DESCRIPTION: Butterworth highpass filter.
**/
void spr_hp_fast_bworth(float *tr, int ndat, float tsa, float fc, int ns, int zph)
{
    int k;                   /* index */
    int n,m,mm;
    static double a[MAX_SEC+1];
    static double b[MAX_SEC+1];
    static double c[MAX_SEC+1];
    static double f[MAX_SEC+1][6];
 
    double temp;
    double wcp,cs;
 
    /* design filter weights */
    wcp=sin(fc*M_PI*tsa)/cos(fc*M_PI*tsa);
    for (k=1;k<=ns;k++)
    {
            cs=cos((2.0*(k+ns)-1.0)*M_PI/(4.0*ns));
            a[k]=1.0/(1.0+wcp*wcp-2.0*wcp*cs);
            b[k]=2.0*(wcp*wcp-1.0)*a[k];
            c[k]=(1.0 +wcp*wcp +2.0*wcp*cs)*a[k];
    }
 
    /* set initial values to 0 */
    for(n=0;n<=MAX_SEC;n++)
    {
            for(m=0;m<=5;m++)
            {
                    f[n][m]=0.0;
            }
    }
    /* set initial values to 0 */
    /* filtering */
    for (m=1;m<=ndat;m++)
    {
            f[1][3]= *(tr+m-1);
            /* go thru ns filter sections */
            for(n=1;n<=ns;n++)
            {
                    temp=a[n]*(f[n][3]-2.0*f[n][2] +f[n][1]);
                    f[n+1][3]=temp-b[n]*f[n+1][2]-c[n]*f[n+1][1];
            }
            /* update past values */
            for(n=1;n<=ns+1;n++)
            {
                    for(mm=1;mm<=2;mm++)
                    {
                            f[n][mm]=f[n][mm+1];
                    }
            }
            /* set present data value and continue */
            *(tr+m-1)= (float) f[ns+1][3];
    }
    if (zph == TRUE)
    {
        /* filtering reverse signal*/
        for (m=ndat;m>=1;m--)
        {
            f[1][3]= *(tr+m-1);
            /* go thru ns filter sections */
            for(n=1;n<=ns;n++)
            {
                    temp=a[n]*(f[n][3]-2.0*f[n][2] +f[n][1]);
                    f[n+1][3]=temp-b[n]*f[n+1][2]-c[n]*f[n+1][1];
            }
            /* update past values */
            for(n=1;n<=ns+1;n++)
            {
                    for(mm=1;mm<=2;mm++)
                    {
                            f[n][mm]=f[n][mm+1];
                    }
            }
            /* set present data value and continue */
            *(tr+m-1)= (float) f[ns+1][3];
        }
    }
}
 
/**
   NAME: spr_lp_fast_bworth
   SYNOPSIS:
   float fc;          corner frequency
   int ns;            number of filter sections
   int zph;          TRUE -> zero phase filter
   spr_lp_bworth(header1,header2,fc,ns,zph);
   DESCRIPTION: Butterworth lowpass filter.
**/
void spr_lp_fast_bworth(float *tr, int ndat, float tsa, float fc, int ns, int zph)
{
    int k;                       /* index */
    int n,m,mm;
    static double a[MAX_SEC+1];
    static double b[MAX_SEC+1];
    static double c[MAX_SEC+1];
    static double f[MAX_SEC+1][6];
 
    double temp;
    double wcp,cs,x;
 
 
    /* Applying Butterworth lowpass: <- fc, sections: ns, ZP: zph */
    wcp=sin(fc*M_PI*tsa)/cos(fc*M_PI*tsa);
    for (k=1;k<=ns;k++)
    {
            cs=cos((2.0*(k+ns)-1.0)*M_PI/(4.0*ns));
            x=1.0/(1.0+wcp*wcp -2.0*wcp*cs);
            a[k]=wcp*wcp*x;
            b[k]=2.0*(wcp*wcp-1.0)*x;
            c[k]=(1.0 +wcp*wcp +2.0*wcp*cs)*x;
    }
    /* set initial values to 0 */
    for(n=0;n<=MAX_SEC;n++)
    {
            for(m=0;m<=5;m++)
            {
                    f[n][m]=0.0;
            }
    }
    /* set initial values to 0 */
    /* filtering */
    for (m=1;m<=ndat;m++)
    {
            f[1][3]= *(tr+m-1);
            /* go thru ns filter sections */
            for(n=1;n<=ns;n++)
            {
                    temp=a[n]*(f[n][3]+2.0*f[n][2] +f[n][1]);
                    f[n+1][3]=temp-b[n]*f[n+1][2]-c[n]*f[n+1][1];
            }
            /* update past values */
            for(n=1;n<=ns+1;n++)
            {
                    for(mm=1;mm<=2;mm++)
                    {
                            f[n][mm]=f[n][mm+1];
                    }
            }
            /* set present data value and continue */
            *(tr+m-1)= (float) f[ns+1][3];
    }
    if (zph == TRUE)
    {
 
        /* filtering reverse signal*/
        for (m=ndat;m>=1;m--)
        {
            f[1][3]= *(tr+m-1);
            /* go thru ns filter sections */
            for(n=1;n<=ns;n++)
            {
                    temp=a[n]*(f[n][3]+2.0*f[n][2] +f[n][1]);
                    f[n+1][3]=temp-b[n]*f[n+1][2]-c[n]*f[n+1][1];
            }
            /* update past values */
            for(n=1;n<=ns+1;n++)
            {
                    for(mm=1;mm<=2;mm++)
                    {
                            f[n][mm]=f[n][mm+1];
                    }
            }
            /* set present data value and continue */
            *(tr+m-1)= (float) f[ns+1][3];
        }
    }
}
 
/**
 * NAME: spr_time_fast_int()
 *      SYNOPSIS:
 *      DT_HEADER *header2;
 *      spr_time_int(header2)
 * VERSION: 2.0
 * DATE: 15-05-1992
 * DESCRIPTION: Integrate trace in the time domain by summing up
**/
void spr_time_fast_int(float *tr, int ndat, float t_samp)
{
        int i;
 
        *tr *= t_samp/2;
 
        for(i=1;i<(ndat-1);i++)
        {
            *(tr+i) = *(tr+i-1) + *(tr+i) * t_samp;
        }
        *(tr+ndat-1) = *(tr+ndat-2) + *(tr+ndat-1)*t_samp/2;
}


void  decim(float *tr1, int ndat, int ndat2, int dec_ratio, int pos)
{
    int j;
    int max_pos;
    float max;
    float *x;


    /* Applying Integer Decimation with decimation ratio dec_ratio */

    max = fabsf(tr1[0]);
    max_pos = 0;

    for (j = 0; j < ndat;j++)
    {
        if(fabsf(tr1[j]) > max){
            max = fabsf(tr1[j]);
            max_pos = j;
        }
    }
    /* only for negative start positions take the 
       position of the maximum */
    if (pos >= 0) max_pos = pos;  
  
    x  = (float *)calloc(ndat2+1,sizeof(float));
    if (x == NULL) {
        fprintf(stderr,"\nMemory allocation error (x)!\n");
        exit(EXIT_FAILURE);
    }
    for (j = max_pos; j < ndat;j = j + dec_ratio)
    {
        if((j/dec_ratio < ndat2) && (j/dec_ratio>=0))
            x[j/dec_ratio] = tr1[j];
    }
    for (j = max_pos - dec_ratio; j >= 0;j = j - dec_ratio)
    {
        if((j/dec_ratio < ndat2) && (j/dec_ratio>=0))
            x[j/dec_ratio] = tr1[j];
    }
    /* copy data back to tr[] */
    for (j = 0; j < ndat;j++)
        tr1[j] = 0.0;
    for (j = 0; j < ndat2;j++)
        tr1[j] = x[j];
    free(x);
   
}

int spr_coef_paz(float *tr,int n,int m,/*@out@*/ float *fp,/*@out@*/ float *coef)
{
    int i,j,k;
    float sqr_sum;
    float *extra_tr1;
    float *extra_tr2;
    float *extra_tr3;
    float num;
    float denom;

    extra_tr1 = (float *) calloc(n, sizeof(float));// allocate extra_tr1
    if (extra_tr1 == NULL) {
        return 13;
    }
    extra_tr2 = (float *) calloc(n, sizeof(float));// allocate extra_tr2
    if (extra_tr2 == NULL) {
        free(extra_tr1);
        return 14;
    }
    extra_tr3 = (float *) calloc(m, sizeof(float));// allocate extra_tr3
    if (extra_tr3 == NULL) {
        free(extra_tr1);
        free(extra_tr2);
        return 15;
    }

    // calculating mean square
    sqr_sum = 0.0;
    for (j=1;j<=n;j++) {
        sqr_sum += tr[j]*tr[j];
    }
    *fp=sqr_sum/n;

    // filling extra_trace one and two by one difference trace
    extra_tr1[1]=tr[1];
    extra_tr2[n-1]=tr[n];
    for (j=2;j<=n-1;j++) {
        extra_tr1[j]=tr[j];
        extra_tr2[j-1]=tr[j];
    }

    // now the more complicated part, calculating the m coefficients
    // return only when the loop is finished, that is k == m
    for (k=1;k<=m;k++) {
        num=0.;
        denom=0.;
        for (j=1;j<=(n-k);j++) {
            num += extra_tr1[j]*extra_tr2[j];
            denom += extra_tr1[j]*extra_tr1[j]+extra_tr2[j]*extra_tr2[j];
        }
        coef[k]=2.0f*num/denom;
        *fp *= (1.0f-coef[k]*coef[k]);
        if (k != 1) {
            for (i=1;i <= (k-1);i++) {
                coef[i] = extra_tr3[i]-coef[k]*extra_tr3[k-i];
            }
        }
        if (k == m) {
            free(extra_tr1);
            free(extra_tr2);
            free(extra_tr3);
            return 0;
        }
        for (i=1;i<=k;i++) {
            extra_tr3[i] = coef[i];
        }
        for (j=1;j<=(n-k-1);j++) {
            extra_tr1[j] -= extra_tr3[k]*extra_tr2[j];
            extra_tr2[j]  = extra_tr2[j+1]-extra_tr3[k]*extra_tr1[j+1];
        }
    }
    // we should never reach this point
    free(extra_tr1);
    free(extra_tr2);
    free(extra_tr3);
    return -1;
}
