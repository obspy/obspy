/*--------------------------------------------------------------------
# Filename: arpicker.c
#  Purpose: Auto Regressive Picker
#   Author: Joachim Wassermann
# Copyright (C) J. Wassermann
#---------------------------------------------------------------------*/
#include "arpicker.h"
#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>


static float calc_aic(double f_err, double b_err) {
    float aic;
    double tmp1;
    double tmp2;
    errno = 0;
    tmp1 = log(f_err*f_err) * -1.0;
    if (errno != 0) {
        fprintf(stderr,"\nError in log calculation for f_err!\n");
    }
    errno = 0;
    tmp2 = log(b_err*b_err);
    if (errno != 0) {
        fprintf(stderr,"\nError in log calculation for f_err!\n");
    }
    aic = (float) (tmp1 - tmp2);
    return aic;
}


int ar_picker(float *tr, float *tr_1, float *tr_2, int ndat, float sample_rate, float f1, float f2, float lta_p, float sta_p, float lta_s, float sta_s, int m1_p, int m1_s, float *ptime, float *stime, double l_p, double l_s, int s_pick)
{
    float *buff1=NULL,*buff2=NULL,*buff3=NULL,*buff4=NULL;
    float *buff1_s=NULL,*buff4_s=NULL;
    float *f_error=NULL,*b_error=NULL;
    float *buf_sta=NULL,*buf_lta=NULL;
    float *ar_f=NULL,*ar_b=NULL;
    float env_max;
    float f,b,lta_max,stlt;
    float u;
    float aic,aic_mini;
    float pm;
    float buff4_max;
    int i1=0,i2=0,i3=0,i4=0,i5=0,i6=0,i7=0;
    int i,j,k;
    int nsta,nlta;
    int nl_p,nl_s;
    int n65,n32;
    int trace_flag=0;
    int errcode = 0;

#define EXIT(code) \
    free(buff1); \
    free(buff1_s); \
    free(buff2); \
    free(buff3); \
    free(buff4); \
    free(buff4_s); \
    free(f_error); \
    free(b_error); \
    free(ar_f); \
    free(ar_b); \
    free(buf_sta); \
    free(buf_lta); \
    return code;

    *ptime = 0.0f;
    *stime = 0.0f;
    buff1 = (float *)calloc(ndat,sizeof(float));
    if (buff1 == NULL) {
        EXIT(1);
    }
    buff1_s = (float *)calloc(ndat,sizeof(float));
    if (buff1_s == NULL) {
        EXIT(2);
    }
    buff2 = (float *)calloc(ndat,sizeof(float));
    if (buff2 == NULL) {
        EXIT(3);
    }
    buff3 = (float *)calloc(ndat,sizeof(float));
    if (buff3 == NULL) {
        EXIT(4);
    }
    buff4 = (float *)calloc(ndat,sizeof(float));
    if (buff4 == NULL) {
        EXIT(5);
    }
    buff4_s = (float *)calloc(ndat,sizeof(float));
    if (buff4_s == NULL) {
        EXIT(6);
    }
    f_error = (float *)calloc(ndat,sizeof(float));
    if (f_error == NULL) {
        EXIT(7);
    }
    b_error = (float *)calloc(ndat,sizeof(float));
    if (b_error == NULL) {
        EXIT(8);
    }
    ar_f = (float *)calloc(ndat/2,sizeof(float));
    if (ar_f == NULL) {
        EXIT(9);
    }
    ar_b = (float *)calloc(ndat/2,sizeof(float));
    if (ar_b == NULL) {
        EXIT(10);
    }
    memcpy(buff1,tr,ndat*sizeof(float));
    memcpy(buff1_s,tr_1,ndat*sizeof(float));
    memcpy(buff4_s,tr_2,ndat*sizeof(float));

    // we follow the recipes of Akazawa and Leonardt & Kennet

    // first we apply a bandpassfilter for lta sta trigger of p wave
    spr_bp_fast_bworth(buff1,ndat,1/sample_rate,f1,f2,2,1);
    memcpy(buff4,buff1,ndat*sizeof(float));

    spr_bp_fast_bworth(buff1_s,ndat,1/sample_rate,f1,f2,2,1);
    spr_bp_fast_bworth(buff4_s,ndat,1/sample_rate,f1,f2,2,1);
    // estimate which of the horizontals has the maximum
    buff4_max = 0.0;
    for(i=0;i<ndat;i++){
        if(fabsf(buff1_s[i]) > buff4_max){
            buff4_max = fabsf(buff1_s[i]);
            trace_flag = 1;
        }
        if(fabsf(buff4_s[i]) > buff4_max){
            buff4_max = fabsf(buff4_s[i]);
            trace_flag = 2;
        }
    }

    if (trace_flag == 2){
        memcpy(buff1_s,buff4_s,ndat*sizeof(float));
    }else{
        memcpy(buff4_s,buff1_s,ndat*sizeof(float));
    }


    // estimate the maximum of the trace
    buff4_max = 0.0;
    for(i=0;i<ndat;i++){
        if(fabsf(buff4[i]) > buff4_max)
            buff4_max = fabsf(buff4[i]);
    }

    // next cummulative envelope function
    env_max = 0.0;
    for(i=0;i<ndat;i++){
        buff2[i] = fabsf(buff4[i])/buff4_max - buff4[i]*buff4[i]/(buff4_max*buff4_max);
        if(i == 0){
            buff3[0] = buff2[i];
        }else{
            if(buff2[i] > buff2[i-1])
                buff3[i] = buff2[i];
            else
                buff3[i] = buff2[i-1];
        }

        if(buff3[i] > env_max){
            env_max = buff3[i];
            i1 = i;
        }
    }


    // now sta/lta
    nsta = (int)(sta_p*sample_rate);
    nlta = (int)(lta_p*sample_rate);
    stlt = 0.;
    buf_sta = (float *)calloc(ndat,sizeof(float));
    if (buf_sta == NULL) {
        EXIT(11);
    }
    buf_lta = (float *)calloc(ndat,sizeof(float));
    if (buf_lta == NULL) {
        EXIT(12);
    }

    for(i=0;i<(i1-nlta);i++){
        for(j=(i+nlta-nsta);j<(i+nlta);j++){
            buf_sta[i+nlta] += buff3[j]/(float)nsta;
        }
        for(j=(i);j<(i+nlta);j++){
            buf_lta[i+nlta] += buff3[j]/(float)nlta;
        }
        if(buf_lta[i+nlta]>0. && (buf_sta[i+nlta]/buf_lta[i+nlta]) > stlt){
            stlt = buf_sta[i+nlta]/buf_lta[i+nlta];
            i2 = i+nlta;
        }

    } 

    nl_p = (int)(l_p*sample_rate);
    nl_s = (int)(l_s*sample_rate);
    if((i2+nl_p+m1_p)>ndat){
        i2 -= nl_p +m1_p;
    }
    i2 += m1_p + nl_p;

    // Leonard & Kennett AIC Criterion - Model2
    for(i=0;i<(i2+nl_p);i++){
        buff1[i] *= buff1[i]*buff1[i];
    }
    //flip the trace in time
    for(i=0;i<(i2+nl_p);i++){
        buff2[i2+nl_p-i-1] = buff1[i];
    }

    errcode = spr_coef_paz(buff1-1,nl_p,m1_p,&pm,ar_f-1);
    if (errcode != 0) {
        EXIT(errcode);
    }
    errcode = spr_coef_paz(buff2-1,nl_p,m1_p,&pm,ar_b-1);
    if (errcode != 0) {
        EXIT(errcode);
    }

    //estimating the Forward-BackwardAIC 
    for(i=m1_p;i<i2-nl_p;i++){
        for(k=0;k<nl_p;k++){
            f = buff1[i+k];
            for(j=0;j<m1_p;j++){
                f -= (buff1[i+k-j]*ar_f[j]); 
            }
            u = f*f * 1.0f/(float)nl_p;
            f_error[i+nl_p] += u;
        }
    }
    //estimating the Backward-AIC 
    for(i=m1_p;i<(i2-nl_p);i++){
        for(k=0;k<nl_p;k++){
            b = buff2[i+k];
            for(j=0;j<m1_p;j++){
                b -= (buff2[i+k-j]*ar_b[j]); 
            }
            u = b*b * 1.0f/(float)nl_p;
            b_error[i+nl_p] += u;
        }
    }
    // Joint AIC forward and backward looking for minimum
    aic_mini = 0.;
    for(i=(m1_p+nl_p),j=(i2-1-m1_p-nl_p);i<=(i2-1-m1_p-nl_p) && j>=(m1_p+nl_p);i++,j--){
        aic = calc_aic(f_error[i], b_error[j]);
        if(aic < aic_mini && b_error[j] > 0. && f_error[i] >0.){
            aic_mini = aic;
            i3 = i;
        }
    }

    memset(f_error,0,ndat*sizeof(float));
    memset(b_error,0,ndat*sizeof(float));

    // changed to rounding, Moritz
    n32 = (int)fmin((i2-i3)*2.0+m1_p+nl_p + 0.5, ndat);
    i3 = i2-n32;
    if(i3 < 0)
        i3 = 0;

    // Leonard & Kennett AIC on original traces
    for(i=0,j=i3;i<n32 && j<i2;i++,j++){
        buff1[i] = tr[j]*tr[j]*tr[j];
    }
    //flip the trace in time
    for(i=0;i<n32;i++){
        buff2[n32-i-1] = buff1[i];
    }

    errcode = spr_coef_paz(buff1-1,nl_p,m1_p,&pm,ar_f-1);
    if (errcode != 0) {
        EXIT(errcode);
    }
    errcode = spr_coef_paz(buff2-1,nl_p,m1_p,&pm,ar_b-1);
    if (errcode != 0) {
        EXIT(errcode);
    }

    //estimating the Forward-AIC 
    for(i=m1_p;i<(n32-nl_p);i++){
        for(k=0;k<nl_p;k++){
            f = buff1[i+k];
            for(j=0;j<m1_p;j++){
                f -= (float)(buff1[i+k-j]*ar_f[j]); 
            }
            u = f*f * 1.0f/(float)nl_p;
            f_error[i+nl_p] += u;
        }
    }
    //estimating the Backward-AIC 
    for(i=m1_p;i<(n32-nl_p);i++){
        for(k=0;k<nl_p;k++){
            b = buff2[i+k];
            for(j=0;j<m1_p;j++){
                b -= (float)(buff2[i+k-j]*ar_b[j]); 
            }
            u = b*b * 1.0f/(float)nl_p;
            b_error[i+nl_p] += u;
        }
    }
    // Joint AIC forward and backward looking for minimum
    aic_mini = 0.;
    for(i=(m1_p+nl_p),j=(n32-1-m1_p-nl_p);i<=(n32-1-m1_p-nl_p) && j>=(m1_p+nl_p);i++,j--){
        aic = calc_aic(f_error[i], b_error[j]);
        if(aic < aic_mini && b_error[j] > 0. && f_error[i] >0.){
            aic_mini = aic;
            i4 = i+i3;
        }
    }

    // estimation of P-Onset
    *ptime = ((float) (i4-nl_p))/sample_rate;

    if(s_pick == 1){
        memset(f_error,0,ndat*sizeof(float));
        memset(b_error,0,ndat*sizeof(float));
        memset(buf_sta,0,ndat*sizeof(float));
        memset(buf_lta,0,ndat*sizeof(float));

        /* let's try this for the moment */

        // now sta-lta
        nsta = (int)(sta_s*sample_rate);
        nlta = (int)(lta_s*sample_rate);
        stlt = 0.;
        for(i=0;i<(ndat-nlta);i++){
            for(j=(i+nlta-nsta);j<(i+nlta);j++){
                buf_sta[i+nlta] += fabsf(buff4_s[j])/(float)nsta;
            }
            for(j=(i);j<(i+nlta);j++){
                buf_lta[i+nlta] += fabsf(buff4_s[j])/(float)nlta;
            }
        }

        // estimation of STA-LTA on horizontal component
        lta_max = 0.;
        for(i = i4;i<(ndat-nlta);i++){
            if((buf_sta[i+nlta] - buf_lta[i+nlta])>lta_max){
                lta_max = buf_sta[i+nlta] - buf_lta[i+nlta];
                i5 = i+nlta;
            }
        }
        i5 += m1_s + nl_s;
        i6 = 0;
        if (i5 > ndat)
            i5 = ndat;
        // we try this for now
        //   i6 = i4 - m1_s - nl_s;
#if 0
#endif
        // STA-LTA in reversed direction
        memset(buf_sta,0,ndat*sizeof(float));
        memset(buf_lta,0,ndat*sizeof(float));
        for(i=(ndat-nlta-1);i>nlta;i--){
            for(j=(i+nsta-1);j>=(i);j--){
                buf_sta[i] += fabsf(buff4_s[j])/(float)nsta;
            }
            for(j=(i+nlta-1);j>=(i);j--){
                buf_lta[i] += fabsf(buff4_s[j])/(float)nlta;
            }
        }
        lta_max = 0.;
        for(i=(i5+nlta);i>=i4;i--){
            if(i < i5 && (buf_sta[i-nlta] - buf_lta[i-nlta])<lta_max){
                lta_max = buf_sta[i-nlta] - buf_lta[i-nlta];
                i6 = i-nlta;
            }
        }

        if(i6 > 0){  
            n65 = (i5-i6);

            if(n65 > ndat)
                n65 = 0;

            // AIC on wideband record
            for(i=0,j=i6;i<n65 && j<i5;i++,j++){
                buff1_s[i] = buff4_s[j]*buff4_s[j]*buff4_s[j];
            }
            //flip the trace in time
            for(i=0;i<n65;i++){
                buff2[n65-i-1] = buff1_s[i];
            }

            errcode = spr_coef_paz(buff1_s-1,nl_s,m1_s,&pm,ar_f-1);
            if (errcode != 0) {
                EXIT(errcode);
            }
            errcode = spr_coef_paz(buff2-1,nl_s,m1_s,&pm,ar_b-1);
            if (errcode != 0) {
                EXIT(errcode);
            }

            //estimating the Forward-AIC 
            for(i=m1_s;i<(n65-nl_s);i++){
                for(k=0;k<nl_s;k++){
                    f = buff1_s[i+k];
                    for(j=0;j<m1_s;j++){
                        f -= (buff1_s[i+k-j]*ar_f[j]); 
                    }
                    u = f*f * 1.0f/(float)nl_s;
                    f_error[i+nl_s] += u;
                }
            }
            //estimating the Backward-AIC 
            for(i=m1_s;i<(n65-nl_s);i++){
                for(k=0;k<nl_s;k++){
                    b = buff2[i+k];
                    for(j=0;j<m1_s;j++){
                        b -= (buff2[i+k-j]*ar_b[j]); 
                    }
                    u = b*b * 1.0f/(float)nl_s;
                    b_error[i+nl_s] += u;
                }
            }
            // Joint AIC forward and backward looking for minimum
            aic_mini = 0.;
            i7 = 0;
            for(i=(m1_s+nl_s),j=(n65-1-m1_s-nl_s);i<=(n65-1-m1_s-nl_s) && j>=(m1_s+nl_s);i++,j--){
                aic = calc_aic(f_error[i], b_error[j]);
                if(aic < aic_mini && b_error[j] > 0. && f_error[i] >0.){
                    aic_mini = aic;
                    i7 = i+i6;
                }
            }
            // S-onsettime, changed to return seconds
            if(i7 > 0)
                *stime = ((float) (i7-nl_s))/sample_rate;
            else
                *stime = 0.0f;
        }else
            *stime = 0.0f;
    }

    EXIT(0);
}
#undef EXIT
