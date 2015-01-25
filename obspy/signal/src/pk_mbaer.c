/*--------------------------------------------------------------------
# Filename: pk_mbaer.c
#  Purpose: C-Verions of Baer Picker based on Fortran code of Baer
#   Author: Andreas Rietbrock, Joachim Wassermann
# Copyright (C) A. Rietbrock, J. Wassermann
#---------------------------------------------------------------------*/
#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


static void preset(float *, int , float *, float *, float *, /*@out@*/ float *, /*@out@*/ float *, float *, int *, /*@out@*/ int *, /*@out@*/ int *, /*@out@*/ int *, char *, /*@out@*/ int *, float );
/*
*******************************************************************************
c====================================================================
c
c p-picker routine by m. baer, schweizer. erdbebendienst
c                              institut fur geophysik
c                              eth-honggerberg
c                              8093 zurich
c                              tel. 01-377 26 27
c                              telex: 823480 eheb ch
c
c see paper by m. baer and u. kradolfer: an automatic phase picker
c                              for local and teleseismic events
c                              bssa vol. 77,4 pp1437-1445
c====================================================================
c
c this subroutine can be call successively for subsequent portions
c of a seismogram. all necessary values are passed back to the calling
c program.
c for the initial call, the following variables have to be set:
c
c      y2                    (e.g. = 1.0, or from any estimate)
c      yt                    (e.g. = 1.0, or from any estimate)
c      sdev                  (e.g. = 1.0)
c      rawold                (e.g. = 0.0)
c      tdownmax              (c.f. paper for possible values)
c      tupevent              (c.f. paper for possible values)
c      thrshl1               (e.g. = 10.0)
c      thrshl2               (e.g. = 20.0)
c      ptime                 must be 0
c      num                   must be 0
c
c subroutine parameters:
c reltrc      : time series as floating data, possibly filtered
c trace       : time series as integer data, unfiltered, used to
c               determine the amplitudes
c npts        : number of datapoints in the time series
c rawold      : last datapoint of time series when leaving the subroutine
c ssx         : sum of characteristic function (CF)
c ssx2        : sum of squares of CF
c mean        : mean of CF
c sdev        : sigma of CF
c num         : number of datapoints of the CF used for sdev & mean
c omega       : weighting factor for the derivative of reltrc
c y2          : last value of amplitude**2
c yt          : last value of derivative**2
c ipkflg      : pick flag
c itar        : first triggered sample
c amp         : maximum value of amplitude
c ptime       : sample number of parrival
c pamp        : maximum amplitude within trigger periode
c preptime    : a possible earlier p-pick
c ifrst       : direction of first motion (1 or -1)
c pfm         :     "          "     "    (U or D)
c dtime       : counts the duration where CF dropped below thrshl1
c noise       : maximum noise amplitude
c uptime      : counter of samples where ipkflg>0 but no p-pick
c test        : variable used for debugging, i.e. printed output
c samplespersec: no of samples per second
c tdownmax    : if dtime exceeds tdownmax, the trigger is examined for
c               validity
c tupevent    : min nr of samples for itrm to be accepted as a pick
c thrshl1     : threshold to trigger for pick (c.f. paper)
c thrshl2     : threshold for updating sigma  (c.f. paper)
c
c preset_len  | no of points taken for the estimation of variance of
                   SF(t) on preset()

*******************************************************************************
int ppick (reltrc,npts,pptime,pfm,
     samplespersec,tdownmax,tupevent,thrshl1,thrshl2,preset_len,p_dur)
float *reltrc;
int npts;
int *pptime;
char *pfm;
float samplespersec;
int tdownmax,tupevent;
float thrshl1,thrshl2;
int preset_len;
int p_dur;
{
*/

int ppick (float *reltrc, int npts, int *pptime, char *pfm, float samplespersec, int tdownmax, int tupevent, float thrshl1, float thrshl2, int preset_len, int p_dur){
      int len2;
      int *trace = NULL;
      int ipkflg;
      int uptime = 0;
      int pamp;
      int preptime,ifrst;
      int noise;

      int ptime;
      int itar;
      int amp,dtime,num,itrm;
      int end_dur;
      float rawold,ssx,ssx2,sum,sdev,mean;
      float edat,rdif,rdat,rda2,rdi2,omega,y2,yt,edev=0.0;
      int i,iamp,picklength;

      float xr;
      int ii;
      float min,max;
      float scale;

      len2 = 2*preset_len;  /*
                            the variance of SF(t) is updated as long as
                            CF(t) is less than S2 ( here thrshl2) and
                            the running index is less than len2
                            */


      /* prepare integer version of float input trace */

      trace = (int *)calloc(npts+1,sizeof(int));
      if (trace == NULL) {
          return -1;
      }

      max = min = reltrc[1];
      for (ii=1; ii<=npts; ii++) {
         if (reltrc[ii] > max ) max = reltrc[ii];
         if (reltrc[ii] < min ) min = reltrc[ii];
      }

      scale = fabsf(max);
      if (fabsf(max) < fabsf(min)) scale = fabsf(min);

      /* scale trace maximum to 10000 */

      ii = 0;

      for (ii=1; ii<=npts; ii++) {
#if 0
         trace[ii]=(int)(reltrc[ii]+0.5);
#endif
         trace[ii]=(int)((256.0*reltrc[ii]/scale)+0.5);
      }

      /* preset some partameters */
      y2 = 1.0;
      yt = 1.0;
      sdev = 1.0;
      rawold = 0.0;
      num = 0;
      mean = 0;
      dtime = 0;

      preset(reltrc,preset_len,&rawold,&y2,&yt,&ssx,&ssx2,&sdev,&num,&itar,
         &ptime,&preptime,pfm,&ipkflg,samplespersec);

      omega = y2/yt;    /* set weighting factor */
      amp = 0;
      /*
      p_dur defines the time interval for which the
      maximum amplitude is evaluated
      Originally set to 6 secs.
      */
#if 0
      p_dur = 6.*samplespersec;
      picklength = 80*samplespersec;
#endif
      picklength = npts; /* check till the end of the trace */
      end_dur = 0;
      noise = 0;
      ipkflg = 0;
      ptime = 0;
      pamp = 0;
      preptime = 0;
      ifrst = 0;
      strcpy(pfm,"");
      noise = 0;
      i = 0;

label160:

      i= i+1;
      if (i > npts)
      {
        /*
           If a P arrival has not been recognized up to now, check whether
           it was triggered just before being forced to leave the routine
           since the end of the trace was reached
        */
        if ( (ptime == 0 ) && (itar != 0))
        {
            /* check how long and how strong the trigger was activated */
            itrm = i-itar-dtime+ipkflg;

            if (itrm >= tupevent) /* triggered for more than tupevent */
            {
               if(ptime == 0)
               {
                  ptime = itar;
                  itar= 0;
                  if (ifrst < 0) pfm[2]='U';  /* ADC inverts signals */
                  if (ifrst > 0) pfm[2]='D';

                  /* check quality */
                  pfm[0] = 'E';
                  pfm[1] = 'P';
                  xr = (float)pamp/(float)noise;
                  pfm[3] = '4';
                  if (xr > 1.5) pfm[3] = '3';
                  if (xr > 4.0) pfm[3] = '2';
                  if (xr > 6.0) pfm[3] = '1';
                  if (xr > 8.0) pfm[3] = '0';
                  if((pfm[3] == '0') || (pfm[3] == '1'))
                      pfm[0] = 'I';
                  pfm[4] = '\0';


                }
            }
        }

        /* pass ptime back to calling routine */

        *pptime = ptime;
        free(trace);
        return 0;

      }

      /*
         once picklength is reached, only amp is updated
         but no further trigger is evaluated
      */
      if ( i > picklength)
      {
        iamp = (int) (abs(trace[i]) + 0.5);

        if(iamp > amp)
        {
           amp= iamp;
        }
        goto label160;
      }

      /*
      calculate CF(t)
      */
      rdat= reltrc[i];
      rdif= (rdat - rawold)*samplespersec;
      rawold= rdat;
      rda2= rdat*rdat;
      rdi2= rdif*rdif;
      y2= y2+rda2;
      yt= yt+rdi2;
      edat= rda2 + omega*rdi2;
      edat= edat*edat;         /* corresponds to SF(t) */
      omega= y2/yt;

      if(sdev > 0)
          edev= (edat-mean)/sdev;  /* corresponds to CF(t), mean corresponds
                                      to S(t)
                                   */

      iamp = (int) (abs(trace[i]) + 0.5);
      if(iamp > amp)
      {
        amp= iamp;
      }
      if(i <= end_dur)
      {
        pamp= amp;
      }

      if ( (edev > thrshl1) && (i > len2)) /*
                                            i > len2  is twice the
                                            region used for noise estimates
                                           */
      {
         if (ipkflg == 0)       /* if trigger has not been up or has been cleared */
         {                      /* save current parameters */
              itar=i;           /* itar is the first triggered sample*/
              ipkflg = 1;
              if(ptime == 0)
              {
                  end_dur = itar + p_dur; /*
                                             define last possible data index
                                             to check maximum amplitude
                                          */
                  if (noise == 0)         /* noise is largest amplitude before
                                             first pick in trace
                                           */
                  {
                     noise = amp;
                  }
                  if (rdif < 0) ifrst=  1;  /* first motion */
                  if (rdif > 0) ifrst= -1;
              }
              if (preptime == 0) /* save this onset as possible P arrival */
                                 /* even if it may not be taken at the end */
              {
                  preptime = itar;
              }
              uptime= 1;
          }
          else if (ptime == 0) /* if trigger has been up, but
                                  no p-arrival has been defined yet */
          {
              if ((edev > 4*thrshl1) && (dtime == 0))    /*
                                                         extra strong signals
                 ipkflg= ipkflg+2;                       get bonus point on
                                                         trigger flag
                                                         */
              uptime= uptime+1;
          }
          dtime = 0;
      }
      else
      {
          /*
          check whether CF drops below the threshold thrshl1
          for more than 'downtime'. In this case ignore pick.
          */
          if (ipkflg != 0)   /* if trigger has been up */
          {
              dtime = dtime+1;
              if (ptime ==0) /* if no p-arrival has been defined yet,
                                keep counting uptime */
              {
                 uptime= uptime+1;
              }
              if(dtime > tdownmax) /*
                                      now check if trigger was on
                                      sufficiently to qualify for an
                                      arrival
                                   */
              {
                  itrm = i-itar-dtime+ipkflg;

                  if(itrm >= tupevent) /* triggered for more than tupevent */
                  {
                     if(ptime == 0)
                     {
                        /* now we got a P arrival */
                        ptime= itar;
                        itar = 0;

                        if(ifrst <0) pfm[2]='U'; /* ADC inverts signals */
                        if(ifrst >0) pfm[2]='D';

                        /* check quality */
                        pfm[0] = 'E';
                        pfm[1] = 'P';
                        xr = (float)pamp/(float)noise;

                        pfm[3] = '4';
                        if (xr > 1.5) pfm[3] = '3';
                        if (xr > 4.0) pfm[3] = '2';
                        if (xr > 6.0) pfm[3] = '1';
                        if (xr > 8.0) pfm[3] = '0';


                        if((pfm[3] == '0') || (pfm[3] == '1'))
                            pfm[0] = 'I';
                        pfm[4] = '\0';

                        /* pass ptime back to calling routine */

                        *pptime = ptime;
                        free(trace);
                        return 0;

                     }
                     /* don't overwrite  already existing phases */
                  } else {

                      /*
                      trigger was not up long enough, so forget about
                      this pick. Just  and reset preliminary amplitude
                      and time values
                      */

                      itar = 0;
                  }
                  /* reset trigger flag and uptime */
                  ipkflg= 0;
                  uptime= 0;
              }
          }
      }

      /* update standard deviation */

      /*
      Sigma is always updated if CF is less than S2 = 2*S1
      (here thrshl2) to allow for variation in noise level.
      However, this is done only for the first len2 points.
      (cf. p. 1440)
      */

      if((edev < thrshl2) || (i <= len2))
      {
          ssx= ssx+edat;
          ssx2= ssx2+edat*edat;
          sum= (float)(num+1);
          if(((sum*ssx2-ssx*ssx)/(sum*sum)) >= 0)
              sdev = sqrtf((sum*ssx2-ssx*ssx)/(sum*sum));
          else
              sdev = 1.;
          mean= ssx/sum;
          num = (int)(sum + 0.5);
      }

      goto label160;

}

/*int preset(rbuf,n,old,y2,yt,sumx,sumx2,sdev,nsum,itar,
     ptime,preptime,pfm,ipkflg,samplespersec)
float *rbuf;
int n;
float *old,*y2,*yt,*sumx,*sumx2,*sdev;
int *nsum, *itar, *ptime,*preptime;
char *pfm;
int *ipkflg;
float samplespersec;*/

/*int preset(float *rbuf, int n, float *old, float *y2, float *yt, float *sumx, float *sumx2, float *sdev, int *nsum, int *itar, int *ptime, int *preptime, char *pfm, int *ipkflg, float samplespersec)*/

static void preset(float *rbuf, int n, float *old, float *y2, float *yt, /*@out@*/ float *sumx, /*@out@*/ float *sumx2, float *sdev, int *nsum, /*@out@*/ int *itar, /*@out@*/ int *ptime, /*@out@*/ int *preptime, char *pfm, /*@out@*/ int *ipkflg, float samplespersec)
{
      int i;
      float yy2,yyt,ysv;

      ysv= rbuf[1];
      *old= ysv;
      *sumx= ysv;
      *y2= 0.0;
      *yt= 0.0;

      for(i=2;i<=n;i++)
      {
          yy2= rbuf[i];
          yyt= (yy2-ysv)*samplespersec;
          ysv= yy2;
          *sumx= *sumx+ysv;
          *y2= (*y2)+yy2*yy2;
          *yt= (*yt)+yyt*yyt;
      }

      if ((n*(*y2)-(*sumx)*(*sumx))/(n*n) > 0)
          *sdev= (float)(sqrt((float)n*(*y2)-(*sumx)*(*sumx))/(n*n));
      else
          *sdev = 1;
      *sumx= 0.0;
      *sumx2= 0.0;
      *nsum= 0;
      *itar= 0;
      *ptime= 0;
      *preptime= 0;
      strcpy(pfm,"");
      *ipkflg= 0;
}
