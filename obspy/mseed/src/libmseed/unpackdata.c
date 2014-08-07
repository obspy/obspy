/************************************************************************
 *  Routines for unpacking INT_16, INT_32, FLOAT_32, FLOAT_64,
 *  STEIM1, STEIM2, GEOSCOPE (24bit and gain ranged), CDSN, SRO
 *  and DWWSSN encoded data records.
 *
 *  Some routines originated and were borrowed from qlib2 by:
 *
 *	Douglas Neuhauser
 *	Seismographic Station
 *	University of California, Berkeley
 *	doug@seismo.berkeley.edu
 *									
 *  Modified by Chad Trabant,
 *  (previously) ORFEUS/EC-Project MEREDIAN
 *  (currently) IRIS Data Management Center
 *
 *  modified: 2012.357
 ************************************************************************/

/*
 * Copyright (c) 1996 The Regents of the University of California.
 * All Rights Reserved.
 * 
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for educational, research and non-profit purposes,
 * without fee, and without a written agreement is hereby granted,
 * provided that the above copyright notice, this paragraph and the
 * following three paragraphs appear in all copies.
 * 
 * Permission to incorporate this software into commercial products may
 * be obtained from the Office of Technology Licensing, 2150 Shattuck
 * Avenue, Suite 510, Berkeley, CA  94704.
 * 
 * IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY
 * FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
 * INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
 * ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE
 * PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
 * CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
 * UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 */

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "libmseed.h"
#include "unpackdata.h"

#define MAX12 0x7ff         /* maximum 12 bit positive # */
#define MAX14 0x1fff        /* maximum 14 bit positive # */
#define MAX16 0x7fff        /* maximum 16 bit positive # */
#define MAX24 0x7fffff      /* maximum 24 bit positive # */

/* For Steim encodings */
#define X0  pf->w[0].fw
#define XN  pf->w[1].fw



/************************************************************************
 *  msr_unpack_int_16:							*
 *                                                                      *
 *  Unpack int_16 miniSEED data and place in supplied buffer.           *
 *                                                                      *
 *  Return: # of samples returned.                                      *
 ************************************************************************/
int msr_unpack_int_16 
 (int16_t      *ibuf,		/* ptr to input data.			*/
  int		num_samples,	/* number of data samples in total.     */
  int		req_samples,	/* number of data desired by caller.	*/
  int32_t      *databuff,	/* ptr to unpacked data array.		*/
  int		swapflag)       /* if data should be swapped.	        */
{
  int		nd = 0;		/* # of data points in packet.		*/
  int16_t	stmp;
  
  if (num_samples < 0) return 0;
  if (req_samples < 0) return 0;
  
  for (nd=0; nd<req_samples && nd<num_samples; nd++) {
    stmp = ibuf[nd];
    if ( swapflag ) ms_gswap2a (&stmp);
    databuff[nd] = stmp;
  }
  
  return nd;
}  /* End of msr_unpack_int_16() */


/************************************************************************
 *  msr_unpack_int_32:							*
 *                                                                      *
 *  Unpack int_32 miniSEED data and place in supplied buffer.           *
 *                                                                      *
 *  Return: # of samples returned.                                      *
 ************************************************************************/
int msr_unpack_int_32
 (int32_t      *ibuf,		/* ptr to input data.			*/
  int		num_samples,	/* number of data samples in total.     */
  int		req_samples,	/* number of data desired by caller.	*/
  int32_t      *databuff,	/* ptr to unpacked data array.		*/
  int		swapflag)	/* if data should be swapped.	        */
{
  int		nd = 0;		/* # of data points in packet.		*/
  int32_t    	itmp;
  
  if (num_samples < 0) return 0;
  if (req_samples < 0) return 0;
  
  for (nd=0; nd<req_samples && nd<num_samples; nd++) {
    itmp = ibuf[nd];
    if ( swapflag) ms_gswap4a (&itmp);
    databuff[nd] = itmp;
  }
  
  return nd;
}  /* End of msr_unpack_int_32() */


/************************************************************************
 *  msr_unpack_float_32:	       				 	*
 *                                                                      *
 *  Unpack float_32 miniSEED data and place in supplied buffer.	        *
 *                                                                      *
 *  Return: # of samples returned.                                      *
 ************************************************************************/
int msr_unpack_float_32
 (float	       *fbuf,		/* ptr to input data.			*/
  int		num_samples,	/* number of data samples in total.     */
  int		req_samples,	/* number of data desired by caller.	*/
  float	       *databuff,	/* ptr to unpacked data array.		*/
  int		swapflag)	/* if data should be swapped.	        */
{
  int		nd = 0;		/* # of data points in packet.		*/
  float    	ftmp;
  
  if (num_samples < 0) return 0;
  if (req_samples < 0) return 0;
  
  for (nd=0; nd<req_samples && nd<num_samples; nd++) {
    memcpy (&ftmp, &fbuf[nd], sizeof(float));
    if ( swapflag ) ms_gswap4a (&ftmp);
    databuff[nd] = ftmp;
  }
  
  return nd;
}  /* End of msr_unpack_float_32() */


/************************************************************************
 *  msr_unpack_float_64:	       					*
 *                                                                      *
 *  Unpack float_64 miniSEED data and place in supplied buffer.	        *
 *                                                                      *
 *  Return: # of samples returned.                                      *
 ************************************************************************/
int msr_unpack_float_64
 (double       *fbuf,		/* ptr to input data.			*/
  int		num_samples,	/* number of data samples in total.     */
  int		req_samples,	/* number of data desired by caller.	*/
  double       *databuff,	/* ptr to unpacked data array.		*/
  int		swapflag)	/* if data should be swapped.	        */
{
  int		nd = 0;		/* # of data points in packet.		*/
  double  	dtmp;
  
  if (num_samples < 0) return 0;
  if (req_samples < 0) return 0;
  
  for (nd=0; nd<req_samples && nd<num_samples; nd++) {
    memcpy (&dtmp, &fbuf[nd], sizeof(double));
    if ( swapflag ) ms_gswap8a (&dtmp);
    databuff[nd] = dtmp;
  }
  
  return nd;
}  /* End of msr_unpack_float_64() */


/************************************************************************
 *  msr_unpack_steim1:							*
 *                                                                      *
 *  Unpack STEIM1 data frames and place in supplied buffer.		*
 *  See the SEED format manual for Steim-1 encoding details.            *
 *                                                                      *
 *  Return: # of samples returned or negative error code.               *
 ************************************************************************/
int msr_unpack_steim1
 (FRAME	       *pf,		/* ptr to Steim1 data frames.		*/
  int		nbytes,		/* number of bytes in all data frames.	*/
  int		num_samples,	/* number of data samples in all frames.*/
  int		req_samples,	/* number of data desired by caller.	*/
  int32_t      *databuff,	/* ptr to unpacked data array.		*/
  int32_t      *diffbuff,	/* ptr to unpacked diff array.		*/
  int32_t      *px0,		/* return X0, first sample in frame.	*/
  int32_t      *pxn,		/* return XN, last sample in frame.	*/
  int		swapflag,	/* if data should be swapped.	        */
  int           verbose)
{
  int32_t      *diff = diffbuff;
  int32_t      *data = databuff;
  int32_t      *prev;
  int	        num_data_frames = nbytes / sizeof(FRAME);
  int		nd = 0;		/* # of data points in packet.		*/
  int		fn;		/* current frame number.		*/
  int		wn;		/* current work number in the frame.	*/
  int		compflag;      	/* current compression flag.		*/
  int		nr, i;
  int32_t	last_data;
  int32_t	itmp;
  int16_t	stmp;
  uint32_t	ctrl;
  
  if (num_samples < 0) return 0;
  if (num_samples == 0) return 0;
  if (req_samples < 0) return 0;
  
  /* Extract forward and reverse integration constants in first frame */
  *px0 = X0;
  *pxn = XN;
  
  if ( swapflag )
    {
      ms_gswap4a (px0);
      ms_gswap4a (pxn);
    }
  
  if ( verbose > 2 )
    ms_log (1, "%s: forward/reverse integration constants:\nX0: %d  XN: %d\n",
	    UNPACK_SRCNAME, *px0, *pxn);
  
  /* Decode compressed data in each frame */
  for (fn = 0; fn < num_data_frames; fn++)
    {
      
      ctrl = pf->ctrl;
      if ( swapflag ) ms_gswap4a (&ctrl);

      for (wn = 0; wn < VALS_PER_FRAME; wn++)
	{
	  if (nd >= num_samples) break;
	  
	  compflag = (ctrl >> ((VALS_PER_FRAME-wn-1)*2)) & 0x3;
	  
	  switch (compflag)
	    {
	      
	    case STEIM1_SPECIAL_MASK:
	      /* Headers info -- skip it */
	      break;
	      
	    case STEIM1_BYTE_MASK:
	      /* Next 4 bytes are 4 1-byte differences */
	      for (i=0; i < 4 && nd < num_samples; i++, nd++)
		*diff++ = pf->w[wn].byte[i];
	      break;
	      
	    case STEIM1_HALFWORD_MASK:
	      /* Next 4 bytes are 2 2-byte differences */
	      for (i=0; i < 2 && nd < num_samples; i++, nd++)
		{
		  if ( swapflag )
		    {
		      stmp = pf->w[wn].hw[i];
		      ms_gswap2a (&stmp);
		      *diff++ = stmp;
		    }
		  else *diff++ = pf->w[wn].hw[i];
		}
	      break;
	      
	    case STEIM1_FULLWORD_MASK:
	      /* Next 4 bytes are 1 4-byte difference */
	      if ( swapflag )
		{
		  itmp = pf->w[wn].fw;
		  ms_gswap4a (&itmp);
		  *diff++ = itmp;
		}
	      else *diff++ = pf->w[wn].fw;
	      nd++;
	      break;
	      
	    default:
	      /* Should NEVER get here */
	      ms_log (2, "msr_unpack_steim1(%s): invalid compression flag = %d\n",
		      UNPACK_SRCNAME, compflag);
	      return MS_STBADCOMPFLAG;
	    }
	}
      ++pf;
    }
  
  /* Test if the number of samples implied by the data frames is the
   * same number indicated in the header.
   */
  if ( nd != num_samples )
    {
      ms_log (1, "Warning: msr_unpack_steim1(%s): number of samples indicated in header (%d) does not equal data (%d)\n",
	      UNPACK_SRCNAME, num_samples, nd);
    }
  
  /*	For now, assume sample count in header to be correct.		*/
  /*	One way of "trimming" data from a block is simply to reduce	*/
  /*	the sample count.  It is not clear from the documentation	*/
  /*	whether this is a valid or not, but it appears to be done	*/
  /*	by other program, so we should not complain about its effect.	*/
  
  nr = req_samples;
  
  /* Compute first value based on last_value from previous buffer.	*/
  /* The two should correspond in all cases EXCEPT for the first	*/
  /* record for each component (because we don't have a valid xn from	*/
  /* a previous record).  Although the Steim compression algorithm	*/
  /* defines x(-1) as 0 for the first record, this only works for the	*/
  /* first record created since coldstart of the datalogger, NOT the	*/
  /* first record of an arbitrary starting record.	                */
  
  /* In all cases, assume x0 is correct, since we don't have x(-1).	*/
  data = databuff;
  diff = diffbuff;
  last_data = *px0;
  if (nr > 0)
    *data = *px0;
  
  /* Compute all but first values based on previous value               */
  prev = data - 1;
  while (--nr > 0 && --nd > 0)
    last_data = *++data = *++diff + *++prev;
  
  /* If a short count was requested compute the last sample in order    */
  /* to perform the integrity check comparison                          */
  while (--nd > 0)
    last_data = *++diff + last_data;
  
  /* Verify that the last value is identical to xn = rev. int. constant */
  if (last_data != *pxn)
    {
      ms_log (1, "%s: Warning: Data integrity check for Steim-1 failed, last_data=%d, xn=%d\n",
	      UNPACK_SRCNAME, last_data, *pxn);
    }
  
  return ((req_samples < num_samples) ? req_samples : num_samples);
}  /* End of msr_unpack_steim1() */


/************************************************************************
 *  msr_unpack_steim2:							*
 *                                                                      *
 *  Unpack STEIM2 data frames and place in supplied buffer.		*
 *  See the SEED format manual for Steim-2 encoding details.            *
 *                                                                      *
 *  Return: # of samples returned or negative error code.               *
 ************************************************************************/
int msr_unpack_steim2 
 (FRAME	       *pf,		/* ptr to Steim2 data frames.		*/
  int		nbytes,		/* number of bytes in all data frames.	*/
  int		num_samples,	/* number of data samples in all frames.*/
  int		req_samples,	/* number of data desired by caller.	*/
  int32_t      *databuff,	/* ptr to unpacked data array.		*/
  int32_t      *diffbuff,	/* ptr to unpacked diff array.		*/
  int32_t      *px0,		/* return X0, first sample in frame.	*/
  int32_t      *pxn,		/* return XN, last sample in frame.	*/
  int		swapflag,	/* if data should be swapped.	        */
  int           verbose)
{
  int32_t      *diff = diffbuff;
  int32_t      *data = databuff;
  int32_t      *prev;
  int		num_data_frames = nbytes / sizeof(FRAME);
  int		nd = 0;		/* # of data points in packet.		*/
  int		fn;		/* current frame number.		*/
  int		wn;		/* current work number in the frame.	*/
  int		compflag;     	/* current compression flag.		*/
  int		nr, i;
  int		n, bits, m1, m2;
  int32_t	last_data;
  int32_t    	val;
  int8_t	dnib;
  uint32_t	ctrl;
  
  if (num_samples < 0) return 0;
  if (num_samples == 0) return 0;
  if (req_samples < 0) return 0;
  
  /* Extract forward and reverse integration constants in first frame.*/
  *px0 = X0;
  *pxn = XN;
  
  if ( swapflag )
    {
      ms_gswap4a (px0);
      ms_gswap4a (pxn);
    }
  
  if ( verbose > 2 )
    ms_log (1, "%s: forward/reverse integration constants:  X0: %d  XN: %d\n",
	    UNPACK_SRCNAME, *px0, *pxn);
  
  /* Decode compressed data in each frame */
  for (fn = 0; fn < num_data_frames; fn++)
    {
      
      ctrl = pf->ctrl;
      if ( swapflag ) ms_gswap4a (&ctrl);
      
      for (wn = 0; wn < VALS_PER_FRAME; wn++)
	{
	  if (nd >= num_samples) break;
	  
	  compflag = (ctrl >> ((VALS_PER_FRAME-wn-1)*2)) & 0x3;
	  
	  switch (compflag)
	    {
	    case STEIM2_SPECIAL_MASK:
	      /* Headers info -- skip it */
	      break;
	      
	    case STEIM2_BYTE_MASK:
	      /* Next 4 bytes are 4 1-byte differences */
	      for (i=0; i < 4 && nd < num_samples; i++, nd++)
		*diff++ = pf->w[wn].byte[i];
	      break;
	      
	    case STEIM2_123_MASK:
	      val = pf->w[wn].fw;
	      if ( swapflag ) ms_gswap4a (&val);
	      dnib =  val >> 30 & 0x3;
	      switch (dnib)
		{
		case 1:	/* 1 30-bit difference */
		  bits = 30; n = 1; m1 = 0x3fffffff; m2 = 0x20000000; break;
		case 2:	/* 2 15-bit differences */
		  bits = 15; n = 2; m1 = 0x00007fff; m2 = 0x00004000; break;
		case 3:	/* 3 10-bit differences */
		  bits = 10; n = 3; m1 = 0x000003ff; m2 = 0x00000200; break;
		default:	/*  should NEVER get here  */
		  ms_log (2, "msr_unpack_steim2(%s): invalid compflag, dnib, fn, wn = %d, %d, %d, %d\n", 
			  UNPACK_SRCNAME, compflag, dnib, fn, wn);
		  return MS_STBADCOMPFLAG;
		}
	      /*  Uncompress the differences */
	      for (i=(n-1)*bits; i >= 0 && nd < num_samples; i-=bits, nd++)
		{
		  *diff = (val >> i) & m1;
		  *diff = (*diff & m2) ? *diff | ~m1 : *diff;
		  diff++;
		}
	      break;
	      
	    case STEIM2_567_MASK:
	      val = pf->w[wn].fw;
	      if ( swapflag ) ms_gswap4a (&val);
	      dnib =  val >> 30 & 0x3;
	      switch (dnib)
		{
		case 0:	/*  5 6-bit differences  */
		  bits = 6; n = 5; m1 = 0x0000003f; m2 = 0x00000020; break;
		case 1:	/*  6 5-bit differences  */
		  bits = 5; n = 6; m1 = 0x0000001f; m2 = 0x00000010; break;
		case 2:	/*  7 4-bit differences  */
		  bits = 4; n = 7; m1 = 0x0000000f; m2 = 0x00000008; break;
		default:
		  ms_log (2, "msr_unpack_steim2(%s): invalid compflag, dnib, fn, wn = %d, %d, %d, %d\n", 
			  UNPACK_SRCNAME, compflag, dnib, fn, wn);
		  return MS_STBADCOMPFLAG;
		}
	      /* Uncompress the differences */
	      for (i=(n-1)*bits; i >= 0 && nd < num_samples; i-=bits, nd++)
		{
		  *diff = (val >> i) & m1;
		  *diff = (*diff & m2) ? *diff | ~m1 : *diff;
		  diff++;
		}
	      break;
	      
	    default:
	      /* Should NEVER get here */
	      ms_log (2, "msr_unpack_steim2(%s): invalid compflag, fn, wn = %d, %d, %d - nsamp: %d\n",
		      UNPACK_SRCNAME, compflag, fn, wn, nd);
	      return MS_STBADCOMPFLAG;
	    }
	}
      ++pf;
    }
    
  /* Test if the number of samples implied by the data frames is the
   * same number indicated in the header.
   */
  if ( nd != num_samples )
    {
      ms_log (1, "Warning: msr_unpack_steim2(%s): number of samples indicated in header (%d) does not equal data (%d)\n",
	      UNPACK_SRCNAME, num_samples, nd);
    }

  /*	For now, assume sample count in header to be correct.		*/
  /*	One way of "trimming" data from a block is simply to reduce	*/
  /*	the sample count.  It is not clear from the documentation	*/
  /*	whether this is a valid or not, but it appears to be done	*/
  /*	by other program, so we should not complain about its effect.	*/
  
  nr = req_samples;
  
  /* Compute first value based on last_value from previous buffer.	*/
  /* The two should correspond in all cases EXCEPT for the first	*/
  /* record for each component (because we don't have a valid xn from	*/
  /* a previous record).  Although the Steim compression algorithm	*/
  /* defines x(-1) as 0 for the first record, this only works for the	*/
  /* first record created since coldstart of the datalogger, NOT the	*/
  /* first record of an arbitrary starting record.	                */
  
  /* In all cases, assume x0 is correct, since we don't have x(-1).	*/
  data = databuff;
  diff = diffbuff;
  last_data = *px0;
  if (nr > 0)
    *data = *px0;

  /* Compute all but first values based on previous value               */
  prev = data - 1;
  while (--nr > 0 && --nd > 0)
    last_data = *++data = *++diff + *++prev;
  
  /* If a short count was requested compute the last sample in order    */
  /* to perform the integrity check comparison                          */
  while (--nd > 0)
    last_data = *++diff + last_data;
  
  /* Verify that the last value is identical to xn = rev. int. constant */
  if (last_data != *pxn)
    {
      ms_log (1, "%s: Warning: Data integrity check for Steim-2 failed, last_data=%d, xn=%d\n",
	      UNPACK_SRCNAME, last_data, *pxn);
    }
  
  return ((req_samples < num_samples) ? req_samples : num_samples);
}  /* End of msr_unpack_steim2() */


/* Defines for GEOSCOPE encoding */
#define GEOSCOPE_MANTISSA_MASK 0x0fff   /* mask for mantissa */
#define GEOSCOPE_GAIN3_MASK 0x7000      /* mask for gainrange factor */
#define GEOSCOPE_GAIN4_MASK 0xf000      /* mask for gainrange factor */
#define GEOSCOPE_SHIFT 12               /* # bits in mantissa */

/************************************************************************
 *  msr_unpack_geoscope:                                                *
 *                                                                      *
 *  Unpack GEOSCOPE gain ranged data (demultiplexed only) encoded       *
 *  miniSEED data and place in supplied buffer.                         *
 *                                                                      *
 *  Return: # of samples returned.                                      *
 ************************************************************************/
int msr_unpack_geoscope
 (const char   *edata,		/* ptr to encoded data.			*/
  int		num_samples,	/* number of data samples in total.     */
  int		req_samples,	/* number of data desired by caller.	*/
  float	       *databuff,	/* ptr to unpacked data array.		*/
  int           encoding,       /* specific GEOSCOPE encoding type      */
  int		swapflag)	/* if data should be swapped.	        */
{
  int nd = 0;		/* # of data points in packet.		*/
  int mantissa;		/* mantissa from SEED data */
  int gainrange;	/* gain range factor */
  int exponent;		/* total exponent */
  int k;
  uint64_t exp2val;
  int16_t sint;
  double dsample = 0.0;
  
  union {
    uint8_t b[4];
    uint32_t i;
  } sample32;
  
  if (num_samples < 0) return 0;
  if (req_samples < 0) return 0;

  /* Make sure we recognize this as a GEOSCOPE encoding format */
  if ( encoding != DE_GEOSCOPE24 &&
       encoding != DE_GEOSCOPE163 &&
       encoding != DE_GEOSCOPE164 )
    {
      ms_log (2, "msr_unpack_geoscope(%s): unrecognized GEOSCOPE encoding: %d\n",
	      UNPACK_SRCNAME, encoding);
      return -1;
    }
  
  for (nd=0; nd<req_samples && nd<num_samples; nd++)
    {
      switch (encoding)
	{
	case DE_GEOSCOPE24:
	  sample32.i = 0;
	  if ( swapflag )
	    for (k=0; k < 3; k++)
	      sample32.b[2-k] = edata[k];
	  else
	    for (k=0; k < 3; k++)
	      sample32.b[1+k] = edata[k];
	  
	  mantissa = sample32.i;

	  /* Take 2's complement for mantissa for overflow */
	  if (mantissa > MAX24) 
	    mantissa -= 2 * (MAX24 + 1);
	  
	  /* Store */
	  dsample = (double) mantissa;
	  
	  break;
	case DE_GEOSCOPE163:
	  memcpy (&sint, edata, sizeof(int16_t));
	  if ( swapflag ) ms_gswap2a(&sint);
	  
	  /* Recover mantissa and gain range factor */
	  mantissa = (sint & GEOSCOPE_MANTISSA_MASK);
	  gainrange = (sint & GEOSCOPE_GAIN3_MASK) >> GEOSCOPE_SHIFT;
	  
	  /* Exponent is just gainrange for GEOSCOPE */
	  exponent = gainrange;
	  
	  /* Calculate sample as mantissa / 2^exponent */
	  exp2val = (uint64_t) 1 << exponent;
	  dsample = ((double) (mantissa-2048)) / exp2val;
	  
	  break;
	case DE_GEOSCOPE164:
	  memcpy (&sint, edata, sizeof(int16_t));
	  if ( swapflag ) ms_gswap2a(&sint);
	  
	  /* Recover mantissa and gain range factor */
	  mantissa = (sint & GEOSCOPE_MANTISSA_MASK);
	  gainrange = (sint & GEOSCOPE_GAIN4_MASK) >> GEOSCOPE_SHIFT;
	  
	  /* Exponent is just gainrange for GEOSCOPE */
	  exponent = gainrange;
	  
	  /* Calculate sample as mantissa / 2^exponent */
	  exp2val = (uint64_t) 1 << exponent;
	  dsample = ((double) (mantissa-2048)) / exp2val;
	  
	  break;
	}
      
      /* Save sample in output array */
      databuff[nd] = (float) dsample;
      
      /* Increment edata pointer depending on size */
      switch (encoding)
	{
	case DE_GEOSCOPE24:
	  edata += 3;
	  break;
	case DE_GEOSCOPE163:
	case DE_GEOSCOPE164:
	  edata += 2;
	  break;
	}
    }
  
  return nd;
}  /* End of msr_unpack_geoscope() */


/* Defines for CDSN encoding */
#define CDSN_MANTISSA_MASK 0x3fff   /* mask for mantissa */
#define CDSN_GAINRANGE_MASK 0xc000  /* mask for gainrange factor */
#define CDSN_SHIFT 14               /* # bits in mantissa */

/************************************************************************
 *  msr_unpack_cdsn:                                                    *
 *                                                                      *
 *  Unpack CDSN gain ranged data encoded miniSEED data and place in     *
 *  supplied buffer.                                                    *
 *                                                                      *
 *  Notes from original rdseed routine:                                 *
 *  CDSN data are compressed according to the formula                   *
 *                                                                      *
 *  sample = M * (2 exp G)                                              *
 *                                                                      *
 *  where                                                               *
 *     sample = seismic data sample                                     *
 *     M      = mantissa; biased mantissa B is written to tape          *
 *     G      = exponent of multiplier (i.e. gain range factor);        *
 *                      key K is written to tape                        *
 *     exp    = exponentiation operation                                *
 *     B      = M + 8191, biased mantissa, written to tape              *
 *     K      = key to multiplier exponent, written to tape             *
 *                      K may have any of the values 0 - 3, as follows: *
 *                      0 => G = 0, multiplier = 2 exp 0 = 1            *
 *                      1 => G = 2, multiplier = 2 exp 2 = 4            *
 *                      2 => G = 4, multiplier = 2 exp 4 = 16           *
 *                      3 => G = 7, multiplier = 2 exp 7 = 128          *
 *     Data are stored on tape in two bytes as follows:                 *
 *             fedc ba98 7654 3210 = bit number, power of two           *
 *             KKBB BBBB BBBB BBBB = form of SEED data                  *
 *             where K = key to multiplier exponent and B = biased mantissa *
 *                                                                      *
 *     Masks to recover key to multiplier exponent and biased mantissa  *
 *     from tape are:                                                   *
 *             fedc ba98 7654 3210 = bit number = power of two          *
 *             0011 1111 1111 1111 = 0x3fff     = mask for biased mantissa *
 *            1100 0000 0000 0000 = 0xc000     = mask for gain range key *
 *                                                                      *
 *  Return: # of samples returned.                                      *
 ************************************************************************/
int msr_unpack_cdsn
 (int16_t      *edata,		/* ptr to encoded data.			*/
  int		num_samples,	/* number of data samples in total.     */
  int		req_samples,	/* number of data desired by caller.	*/
  int32_t      *databuff,	/* ptr to unpacked data array.		*/
  int		swapflag)	/* if data should be swapped.	        */
{
  int32_t nd = 0;	/* sample count */
  int32_t mantissa;	/* mantissa */
  int32_t gainrange;	/* gain range factor */
  int32_t mult = -1;    /* multiplier for gain range */
  uint16_t sint;
  int32_t sample;
  
  if (num_samples < 0) return 0;
  if (req_samples < 0) return 0;
  
  for (nd=0; nd<req_samples && nd<num_samples; nd++)
    {
      memcpy (&sint, &edata[nd], sizeof(int16_t));
      if ( swapflag ) ms_gswap2a(&sint);
      
      /* Recover mantissa and gain range factor */
      mantissa = (sint & CDSN_MANTISSA_MASK);
      gainrange = (sint & CDSN_GAINRANGE_MASK) >> CDSN_SHIFT;
      
      /* Determine multiplier from the gain range factor and format definition
       * because shift operator is used later, these are powers of two */
      if ( gainrange == 0 ) mult = 0;
      else if ( gainrange == 1 ) mult = 2;
      else if ( gainrange == 2 ) mult = 4;
      else if ( gainrange == 3 ) mult = 7;
      
      /* Unbias the mantissa */
      mantissa -= MAX14;
      
      /* Calculate sample from mantissa and multiplier using left shift
       * mantissa << mult is equivalent to mantissa * (2 exp (mult)) */
      sample = (mantissa << mult);
      
      /* Save sample in output array */
      databuff[nd] = sample;
    }
  
  return nd;
}  /* End of msr_unpack_cdsn() */


/* Defines for SRO encoding */
#define SRO_MANTISSA_MASK 0x0fff   /* mask for mantissa */
#define SRO_GAINRANGE_MASK 0xf000  /* mask for gainrange factor */
#define SRO_SHIFT 12               /* # bits in mantissa */

/************************************************************************
 *  msr_unpack_sro:                                                     *
 *                                                                      *
 *  Unpack SRO gain ranged data encoded miniSEED data and place in      *
 *  supplied buffer.                                                    *
 *                                                                      *
 *  Notes from original rdseed routine:                                 *
 *  SRO data are represented according to the formula                   *
 *                                                                      *
 *  sample = M * (b exp {[m * (G + agr)] + ar})                         *
 *                                                                      *
 *  where                                                               *
 *	sample = seismic data sample                                    *
 *	M      = mantissa                                               *
 *	G      = gain range factor                                      *
 *	b      = base to be exponentiated = 2 for SRO                   *
 *	m      = multiplier  = -1 for SRO                               *
 *	agr    = term to be added to gain range factor = 0 for SRO      *
 *	ar     = term to be added to [m * (gr + agr)]  = 10 for SRO     *
 *	exp    = exponentiation operation                               *
 *	Data are stored in two bytes as follows:                        *
 *		fedc ba98 7654 3210 = bit number, power of two          *
 *		GGGG MMMM MMMM MMMM = form of SEED data                 *
 *		where G = gain range factor and M = mantissa            *
 *	Masks to recover gain range and mantissa:                       *
 *		fedc ba98 7654 3210 = bit number = power of two         *
 *		0000 1111 1111 1111 = 0x0fff     = mask for mantissa    *
 *		1111 0000 0000 0000 = 0xf000     = mask for gain range  *
 *                                                                      *
 *  Return: # of samples returned.                                      *
 ************************************************************************/
int msr_unpack_sro
 (int16_t      *edata,		/* ptr to encoded data.			*/
  int		num_samples,	/* number of data samples in total.     */
  int		req_samples,	/* number of data desired by caller.	*/
  int32_t      *databuff,	/* ptr to unpacked data array.		*/
  int		swapflag)	/* if data should be swapped.	        */
{
  int32_t nd = 0;	/* sample count */
  int32_t mantissa;	/* mantissa */
  int32_t gainrange;	/* gain range factor */
  int32_t add2gr;       /* added to gainrage factor */
  int32_t mult;         /* multiplier for gain range */
  int32_t add2result;   /* added to multiplied gain rage */
  int32_t exponent;	/* total exponent */
  uint16_t sint;
  int32_t sample;
  
  if (num_samples < 0) return 0;
  if (req_samples < 0) return 0;
  
  add2gr = 0;
  mult = -1;
  add2result = 10;
  
  for (nd=0; nd<req_samples && nd<num_samples; nd++)
    {
      memcpy (&sint, &edata[nd], sizeof(int16_t));
      if ( swapflag ) ms_gswap2a(&sint);
      
      /* Recover mantissa and gain range factor */
      mantissa = (sint & SRO_MANTISSA_MASK);
      gainrange = (sint & SRO_GAINRANGE_MASK) >> SRO_SHIFT;
      
      /* Take 2's complement for mantissa */
      if ( mantissa > MAX12 )
	mantissa -= 2 * (MAX12 + 1);
      
      /* Calculate exponent, SRO exponent = 0..10 */
      exponent = (mult * (gainrange + add2gr)) + add2result;
      
      if ( exponent < 0 || exponent > 10 )
	{
	  ms_log (2, "msr_unpack_sro(%s): SRO gain ranging exponent out of range: %d\n",
		  UNPACK_SRCNAME, exponent);
	  return MS_GENERROR;
	}
      
      /* Calculate sample as mantissa * 2^exponent */
      sample = mantissa * ( (uint64_t) 1 << exponent );
      
      /* Save sample in output array */
      databuff[nd] = sample;
    }
  
  return nd;
}  /* End of msr_unpack_sro() */


/************************************************************************
 *  msr_unpack_dwwssn:                                                  *
 *                                                                      *
 *  Unpack DWWSSN encoded miniSEED data and place in supplied buffer.   *
 *                                                                      *
 *  Return: # of samples returned.                                      *
 ************************************************************************/
int msr_unpack_dwwssn
 (int16_t      *edata,		/* ptr to encoded data.			*/
  int		num_samples,	/* number of data samples in total.     */
  int		req_samples,	/* number of data desired by caller.	*/
  int32_t      *databuff,	/* ptr to unpacked data array.		*/
  int		swapflag)	/* if data should be swapped.	        */
{
  int32_t nd = 0;	/* sample count */
  int32_t sample;
  uint16_t sint;
  
  if (num_samples < 0) return 0;
  if (req_samples < 0) return 0;
  
  for (nd=0; nd<req_samples && nd<num_samples; nd++)
    {
      memcpy (&sint, &edata[nd], sizeof(uint16_t));
      if ( swapflag ) ms_gswap2a(&sint);
      sample = (int32_t) sint;
      
      /* Take 2's complement for sample */
      if ( sample > MAX16 )
	sample -= 2 * (MAX16 + 1);
      
      /* Save sample in output array */
      databuff[nd] = sample;
    }
  
  return nd;
}  /* End of msr_unpack_dwwssn() */
