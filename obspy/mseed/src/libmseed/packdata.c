/***********************************************************************
 *  Routines for packing INT_32, INT_16, FLOAT_32, FLOAT_64,
 *  STEIM1 and STEIM2 data records.
 *
 *	Douglas Neuhauser						
 *	Seismological Laboratory					
 *	University of California, Berkeley				
 *	doug@seismo.berkeley.edu					
 *
 *
 * modified Aug 2008:
 *  - Optimize Steim 1 & 2 packing routines using small, re-used buffers.
 *
 * modified Sep 2004:
 *  - Reworked and cleaned routines for use within libmseed.
 *  - Added float32 and float64 packing routines.
 *
 * Modified by Chad Trabant, IRIS Data Management Center
 *
 * modified: 2009.111
 ************************************************************************/

/*
 * Copyright (c) 1996-2004 The Regents of the University of California.
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
#include "packdata.h"

static int pad_steim_frame (DFRAMES*, int, int, int, int, int);

#define	EMPTY_BLOCK(fn,wn) (fn+wn == 0)

#define	X0  dframes->f[0].w[0].fw
#define	XN  dframes->f[0].w[1].fw

#define	BIT4PACK(i,points_remaining)		   \
  (points_remaining >= 7 &&			   \
   (minbits[i] <= 4) && (minbits[i+1] <= 4) &&	   \
   (minbits[i+2] <= 4) && (minbits[i+3] <= 4) &&   \
   (minbits[i+4] <= 4) && (minbits[i+5] <= 4) &&   \
   (minbits[i+6] <= 4))

#define	BIT5PACK(i,points_remaining)		   \
  (points_remaining >= 6 &&			   \
   (minbits[i] <= 5) && (minbits[i+1] <= 5) &&	   \
   (minbits[i+2] <= 5) && (minbits[i+3] <= 5) &&   \
   (minbits[i+4] <= 5) && (minbits[i+5] <= 5))

#define	BIT6PACK(i,points_remaining)		   \
  (points_remaining >= 5 &&			   \
   (minbits[i] <= 6) && (minbits[i+1] <= 6) &&	   \
   (minbits[i+2] <= 6) && (minbits[i+3] <= 6) &&   \
   (minbits[i+4] <= 6))

#define	BYTEPACK(i,points_remaining)		 \
  (points_remaining >= 4 &&			 \
   (minbits[i] <= 8) && (minbits[i+1] <= 8) &&	 \
   (minbits[i+2] <= 8) && (minbits[i+3] <= 8))

#define	BIT10PACK(i,points_remaining)		   \
  (points_remaining >= 3 &&			   \
   (minbits[i] <= 10) && (minbits[i+1] <= 10) &&   \
   (minbits[i+2] <= 10))

#define	BIT15PACK(i,points_remaining)		\
  (points_remaining >= 2 &&			\
   (minbits[i] <= 15) && (minbits[i+1] <= 15))

#define	HALFPACK(i,points_remaining)					\
  (points_remaining >= 2 && (minbits[i] <= 16) && (minbits[i+1] <= 16))

#define	BIT30PACK(i,points_remaining)  \
  (points_remaining >= 1 &&	       \
   (minbits[i] <= 30))

#define	MINBITS(diff,minbits)					       \
  if (diff >= -8 && diff < 8) minbits = 4;			       \
  else if (diff >= -16 && diff < 16) minbits = 5;		       \
  else if (diff >= -32 && diff < 32) minbits = 6;		       \
  else if (diff >= -128 && diff < 128) minbits = 8;		       \
  else if (diff >= -512 && diff < 512) minbits = 10;		       \
  else if (diff >= -16384 && diff < 16384) minbits = 15;	       \
  else if (diff >= -32768 && diff < 32768) minbits = 16;	       \
  else if (diff >= -536870912 && diff < 536870912) minbits = 30;       \
  else minbits = 32;

#define PACK(bits,n,m1,m2)  {			\
    int i = 0;					\
    unsigned int val = 0;			\
    for (i=0;i<n;i++) {				\
      val = (val<<bits) | (diff[i]&m1); 	\
    }						\
    val |= ((unsigned int)m2 << 30);		\
    dframes->f[fn].w[wn].fw = val; }


/************************************************************************
 *  msr_pack_int_16:							*
 *	Pack integer data into INT_16 format.				*
 *	Return: 0 on success, -1 on failure.				*
 ************************************************************************/
int msr_pack_int_16
 (int16_t    *packed,           /* output data array - packed           */
  int32_t    *data,             /* input data array                     */
  int         ns,               /* desired number of samples to pack    */
  int         max_bytes,        /* max # of bytes for output buffer     */
  int         pad,              /* flag to specify padding to max_bytes */
  int        *pnbytes,          /* number of bytes actually packed      */
  int        *pnsamples,        /* number of samples actually packed    */
  int         swapflag)         /* if data should be swapped            */
{
  int points_remaining = ns;    /* number of samples remaining to pack  */
  int i = 0;
  
  while (points_remaining > 0 && max_bytes >= 2)
    {  /* Pack the next available data into INT_16 format */
      if ( data[i] < -32768 || data[i] > 32767 )
	ms_log (2, "msr_pack_int_16(%s): input sample out of range: %d\n",
		PACK_SRCNAME, data[i]);
      
      *packed = data[i];      
      if ( swapflag ) ms_gswap2 (packed);
      
      packed++;
      max_bytes -= 2;
      points_remaining--;
      i++;
    }
  
  *pnbytes = (ns - points_remaining) * 2;
  
  /* Pad miniSEED block if necessary */
  if (pad)
    {
      memset ((void *)packed, 0, max_bytes);
      *pnbytes += max_bytes;
    }

  *pnsamples = ns - points_remaining;

  return 0;
}


/************************************************************************
 *  msr_pack_int_32:							*
 *	Pack integer data into INT_32 format.				*
 *	Return: 0 on success, -1 on failure.				*
 ************************************************************************/
int msr_pack_int_32 
 (int32_t  *packed,          /* output data array - packed              */
  int32_t  *data,            /* input data array - unpacked             */
  int       ns,              /* desired number of samples to pack       */
  int       max_bytes,       /* max # of bytes for output buffer        */
  int       pad,             /* flag to specify padding to max_bytes    */
  int      *pnbytes,         /* number of bytes actually packed         */
  int      *pnsamples,       /* number of samples actually packed       */
  int       swapflag)        /* if data should be swapped               */
{
  int points_remaining = ns; /* number of samples remaining to pack */
  int i = 0;

  while (points_remaining > 0 && max_bytes >= 4)
    { /* Pack the next available data into INT_32 format */
      *packed = data[i];
      if ( swapflag ) ms_gswap4 (packed);
      
      packed++;
      max_bytes -= 4;
      points_remaining--;
      i++;
    }
  
  *pnbytes = (ns - points_remaining) * 4;
  
  /* Pad miniSEED block if necessary */
  if (pad)
    {
      memset ((void *)packed, 0, max_bytes);
      *pnbytes += max_bytes;
    }
  
  *pnsamples = ns - points_remaining;

  return 0;
}


/************************************************************************
 *  msr_pack_float_32:							*
 *	Pack float data into FLOAT32 format.				*
 *	Return: 0 on success, -1 on error.				*
 ************************************************************************/
int msr_pack_float_32 
 (float    *packed,          /* output data array - packed              */
  float    *data,            /* input data array - unpacked             */
  int       ns,              /* desired number of samples to pack       */
  int       max_bytes,       /* max # of bytes for output buffer        */
  int       pad,             /* flag to specify padding to max_bytes    */
  int      *pnbytes,         /* number of bytes actually packed         */
  int      *pnsamples,       /* number of samples actually packed       */
  int       swapflag)        /* if data should be swapped               */
{
  int points_remaining = ns; /* number of samples remaining to pack */
  int i = 0;
  
  while (points_remaining > 0 && max_bytes >= 4)
    {
      *packed = data[i];
      if ( swapflag ) ms_gswap4 (packed);
      
      packed++;
      max_bytes -= 4;
      points_remaining--;
      i++;
    }
  
  *pnbytes = (ns - points_remaining) * 4;
  
  /* Pad miniSEED block if necessary */
  if (pad)
    {
      memset ((void *)packed, 0, max_bytes);
      *pnbytes += max_bytes;
    }
  
  *pnsamples = ns - points_remaining;
  
  return 0;
}


/************************************************************************
 *  msr_pack_float_64:							*
 *	Pack double data into FLOAT64 format.				*
 *	Return: 0 on success, -1 on error.				*
 ************************************************************************/
int msr_pack_float_64 
 (double   *packed,          /* output data array - packed              */
  double   *data,            /* input data array - unpacked             */
  int       ns,              /* desired number of samples to pack       */
  int       max_bytes,       /* max # of bytes for output buffer        */
  int       pad,             /* flag to specify padding to max_bytes    */
  int      *pnbytes,         /* number of bytes actually packed         */
  int      *pnsamples,       /* number of samples actually packed       */
  int       swapflag)        /* if data should be swapped               */
{
  int points_remaining = ns; /* number of samples remaining to pack */
  int i = 0;
  
  while (points_remaining > 0 && max_bytes >= 8)
    {
      *packed = data[i];
      if ( swapflag ) ms_gswap8 (packed);
      
      packed++;
      max_bytes -= 8;
      points_remaining--;
      i++;
    }
  
  *pnbytes = (ns - points_remaining) * 8;
  
  /* Pad miniSEED block if necessary */
  if (pad)
    {
      memset ((void *)packed, 0, max_bytes);
      *pnbytes += max_bytes;
    }
  
  *pnsamples = ns - points_remaining;
  
  return 0;
}


/************************************************************************
 *  msr_pack_steim1:							*
 *	Pack data into STEIM1 data frames.				*
 *  return:								*
 *	0 on success.							*
 *	-1 on error.		           	         	        *
 ************************************************************************/
int msr_pack_steim1
 (DFRAMES      *dframes,       	/* ptr to data frames                   */
  int32_t      *data,		/* ptr to unpacked data array           */
  int32_t       d0,		/* first difference value               */
  int		ns,		/* number of samples to pack            */
  int		nf,		/* total number of data frames          */
  int		pad,		/* flag to specify padding to nf        */
  int	       *pnframes,	/* number of frames actually packed     */
  int	       *pnsamples,	/* number of samples actually packed    */
  int           swapflag)       /* if data should be swapped            */
{
  int		points_remaining = ns;
  int           points_packed = 0;
  int32_t       diff[4];        /* array of differences                 */
  uint8_t       minbits[4];     /* array of minimum bits for diffs      */
  int		i, j;
  int		mask;
  int		ipt = 0;	/* index of initial data to pack.	*/
  int		fn = 0;		/* index of initial frame to pack.	*/
  int		wn = 2;		/* index of initial word to pack.	*/
  int32_t      	itmp;
  int16_t	stmp;
  
  /* Calculate initial difference and minbits buffers */
  diff[0] = d0;
  MINBITS(diff[0],minbits[0]);
  for (i=1; i < 4 && i < ns; i++)
    {
      diff[i] = data[i] - data[i-1];
      MINBITS(diff[i],minbits[i]);
    }
  
  dframes->f[fn].ctrl = 0;
  
  /* Set X0 and XN values in first frame */
  X0 = data[0];
  if ( swapflag ) ms_gswap4 (&X0);
  dframes->f[0].ctrl = (dframes->f[0].ctrl<<2) | STEIM1_SPECIAL_MASK;
  XN = data[ns-1];
  if ( swapflag ) ms_gswap4 (&XN);
  dframes->f[0].ctrl = (dframes->f[0].ctrl<<2) | STEIM1_SPECIAL_MASK;
  
  while (points_remaining > 0)
    {
      points_packed = 0;
      
      /* Pack the next available data into the most compact form */
      if (BYTEPACK(0,points_remaining))
	{
	  mask = STEIM1_BYTE_MASK;
	  for (j=0; j<4; j++) { dframes->f[fn].w[wn].byte[j] = diff[j]; }
	  points_packed = 4;
	}
      else if (HALFPACK(0,points_remaining))
	{
	  mask = STEIM1_HALFWORD_MASK;
	  for (j=0; j<2; j++)
	    {
	      stmp = diff[j];
	      if ( swapflag ) ms_gswap2 (&stmp);
	      dframes->f[fn].w[wn].hw[j] = stmp;
	    }
	  points_packed = 2;
	}
      else
	{
	  mask = STEIM1_FULLWORD_MASK;
	  itmp = diff[0];
	  if ( swapflag ) ms_gswap4 (&itmp);
	  dframes->f[fn].w[wn].fw = itmp;
	  points_packed = 1;
	}
      
      /* Append mask for this word to current mask */
      dframes->f[fn].ctrl = (dframes->f[fn].ctrl<<2) | mask;
      
      points_remaining -= points_packed;
      ipt += points_packed;
      
      /* Check for full frame or full block */
      if (++wn >= VALS_PER_FRAME)
	{
	  if ( swapflag ) ms_gswap4 (&dframes->f[fn].ctrl);
	  /* Reset output index to beginning of frame */
	  wn = 0;
	  /* If block is full, output block and reinitialize */
	  if (++fn >= nf) break;
	  dframes->f[fn].ctrl = 0;
	}
      
      /* Shift and re-fill difference and minbits buffers */
      for ( i=points_packed; i < 4; i++ )
	{
	  /* Shift remaining buffer entries */
	  diff[i-points_packed] = diff[i];
	  minbits[i-points_packed] = minbits[i];
	}
      for ( i=4-points_packed,j=ipt+(4-points_packed); i < 4 && j < ns; i++,j++ )
	{
	  /* Re-fill entries */
	  diff[i] = data[j] - data[j-1];
	  MINBITS(diff[i],minbits[i]);
	}
    }
  
  /* Update XN value in first frame */
  XN = data[(ns-1)-points_remaining];
  if ( swapflag ) ms_gswap4 (&XN);
  
  /* End of data.  Pad current frame and optionally rest of block */
  /* Do not pad and output a completely empty block */
  if ( ! EMPTY_BLOCK(fn,wn) )
    {
      *pnframes = pad_steim_frame (dframes, fn, wn, nf, swapflag, pad);
    }
  else
    {
      *pnframes = 0;
    }
  
  *pnsamples = ns - points_remaining;
  
  return 0;
}


/************************************************************************
 *  msr_pack_steim2:							*
 *	Pack data into STEIM1 data frames.				*
 *  return:								*
 *	0 on success.							*
 *	-1 on error.                                                    *
 ************************************************************************/
int msr_pack_steim2
 (DFRAMES      *dframes,	/* ptr to data frames                   */
  int32_t      *data,		/* ptr to unpacked data array           */
  int32_t       d0,		/* first difference value               */
  int		ns,		/* number of samples to pack            */
  int		nf,		/* total number of data frames to pack  */
  int		pad,		/* flag to specify padding to nf        */
  int	       *pnframes,	/* number of frames actually packed     */
  int	       *pnsamples,	/* number of samples actually packed    */
  int           swapflag)       /* if data should be swapped            */
{
  int		points_remaining = ns;
  int           points_packed = 0;
  int32_t       diff[7];        /* array of differences                 */
  uint8_t       minbits[7];     /* array of minimum bits for diffs      */
  int		i, j;
  int		mask;
  int		ipt = 0;	/* index of initial data to pack.	*/
  int		fn = 0;		/* index of initial frame to pack.	*/
  int		wn = 2;		/* index of initial word to pack.	*/
  
  /* Calculate initial difference and minbits buffers */
  diff[0] = d0;
  MINBITS(diff[0],minbits[0]);
  for (i=1; i < 7 && i < ns; i++)
    {
      diff[i] = data[i] - data[i-1];
      MINBITS(diff[i],minbits[i]);
    }
  
  dframes->f[fn].ctrl = 0;
  
  /* Set X0 and XN values in first frame */
  X0 = data[0];
  if ( swapflag ) ms_gswap4 (&X0);
  dframes->f[0].ctrl = (dframes->f[0].ctrl<<2) | STEIM2_SPECIAL_MASK;
  XN = data[ns-1];
  if ( swapflag ) ms_gswap4 (&XN);
  dframes->f[0].ctrl = (dframes->f[0].ctrl<<2) | STEIM2_SPECIAL_MASK;
  
  while (points_remaining > 0)
    {
      points_packed = 0;
      
      /* Pack the next available datapoints into the most compact form */
      if (BIT4PACK(0,points_remaining))
	{
	  PACK(4,7,0x0000000f,02)
	  if ( swapflag ) ms_gswap4 (&dframes->f[fn].w[wn].fw);
	  mask = STEIM2_567_MASK;
	  points_packed = 7;
	}
      else if (BIT5PACK(0,points_remaining))
	{
	  PACK(5,6,0x0000001f,01)
	    if ( swapflag ) ms_gswap4 (&dframes->f[fn].w[wn].fw);
	  mask = STEIM2_567_MASK;
	  points_packed = 6;
	}
      else if (BIT6PACK(0,points_remaining))
	{
	  PACK(6,5,0x0000003f,00)
	  if ( swapflag ) ms_gswap4 (&dframes->f[fn].w[wn].fw);
	  mask = STEIM2_567_MASK;
	  points_packed = 5;
	}
      else if (BYTEPACK(0,points_remaining))
	{
	  mask = STEIM2_BYTE_MASK;
	  for (j=0; j<4; j++) dframes->f[fn].w[wn].byte[j] = diff[j];
	  points_packed = 4;
	}
      else if (BIT10PACK(0,points_remaining))
	{
	  PACK(10,3,0x000003ff,03)
	  if ( swapflag ) ms_gswap4 (&dframes->f[fn].w[wn].fw);
	  mask = STEIM2_123_MASK;
	  points_packed = 3;
	}
      else if (BIT15PACK(0,points_remaining))
	{
	  PACK(15,2,0x00007fff,02)
	  if ( swapflag ) ms_gswap4 (&dframes->f[fn].w[wn].fw);
	  mask = STEIM2_123_MASK;
	  points_packed = 2;
	}
      else if (BIT30PACK(0,points_remaining))
	{
	  PACK(30,1,0x3fffffff,01)
	  if ( swapflag ) ms_gswap4 (&dframes->f[fn].w[wn].fw);
	  mask = STEIM2_123_MASK;
	  points_packed = 1;
	}
      else
	{
	  ms_log (2, "msr_pack_steim2(%s): Unable to represent difference in <= 30 bits\n",
		  PACK_SRCNAME);
	  return -1;
	}
      
      /* Append mask for this word to current mask */
      dframes->f[fn].ctrl = (dframes->f[fn].ctrl<<2) | mask;
      
      points_remaining -= points_packed;
      ipt += points_packed;
      
      /* Check for full frame or full block */
      if (++wn >= VALS_PER_FRAME)
	{
	  if ( swapflag ) ms_gswap4 (&dframes->f[fn].ctrl);
	  /* Reset output index to beginning of frame */
	  wn = 0;
	  /* If block is full, output block and reinitialize */
	  if (++fn >= nf) break;
	  dframes->f[fn].ctrl = 0;
	}
      
      /* Shift and re-fill difference and minbits buffers */
      for ( i=points_packed; i < 7; i++ )
	{
	  /* Shift remaining buffer entries */
	  diff[i-points_packed] = diff[i];
	  minbits[i-points_packed] = minbits[i];
	}
      for ( i=7-points_packed,j=ipt+(7-points_packed); i < 7 && j < ns; i++,j++ )
	{
	  /* Re-fill entries */
	  diff[i] = data[j] - data[j-1];
	  MINBITS(diff[i],minbits[i]);
	}
    }
  
  /* Update XN value in first frame */
  XN = data[(ns-1)-points_remaining];
  if ( swapflag ) ms_gswap4 (&XN);
  
  /* End of data.  Pad current frame and optionally rest of block */
  /* Do not pad and output a completely empty block */
  if ( ! EMPTY_BLOCK(fn,wn) )
    {
      *pnframes = pad_steim_frame (dframes, fn, wn, nf, swapflag, pad);
    }
  else
    {
      *pnframes = 0;
    }
  
  *pnsamples = ns - points_remaining;
  
  return 0;
}


/************************************************************************
 *  pad_steim_frame:							*
 *	Pad the rest of the data record with null values,		*
 *	and optionally the rest of the total number of frames.		*
 *  return:								*
 *	total number of frames in record.				*
 ************************************************************************/
static int pad_steim_frame
 (DFRAMES      *dframes,
  int		fn,	    	/* current frame number.		*/
  int	    	wn,		/* current work number.			*/
  int	    	nf,		/* total number of data frames.		*/
  int		swapflag,	/* flag to swap byte order of data.	*/
  int	    	pad)		/* flag to pad # frames to nf.		*/
{
  /* Finish off the current frame */
  if (wn < VALS_PER_FRAME && fn < nf)
    {
      for (; wn < VALS_PER_FRAME; wn++)
	{
	  dframes->f[fn].w[wn].fw = 0;
	  dframes->f[fn].ctrl = (dframes->f[fn].ctrl<<2) | STEIM1_SPECIAL_MASK;
	}
      if ( swapflag ) ms_gswap4 (&dframes->f[fn].ctrl);
      fn++;
    }
  
  /* Fill the remaining frames in the block */
  if (pad)
    {
      for (; fn<nf; fn++)
	{
	  dframes->f[fn].ctrl = STEIM1_SPECIAL_MASK;	/* mask for ctrl */
	  for (wn=0; wn<VALS_PER_FRAME; wn++)
	    {
	      dframes->f[fn].w[wn].fw = 0;
	      dframes->f[fn].ctrl = (dframes->f[fn].ctrl<<2) | STEIM1_SPECIAL_MASK;
	    }
	  
	  if ( swapflag ) ms_gswap4 (&dframes->f[fn].ctrl);
	}
    }
  
  return fn;
}


/************************************************************************
 *  msr_pack_text:						       	*
 *	Pack text data into text format.  Split input data on line	*
*	breaks so as to not split lines between records.		* 
*	Return: 0 on success, -1 on error.				*
 ************************************************************************/
int msr_pack_text
 (char 	       *packed,         /* output data array - packed.		*/
  char	       *data,		/* input data array - unpacked.		*/
  int		ns,		/* desired number of samples to pack.	*/
  int		max_bytes,	/* max # of bytes for output buffer.	*/
  int		pad,		/* flag to specify padding to max_bytes.*/
  int	       *pnbytes,	/* number of bytes actually packed.	*/
  int	       *pnsamples)	/* number of samples actually packed.	*/
{
  int points_remaining = ns;	/* number of samples remaining to pack.	*/
  int		last = -1;
  int		nbytes;
  int		i;
  
  /* Split lines only if a single line will not fit in 1 record */
  if (points_remaining > max_bytes)
    {
      /* Look for the last newline that will fit in output buffer */
      for (i=max_bytes-1; i>=0; i--)
	{
	  if (data[i] == '\n') {
	    last = i;
	    break;
	  }
	}
      if (last < 0) last = max_bytes - 1;
    }
  
  if (last < 0) last = points_remaining - 1;
  nbytes = last + 1;
  memcpy (packed, data, nbytes);
  packed += nbytes;
  max_bytes -= nbytes;
  *pnbytes = nbytes;
  *pnsamples = nbytes;
  points_remaining -= nbytes;
  
  /* Pad miniSEED block if necessary */
  if (pad)
    {
      memset ((void *)packed, 0, max_bytes);
      *pnbytes += max_bytes;
    }
  
  *pnsamples = ns - points_remaining;
  
  return 0;
}
