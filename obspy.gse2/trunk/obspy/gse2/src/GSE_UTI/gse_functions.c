#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "gse_header.h"
#include "gse_types.h"
#include "buf.h"
#define     MODULO_VALUE 100000000

 
/*********************************************************************
  Function: check_sum
    This function computes the GSE2.0 checksum used in the CHK2 line
    and is based on the original algorithm (compute_checksum). 
    The funktionality is changed, so that now the function itself
    returns the checksum value and the input contains a previously
    computed checksum (or zero). Hence, a discontineous data stream
    can be check-summed, e.g. the block data structure of the 
    lennartz MARS-system. 

    St. Stange, 21.4.1998 , last change 12.4.2001
*
*	980617
*	dstoll: change abs() to labs() to keep gcc happy
*	dstoll:	do not return absolute value! This will lead to a wrong
*	        start value for checksum at subsequent invocations of check_sum!
*********************************************************************/

long check_sum (signal_int, number_of_samples, checksum)
        long     *signal_int;
        int     number_of_samples;
        long     checksum;
{
        int     i_sample;
        long     sample_value;
        long     modulo;
 
        modulo = MODULO_VALUE;
        for (i_sample=0; i_sample < number_of_samples; i_sample++)
        {
 
               /* check on sample value overflow */
 
                sample_value = signal_int[i_sample];
 
                if (labs(sample_value) >= modulo)
                {
                        sample_value = sample_value -
                                (sample_value/modulo)*modulo;
                }
 
                /* add the sample value to the checksum */
 
                checksum += sample_value;
 
                /* apply modulo division to the checksum */
 
                if (labs(checksum) >= modulo)
                {
                        checksum = checksum -
                                (checksum/modulo)*modulo;
                }
        }
 
        /* return value of the checksum */
/*
 *	dstoll 980617: do not return abs value but "as is"
 */
        return (checksum);

}	/* end of check_sum */
 
/*********************************************************************
  Function: compress_6b
    This routine computes the 6Byte encoding of integer data according
    GSE2.0 based on cmprs6.f in CODECO by Urs Kradolfer. Again, here we
    can cope with consecutive chunks of a data series.
    Input is the data series (integer) and the # of samples. The character
    representation of the data is successively stored to the dynamic
    character buffer written by Andreas Greve.
    Attention: Clipping is at 2**27 - 1 although it looks like 2**28 -1
    in the FORTRAN-Code.

    St. Stange, 28.4.1998
*********************************************************************/

int compress_6b (long *data, int n_of_samples)
{
  static char achar[] =
       " +-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
/*                    2**5 2**10 2**15  2**20    2**25     2**27     */ 
  static long expo_2[] = { 0, 32, 1024, 32768, 1048576, 33554432, 134217728 };
/*              -1 +       2**5  2**10  2**15   2**20     2**25     */ 
  static long expo_2m1_o[] = { 01, 037, 01777, 077777, 03777777, 0177777777 };
  int nflag;
  int mflag = 32;
  long jc, value, si;
  int case_expo;

  for (si = 0; si < n_of_samples; si++)
  {
	value = data[si];
	nflag = 1;
	if (value < 0 ) 	/* convert negative numbers */
		{ nflag += 16; value = -value; }

				/* clip at 2**27 -1 */
	value = (value >= expo_2[6]) ? expo_2[6] - 1 : value;

	frexp ((double)value, &case_expo);  /* compute the exponent (base 2) */
	case_expo = case_expo/5;	/* and reduce by integer division */

	if (case_expo > 5 || case_expo < 0) return -1;

	for ( ; case_expo > 0; case_expo--)
	{				/* one character per turn */
		jc = value/expo_2[case_expo] + nflag + mflag;
		/*if (jc > 64 || jc < 1) return jc;*/
		buf_putchar(achar[jc]);		/* store a character */
		value = value & expo_2m1_o[case_expo];
		nflag = 1;
	}
		
	jc = value + nflag;		/* one character to go */
	buf_putchar(achar[jc]);		/* store a character */

  }
        return 0;

}	/* end of compress_6b */

/*********************************************************************
  Function: diff_2nd
    This routine computes the second differences of a data stream
    according to the format GSE2.0, based on dif1.f by Urs Kradolfer.
    The data stream is an long vector, the 2nd differences are
    returned in the same vector.
    The cont_flag and the static variables enable a continuation of
    the computation of the 2nd differences if the data comes in chunks.
    cont_flag = 0 indicates a new data set
	
    St. Stange, 21.4.1998
*********************************************************************/

void diff_2nd (long *data, int n_of_samples, int cont_flag)
{
  static long t1, t2, t3;	/* internal temporary variables */
  int si = 0;			/* index set to 0 for continuation*/

  if (cont_flag == 0) 		/* initialize for new data set */
	{ t3 = data[0]; t2 = -2*t3; si = 1;}

  for ( ; si < n_of_samples; si++)	
  {	t1 = data[si];
	data[si] = t1 + t2;
	t2 = t3 - 2*t1;
	t3 = t1;
  }

}	/* end of diff_2nd */
 
/*********************************************************************
  Function: write_header
    This function dumps the structure header to the specified file.
    Format is according to GSE2.0.

    St. Stange, 27.4.1998
*********************************************************************/

void write_header(FILE *fp, struct header *head)
{
  fprintf(fp,"WID2 %4d/%02d/%02d %02d:%02d:%06.3f %-5s %-3s %-4s %-3s %8d %11.6f %10.4e %7.3f %-6s %5.1f %4.1f\n",
	head->d_year, head->d_mon, head->d_day, head->t_hour,
	head->t_min, head->t_sec, head->station, head->channel, head->auxid,
	head->datatype, head->n_samps, head->samp_rate, head->calib,
	head->calper, head->instype, head->hang, head->vang);

}	/* end of write_header */

/*********************************************************************
* Function: decomp_6b
*   This routine evolves the data series from the 6Byte encoding according
*   GSE2.0 based on dcomp6.f in CODECO by Urs Kradolfer. 
*   Input is the character representation (meaning the file pointer to it),
*   the number of samples to be expected and the pointer to the data.
*   Output is the data series in LONG (has to be allocated elsewhere!). 
*   The GSE file must be opened and positioned to or before the "DAT2" line.
*   Returns actual # of samples or -1 as error code.
*   Calls no other routines.
*   St. Stange, 1.10.1998  , verified for PC-byte-sex 11.4.2001
*   6.12.2002: skips blancs and new lines within the data body. Unfortunately this
*   disables it to tell the end of the data stream, hence n_of_samples must be correct!
*********************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int decomp_6b (FILE *fop, int n_of_samples, long *dta)
{
  static int ichar[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,2,3,4,5,6,7,
             8,9,10,11,0,0,0,0,0,0,0,12,13,14,15,16,17,18,19,20,21,22,
             23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,0,0,0,0,0,0,
             38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,
             57,58,59,60,61,62,63,0,0,0,0,0,0},/*1 more than in FORTRAN*/
             isign=020, ioflow=040, mask1=017, mask2=037, m127=0177;

  int i, ibuf=-1, k, inn, jsign=0, joflow=0;
  long itemp;
  char cbuf[82]=" ";
  
  if (n_of_samples == 0) { printf ("decomp_6b: no action.\n"); return 0; }
  
  					/* read until we encounter DAT2 */
  while (isspace(cbuf[0]) || strncmp(cbuf,"DAT2",4)) {
    if (fgets (cbuf,82,fop) == NULL) { 	printf ("decomp_6b: No DAT2 found!\n"); 
    					return -1; } }
  	
  if (fgets (cbuf,82,fop) == NULL) 	/* read first char line */
  	{printf ("decomp_6b: Whoops! No data after DAT2.\n"); return -1; }
  for (i = 0; i < n_of_samples; i++)		/* loop over expected samples */
  {
  	ibuf += 1;
  	if (ibuf > 79 || isspace(cbuf[ibuf])) { 
  	  if (fgets (cbuf,82,fop) == NULL) 	/* get next line */
  		{printf ("decomp_6b: missing input line?\n"); return -1; }
  	  if (!(strncmp(cbuf,"CHK2",4)))
  		{printf ("decomp_6b: CHK2 reached prematurely!\n"); return i; }
  	  ibuf = 0;
  	}

  	/* get ascii code of input character, strip off any higher bits
  	(don't know whether it does what it says) and get number representation */
  	
  	k = (int)((int)cbuf[ibuf] & m127); inn = ichar[k];
  	
  	jsign = (inn & isign);			/* get sign bit */
  	joflow = (inn & ioflow); 	/* get continuation bit if any */
  	itemp = (long)(inn & mask1);	/* remove dispensable bits and store */
  	
  	while (joflow != 0) 		/* loop over other bytes in sample */
  	{
  	  itemp <<= 5;			/* multiply with 32 for next byte */
  	  ibuf += 1;
  	  if (ibuf > 79 || isspace(cbuf[ibuf])) {
  	    if (fgets (cbuf,82,fop) == NULL) 	/* get next line */
  		{printf ("decomp_6b: missing input line.\n"); return -1; }
  	    ibuf = 0;
	  }
  					/* now the same procedure as above */
  	  k = (int)((int)cbuf[ibuf] & m127); inn = ichar[k];
	  joflow = (inn & ioflow); 	/* get continuation bit if any */
	  itemp = itemp + (long)(inn & mask2);	/* remove bits and store */
	  
	} /* finish up sample if there is no further continuation bit */
	
	*(dta+i) = itemp;		/* store data */

	if (jsign != 0) *(dta+i) = -*(dta+i);	/* evaluate sign bit */
	
  }		/* end of loop over samples */

  return i;				/* return actual # of samples read */

}	/* end of decomp_6b */
/*********************************************************************
  Function: rem_2nd_diff
    This routine removes the second differences of a data stream
    according to the format GSE2.0, based on remdif1.f by Urs Kradolfer.
    The data stream is a long vector, the 2nd differences are
    returned in the same vector.
	
    St. Stange, 2.10.1998
*********************************************************************/

void rem_2nd_diff (long *data, int n_of_samples)
{

 int idx;

 /* the first sample (data[0]) remains the same */

 data[1] = data[1] + data[0];  /* first grip on second sample */

 /* loop over all others: */

 for ( idx=2; idx < n_of_samples; idx++)
 {	data[idx] = data[idx] + data[idx-1];
	data[idx-1] = data[idx-1] + data[idx-2];	/* back one step */
 }
 		/* don't forget the last sample! */
 data[n_of_samples -1] = data[n_of_samples -1] + data[n_of_samples -2];

}	/* end of rem_2nd_diff */
 
/*********************************************************************
*  Function: read_header
*    This function looks for the next "WID2"-line in the specified file.
*    The information according to GSE2.0 is extracted to the header structure.
*
*    St. Stange, 11.4.2001
*********************************************************************/

int read_header(FILE *fop, struct header *hed)
{
        char iline[121];

	while (fgets(iline,120,fop) != NULL)
	{
	  if (!strncmp(iline,"WID2",4))
	  {		/* found a WID2 line */
	  strcpy(hed->station,"     ");    /* "initialize" characters */
          strcpy(hed->channel,"   ");
          strcpy(hed->auxid,"    ");
          strcpy(hed->datatype,"   ");
          strcpy(hed->instype,"      ");

	  sscanf(iline,"%*s%4d%*1c%2d%*1c%2d%*1c%2d%*1c%2d%*1c%6f",&hed->d_year,
                &hed->d_mon,&hed->d_day,&hed->t_hour,&hed->t_min,&hed->t_sec);
          strncpy(hed->station,&iline[29],5);
          strncpy(hed->channel,&iline[35],3);
          strncpy(hed->auxid,&iline[39],4);
          strncpy(hed->datatype,&iline[44],3);
          strncpy(hed->instype,&iline[88],6);
	  sscanf(iline,"%*48c%8d%*1c%11f%*1c%10f%*1c%7f%*8c%5f%*1c%4f",&hed->n_samps,
                &hed->samp_rate,&hed->calib,&hed->calper,&hed->hang,&hed->vang);
	  return 0;
	  }
	}		/* next line */
	/*printf ("read_header: EndOfFile reached!\n");*/
	return -1;
}       /* end of read_header */
