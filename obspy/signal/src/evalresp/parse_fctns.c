/* parse_fctns.c */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

/*
    8/28/2001 -- [ET]  Added "!= 0" to several conditionals to squelch
                       "possibly incorrect assignment" warnings.
   10/19/2005 -- [ET]  Modified to handle case where file contains
                      "B052F03 Location:" and nothing after it on line.
 */

#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif

#include <string.h>
#include <stdlib.h>

#include "./evresp.h"
double atof();

/* parse_pref:  used to parse the prefix from a non-comment RESP file line (returns the blockette
                number and starting field number for that line as arguments, return values are -1
                for comment lines (if they sneak through), 1 for lines that start with 'B' and
                zero for lines that are not comments and don't start with a 'B' */

int parse_pref(int *blkt_no, int *fld_no, char *line) {
  char fldstr[FLDSTRLEN], blktstr[BLKTSTRLEN];

  strncpy(fldstr,"",FLDSTRLEN);
  strncpy(blktstr,"",BLKTSTRLEN);
  if(*line != 'B' || strlen(line) < 7)
    return(0);

  strncpy(blktstr,(line+1),3);
  strncpy(fldstr,(line+5),2);
  *(blktstr+3) = '\0';
  *(fldstr+2) = '\0';

  if(!is_int(blktstr))
    error_return(UNDEF_PREFIX,"parse_pref; prefix '%s' cannot be %s",
                 blktstr,"converted to a blockette number");
  *blkt_no = atoi(blktstr);
  if(!is_int(fldstr))
    error_return(UNDEF_PREFIX,"parse_pref; prefix '%s' cannot be %s",
                 fldstr,"converted to a blockette number");
  *fld_no = atoi(fldstr);
  return(1);
}

/* parse_pz:  parses RESP file poles & zeros (in a Blockette [53] or [43]). Errors cause program
              termination.  The blockette and field numbers are checked as the file is parsed.
              The field-format of the RESP file is assumed to be fixed (i.e. the same number of
              fields per line in the same order), but the actual positions of those fields on the
              line will not effect the parsing routines.  For this routine to work, the lines must
              contain evalresp-3.0 style prefixes */

void parse_pz(FILE *fptr, struct blkt *blkt_ptr, struct stage *stage_ptr) {
  int i, blkt_typ, check_fld;
  int blkt_read;
  int npoles, nzeros;
  char field[MAXFLDLEN], line[MAXLINELEN];

  /* first get the response type (from the input line).  Note: if is being called from
     a blockette [53] the first field expected is a F03, if from a blockette [43], the
     first field should be a F05 */

  if(FirstField != 3 && FirstField != 5) {
    error_return(PARSE_ERROR,"parse_pz; %s%s%s%2.2d","(return_field) fld ",
                 "number does not match expected value\n\tfld_xpt=F03 or F05",
                 ", fld_found=F", FirstField);
  }

  if(FirstField == 3)
    blkt_read = 53;
  else
    blkt_read = 43;

  parse_field(FirstLine,0,field);
  if(strlen(field) != 1) {
    error_return(PARSE_ERROR,"parse_pz; parsing (Poles & Zeros), illegal filter type ('%s')",
                 field);
  }
  blkt_typ = *field;
  switch (blkt_typ) {
  case 'A':
    blkt_ptr->type = LAPLACE_PZ;
    break;
  case 'B':
    blkt_ptr->type = ANALOG_PZ;
    break;
  case 'D':
    blkt_ptr->type = IIR_PZ;
    break;
  default:
    error_return(PARSE_ERROR, "parse_pz; parsing (Poles & Zeros), unexpected filter type ('%c')",
                 *field);
  }

  /* set the check-field counter to the next expected field */

  check_fld = FirstField+1;

  /* then, if is a B053F04, get the stage sequence number (from the file) */

  if(check_fld == 4) {
    get_field(fptr,field,blkt_read,check_fld++,":",0);
    stage_ptr->sequence_no = get_int(field);
    curr_seq_no = stage_ptr->sequence_no;
  }

  /* next should be the units (in first, then out) */

  get_line(fptr,line,blkt_read,check_fld++,":");
  stage_ptr->input_units = check_units(line);

  get_line(fptr,line,blkt_read,check_fld++,":");
  stage_ptr->output_units = check_units(line);

  /* then the A0 normalization factor */

  get_field(fptr,field,blkt_read,check_fld++,":",0);
  blkt_ptr->blkt_info.pole_zero.a0 = get_double(field);

  /* the A0 normalization frequency */

  get_field(fptr,field,blkt_read,check_fld++,":",0);
  blkt_ptr->blkt_info.pole_zero.a0_freq = get_double(field);

  /* the number of zeros */

  get_field(fptr,field,blkt_read,check_fld,":",0);
  nzeros = get_int(field);
  blkt_ptr->blkt_info.pole_zero.nzeros = nzeros;

  /* remember to allocate enough space for the number of zeros to follow */

  blkt_ptr->blkt_info.pole_zero.zeros = alloc_complex(nzeros);

  /* set the expected field to the current value (9 or 10 for [53] or [43])
     to the current value + 5 (14 or 15 for [53] or [43] respectively) */

  check_fld += 5;

  /* the number of poles */

  get_field(fptr,field,blkt_read,check_fld,":",0);
  npoles = get_int(field);
  blkt_ptr->blkt_info.pole_zero.npoles = npoles;

  /* remember to allocate enough space for the number of poles to follow */

  blkt_ptr->blkt_info.pole_zero.poles = alloc_complex(npoles);

  /* set the expected field to the current value (14 or 15 for [53] or [43])
     to the current value - 4 (10 or 11 for [53] or [43] respectively) */

  check_fld -= 4;

  /* get the zeros */

  for(i = 0; i < nzeros; i++) {
    get_line(fptr,line,blkt_read,check_fld," ");
    parse_field(line,1,field);
    if(!is_real(field))
      error_return(PARSE_ERROR,"parse_pz: %s%s%s",
                   "zeros must be real numbers (found '",field,"')");
    blkt_ptr->blkt_info.pole_zero.zeros[i].real = atof(field);
    parse_field(line,2,field);
    if(!is_real(field))
      error_return(PARSE_ERROR,"parse_pz: %s%s%s",
                   "zeros must be real numbers (found '",field,"')");
    blkt_ptr->blkt_info.pole_zero.zeros[i].imag = atof(field);
  }

  /* set the expected field to the current value (10 or 11 for [53] or [43])
     to the current value + 5 (15 or 16 for [53] or [43] respectively) */

  check_fld += 5;

  /* and then get the poles */

  for(i = 0; i < npoles; i++) {
    get_line(fptr,line,blkt_read,check_fld," ");
    parse_field(line,1,field);
    if(!is_real(field))
      error_return(PARSE_ERROR,"parse_pz: %s%s%s",
                   "poles must be real numbers (found '",field,"')");
    blkt_ptr->blkt_info.pole_zero.poles[i].real = atof(field);
    parse_field(line,2,field);
    if(!is_real(field))
      error_return(PARSE_ERROR,"parse_pz: %s%s%s",
                   "poles must be real numbers (found '",field,"')");
    blkt_ptr->blkt_info.pole_zero.poles[i].imag = atof(field);
  }

}

/* parse_iir_coeff:  parses the RESP file blockettes for COEFF filters (in a Blockette [54] or [44]).
	Handles the case of a digital IIR filtering
	Derived from parse_coeff : see below by IGD
                 Errors cause the program to terminate.  The blockette and field numbers are
                 checked as the file is parsed. As with parse_pz(), for this routine to work,
                 the lines must contain evalresp-3.0 style prefixes 
	I.Dricker (i.dricker@isti.com)	06/27/00		  		*/

void parse_iir_coeff(FILE *fptr, struct blkt *blkt_ptr, struct stage *stage_ptr) {
  int i, blkt_typ, check_fld;
  int blkt_read;
  int ncoeffs, ndenom;
  char field[MAXFLDLEN], line[MAXLINELEN];

  /* first get the response type (from the input line).  Note: if is being called from
     a blockette [54] the first field expected is a F03, if from a blockette [44], the
     first field should be a F05 */

  if(FirstField != 3 && FirstField != 5) {
    error_return(PARSE_ERROR,"parse_coeff; %s%s%s%2.2d","(return_field) fld ",
                 "number does not match expected value\n\tfld_xpt=F03 or F05",
                 ", fld_found=F", FirstField);
  }

  if(FirstField == 3)
    blkt_read = 54;
  else
    blkt_read = 44;

  parse_field(FirstLine,0,field);
  if(strlen(field) != 1) {
    error_return(PARSE_ERROR, "parse_coeff; parsing (IIR_COEFFS), illegal filter type ('%s')",
                 field);
  }
  blkt_typ = *field;
  switch (blkt_typ) {
  case 'D':
    blkt_ptr->type = IIR_COEFFS;
    break;
  default:
    error_return(PARSE_ERROR,"parse_coeff; parsing (IIR_COEFFS), unexpected filter type ('%c')",
                 *field);
  }

  /* set the check-field counter to the next expected field */

  check_fld = FirstField+1;

  /* then, if is a B054F04, get the stage sequence number (from the file) */

  if(check_fld == 4) {
    get_field(fptr,field,blkt_read,check_fld++,":",0);
    stage_ptr->sequence_no = get_int(field);
    curr_seq_no = stage_ptr->sequence_no;
  }

  /* next should be the units (in first, then out) */

  get_line(fptr,line,blkt_read,check_fld++,":");
  stage_ptr->input_units = check_units(line);

  get_line(fptr,line,blkt_read,check_fld++,":");
  stage_ptr->output_units = check_units(line);

  /* the number of coefficients */

  get_field(fptr,field,blkt_read,check_fld++,":",0);
  ncoeffs = get_int(field);
  blkt_ptr->blkt_info.coeff.nnumer = ncoeffs;


  /* remember to allocate enough space for the number of coefficients to follow */

  blkt_ptr->blkt_info.coeff.numer = alloc_double(ncoeffs);

  /* set the expected field to the current value (8 or 9 for [54] or [44])
     to the current value + 2 (10 or 11 for [54] or [44] respectively) */

  check_fld += 2;

  /* the number of denominators */

  get_field(fptr,field,blkt_read,check_fld,":",0);
  ndenom = get_int(field);

  /* if the number of denominators is zero, then is not an IIR filter. */
      
  if(ndenom == 0) {
    error_return(UNRECOG_FILTYPE, "%s%s",
                 "parse_coeff; This is not IIR filter , because number of denominators is zero!\n",
                 "\tshould be represented as blockette [53] filters");
  }
  blkt_ptr->blkt_info.coeff.ndenom = ndenom;


  /* remember to allocate enough space for the number of coefficients to follow */

  blkt_ptr->blkt_info.coeff.denom = alloc_double(ndenom);


  /* set the expected field to the current value (10 or 11 for [54] or [44])
     to the current value - 2 (8 or 9 for [54] or [44] respectively) */

  check_fld -= 2;

  /* the coefficients */

  for(i = 0; i < ncoeffs; i++) {
    get_field(fptr,field,blkt_read,check_fld," ",1);
    if(!is_real(field))
      error_return(PARSE_ERROR,"parse_coeff: %s%s%s",
                   "numerators must be real numbers (found '",field,"')");
    blkt_ptr->blkt_info.coeff.numer[i] = atof(field);
  }

check_fld += 3;  /*IGD : we need field 11 now */

   for(i = 0; i < ndenom; i++) {
    get_field(fptr,field,blkt_read,check_fld," ",1);
    if(!is_real(field))
      error_return(PARSE_ERROR,"parse_coeff: %s%s%s",
                   "denominators must be real numbers (found '",field,"')");
    blkt_ptr->blkt_info.coeff.denom[i] = atof(field);
  } 

}



/* parse_coeff:  parses the RESP file blockettes for COEFF filters (in a Blockette [54] or [44]).
                 Errors cause the program to terminate.  The blockette and field numbers are
                 checked as the file is parsed. As with parse_pz(), for this routine to work,
                 the lines must contain evalresp-3.0 style prefixes */

void parse_coeff(FILE *fptr, struct blkt *blkt_ptr, struct stage *stage_ptr) {
  int i, blkt_typ, check_fld;
  int blkt_read;
  int ncoeffs, ndenom;
  char field[MAXFLDLEN], line[MAXLINELEN];

  /* first get the response type (from the input line).  Note: if is being called from
     a blockette [54] the first field expected is a F03, if from a blockette [44], the
     first field should be a F05 */

  if(FirstField != 3 && FirstField != 5) {
    error_return(PARSE_ERROR,"parse_coeff; %s%s%s%2.2d","(return_field) fld ",
                 "number does not match expected value\n\tfld_xpt=F03 or F05",
                 ", fld_found=F", FirstField);
  }

  if(FirstField == 3)
    blkt_read = 54;
  else
    blkt_read = 44;

  parse_field(FirstLine,0,field);
  if(strlen(field) != 1) {
    error_return(PARSE_ERROR, "parse_coeff; parsing (FIR_ASYM), illegal filter type ('%s')",
                 field);
  }
  blkt_typ = *field;
  switch (blkt_typ) {
  case 'D':
    blkt_ptr->type = FIR_ASYM;
    break;
  default:
    error_return(PARSE_ERROR,"parse_coeff; parsing (FIR_ASYM), unexpected filter type ('%c')",
                 *field);
  }

  /* set the check-field counter to the next expected field */

  check_fld = FirstField+1;

  /* then, if is a B054F04, get the stage sequence number (from the file) */

  if(check_fld == 4) {
    get_field(fptr,field,blkt_read,check_fld++,":",0);
    stage_ptr->sequence_no = get_int(field);
    curr_seq_no = stage_ptr->sequence_no;
  }

  /* next should be the units (in first, then out) */

  get_line(fptr,line,blkt_read,check_fld++,":");
  stage_ptr->input_units = check_units(line);

  get_line(fptr,line,blkt_read,check_fld++,":");
  stage_ptr->output_units = check_units(line);

  /* the number of coefficients */

  get_field(fptr,field,blkt_read,check_fld++,":",0);
  ncoeffs = get_int(field);
  blkt_ptr->blkt_info.fir.ncoeffs = ncoeffs;

  /* remember to allocate enough space for the number of coefficients to follow */

  blkt_ptr->blkt_info.fir.coeffs = alloc_double(ncoeffs);

  /* set the expected field to the current value (8 or 9 for [54] or [44])
     to the current value + 2 (10 or 11 for [54] or [44] respectively) */

  check_fld += 2;

  /* the number of denominators */

  get_field(fptr,field,blkt_read,check_fld,":",0);
  ndenom = get_int(field);

  /* if the number of denominators is not zero, then is not a FIR filter.
     evalresp cannot evaluate IIR and Analog filters that are represented
     as numerators and denominators (and it is not recommended that these
     filter types be represented this way because of the loss of accuracy).
     These filter types should be represented as poles and zeros.  If one
     of these unsupported filters is detected, print error and exit. */

  if(ndenom) {
    error_return(UNRECOG_FILTYPE, "%s%s",
                 "parse_coeff; Unsupported filter type, IIR and Analog filters\n",
                 "\tshould be represented as blockette [53] filters");
  }

  /* set the expected field to the current value (10 or 11 for [54] or [44])
     to the current value - 2 (8 or 9 for [54] or [44] respectively) */

  check_fld -= 2;

  /* the coefficients */

  for(i = 0; i < ncoeffs; i++) {
    get_field(fptr,field,blkt_read,check_fld," ",1);
    if(!is_real(field))
      error_return(PARSE_ERROR,"parse_coeff: %s%s%s",
                   "coeffs must be real numbers (found '",field,"')");
    blkt_ptr->blkt_info.fir.coeffs[i] = atof(field);
  }
  
}

/* parse_list:  parses the RESP file blockettes for list filters (in a Blockette [55] or [45]).
                Errors cause the program to terminate.  The blockette and field numbers are
                checked as the file is parsed. As with parse_pz(), for this routine to work,
                the lines must contain evalresp-3.0 style prefixes */
/* Ilya Dricker ISTI 06/21/00. I modify this routine to accomodate the form of the parsed blockette
		55 generated by SeismiQuery. Since currently the blockette 55 is not supported,
		we do not anicipate problems caused by this change */
void parse_list(FILE *fptr, struct blkt *blkt_ptr, struct stage *stage_ptr) {
  int i, blkt_typ = LIST;
  int blkt_read, check_fld;
  int nresp;
  char field[MAXFLDLEN], line[MAXLINELEN];
  int marker;
  int format = -1;
  /* set the filter type */

  blkt_ptr->type = blkt_typ;

  if(FirstField != 3 && FirstField != 5) {
    error_return(PARSE_ERROR,"parse_list; %s%s%s%2.2d","(return_field) fld ",
                 "number does not match expected value\n\tfld_xpt=F03 or F05",
                 ", fld_found=F", FirstField);
  }

  if(FirstField == 3)
    blkt_read = 55;
  else
    blkt_read = 45;

  /* set the check-field counter to the next expected field */

  check_fld = FirstField;

  /* then, if first is a B055F03, get the stage sequence number (from FirstLine)
     and the input units (from the file), otherwise, get the input units (from FirstLine) */

  if(check_fld == 3) {
    parse_field(FirstLine,0,field);
    stage_ptr->sequence_no = get_int(field);
    curr_seq_no = stage_ptr->sequence_no;
    check_fld++;
    get_line(fptr,line,blkt_read,check_fld++,":");
  }
  else {
    strncpy(line, FirstLine, MAXLINELEN);
    check_fld++;
  }

  /* the units (in first, then out) */

  stage_ptr->input_units = check_units(line);

  get_line(fptr,line,blkt_read,check_fld++,":");
  stage_ptr->output_units = check_units(line);

  /* the number of responses */

  get_field(fptr,field,blkt_read,check_fld++,":",0);
  nresp = get_int(field);
  blkt_ptr->blkt_info.list.nresp = nresp;

  /* remember to allocate enough space for the number frequency, amplitude, phase tuples
     that follow */

  blkt_ptr->blkt_info.list.freq = alloc_double(nresp);
  blkt_ptr->blkt_info.list.amp = alloc_double(nresp);
  blkt_ptr->blkt_info.list.phase = alloc_double(nresp);

  /* then get the response information */

  if (blkt_read == 55)	{ /* This is blockette 55 */

	/*we now check if the B055F07-11 has a numbering field and set format accordingly */
	marker = ftell(fptr);
  	get_line(fptr,line,blkt_read,check_fld," ");
	format = count_fields(line)-5; /*format == 0 if no number of responses in the file */
					/*format == 1 if  number of responses in the sec column*/
	fseek(fptr, marker, SEEK_SET); /* rewind back to we we've been before the test */
  	if (format != 0 && format != 1) /*Wrong format */
		      error_return(PARSE_ERROR,"parse_list: %s",
        	           "Unknown format for B055F07-11");
	
 	for(i = 0; i < nresp; i++) {
    	    get_line(fptr,line,blkt_read,check_fld," ");
	    parse_field(line,format,field);   /* Frequency */
	    if(!is_real(field))
	      error_return(PARSE_ERROR,"parse_list: %s%s%s",
        	           "freq vals must be real numbers (found '",field,"')");
	    blkt_ptr->blkt_info.list.freq[i] = atof(field);	      
            parse_field(line,1+format,field);  /* the amplitude of the Fourier transform */
    	    if(!is_real(field))
      			error_return(PARSE_ERROR,"parse_list: %s%s%s",
        		           "amp vals must be real numbers (found '",field,"')");
            blkt_ptr->blkt_info.list.amp[i] = atof(field);
            parse_field(line,3+format,field);   /* Phase of the transform */ 
            if(!is_real(field))
      			error_return(PARSE_ERROR,"parse_list: %s%s%s",
        		           "phase vals must be real numbers (found '",field,"')");
           blkt_ptr->blkt_info.list.phase[i] = atof(field);
  	}
  }
  else	{  /* This is blockette 45 - leave at as in McSweeny's version */
  	for(i = 0; i < nresp; i++) {
    	    get_line(fptr,line,blkt_read,check_fld," ");
	    parse_field(line,0,field);
	    if(!is_real(field))
	      error_return(PARSE_ERROR,"parse_list: %s%s%s",
        	           "freq vals must be real numbers (found '",field,"')");
	    blkt_ptr->blkt_info.list.freq[i] = atof(field);
	    parse_field(line,1,field);
    	    if(!is_real(field))
      			error_return(PARSE_ERROR,"parse_list: %s%s%s",
        		           "amp vals must be real numbers (found '",field,"')");
            blkt_ptr->blkt_info.list.amp[i] = atof(field);
            parse_field(line,3,field);
            if(!is_real(field))
      			error_return(PARSE_ERROR,"parse_list: %s%s%s",
        		           "phase vals must be real numbers (found '",field,"')");
           blkt_ptr->blkt_info.list.phase[i] = atof(field);
  	}
  }
}

/* parse_generic:  parses the RESP file for generic filters (in a Blockette [56] or [46]).
                   Errors cause the program to terminate.  The blockette and field numbers are
                   checked as the file is parsed. As with parse_pz(), for this routine to work,
                   the lines must contain evalresp-3.0 style prefixes */

void parse_generic(FILE *fptr, struct blkt *blkt_ptr, struct stage *stage_ptr) {
  int i, blkt_typ = GENERIC;
  int blkt_read, check_fld;
  int ncorners;
  char field[MAXFLDLEN], line[MAXLINELEN];

  /* set the filter type */

  blkt_ptr->type = blkt_typ;

  /* first get the stage sequence number (from the input line) */

  if(FirstField != 3 && FirstField != 5) {
    error_return(PARSE_ERROR,"parse_generic; %s%s%s%2.2d","(return_field) fld ",
                 "number does not match expected value\n\tfld_xpt=F03 or F05",
                 ", fld_found=F", FirstField);
  }
  if(FirstField == 3)
    blkt_read = 56;
  else
    blkt_read = 46;

  /* set the check-field counter to the next expected field */

  check_fld = FirstField;

  /* then, if first is a B056F03, get the stage sequence number (from FirstLine)
     and the input units (from the file), otherwise, get the input units (from FirstLine) */

  if(check_fld == 3) {
    parse_field(FirstLine,0,field);
    stage_ptr->sequence_no = get_int(field);
    curr_seq_no = stage_ptr->sequence_no;
    check_fld++;
    get_line(fptr,line,blkt_read,check_fld++,":");
  }
  else {
    strncpy(line, FirstLine, MAXLINELEN);
    check_fld++;
  }

  /* next (from the file) should be the units (in first, then out) */

  stage_ptr->input_units = check_units(line);

  get_line(fptr,line,blkt_read,check_fld++,":");
  stage_ptr->output_units = check_units(line);

  /* the number of responses */

  get_field(fptr,field,blkt_read,check_fld++,":",0);
  ncorners = get_int(field);
  blkt_ptr->blkt_info.generic.ncorners = ncorners;

  /* remember to allocate enough space for the number corner_frequency, corner_slope pairs
     that follow */

  blkt_ptr->blkt_info.generic.corner_freq = alloc_double(ncorners);
  blkt_ptr->blkt_info.generic.corner_slope = alloc_double(ncorners);

  /* then get the response information */

  for(i = 0; i < ncorners; i++) {
    get_line(fptr,line,blkt_read,check_fld," ");
    parse_field(line,1,field);
    if(!is_real(field))
      error_return(PARSE_ERROR,"parse_generic: %s%s%s",
                   "corner_freqs must be real numbers (found '",field,"')");
    blkt_ptr->blkt_info.generic.corner_freq[i] = atof(field);
    parse_field(line,2,field);
    if(!is_real(field))
      error_return(PARSE_ERROR,"parse_generic: %s%s%s",
                   "corner_slopes must be real numbers (found '",field,"')");
    blkt_ptr->blkt_info.generic.corner_slope[i] = atof(field);
  }
}

/* parse_deci:  parses the RESP file for decimation filters (in a Blockette [57] or [47]).
                Errors cause the program to terminate.  The blockette and field numbers are
                checked as the file is parsed. As with parse_pz(), for this routine to work,
                the lines must contain evalresp-3.0 style prefixes */

int parse_deci(FILE *fptr, struct blkt *blkt_ptr) {
  int  blkt_typ = DECIMATION;
  int blkt_read, check_fld;
  int sequence_no = 0;
  double srate;
  char field[MAXFLDLEN];

  /* set the filter type */

  blkt_ptr->type = blkt_typ;

  /* first get the stage sequence number (from the input line) */

  if(FirstField != 3 && FirstField != 5) {
    error_return(PARSE_ERROR,"parse_deci; %s%s%s%2.2d","(return_field) fld ",
                 "number does not match expected value\n\tfld_xpt=F03 or F05",
                 ", fld_found=F", FirstField);
  }
  if(FirstField == 3)
    blkt_read = 57;
  else
    blkt_read = 47;

  /* set the check-field counter to the next expected field */

  check_fld = FirstField;

  /* then, if first is a B057F03, get the stage sequence number (from FirstLine)
     and the input units (from the file), otherwise, get the input units (from FirstLine) */

  if(check_fld == 3) {
    parse_field(FirstLine,0,field);
    sequence_no = get_int(field);
    check_fld++;
    get_field(fptr,field,blkt_read,check_fld++,":",0);
  }
  else {
    parse_field(FirstLine,0,field);
    check_fld++;
  }

  /* next (from the file) input sample rate, convert to input sample interval */

  srate = get_double(field);
  if(srate)
    blkt_ptr->blkt_info.decimation.sample_int = 1.0 / srate;

  /* get the decimation factor and decimation offset */

  get_field(fptr,field,blkt_read,check_fld++,":",0);
  blkt_ptr->blkt_info.decimation.deci_fact = get_int(field);
  get_field(fptr,field,blkt_read,check_fld++,":",0);
  blkt_ptr->blkt_info.decimation.deci_offset = get_int(field);

  /* the estimated delay */

  get_field(fptr,field,blkt_read,check_fld++,":",0);
  blkt_ptr->blkt_info.decimation.estim_delay = get_double(field);

  /* and, finally, the applied correction.  Note:  the calculated delay is left undefined
     by this routine, although space does exist in this filter type for this parameter */

  get_field(fptr,field,blkt_read,check_fld++,":",0);
  blkt_ptr->blkt_info.decimation.applied_corr = get_double(field);

  /* return the sequence number of the stage for verification */

  return(sequence_no);

}

/* parse_gain:  parses the RESP file blockettes for gain filters (in a Blockette [58] or [48]).
                Errors cause the program to terminate.  The blockette and field numbers are
                checked as the file is parsed. As with parse_pz(), for this routine to work,
                the lines must contain evalresp-3.0 style prefixes */

int parse_gain(FILE *fptr, struct blkt *blkt_ptr) {
  int i, blkt_typ = GAIN;
  int blkt_read, check_fld;
  int sequence_no = 0;
  int nhist = 0;
  char field[MAXFLDLEN], line[MAXLINELEN];
 
 /* set the filter type */

  blkt_ptr->type = blkt_typ;

  /* first get the stage sequence number (from the input line) */

  if(FirstField != 3 && FirstField != 5) {
    error_return(PARSE_ERROR,"parse_gain; %s%s%s%2.2d","(return_field) fld ",
                 "number does not match expected value\n\tfld_xpt=F03 of F05",
                 ", fld_found=F", FirstField);
  }

  if(FirstField == 3)
    blkt_read = 58;
  else
    blkt_read = 48;

  /* set the check-field counter to the next expected field */

  check_fld = FirstField;

  /* then, if first is a B058F03, get the stage sequence number (from FirstLine)
     and the symmetry type (from the file), otherwise, get the symmetry (from FirstLine) */

  if(check_fld == 3) {
    parse_field(FirstLine,0,field);
    sequence_no = get_int(field);
    check_fld++;
    get_field(fptr,field,blkt_read,check_fld++,":",0);
  }
  else {
    parse_field(FirstLine,0,field);
    check_fld++;
  }

  /* then get the gain and frequency of gain (these correspond to sensitivity and frequency of
     sensitivity for stage 0 Sensitivity/Gain filters) */

  blkt_ptr->blkt_info.gain.gain = get_double(field);
  get_field(fptr,field,blkt_read,check_fld++,":",0);
  blkt_ptr->blkt_info.gain.gain_freq = get_double(field);

  /* if there is a history, skip it. First determine number of lines to skip */

  get_field(fptr,field,blkt_read,check_fld++,":",0);
  nhist = get_int(field);

  /* then skip them  */

  for(i = 0; i < nhist; i++) {
    get_line(fptr,line,blkt_read,check_fld," ");
  }

  /* return the sequence number of the stage for verification */

  return(sequence_no);

}

/* parse_fir:  parses the RESP file blockettes for FIR filters (in a Blockette [61] or [41]).
               Errors cause the program to terminate.  The blockette and field numbers are
               checked as the file is parsed. As with parse_pz(), for this routine to work,
               the lines must contain evalresp-3.0 style prefixes */

void parse_fir (FILE *fptr, struct blkt *blkt_ptr, struct stage *stage_ptr) {
  int i, blkt_typ;
  int blkt_read, check_fld;
  int ncoeffs;
  char field[MAXFLDLEN], line[MAXLINELEN];

  /* first get the stage sequence number (from the input line) */

  if(FirstField != 3 && FirstField != 5) {
    error_return(PARSE_ERROR,"parse_fir; %s%s%s%2.2d","(return_field) fld ",
                 "number does not match expected value\n\tfld_xpt=F03 or F05",
                 ", fld_found=F", FirstField);
  }

  if(FirstField == 3)
    blkt_read = 61;
  else
    blkt_read = 41;

  /* set the check-field counter to the next expected field */

  check_fld = FirstField;

  /* then, if first is a B061F03, get the stage sequence number (from FirstLine)
     and the symmetry type (from the file), otherwise, get the symmetry (from FirstLine) */

  if(check_fld == 3) {
    parse_field(FirstLine,0,field);
    stage_ptr->sequence_no = get_int(field);
    curr_seq_no = stage_ptr->sequence_no;
    check_fld+=2;
    get_field(fptr,field,blkt_read,check_fld++,":",0);
  }
  else {
    parse_field(FirstLine, 0, field);
    check_fld++;
  }

  /* then get the symmetry type */

  if(strlen(field) != 1) {
    error_return(PARSE_ERROR, "parse_fir; parsing (FIR), illegal symmetry type ('%s')",
                 field);
  }
  blkt_typ = *field;
  switch (blkt_typ) {
  case 'A':
    blkt_ptr->type = FIR_ASYM;               /* no symmetry */
    break;
  case 'B':
    blkt_ptr->type = FIR_SYM_1;               /* odd number coefficients with symmetry */
    break;
  case 'C':
    blkt_ptr->type = FIR_SYM_2;               /* even number coefficients with symmetry */
    break;
  default:
    error_return(PARSE_ERROR, "parse_fir; parsing (FIR), unexpected symmetry type ('%c')",
                 *field);
  }
  /* next should be the units (in first, then out) */

  get_line(fptr,line,blkt_read,check_fld++,":");
  stage_ptr->input_units = check_units(line);

  get_line(fptr,line,blkt_read,check_fld++,":");
  stage_ptr->output_units = check_units(line);

  /* the number of coefficients */

  get_field(fptr,field,blkt_read,check_fld++,":",0);
  ncoeffs = get_int(field);
  blkt_ptr->blkt_info.fir.ncoeffs = ncoeffs;

  /* remember to allocate enough space for the number of coefficients to follow */

  blkt_ptr->blkt_info.fir.coeffs = alloc_double(ncoeffs);

  /* the coefficients */

  for(i = 0; i < ncoeffs; i++) {
    get_field(fptr,field,blkt_read,check_fld," ",1);
    if(!is_real(field))
      error_return(PARSE_ERROR,"parse_fir: %s%s%s",
                   "coeffs must be real numbers (found '",field,"')");
    blkt_ptr->blkt_info.fir.coeffs[i] = atof(field);
  }

}

/* parse_ref:  parses the RESP file blockettes for Response Reference type filters (in a
               Blockette [60]).  Errors cause the program to terminate.  The blockette and
               field numbers are checked as the file is parsed. As with parse_pz(), for
               this routine to work, the lines must contain evalresp-3.0 style prefixes */

void parse_ref(FILE *fptr, struct blkt *blkt_ptr, struct stage *stage_ptr) { 
  int this_blkt_no = 60, blkt_no, fld_no, i, j, prev_blkt_no = 60;
  int nstages, stage_num, nresps, lcl_nstages;
  char field[MAXFLDLEN];
  struct blkt  *last_blkt;
  struct stage *last_stage, *this_stage;

  /* set the filter type for, then start parsing the stages */

  blkt_ptr->type = REFERENCE;

  /* set up a local variable for the input 'stage pointer' */

  this_stage = stage_ptr;

  /* first get the number of stages (from the input line) */

  if(FirstField != 3) {
    error_return(PARSE_ERROR,"parse_ref; %s%s%s%2.2d","(return_field) fld ",
                 "number does not match expected value\n\tfld_xpt=F03",
                 ", fld_found=F", FirstField);
  }
  parse_field(FirstLine,0,field);
  if(!is_int(field))
    error_return(PARSE_ERROR,"parse_ref; value '%s' %s",
                 field," cannot be converted to the number of stages");
  nstages = atoi(field);
  blkt_ptr->blkt_info.reference.num_stages = nstages;

  /* then (from the file) read all of the stages in sequence */

  for(i = 0; i < nstages; i++) {

    /* determine the stage number in the sequence of stages */

    get_field(fptr,field,this_blkt_no,4,":",0);
    if(!is_int(field))
      error_return(PARSE_ERROR,"parse_ref; value '%s' %s",
                   field," cannot be converted to the stage sequence number");
    stage_num = atoi(field);
    blkt_ptr->blkt_info.reference.stage_num = stage_num;

    /* set the stage sequence number and the pointer to the first blockette */

    this_stage->sequence_no = stage_num;
    curr_seq_no = this_stage->sequence_no;

    /* then the number of responses in this stage */

    get_field(fptr,field,this_blkt_no,5,":",0);
    if(!is_int(field))
      error_return(PARSE_ERROR,"parse_ref; value '%s' %s",
                   field," cannot be converted to the number of responses");
    nresps = atoi(field);
    blkt_ptr->blkt_info.reference.num_responses = nresps;

    /* then, for each of the responses in this stage, get the first line of the next
       response to determine the type of response to read */
    
    for(j = 0; j < nresps; j++) {
      FirstField = next_line(fptr, FirstLine, &blkt_no, &fld_no, ":");
      last_blkt = blkt_ptr;
      switch (blkt_no) {
      case 43:
        blkt_ptr = alloc_pz();
        parse_pz(fptr, blkt_ptr, this_stage);
        break;
      case 44:
        blkt_ptr = alloc_fir();
        parse_coeff(fptr, blkt_ptr, this_stage);
        break;
      case 45: 
        blkt_ptr = alloc_list();
        parse_list(fptr, blkt_ptr, this_stage);
        break;
      case 46:
        blkt_ptr = alloc_generic();
        parse_generic(fptr, blkt_ptr, this_stage);
        break;
      case 47:
        blkt_ptr = alloc_deci();
        parse_deci(fptr, blkt_ptr);
        break;
      case 48:
        blkt_ptr = alloc_gain();
        parse_gain(fptr, blkt_ptr);
        break;
      case 41:
        blkt_ptr = alloc_fir();
        parse_fir(fptr, blkt_ptr, this_stage);
        break;
      case 60:
        error_return(PARSE_ERROR,"parse_ref; unexpected end of stage (at blockette [%3.3d])",
                     prev_blkt_no);
        break;
      default:
	/* code to ignore unexected blockette/field lines might be useful here, but need 
	   example code to test it - SBH. 2004.079
	   If we want it, replace this error_return and break with a "continue;"
	 */
        error_return(UNRECOG_FILTYPE, "parse_ref; unexpected filter type (blockette [%3.3d])",
                     blkt_no);
        break;
      }
      last_blkt->next_blkt = blkt_ptr;
      prev_blkt_no = blkt_no;
    }
    if(i != (nstages-1)) {          /* if not the last stage in this filter */

      /* set the last stage's next_stage to point to the new stage and the first_blkt
         for that stage to point to a new blockette [60] type filter */

      last_stage = this_stage;
      this_stage = alloc_stage();
      blkt_ptr = alloc_ref();
      last_stage->next_stage = this_stage;
      this_stage->first_blkt = blkt_ptr;

      /* set the filter type for the new blockette [60] filter stage */

      blkt_ptr->type = REFERENCE;

      /* and set the number of stages again ... */

      get_field(fptr,field,this_blkt_no,3,":",0);
      if(!is_int(field))
        error_return(PARSE_ERROR,"parse_ref; value '%s' %s",
                     field," cannot be converted to the new stage sequence number");
 
      lcl_nstages = atoi(field);
      if(lcl_nstages != nstages) {
        error_return(PARSE_ERROR, "parse_ref; internal RESP format error, %s%d%s%d",
                     "\n\tstage expected = ", nstages, ", stage found = ", lcl_nstages);
      }
      blkt_ptr->blkt_info.reference.num_stages = nstages;

    }
  }
}

/* parse_channel:  parses the RESP file blockettes for a channel. Errors cause the program to
                   terminate.  The blockette and field numbers are checked as the file is parsed.
                   As with the filter blockettes, the actual positions of the fields on a line
                   will not effect this routine.  This routine constructs a linked list of filters
                   until a non-filter blockette (or the EOF) is found.  For this routine to work,
                   the lines must contain evalresp-3.0 style prefixes.  Also, it is assumed
                   that the file pointer has been prepositioned at the end of the
                   station/channel/time information in the RESP file for the channel of interest,
                   (i.e. that the next non-comment line starts the actual response information) */

int parse_channel(FILE *fptr, struct channel* chan) {
  // TODO - assigments for no_units and tmp_stage2 made blindly to fix compiler warning.  bug?
  int blkt_no, read_blkt = 0, fld_no, no_units = 0;
  int curr_seq_no, last_seq_no;
  struct blkt *blkt_ptr, *last_blkt = (struct blkt *)NULL;
  struct stage *this_stage, *last_stage, *tmp_stage, *tmp_stage2 = NULL;

  /* initialize the channel's sequence of stages */

  last_stage = (struct stage *)NULL;
  curr_seq_no = last_seq_no = 0;
  this_stage = alloc_stage();
  chan->first_stage = this_stage;
  chan->nstages++;
  tmp_stage = alloc_stage();

  /* start processing the response information */

  while((FirstField = next_line(fptr, FirstLine, &blkt_no, &fld_no, ":")) != 0 && blkt_no != 50) {
    switch (blkt_no) {
    case 53:
      blkt_ptr = alloc_pz();
      parse_pz(fptr, blkt_ptr, tmp_stage);
      curr_seq_no = tmp_stage->sequence_no;
      break;
    case 54:
      /* IGD : as we add an IIR case to this blockette, we cannot simply assume that blockette 54 is FIR */
     /*The field 10 should be distinguish between the IIR and FIR */
     if (is_IIR_coeffs (fptr, ftell(fptr)))	{ /*IGD New IIR case */
     	blkt_ptr = alloc_coeff();
	parse_iir_coeff(fptr, blkt_ptr, tmp_stage);
     }
      else	{			/*IGD this is the original case here */
      	blkt_ptr = alloc_fir();
      	parse_coeff(fptr, blkt_ptr, tmp_stage);
     }
      curr_seq_no = tmp_stage->sequence_no;
      break;
    case 55:
      blkt_ptr = alloc_list();
      parse_list(fptr, blkt_ptr, tmp_stage);
      curr_seq_no = tmp_stage->sequence_no;
      break;
    case 56:
      blkt_ptr = alloc_generic();
      parse_generic(fptr, blkt_ptr, tmp_stage);
      curr_seq_no = tmp_stage->sequence_no;
      break;
    case 57:
      blkt_ptr = alloc_deci();
      curr_seq_no = parse_deci(fptr, blkt_ptr);
      break;
    case 58:
      blkt_ptr = alloc_gain();
      curr_seq_no = parse_gain(fptr, blkt_ptr);
      break;
    case 61:
      blkt_ptr = alloc_fir();
      parse_fir(fptr, blkt_ptr, tmp_stage);
      curr_seq_no = tmp_stage->sequence_no;
      break;
    case 60:  /* never see a blockette [41], [43]-[48] without a [60], parse_ref handles these */
      blkt_ptr = alloc_ref();
      tmp_stage2 = alloc_stage();
      parse_ref(fptr, blkt_ptr, tmp_stage2);
      curr_seq_no = tmp_stage2->sequence_no;
      tmp_stage2->first_blkt = blkt_ptr;
      break;
    default:
      /*
	2004.079 - SBH changed to allow code to skip unrecognized lines in RESP file. Just continue
	to the next line.
	error_return(UNRECOG_FILTYPE, "parse_chan; unrecognized filter type (blockette [%c])",
	blkt_no);
      break;
      */
      continue;
    }
    if(blkt_no != 60) {
      if(!read_blkt++) {
        this_stage->first_blkt = blkt_ptr;
        this_stage->sequence_no = curr_seq_no;
        last_stage = this_stage;
        no_units = 1;
      }
      else if(last_seq_no != curr_seq_no) {
        chan->nstages++;
        last_stage = this_stage;
        this_stage = alloc_stage();
        this_stage->sequence_no = curr_seq_no;
        last_stage->next_stage = this_stage;
        this_stage->first_blkt = blkt_ptr;
        last_stage = this_stage;
        no_units = 1;
      }
      else
        last_blkt->next_blkt = blkt_ptr;
  
      if(no_units && blkt_no != 57 && blkt_no != 58) {
        this_stage->input_units = tmp_stage->input_units;
        this_stage->output_units = tmp_stage->output_units;
        no_units = 0;
      }
  
      last_blkt = blkt_ptr;
      last_seq_no = curr_seq_no;
    }
    else {
        if(!read_blkt++) {
            this_stage = tmp_stage2;
            free_stages(chan->first_stage);
            chan->first_stage = this_stage;
        }
        else if(last_seq_no != curr_seq_no) {
            this_stage = tmp_stage2;
            last_stage->next_stage = this_stage;
            chan->nstages++;
        }
        else {
            blkt_ptr = tmp_stage2->first_blkt;
            last_blkt->next_blkt = blkt_ptr;
            if (this_stage != NULL && tmp_stage2->next_stage != NULL) {
                this_stage->next_stage = tmp_stage2->next_stage;
            }
        }
            
        while(this_stage->next_stage != (struct stage *)NULL) {
            this_stage = this_stage->next_stage;
            chan->nstages++;
        }
        blkt_ptr = this_stage->first_blkt;
        while(blkt_ptr->next_blkt != (struct blkt *)NULL) {
            blkt_ptr = blkt_ptr->next_blkt;
        }
        last_blkt = blkt_ptr;
        last_stage = this_stage;
        curr_seq_no = this_stage->sequence_no;
        last_seq_no = curr_seq_no;
    }
  }
  free_stages(tmp_stage);
  return(FirstField);
}

/* get_channel:  retrieves the info from the  RESP file blockettes for a channel. Errors cause the
                 program to  terminate.  The blockette and field numbers are checked as the file
                 is parsed. Only the station/channel/date info is returned. The file pointer is
                 left at the end of the date info, where it can be used to read the response
                 information into the filter structures. */

int get_channel(FILE *fptr, struct channel* chan) {
  int blkt_no, fld_no;
  char field[MAXFLDLEN], line[MAXLINELEN];

  /* check to make sure a non-comment field exists and it is the sta/chan/date info.
     Note:  If it is the first channel (and, as a result, FirstLine contains a null
     string), then we have to get the next line.  Otherwise, the FirstLine argument
     has been set by a previous call to 'next_line' in another routine (when parsing
     a previous station-channel-network tuple's information) and this is the
     line that should be parsed to get the station name */

  chan->nstages = 0;
  chan->sensfreq = 0.0;
  chan->sensit = 0.0;
  chan->calc_sensit = 0.0;
  chan->calc_delay = 0.0;
  chan->estim_delay = 0.0;
  chan->applied_corr = 0.0;
  chan->sint = 0.0;

  if(!strlen(FirstLine))
    get_field(fptr,field,50,3,":",0);
  else
    parse_field(FirstLine,0,field);

  strncpy(chan->staname,field,STALEN);

  /* then (from the file) the Network ID */

  get_field(fptr,field,50,16,":",0);
  if(!strncmp(field,"??",2))
    strncpy(chan->network,"",NETLEN);
  else
    strncpy(chan->network,field,NETLEN);

  /* then (from the file) the Location Identifier (if it exists ... it won't for
     "old style" RESP files so assume old style RESP files contain a null location
     identifier) and the channel name */

/* Modified to use 'next_line()' and 'parse_field()' directly
   to handle case where file contains "B052F03 Location:" and
   nothing afterward -- 10/19/2005 -- [ET] */
/*  test_field(fptr,field,&blkt_no,&fld_no,":",0); */

  next_line(fptr, line, &blkt_no, &fld_no, ":");
  if(strlen(line) > 0)                 /* if data after "Location:" then */
    parse_field(line, 0, field);       /* parse location data */
  else                            /* if no data after "Location:" then */
    field[0] = '\0';              /* clear 'field' string */

  if(blkt_no == 52 && fld_no == 3) {
    if(strlen(field) <= 0 || !strncmp(field,"??",2))
      strncpy(chan->locid,"",LOCIDLEN);
    else
      strncpy(chan->locid,field,LOCIDLEN);
    get_field(fptr,field,52,4,":",0);
    strncpy(chan->chaname,field,CHALEN);
  }
  else if(blkt_no == 52 && fld_no == 4) {
    strncpy(chan->locid,"",LOCIDLEN);
    strncpy(chan->chaname,field,CHALEN);
  }
  else {
#ifdef LIB_MODE
    return 0;
#else
    error_return(PARSE_ERROR,"get_line; %s%s%3.3d%s%3.3d%s[%2.2d|%2.2d]%s%2.2d","blkt",
                 " and fld numbers do not match expected values\n\tblkt_xpt=B",
                 52, ", blkt_found=B", blkt_no, "; fld_xpt=F", 3, 4,
                 ", fld_found=F", fld_no);
#endif
  }

  /* get the Start Date */

  get_line(fptr,line,52,22,":");
  strncpy(chan->beg_t,line,DATIMLEN);

  /* get the End Date */

  get_line(fptr,line,52,23,":");
  strncpy(chan->end_t,line,DATIMLEN);

  return(1);

}

/* timecmp:  returns an integer indicating whether the time in the input argument
             "dt1" is greater than (1), equal to (0), or less than (-1) the time
             in the input argument "dt2" */

int timecmp(struct dateTime *dt1, struct dateTime *dt2) {

  /* check year */
  if (dt1->year < dt2->year) return (-1);
  if (dt1->year > dt2->year) return (1);
 
  /* check day */
  if (dt1->jday < dt2->jday)  return (-1);
  if (dt1->jday > dt2->jday)  return (1);
 
  /* check hour */
  if (dt1->hour < dt2->hour) return (-1);
  if (dt1->hour > dt2->hour) return (1);
 
  /* check minute */
  if (dt1->min < dt2->min) return (-1);
  if (dt1->min > dt2->min) return (1);
 
  /* check second */
  if (dt1->sec < dt2->sec) return (-1);
  if (dt1->sec > dt2->sec) return (1);
 
  /* if I got this far, times are equal */
  return (0);

}

/* in_epoch:  determines if an input date-time lies within the response epoch
              defined by the strings "beg_t" and "end_t" */

int in_epoch(const char *datime, const char *beg_t, const char *end_t) {
  char *start_pos;
  char temp_str[DATIMLEN];
  int len;
  struct dateTime start_time, end_time, this_time;

  /* parse the "datime" argument */

  this_time.hour = this_time.min = 0;
  this_time.sec = 0.0;
  strncpy(temp_str, datime, DATIMLEN);
  start_pos = temp_str;
  len = strcspn(start_pos, ",");
  *(start_pos+len) = '\0';
  this_time.year = atoi(start_pos);
  start_pos += (strlen(start_pos)+1);
  len = strcspn(start_pos, ",");
  *(start_pos+len) = '\0';
  this_time.jday = atoi(start_pos);
  start_pos += (strlen(start_pos)+1);
  len = strcspn(start_pos, ":");
  *(start_pos+len) = '\0';
  this_time.hour = atoi(start_pos);
  start_pos += (strlen(start_pos)+1);
  len = strcspn(start_pos, ":");
  *(start_pos+len) = '\0';
  this_time.min = atoi(start_pos);
  start_pos += (strlen(start_pos)+1);
  this_time.sec = atof(start_pos);

  /* parse the "beg_t" argument */

  start_time.hour = start_time.min = 0;
  start_time.sec = 0.0;
  strncpy(temp_str, beg_t, DATIMLEN);
  start_pos = temp_str;
  len = strcspn(start_pos, ",");
  *(start_pos+len) = '\0';
  start_time.year = atoi(start_pos);
  start_pos += (strlen(start_pos)+1);
  len = strcspn(start_pos, ",");
  *(start_pos+len) = '\0';
  start_time.jday = atoi(start_pos);
  start_pos += (strlen(start_pos)+1);
  if(strlen(start_pos)) {
    len = strcspn(start_pos, ":");
    *(start_pos+len) = '\0';
    start_time.hour = atoi(start_pos);
    start_pos += (strlen(start_pos)+1);
    if(strlen(start_pos)) {
      len = strcspn(start_pos, ":");
      *(start_pos+len) = '\0';
      start_time.min = atoi(start_pos);
      start_pos += (strlen(start_pos)+1);
      if(strlen(start_pos)) {
	start_time.sec = atof(start_pos);
      }
    }
  }

  /* parse the "end_t" argument.  If there is no ending time, only check to see if
     the start time is before this time */

  if(0 != strncmp(end_t, "No Ending Time", 14)) {  /* Bad bug fixed 06/26/08 */
    end_time.hour = end_time.min = 0;
    end_time.sec = 0.0;
    strncpy(temp_str, end_t, DATIMLEN);
    start_pos = temp_str;
    len = strcspn(start_pos, ",");
    *(start_pos+len) = '\0';
    end_time.year = atoi(start_pos);
    start_pos += (strlen(start_pos)+1);
    len = strcspn(start_pos, ",");
    *(start_pos+len) = '\0';
    end_time.jday = atoi(start_pos);
    start_pos += (strlen(start_pos)+1);
    if(strlen(start_pos)) {
      len = strcspn(start_pos, ":");
      *(start_pos+len) = '\0';
      end_time.hour = atoi(start_pos);
      start_pos += (strlen(start_pos)+1);
      if(strlen(start_pos)) {
	len = strcspn(start_pos, ":");
	*(start_pos+len) = '\0';
	end_time.min = atoi(start_pos);
	start_pos += (strlen(start_pos)+1);
	if(strlen(start_pos)) {
	  end_time.sec = atof(start_pos);
	}
      }
    }
    return((timecmp(&start_time, &this_time) <= 0 && timecmp(&end_time, &this_time) > 0));
  }
  else {
    return((timecmp(&start_time, &this_time) <= 0));
  }

}

/* find_resp:  finds the location of a response for one of the input station-channels
               at the specified date.  If no response is available in the file
               pointed to by fptr, then a -1 value is returned (indicating
               failure), otherwise the index of the matching station-channel pair.
               The matching beg_t, end_t and station info are returned as part of
               the input structure "this_channel".
               The pointer to the file (fptr) is left in position for the
               "parse_channel" routine to grab the response information for that
               station.  Note: the station information is preloaded into "this_channel",
               so the file pointer does not need to be repositioned to allow for this
               information to be reread. */

int find_resp(FILE *fptr, struct scn_list *scn_lst, char *datime,
              struct channel *this_channel) {
  int test, i;
  struct scn *scn = NULL;
  short int testSta = 0;
  short int testChan = 0;
  short int testNet = 0;
  short int testLoc = 0;
  short int testTime = 0;

  while((test = get_channel(fptr, this_channel)) != 0) {
    for(i = 0; i < scn_lst->nscn; i++) {
      scn = scn_lst->scn_vec[i];
      testSta = string_match(this_channel->staname,scn->station,"-g");
      testNet = (!strlen(scn->network) && !strlen(this_channel->network)) ||
          string_match(this_channel->network,scn->network,"-g");
      testLoc = string_match(this_channel->locid,scn->locid,"-g");
      testChan = string_match(this_channel->chaname,scn->channel,"-g");
      testTime = in_epoch(datime, this_channel->beg_t, this_channel->end_t);  

      if(testSta && testNet && testLoc && testChan && testTime) {
        scn->found = 1;
        return(i);
      }
    }
    if(!(test = next_resp(fptr))) {
      return(-1);
    }
  }
  return(-1);
}

/* get_resp:  finds the location of a response for the input station-channel
              at the specified date.  If no response is available in the file
              pointed to by fptr, then a -1 value is returned (indicating
              failure), otherwise a value of 1 is returned (indicating success),
              The matching beg_t, end_t and station info are returned as part of
              the input structure "this_channel".
              The pointer to the file (fptr) is left in position for the
              "parse_channel" routine to grab the response information for that
              station.  Note: the station information is preloaded into "this_channel",
              so the file pointer does not need to be repositioned to allow for this
              information to be reread. */

int get_resp(FILE *fptr, struct scn *scn, char *datime,
              struct channel *this_channel) {
  int test;

  while((test = get_channel(fptr, this_channel)) != 0) {

    if(string_match(this_channel->staname,scn->station,"-g") &&
       ((!strlen(scn->network) && !strlen(this_channel->network)) ||
        string_match(this_channel->network,scn->network,"-g")) &&
       string_match(this_channel->locid,scn->locid,"-g") &&
       string_match(this_channel->chaname,scn->channel,"-g") &&
	 in_epoch(datime, this_channel->beg_t, this_channel->end_t)) {
      scn->found = 1;
      return(1);
    }
    else {
      if(!(test = next_resp(fptr))) {
        return(-1);
      }
    }
  }
  return(-1);
}

/* next_resp:  finds the location of the start of the next response in the file
               and positions the file pointer there. If no more responses are available
               then a zero value is returned (indicating failure to reposition the file
               pointer), otherwise the field number for that line  is returned and
               that first line is returned in the input argument "FirstLine".
               The pointer to the file (fptr) is left in position for the get_channel
               routine to grab the channel information */

int next_resp(FILE *fptr) {
  int blkt_no, fld_no, test;
  char tmp_line[MAXLINELEN];

  while((test = check_line(fptr, &blkt_no, &fld_no, tmp_line)) != 0 && blkt_no != 50)
    ;

  if(test && blkt_no == 50) {
    parse_field(tmp_line,2,FirstLine);
    return(1);
  }
  else
    return(0);
}

