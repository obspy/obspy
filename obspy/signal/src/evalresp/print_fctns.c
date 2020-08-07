/* print_fctns.c */

/*
    11/3/2005 -- [ET]  Added 'print_resp_itp()' function with support for
                       List blockette interpolation; made 'print_resp()'
                       call 'print_resp_itp()' function with default
                       values for List blockette interpolation parameters;
                       added List-blockette interpolation parameters to
                       'print_chan()' function and modified message shown
                       when List blockette encountered.
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "./evresp.h"
#include <string.h>
#include <stdlib.h>

/* function declarations for forward references */
int arrays_equal(double *arr1, double *arr2, int arr_size);
void evresp_adjust_phase(double *pha, int len, double min, double max);
int evresp_vector_minmax(double *pha, int len, double *min, double *max);

/* print_chan:  prints a summary of the channel's response information to stderr */

void print_chan(struct channel *chan, int start_stage, int stop_stage,
          int stdio_flag, int listinterp_out_flag, int listinterp_in_flag, 
          int useTotalSensitivityFlag) {
  struct stage *this_stage, *last_stage, *first_stage;
  struct blkt *this_blkt;
  char tmp_str[TMPSTRLEN], out_str[OUTPUTLEN];
  int in_units = 0;
  int out_units = 0;
  int first_blkt;

  /* determine what the input units of the first stage and output units
     of the last stage are */

  this_stage = first_stage = chan->first_stage;
  while(this_stage) {
    if(start_stage >= 0 && stop_stage && (this_stage->sequence_no < start_stage ||
       this_stage->sequence_no > stop_stage)) {
      this_stage = this_stage->next_stage;
      continue;
    }
    else if(start_stage >= 0 && !stop_stage && this_stage->sequence_no != start_stage) {
      this_stage = this_stage->next_stage;
      continue;
    }
    last_stage = this_stage;
    if(!in_units) in_units = this_stage->input_units;
    if(last_stage->output_units)
      out_units = last_stage->output_units;
    this_stage = last_stage->next_stage;
    if(this_stage != (struct stage *)NULL)
      this_blkt = this_stage->first_blkt;
  }

  /* inform the user of which file has been evaluated */

  fprintf(stderr, "%s --------------------------------------------------\n", myLabel);
  if(!stdio_flag) {
    fprintf(stderr, "%s  %s\n", myLabel, curr_file);
  }
  else {
    if(strlen(chan->network)) {
      fprintf(stderr, "%s  RESP.%s.%s.%s.%s (from stdin)\n", myLabel, chan->network,
                      chan->staname,chan->locid,chan->chaname);
    }
    else {
      fprintf(stderr, "%s  RESP..%s.%s.%s (from stdin)\n",myLabel, chan->staname,
                      chan->locid,chan->chaname);
    }
  }
  fprintf(stderr, "%s --------------------------------------------------\n", myLabel);
  fprintf(stderr, "%s  %s %s %s %s ", myLabel, (strlen(chan->network) ? chan->network : "??"),
          chan->staname, (strlen(chan->locid) ? chan->locid : "??"), chan->chaname);
  if(!def_units_flag)
    fprintf(stderr, "%s %s %s\n%s   Seed units: %s(in)->%s(out)\n",
            myLabel, chan->beg_t, chan->end_t, myLabel, SEEDUNITS[in_units],
            SEEDUNITS[out_units]);
  else
    fprintf(stderr, "%s %s %s\n%s   Seed units: %s(in)->%s(out)\n",
            myLabel, chan->beg_t, chan->end_t, myLabel, chan->first_units,
            chan->last_units);

  fprintf(stderr, "%s   computed sens=%.5E (reported=%.5E) @ %.5E Hz\n",
          myLabel, chan->calc_sensit, chan->sensit, chan->sensfreq);
  fprintf(stderr, "%s   calc_del=%.5E  corr_app=%.5E  est_delay=%.5E  final_sint=%.3g(sec/sample)\n",
          myLabel, chan->calc_delay, chan->applied_corr, chan->estim_delay, chan->sint);
  if (1 == useTotalSensitivityFlag)
    fprintf(stderr, "%s   (reported sensitivity was used to compute response (-ts option enabled))\n", myLabel);

/* then print the parameters for each stage (stage number, type of stage, number
     of coefficients [or number of poles and zeros], gain, and input sample interval
     if it is defined for that stage */

  this_stage = first_stage;
  while(this_stage) {
    if(start_stage >= 0 && stop_stage && (this_stage->sequence_no < start_stage ||
       this_stage->sequence_no > stop_stage)) {
      this_stage = this_stage->next_stage;
      continue;
    }
    else if(start_stage >= 0 && !stop_stage && this_stage->sequence_no != start_stage) {
      this_stage = this_stage->next_stage;
      continue;
    }
    this_blkt = this_stage->first_blkt;
    if(this_stage->sequence_no) {
      strncpy(tmp_str,"",TMPSTRLEN);
      sprintf(tmp_str,"     stage %2d:",this_stage->sequence_no);
      strcpy(out_str,tmp_str);
    }
    first_blkt = 1;
    while(this_blkt) {
      strncpy(tmp_str,"",TMPSTRLEN);
      switch (this_blkt->type) {
      case LAPLACE_PZ:
        sprintf(tmp_str," LAPLACE     A0=%E NZeros= %2d NPoles= %2d",
                  this_blkt->blkt_info.pole_zero.a0,
                  this_blkt->blkt_info.pole_zero.nzeros,
                  this_blkt->blkt_info.pole_zero.npoles);
        break;
      case ANALOG_PZ:
        sprintf(tmp_str," ANALOG      A0=%E NZeros= %2d NPoles= %2d",
                          this_blkt->blkt_info.pole_zero.a0,
                          this_blkt->blkt_info.pole_zero.nzeros,
                          this_blkt->blkt_info.pole_zero.npoles);
        break;
      case FIR_SYM_1:
        sprintf(tmp_str," FIR_SYM_1   H0=%E Ncoeff=%3d",
                          this_blkt->blkt_info.fir.h0,
                          this_blkt->blkt_info.fir.ncoeffs*2-1);
        break;
     case FIR_SYM_2:
        sprintf(tmp_str," FIR_SYM_2   H0=%E Ncoeff=%3d",
                          this_blkt->blkt_info.fir.h0,
                          this_blkt->blkt_info.fir.ncoeffs*2);
        strcat(out_str,tmp_str);
        strncpy(tmp_str,"",TMPSTRLEN);
        break;
      case FIR_ASYM:
        sprintf(tmp_str," FIR_ASYM    H0=%E Ncoeff=%3d",
                          this_blkt->blkt_info.fir.h0,
                          this_blkt->blkt_info.fir.ncoeffs);
        break;
      case IIR_PZ:
        sprintf(tmp_str," IIR_PZ      A0=%E NZeros= %2d NPoles= %2d",
                          this_blkt->blkt_info.pole_zero.a0,
                          this_blkt->blkt_info.pole_zero.nzeros,
                          this_blkt->blkt_info.pole_zero.npoles);
	break;
      case IIR_COEFFS:
	sprintf(tmp_str, "IIR_COEFFS   H0=%E NNumers=%2d NDenums= %2d",
			  this_blkt->blkt_info.coeff.h0,
                          this_blkt->blkt_info.coeff.nnumer,
                          this_blkt->blkt_info.coeff.ndenom);
        break;
      case GAIN:
        if(first_blkt && this_stage->sequence_no)
          sprintf(tmp_str," GAIN        Sd=%E",this_blkt->blkt_info.gain.gain);
        else if(this_stage->sequence_no)
          sprintf(tmp_str," Sd=%E",this_blkt->blkt_info.gain.gain);
        break;
      case DECIMATION:
        sprintf(tmp_str," SamInt=%E",this_blkt->blkt_info.decimation.sample_int);
	if (this_blkt->blkt_info.decimation.applied_corr < 0)
	  fprintf(stderr, "%s WARNING Stage %d: Negative correction_applied=%.5E is likely to be incorrect\n",
		  myLabel, this_stage->sequence_no, this_blkt->blkt_info.decimation.applied_corr);
	if (this_blkt->blkt_info.decimation.estim_delay < 0)
	  fprintf(stderr, "%s WARNING Stage %d: Negative estimated_delay=%.5E is likely to be incorrect\n",
		  myLabel, this_stage->sequence_no, this_blkt->blkt_info.decimation.estim_delay);
        break;
      case GENERIC:
        sprintf(tmp_str," Generic blockette is ignored; ");
        break;

      case FIR_COEFFS:
      case LIST:
      case REFERENCE:
        break;
      default:
        fprintf(stderr, "%s .........", myLabel);
      }
      strcat(out_str,tmp_str);
      if(first_blkt)
        first_blkt = 0;
      this_blkt = this_blkt->next_blkt;
    }
    if(this_stage->sequence_no)
      fprintf(stderr,"%s %s\n",myLabel, out_str);
    this_stage = this_stage->next_stage;
  }
  fprintf(stderr, "%s--------------------------------------------------\n", myLabel);
  /* IGD : here we print a notice about blockette 55: evalresp v. 2.3.17+*/
  /* ET:  Notice modified, with different notice if freqs interpolated */
  if(chan->first_stage->first_blkt->type == LIST) {
    if(listinterp_in_flag) {
      fprintf(stderr, "%s Note:  The input has been interpolated from the response List stage\n", myLabel);
      fprintf(stderr, "%s (blockette 55) to generate output for the %d frequencies requested\n",
                       myLabel, chan->first_stage->first_blkt->blkt_info.list.nresp);
    }
    else if(listinterp_out_flag) {
      fprintf(stderr, "%s Note:  The output has been interpolated from the %d frequencies\n",
                       myLabel, chan->first_stage->first_blkt->blkt_info.list.nresp);
      fprintf(stderr, "%s defined in the response List stage (blockette 55)\n", myLabel);
    }
    else {
      fprintf(stderr, "%s ++++++++ WARNING ++++++++++++++++++++++++++++\n", myLabel);
      fprintf(stderr, "%s Response contains a List stage (blockette 55)--the output has\n", myLabel);
      fprintf(stderr, "%s been generated for those %d frequencies defined in the blockette\n",
                       myLabel, chan->first_stage->first_blkt->blkt_info.list.nresp);
      fprintf(stderr, "%s +++++++++++++++++++++++++++++++++++++++++++++\n", myLabel);
    }
  }
  fflush(stderr);
}

/* print_resp:  prints the response information in the fashion that the
                user requested it.  The response is either in the form of
                a complex spectra (freq, real_resp, imag_resp) to the
                file SPECTRA.NETID.STANAME.CHANAME (if rtype = "cs")
                or in the form of seperate amplitude and phase files
                (if rtype = "ap") with names like AMP.NETID.STANAME.CHANAME
                and PHASE.NETID.STANAME.CHANAME.  In all cases, the pointer to
                the channel is used to obtain the NETID, STANAME, and CHANAME
                values.  If the 'stdio_flag' is set to 1, then the response
                information will be output to stdout, prefixed by a header that
                includes the NETID, STANAME, and CHANAME, as well as whether
                the response given is in amplitude/phase or complex response
                (real/imaginary) values.  If either case, the output to stdout
                will be in the form of three columns of real numbers, in the
                former case they will be freq/amp/phase tuples, in the latter
                case freq/real/imaginary tuples.
                This version of the function includes the 'listinterp...'
                parameters */

void print_resp_itp(double *freqs, int nfreqs, struct response *first,
                char *rtype, int stdio_flag, int listinterp_out_flag,
                double listinterp_tension, int unwrap_flag) {
  int i;
  double amp, pha;
  /* Space for "SPECTRA." + station.network.location.channel + NUL */
  char filename[8+STALEN+1+NETLEN+1+LOCIDLEN+1+CHALEN+1];
  FILE *fptr1, *fptr2;
  struct response *resp;
  struct complex *output;
  double *amp_arr;
  double *pha_arr;
  double *freq_arr;
  int freqarr_alloc_flag;
  int num_points;

  double added_value = 0.0;
  double prev_phase = 0.0;
  double phas1 = 0.0;


  resp = first;
  while(resp != (struct response *)NULL) {
    output = resp->rvec;
    if((0 == strcasecmp(rtype,"AP")) || (0 == strcasecmp(rtype,"FAP"))) {
         /* use count from 'response' block to support List blockette */
      num_points = resp->nfreqs;
         /* convert complex-spectra to amp/phase and load into arrays */
      amp_arr = (double *)calloc(num_points,sizeof(double));
      pha_arr = (double *)calloc(num_points,sizeof(double));
      for(i = 0; i < num_points; i++) {
        amp_arr[i] = sqrt(output[i].real*output[i].real+
                                             output[i].imag*output[i].imag);
        pha_arr[i] = atan2(output[i].imag,output[i].real+1.e-200)*180.0/Pi;
      }
      if(listinterp_out_flag && (nfreqs != resp->nfreqs ||
                                 !arrays_equal(freqs,resp->freqs,nfreqs))) {
              /* flag set for interpolating List blockette entries and
                 requested vs response frequency arrays are not identical */
                                       /* copy List freqs into new array */
        freq_arr = (double *)calloc(num_points,sizeof(double));
        memcpy(freq_arr,resp->freqs,sizeof(double)*num_points);
        freqarr_alloc_flag = 1;        /* indicate freq array allocated */
                                       /* interpolate to given freqs */
        interpolate_list_blockette(&freq_arr,&amp_arr,&pha_arr,
                               &num_points,freqs,nfreqs,listinterp_tension);
      }
      else {                 /* not interpolating List blockette entries */
              /* use freqs from 'response' blk to support List blockette */
        freq_arr = resp->freqs;
        freqarr_alloc_flag = 0;        /* indicate freq array not alloc'd */
      }
      if(!stdio_flag) {
	if (0 == strcasecmp(rtype,"AP")) {
          sprintf(filename,"AMP.%s.%s.%s.%s",resp->network,resp->station,resp->locid,resp->channel);
          if((fptr1 = fopen(filename,"w")) == (FILE *)NULL) {
            error_exit(OPEN_FILE_ERROR,"print_resp; failed to open file %s", filename);
          }
          sprintf(filename,"PHASE.%s.%s.%s.%s",resp->network,resp->station,resp->locid,resp->channel);
          if((fptr2 = fopen(filename,"w")) == (FILE *)NULL) {
            error_exit(OPEN_FILE_ERROR,"print_resp; failed to open file %s", filename);
          }
	  if (1 == unwrap_flag) {
          /* 04/27/2010 unwraped phases should only start causal! - Johannes Schweitzer*/
	  phas1 = 0.0;
	  if(pha_arr[0] < 0.0 ) { phas1 = 360.0; }
	  prev_phase = pha_arr[0] + phas1;	    
	    
	    for(i = 0; i < num_points; i++) {
	      pha = pha_arr[i] + phas1;
	      pha = unwrap_phase(pha, prev_phase, 360.0, &added_value);
	      pha_arr[i] = pha;
	      prev_phase = pha;
	    }
	    /* Next function attempts to put phase withing -360:360 bounds
	    * this is requested by AFTAC
	    */
	   /* Next line is removed at request of Chad */
	   /* (void) evresp_adjust_phase(pha_arr, num_points, -360.0, 360.0); */
	  }
	  else {
#ifdef UNWRAP_PHASE
	    for(i = 0; i < num_points; i++) {
	      pha = pha_arr[i];
	      pha = unwrap_phase(pha, prev_phase, 360.0, &added_value);
	      pha_arr[i] = pha;
	      prev_phase = pha;
	    }
	    /* Next function attempts to put phase withing -360:360 bounds
	    * this is requested by AFTAC
	    */
	    (void) evresp_adjust_phase(pha_arr, num_points, -360.0, 360.0);
#else
            /* Do not unwrap */ ;
#endif
	  }
	   	  
          for(i = 0; i < num_points; i++) {
            fprintf(fptr1,"%.6E %.6E\n",freq_arr[i],amp_arr[i]);
            fprintf(fptr2,"%.6E %.6E\n",freq_arr[i],pha_arr[i]);
          }
          fclose(fptr1);
          fclose(fptr2);
	}  /* End of AP CASE */
	if (0 == strcasecmp(rtype,"FAP")) {
          sprintf(filename,"FAP.%s.%s.%s.%s",resp->network,resp->station,resp->locid,resp->channel);
          if((fptr1 = fopen(filename,"w")) == (FILE *)NULL)
            error_exit(OPEN_FILE_ERROR,"print_resp; failed to open file %s", filename);

          /* 04/27/2010 unwraped phases should only start causal! - Johannes Schweitzer*/
	  phas1 = 0.0;
	  if(pha_arr[0] < 0.0 ) { phas1 = 360.0; }
	  prev_phase = pha_arr[0] + phas1;

	  /* Unwrap phase regardless of compile option */
	  for(i = 0; i < num_points; i++) {
	    pha = pha_arr[i] + phas1;
	    pha = unwrap_phase(pha, prev_phase, 360.0, &added_value);
	    pha_arr[i] = pha;
	    prev_phase = pha;
	  }           
          
	  for(i = 0; i < num_points; i++) {
            fprintf(fptr1,"%.6E  %.6E  %.6E\n",freq_arr[i],amp_arr[i], pha_arr[i]);
          }
          fclose(fptr1);
	}  /* End of new FAP CASE */
      }  /* End of AP or FAP case */
      else {
        fprintf(stdout, "%s --------------------------------------------------\n", myLabel);
        fprintf(stdout,"%s AMP/PHS.%s.%s.%s.%s\n",myLabel, resp->network,resp->station,resp->locid,resp->channel);
        fprintf(stdout, "%s --------------------------------------------------\n", myLabel);
        for(i = 0; i < num_points; i++) {
          amp = amp_arr[i];
          pha = pha_arr[i];
          fprintf(stdout,"%s %.6E %.6E %.6E\n",myLabel, freq_arr[i],amp,pha);
        }
        fprintf(stdout, "%s --------------------------------------------------\n", myLabel);
      }
      if(freqarr_alloc_flag)      /* if freq array was allocated then */
        free(freq_arr);           /* free freq array */
      free(pha_arr);              /* free allocated arrays */
      free(amp_arr);
    }
    else {
      if(!stdio_flag) {
        sprintf(filename,"SPECTRA.%s.%s.%s.%s",resp->network,resp->station,resp->locid,resp->channel);
        if((fptr1 = fopen(filename,"w")) == (FILE *)NULL) {
          error_exit(OPEN_FILE_ERROR,"print_resp; failed to open file %s", filename);
        }
      }
      else {
        fptr1 = stdout;
        fprintf(stdout, "%s --------------------------------------------------\n", myLabel);
        fprintf(stdout,"%s SPECTRA.%s.%s.%s.%s\n",myLabel, resp->network,resp->station,resp->locid,resp->channel);
        fprintf(stdout, "%s --------------------------------------------------\n", myLabel);
      }
      for(i = 0; i < resp->nfreqs; i++)
        fprintf(fptr1,"%.6E %.6E %.6E\n",resp->freqs[i],output[i].real,output[i].imag);
      if(!stdio_flag) {
        fclose(fptr1);
      }
    }
    resp = resp->next;
  }
}

/* print_resp:  prints the response information in the fashion that the
                user requested it.  The response is either in the form of
                a complex spectra (freq, real_resp, imag_resp) to the
                file SPECTRA.NETID.STANAME.CHANAME (if rtype = "cs")
                or in the form of seperate amplitude and phase files
                (if rtype = "ap") with names like AMP.NETID.STANAME.CHANAME
                and PHASE.NETID.STANAME.CHANAME.  In all cases, the pointer to
                the channel is used to obtain the NETID, STANAME, and CHANAME
                values.  If the 'stdio_flag' is set to 1, then the response
                information will be output to stdout, prefixed by a header that
                includes the NETID, STANAME, and CHANAME, as well as whether
                the response given is in amplitude/phase or complex response
                (real/imaginary) values.  If either case, the output to stdout
                will be in the form of three columns of real numbers, in the
                former case they will be freq/amp/phase tuples, in the latter
                case freq/real/imaginary tuples.
                This version of the function does not include the
                'listinterp...' parameters */

void print_resp(double *freqs, int nfreqs, struct response *first,
                char *rtype, int stdio_flag) {
  print_resp_itp(freqs,nfreqs,first,rtype,stdio_flag,0,0.0, 0);
}

/* Compares the entries in the given arrays.
     arr1 - first array of double values.
     arr2 - second array of double values.
     arr_size - number of entries in array.
   Returns 1 if the all the entries in the arrays are equal, 0 if not. */
int arrays_equal(double *arr1, double *arr2, int arr_size) {
  int i;
  for(i=0; i<arr_size; ++i) {
    if(arr1[i] != arr2[i])
      return 0;
  }
  return 1;
}

void evresp_adjust_phase(double *pha, int len, double min, double max)
	{
		double vector_min;
		double vector_max;
		int retVal;
		int cycles = 0;
		int i;
		if (!pha)
			return;

		retVal = evresp_vector_minmax(pha, len, &vector_min, &vector_max);
		if (0 == retVal)
			return;
		/* See if the data range is within the requested */
		if ((max - min) < (vector_max - vector_min))
			return; /* range is larger that requested */
		/* see if shifting down puts vector inside the requested range */
		if (vector_max > max)
		{
			cycles = (vector_max-max)/180 + 1;
		}
		/* see if shifting up puts vector inside the requested range */
		if (vector_min < min)
		{
			cycles = (vector_min-min)/180 - 1;
		}
		/* Do actual shift: note if we were within the ranged originally , cycles == 0; */
		for (i = 0; i < len; i++)
			pha[i] = pha[i] - cycles * 180;
		return;
	}

int evresp_vector_minmax(double *pha, int len, double *min, double *max)
	{
		int i;
		if (!pha)
			return 0;
		*min = pha[0];
		*max = pha[0];
		for (i = 0; i < len; i++)
		{
			if (pha[i] > *max)
				*max = pha[i];
			if (pha[i] < *min)
				*min = pha[i];
		}
		return 1;
	}

