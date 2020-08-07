/* calc_fctns.c */

/*
    08/28/2008 -- [IGD] Fixed a bug which used calucalated delay instead of estimated in
                          computation of phase
    11/3/2005 -- [ET]  Added 'wrap_phase()' function; moved 'use_delay()'
                       function from 'calc_fctns.c' to 'evalresp.c'.
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


/*------------------------------------------------------------------------------------*/
/* LOG:                                                                               */
/* Version 3.2.20:  iir_pz_trans()   modified IGD 09/20/01                            */
/* Starting with version 3.2.17, evalresp supports IIR type blockettes 54             */
/* Starting with version 3.2.17, evalresp supports blockette 55 of SEED               */
/* which is Response List Blockette (LIST). List contains the frequency,              */
/* amplitude and phase of the filter for a selected set of frequencies ONLY           */
/* The current suggestion of DMC IRIS is not to do any interpolation of the           */
/* and just leave as many frequencies in the output file as it used to be in          */
/* the blockette 55. This suggestion leads to the obvious fact that the               */
/* frequencies sampling requested by the user cannot be served if bl. 55 is used      */
/* and the frequency vector should be overritten by what we actually do have in       */
/* the blockette 55. Below we check if we use blockette 55 and if we do, we change    */
/* the frequency vector                                                               */
/* Note that we assume that blockette 55 can occur only as a single filtering stage   */
/* in the filter, cannot be mixed with the other types of the filters, therefore,     */
/* blockette 55 should be the first stage only                                        */
/* I. Dricker, ISTI, 06/22/00 for version 2.3.17 of evalresp                          */
/*------------------------------------------------------------------------------------*/

/* Notice: this version is modified by I.Dricker, ISTI i.dricker@isti.com 06/22/00 */
#include "./evresp.h"

#include <stdlib.h>
#include <string.h>

/*=================================================================
*                   Calculate response
*=================================================================*/
void calc_resp(struct channel *chan, double *freq, int nfreqs, struct complex *output,
          char *out_units, int start_stage, int stop_stage, int useTotalSensitivityFlag) {
  struct blkt *blkt_ptr;
  struct stage *stage_ptr;
  int i, j, units_code, eval_flag = 0, nc = 0, sym_fir = 0;
  double w;
  int matching_stages = 0, has_stage0 = 0;
  struct complex of, val;
  double corr_applied, estim_delay, delay;

/*  if(start_stage && start_stage > chan->nstages) {
    error_return(NO_STAGE_MATCHED, "calc_resp: %s start_stage=%d, highest stage found=%d)",
                 "No Matching Stages Found (requested",start_stage, chan->nstages);
  } */

  /* for each frequency */

  for(i = 0; i < nfreqs; i++) {
    w = twoPi * freq[i];
    val.real = 1.0; val.imag =  0.0;

    /* loop through the stages and filters for each stage, calculating
       the response for each frequency for all stages */

    stage_ptr = chan->first_stage;
    units_code = stage_ptr->input_units;
    for(j = 0; j < chan->nstages; j++) {
      nc = 0;
      sym_fir = 0;
      if(!stage_ptr->sequence_no)
        has_stage0 = 1;
      if(start_stage >= 0 && stop_stage && (stage_ptr->sequence_no < start_stage ||
         stage_ptr->sequence_no > stop_stage)) {
        stage_ptr = stage_ptr->next_stage;
        continue;
      }
      else if(start_stage >= 0 && !stop_stage && stage_ptr->sequence_no != start_stage) {
        stage_ptr = stage_ptr->next_stage;
        continue;
      }
      matching_stages++;
      blkt_ptr = stage_ptr->first_blkt;
      while(blkt_ptr) {
        eval_flag = 0;
        switch(blkt_ptr->type) {
        case ANALOG_PZ:
        case LAPLACE_PZ:
          analog_trans(blkt_ptr, freq[i], &of);
          eval_flag = 1;
          break;
        case IIR_PZ:
          if(blkt_ptr->blkt_info.pole_zero.nzeros || blkt_ptr->blkt_info.pole_zero.npoles) {
            iir_pz_trans(blkt_ptr, w, &of);
            eval_flag = 1;
          }
          break;
        case FIR_SYM_1:
        case FIR_SYM_2:
	  if(blkt_ptr->type == FIR_SYM_1)
	    nc = (double) blkt_ptr->blkt_info.fir.ncoeffs*2 - 1;
	  else if(blkt_ptr->type == FIR_SYM_2)
	    nc = (double) blkt_ptr->blkt_info.fir.ncoeffs*2;
          if(blkt_ptr->blkt_info.fir.ncoeffs) {
            fir_sym_trans(blkt_ptr, w, &of);
	    sym_fir = 1;
            eval_flag = 1;
          }
          break;
        case FIR_ASYM:
	  nc = (double) blkt_ptr->blkt_info.fir.ncoeffs;
          if(blkt_ptr->blkt_info.fir.ncoeffs) {
            fir_asym_trans(blkt_ptr, w, &of);
	    sym_fir = -1;
            eval_flag = 1;
          }
          break;
        case DECIMATION:
	  if(nc != 0) {
	    /* IGD 08/27/08 Use estimated delay instead of calculated */
	    estim_delay = (double) blkt_ptr->blkt_info.decimation.estim_delay;
	    corr_applied = blkt_ptr->blkt_info.decimation.applied_corr;

	    /* Asymmetric FIR coefficients require a delay correction */
	    if ( sym_fir == -1 ) {
	      if (TRUE == use_delay(QUERY_DELAY))
		delay = estim_delay;
	      else
		delay = corr_applied;
	    }
	    /* Otherwise delay has already been handled in fir_sym_trans() */
	    else {
	      delay = 0;
	    }

	    calc_time_shift (delay, w, &of);

	    eval_flag = 1;
	  }
          break;
	case LIST: /* This option is added in version 2.3.17 I.Dricker*/
		calc_list (blkt_ptr, i, &of); /*compute real and imag parts for the i-th ampl and phase */
		eval_flag = 1;
		break;
	case IIR_COEFFS: /* This option is added in version 2.3.17 I.Dricker*/
		iir_trans(blkt_ptr, w, &of);
		eval_flag = 1;
		break;
        default:
          break;
        }
        if(eval_flag)
          zmul(&val, &of);
        blkt_ptr = blkt_ptr->next_blkt;
      }
      stage_ptr = stage_ptr->next_stage;
    }

    /* if no matching stages were found, then report the error */

    if(!matching_stages && !has_stage0) {
      error_return(NO_STAGE_MATCHED, "calc_resp: %s start_stage=%d, highest stage found=%d)",
                   "No Matching Stages Found (requested",start_stage, chan->nstages);
    }
    else if(!matching_stages) {
      error_return(NO_STAGE_MATCHED, "calc_resp: %s start_stage=%d, highest stage found=%d)",
                   "No Matching Stages Found (requested",start_stage, chan->nstages-1);
    }

    /*  Write output for freq[i] in output[i] (note: unitScaleFact is a global variable
        set by the 'check_units' function that is used to convert to 'MKS' units when the
        the response was given as a displacement, velocity, or acceleration in units other
        than meters) */
    if (0 == useTotalSensitivityFlag) {
      output[i].real = val.real * chan->calc_sensit * unitScaleFact;
      output[i].imag = val.imag * chan->calc_sensit * unitScaleFact;
    }
    else  {
      output[i].real = val.real * chan->sensit * unitScaleFact;
      output[i].imag = val.imag * chan->sensit * unitScaleFact;
    }

    convert_to_units(units_code, out_units, &output[i], w);
  }

}

/*==================================================================
 * Convert response to velocity first, then to specified units
 *=================================================================*/
void convert_to_units(int inp, char *out_units, struct complex *data, double w) {
  int out = VEL, l;
  struct complex scale_val;

  /* if default units were specified by the user, no conversion is made,
     otherwise convert to unit the user specified. */

  if (out_units != NULL && (l=strlen(out_units)) > 0) {
    curr_seq_no = -1;
    if(!strncmp(out_units, "DEF", 3))
      return;
    else if(!strncmp(out_units, "DIS", 3)) out = DIS;
    else if(!strncmp(out_units, "VEL", 3)) out = VEL;
    else if(!strncmp(out_units, "ACC", 3)) out = ACC;
    else {
      error_return(BAD_OUT_UNITS, "convert_to_units: bad output units");
    }
  }
  else out = VEL;

  if (inp == DIS) {
    if (out == DIS) return;
    if (w != 0.0) {
      scale_val.real = 0.0; scale_val.imag = -1.0/w;
      zmul(data, &scale_val);
    }
    else data->real = data->imag = 0.0;
  }
  else if (inp == ACC) {
    if (out == ACC) return;
    scale_val.real = 0.0; scale_val.imag = w;
    zmul(data, &scale_val);
  }

  if (out == DIS) {
    scale_val.real = 0.0; scale_val.imag = w;
    zmul(data, &scale_val);
  }
  else if (out == ACC) {
    if (w != 0.0) {
      scale_val.real = 0.0; scale_val.imag = -1.0/w;
      zmul(data, &scale_val);
    }
    else data->real = data->imag = 0.0;
  }

}


/*==================================================================
 *                Response of a digital IIR filter
 * This code is modified fom the FORTRAN subroutine written and tested by
 * Bob Hutt (ASL USGS). Evaluates phase directly from imaginary and real
 *    parts of IIR filter coefficients.
 * C translation from FORTRAN function: Ilya Dricker (ISTI), i.dricker@isti.com
 * Version 0.2 07/12/00
 *================================================================*/
void iir_trans(struct blkt *blkt_ptr, double wint, struct complex *out) {


	 double h0;
         double  xre, xim, phase;
         double  amp;
	 double t, w;
	 double *cn, *cd; /* numerators and denominators */
	 int nn, nd, in, id;

   	  struct blkt *next_ptr;


  h0 = blkt_ptr->blkt_info.coeff.h0;  /* set a sensitivity */
  next_ptr = blkt_ptr->next_blkt;

	/* Sample interaval = 1/sample rate */
  t = next_ptr->blkt_info.decimation.sample_int;

	/* Numerator coeffs number */
  nn = blkt_ptr->blkt_info.coeff.nnumer;
	/* Denominator coeffs number */
  nd = blkt_ptr->blkt_info.coeff.ndenom;

	/* Numerators */
  cn = blkt_ptr->blkt_info.coeff.numer;
        /* Denominators */
  cd = blkt_ptr->blkt_info.coeff.denom;

/* Calculate radial freq. time sample interval */
  w = wint * t;

/* Handle numerator */
  xre= cn[0];
  xim = 0.0;

  if (nn != 1)	{
	 for (in=1; in<nn; in++)	{
		xre+=cn[in]*cos(-(in*w));
		xim+=cn[in]*sin(-(in*w));
 	}
 }
  amp=sqrt(xre*xre + xim*xim);
  phase=atan2(xim,xre);



/* handle denominator */

  xre= cd[0];
  xim = 0.0;

  if(nd !=1 )	{
         for (id = 1; id < nd; id ++)	{
         xre+=cd[id]*cos(-(id*w));
         xim+=cd[id]*sin(-(id*w));
	}
 }

   amp /= (sqrt(xre*xre+xim*xim));
   phase = (phase - atan2(xim,xre)) ;




  out->real = amp * cos(phase) * h0;
  out->imag = amp * sin(phase) * h0;

 }


/*================================================================
 *	Response of blockette 55 (Response List Blockette)
 * Function introduced in version 3.2.17 of evalresp
 * Ilya Dricker ISTI (.dricker@isti.com) 06/22/00
 *===============================================================*/
void calc_list (struct blkt *blkt_ptr, int i, struct complex *out)	{
  double  amp, phase;
  double halfcirc = 180;

  amp = blkt_ptr->blkt_info.list.amp[i];
  phase = blkt_ptr->blkt_info.list.phase[i];
  phase = phase/halfcirc * (double) Pi; /*convert the phase to radians from the default degrees */
  out->real = amp * cos(phase);
  out->imag = amp * sin(phase);

}

/*==================================================================
 *                Response of analog filter
 *=================================================================*/
void analog_trans(struct blkt *blkt_ptr, double freq, struct complex *out) {
  int nz, np, i;
  struct complex *ze, *po, denom, num, omega, temp;
  double h0, mod_squared;

  if (blkt_ptr->type == LAPLACE_PZ) freq = twoPi * freq;
  omega.imag = freq;
  omega.real = 0.0;
  denom.real = denom.imag = num.real = num.imag = 1.0;

  ze = blkt_ptr->blkt_info.pole_zero.zeros;
  nz = blkt_ptr->blkt_info.pole_zero.nzeros;
  po = blkt_ptr->blkt_info.pole_zero.poles;
  np = blkt_ptr->blkt_info.pole_zero.npoles;
  h0 = blkt_ptr->blkt_info.pole_zero.a0;

  for (i = 0; i < nz; i++) {
	/* num=num*(omega-zero[i]) */
	temp.real = omega.real - ze[i].real;
	temp.imag = omega.imag - ze[i].imag;
    zmul(&num, &temp);
  }
  for (i = 0; i < np; i++) {
	/* denom=denom*(omega-pole[i]) */
	temp.real = omega.real - po[i].real;
	temp.imag = omega.imag - po[i].imag;
    zmul(&denom, &temp);
  }

  /* gain*num/denum */

  temp.real = denom.real;
  temp.imag = -denom.imag;
  zmul(&temp, &num);
  mod_squared = denom.real*denom.real + denom.imag*denom.imag;

  if (mod_squared != 0.) {
      temp.real /= mod_squared;
      temp.imag /= mod_squared;
  }
  else if (temp.real != 0.0 || temp.imag != 0.0) {
      fprintf(stderr,"%s WARNING (analog_trans): Numerical problem detected. Result might be wrong.", myLabel);
      fprintf(stderr,"%s\t Execution continuing.\n", myLabel);
  }

  out->real = h0 * temp.real;
  out->imag = h0 * temp.imag;
}

/*==================================================================
 *                Response of symetrical FIR filters
 *=================================================================*/
void fir_sym_trans(struct blkt *blkt_ptr, double w, struct complex *out) {
  double *a, h0, wsint;
  struct blkt *next_ptr;
  int na;
  int k, fact;
  double R = 0.0, sint;

  a = blkt_ptr->blkt_info.fir.coeffs;
  na = blkt_ptr->blkt_info.fir.ncoeffs;
  next_ptr = blkt_ptr->next_blkt;
  h0 = blkt_ptr->blkt_info.fir.h0;
  sint = next_ptr->blkt_info.decimation.sample_int;
  wsint = w * sint;

  if (blkt_ptr->type == FIR_SYM_1) {
    for (k=0; k<(na-1); k++) {
      fact = na-(k+1);
      R += a[k] * cos(wsint * fact);
    }
    out->real = (a[k] + 2.0*R) * h0;
    out->imag = 0.;
  }
  else if (blkt_ptr->type == FIR_SYM_2) {
    for (k=0; k<na; k++) {
      fact = na-(k+1);
      R += a[k] * cos(wsint * (fact+0.5));
    }
    out->real = 2.0*R * h0;
    out->imag = 0.;
  }
}

/*==================================================================
 *                Response of asymetrical FIR filters
 *=================================================================*/
void fir_asym_trans(struct blkt *blkt_ptr, double w, struct complex *out) {
  double *a, h0, sint;
  struct blkt *next_ptr;
  int na;
  int k;
  double R = 0.0, I = 0.0;
  double wsint, y;
  double mod, pha;

  a = blkt_ptr->blkt_info.fir.coeffs;
  na = blkt_ptr->blkt_info.fir.ncoeffs;
  next_ptr = blkt_ptr->next_blkt;
  h0 = blkt_ptr->blkt_info.fir.h0;
  sint = next_ptr->blkt_info.decimation.sample_int;
  wsint = w * sint;

  for (k = 1; k < na; k++) {
    if (a[k] != a[0]) break;
  }
  if (k == na) {
    if (wsint == 0.0) out->real = 1.;
    else out->real = (sin(wsint/2.*na) / sin(wsint/2.)) * a[0];
    out->imag = 0;
    return;
  }

  for (k = 0; k < na; k++) {
    y = wsint * k;
    R += a[k] * cos(y);
    I += a[k] * -sin(y);
  }

  mod = sqrt(R*R + I*I);
  pha = atan2(I,R);
  R = mod * cos(pha);
  I = mod * sin(pha);
  out->real = R * h0;
  out->imag = I * h0;
}

/*==================================================================
 *                Response of IIR filters
 *=================================================================*/
void iir_pz_trans(struct blkt *blkt_ptr, double w, struct complex *out) {
  struct complex *ze, *po;
  double h0, sint, wsint;
  struct blkt *next_ptr;
  int nz, np;
  int i;
  double mod = 1.0, pha = 0.0;
  double R, I;
  double c, s;

  ze = blkt_ptr->blkt_info.pole_zero.zeros;
  nz = blkt_ptr->blkt_info.pole_zero.nzeros;
  po = blkt_ptr->blkt_info.pole_zero.poles;
  np = blkt_ptr->blkt_info.pole_zero.npoles;
  h0 = blkt_ptr->blkt_info.pole_zero.a0;
  next_ptr = blkt_ptr->next_blkt;
  sint = next_ptr->blkt_info.decimation.sample_int;
  wsint = w * sint;

  c = cos(wsint);
  s = sin(wsint);  /* IGD 10/21/02 instead of -: pointed by Sleeman */
  for (i = 0; i < nz; i++) {
    R = c - ze[i].real;    /* IGD 09/20/01 instead of + */
    I = s - ze[i].imag;    /* IGD 09/20/01 instead of + */
    mod *= sqrt(R*R + I*I);
    if (R == 0.0 && I == 0.0) pha += 0.0;
    else                      pha += atan2(I, R);
  }
  for (i = 0; i < np; i++) {
    R = c - po[i].real;    /* IGD 09/20/01 instead of + */
    I = s - po[i].imag;    /* IGD 09/20/01 instead of + */
    mod /= sqrt(R*R +I*I);
    if (R == 0.0 && I == 0.0) pha += 0.0;
    else                      pha -= atan2(I, R);
  }
  out->real = mod * cos(pha) * h0;
  out->imag = mod * sin(pha) * h0;
 }

/*==================================================================
 *      calculate the phase shift equivalent to the time shift
 *      delta at the frequence w (rads/sec)
 *=================================================================*/
void calc_time_shift(double delta, double w, struct complex *out) {
  out->real = cos(w*delta);
  out->imag = sin(w*delta);
}



/*==================================================================
 *    Complex multiplication:  complex version of val1 *= val2;
 *=================================================================*/
void zmul(struct complex *val1, struct complex *val2) {
double r, i;
    r = val1->real*val2->real - val1->imag*val2->imag;
    i = val1->imag*val2->real + val1->real*val2->imag;
    val1->real = r;
    val1->imag = i;
}

/*=================================================================
*                   Normalize response
*=================================================================*/
void norm_resp(struct channel *chan, int start_stage, int stop_stage,
               int hide_sensitivity_mismatch_warning) {
  // TODO - NULL assignments below made blindly to fix compiler warning.  bug?
  struct stage *stage_ptr;
  struct blkt *fil, *last_fil = NULL, *main_filt = NULL;
  int i, main_type, reset_gain, skipped_stages = 0;
  double w,  f;
  double  percent_diff;
  struct complex of, df;

  /* -------- TEST 1 -------- */
  /*
      A single stage response must specify a stage gain, a stage zero
      sensitivity, or both.  If the gain has been set, simply drop through
      to the next test. If the gain is zero, set the gain equal to the
      sensitivity and then go to the next test. */

  if (chan->nstages == 1 ) {			/* has no stage 0, does it have a gain??? */
    stage_ptr = chan->first_stage;
    fil = stage_ptr->first_blkt;
    while(fil != (struct blkt *)NULL && fil->type != GAIN) {
      last_fil = fil;
      fil = last_fil->next_blkt;
    }
    if (fil == (struct blkt *)NULL) {
      error_return(ILLEGAL_RESP_FORMAT, "norm_resp; no stage gain defined, zero sensitivity");
    }
  }
  else if (chan->nstages == 2 ) {		/* has a stage 0??? */
    stage_ptr = chan->first_stage;
    fil = stage_ptr->first_blkt;
    while(fil != (struct blkt *)NULL && fil->type != GAIN) {
      last_fil = fil;
      fil = last_fil->next_blkt;
    }
    if (fil == (struct blkt *)NULL) {
      if (chan->sensit == 0.0) {
        error_return(ILLEGAL_RESP_FORMAT, "norm_resp; no stage gain defined, zero sensitivity");
      }
      else {
        fil = alloc_gain();
        fil->blkt_info.gain.gain      = chan->sensit;
        fil->blkt_info.gain.gain_freq = chan->sensfreq;
        last_fil->next_blkt = fil;
      }
    }
  }

  /* -------- TEST 2 -------- */
  /*

     Compute the variable chan->calc_sensit.  This value would normally be the
     same as chan->sensit (the stage zero sensitivity) if the sensitivity
     has been given and all the channel gains are at the sensitivity frequency.

     Note that we must verify that each filter gain and normalization is
     at the sensitivity frequency, before we can compute the overall
     channel sensitivity.  If the frequencies are different, calculate the
     gain for the stage at the same frequency as the channel sensitivity
     frequency.

     Note on terminology:

     chan->sensit      = stage zero sensitivity read from input file
     (i.e. the total sensitivity for a given channel).
     chan->sensfreq    = frequency at which chan-sensit is reported.
     chan->calc_sensit = product of the individual stage gains. This is computed
     below. The computation is done at the frequency
     chan->sensfreq. */

  /* Loop thru stages, checking for stage gains of zero */

  stage_ptr = chan->first_stage;
  for(i = 0; i < chan->nstages; i++) {
    fil = stage_ptr->first_blkt;
    curr_seq_no = stage_ptr->sequence_no;
    while(fil) {
      if (fil->type == GAIN && fil->blkt_info.gain.gain == 0.0 ) {
        error_return(ILLEGAL_RESP_FORMAT, "norm_resp; zero stage gain");
      }
      fil = fil->next_blkt;
    }
    stage_ptr = stage_ptr->next_stage;
  }

  /*
     If necessary, loop thru filters, looking for a non-zero frequency to
     use for chan->sensfreq. If the channel sensitivity is defined, then
     we will use chan->sensfreq, regardless of whether it is zero or not.
     Otherwise, we use the last non-zero filter gain frequency. Since
     filters are  typically for low pass purposes, the last non-zero
     frequency is likely the best choice, as its pass band is the
     narrowest. If all the frequencies are zero, then use zero! */

  if ( chan->sensit == 0.0 ) {
    stage_ptr = chan->first_stage;
    for(i = 0; i < chan->nstages; i++) {
      fil = stage_ptr->first_blkt;
      while(fil) {
        if (fil->type == GAIN && fil->blkt_info.gain.gain_freq != 0.0 )
          chan->sensfreq = fil->blkt_info.gain.gain_freq;
        fil = fil->next_blkt;
      }
      stage_ptr = stage_ptr->next_stage;
    }
  }

  /*
     Loop thru filters, evaluate the filter gains and normalizations at
     the single frequency,  chan->sensfreq. Use the stage gains to
     calculate a total channel sensitivity. */

  chan->calc_sensit = 1.0;
  f = chan->sensfreq;
  w = twoPi * f;

  stage_ptr = chan->first_stage;
  for(i = 0; i < chan->nstages; i++) {
    if(start_stage >= 0 && stop_stage && (stage_ptr->sequence_no < start_stage ||
					  stage_ptr->sequence_no > stop_stage)) {
      if(stage_ptr->sequence_no)
	skipped_stages = 1;
      stage_ptr = stage_ptr->next_stage;
      continue;
    }
    else if(start_stage >= 0 && !stop_stage && stage_ptr->sequence_no != start_stage) {
      if(stage_ptr->sequence_no)
	skipped_stages = 1;
      stage_ptr = stage_ptr->next_stage;
      continue;
    }
    fil = stage_ptr->first_blkt;
    curr_seq_no = stage_ptr->sequence_no;
    main_type = 0;
    while(fil) {
      switch(fil->type) {
      case DECIMATION:
      case REFERENCE:
        break;
      case ANALOG_PZ:
      case LAPLACE_PZ:
      case IIR_PZ:
      case FIR_SYM_1:
      case FIR_SYM_2:
      case FIR_ASYM:
      case IIR_COEFFS:   /* IGD New type from v 3.2.17 */
        main_filt = fil;
        main_type = fil->type;
        break;
      case GAIN:
        if(curr_seq_no) {
          if((fil->blkt_info.gain.gain_freq != chan->sensfreq) ||
	     ((main_type == ANALOG_PZ ||
	       main_type == LAPLACE_PZ ||
	       main_type == IIR_PZ) &&
	      (main_filt->blkt_info.pole_zero.a0_freq != chan->sensfreq))) {

            reset_gain = 1;
            if(main_type == ANALOG_PZ || main_type == LAPLACE_PZ) {
              main_filt->blkt_info.pole_zero.a0 = 1.0;
              analog_trans(main_filt, fil->blkt_info.gain.gain_freq, &df);
              if(df.real == 0.0 && df.imag == 0.0) {
                error_return(ILLEGAL_FILT_SPEC,
                             "norm_resp: Gain frequency of zero found in bandpass analog filter");
              }
              analog_trans(main_filt, f, &of);
              if(of.real == 0.0 && of.imag == 0.0) {
                error_return(ILLEGAL_FILT_SPEC,
                             "norm_resp: Chan. Sens. frequency found with bandpass analog filter");
              }
            }
            else if(main_type == IIR_PZ) {
              main_filt->blkt_info.pole_zero.a0 = 1.0;
              iir_pz_trans(main_filt, twoPi * fil->blkt_info.gain.gain_freq, &df);
              iir_pz_trans(main_filt, w, &of);
            }
            else if((main_type == FIR_SYM_1 || main_type == FIR_SYM_2) &&
                     main_filt->blkt_info.fir.ncoeffs) {
              main_filt->blkt_info.fir.h0 = 1.0;
              fir_sym_trans(main_filt, twoPi * fil->blkt_info.gain.gain_freq, &df);
              fir_sym_trans(main_filt, w, &of);
            }
            else if(main_type == FIR_ASYM && main_filt->blkt_info.fir.ncoeffs) {
              main_filt->blkt_info.fir.h0 = 1.0;
              fir_asym_trans(main_filt, twoPi * fil->blkt_info.gain.gain_freq, &df);
              fir_asym_trans(main_filt, w, &of);
            }
            else if(main_type == IIR_COEFFS) {   /*IGD - new case for 3.2.17 */
              main_filt->blkt_info.coeff.h0 = 1.0;
              iir_trans(main_filt, twoPi * fil->blkt_info.gain.gain_freq, &df);
              iir_trans(main_filt, w, &of);
            }

            else
              reset_gain = 0;

            if(reset_gain) {
              fil->blkt_info.gain.gain /= sqrt(df.real*df.real + df.imag*df.imag);
              fil->blkt_info.gain.gain *= sqrt(of.real*of.real + of.imag*of.imag);
              fil->blkt_info.gain.gain_freq = f;
              if(main_type == ANALOG_PZ || main_type == LAPLACE_PZ ||
                 main_type == IIR_PZ) {
                main_filt->blkt_info.pole_zero.a0 = 1.0 / sqrt(of.real*of.real + of.imag*of.imag);
                main_filt->blkt_info.pole_zero.a0_freq = f;
              }
              else if(main_type == FIR_SYM_1 || main_type == FIR_SYM_2 || main_type == FIR_ASYM) {
                main_filt->blkt_info.fir.h0 = 1.0 / sqrt(of.real*of.real + of.imag*of.imag);
              }
              else if(main_type == IIR_COEFFS) {
                main_filt->blkt_info.coeff.h0 = 1.0 / sqrt(of.real*of.real + of.imag*of.imag);
              }
            }
          }
          /* compute new overall sensitivity. */
          chan->calc_sensit *= fil->blkt_info.gain.gain;
          if(chan->nstages == 1) {
            chan->sensit = chan->calc_sensit;
          }
        }
        break;
      default:
        break;
      }
      fil = fil->next_blkt;
    }
    stage_ptr = stage_ptr->next_stage;
  }

  /* -------- TEST 3 -------- */
  /* Finally, print a warning, if necessary */

  if(!hide_sensitivity_mismatch_warning && !skipped_stages && chan->sensit != 0.0) {
    percent_diff = fabs((chan->sensit - chan->calc_sensit) / chan->sensit);

    if (percent_diff >= 0.05) {
      fprintf(stderr,"%s WARNING (norm_resp): computed and reported sensitivities", myLabel);
      fprintf(stderr,"%s differ by more than 5 percent. \n", myLabel);
      fprintf(stderr,"%s\t Execution continuing.\n", myLabel);
      fflush(stderr);
    }
  }
}

/* IGD 04/05/04 Phase unwrapping function
 * It works only inside a loop over phases.
 *
 */
double
        unwrap_phase(double phase, double prev_phase, double range, double *added_value)
	{
		phase += *added_value;
		if (fabs(phase - prev_phase) > range/2)
		{
			if (phase - prev_phase > 0)
			{
				*added_value -= range;
				phase -= range;
			}
			else
			{
				*added_value += range;
				phase += range;
			}
		}
		return phase;
	}

/*
 * wrap_phase:  Wraps a set of phase values so that all of the values are
 *      between "-range" and "+range" (inclusive).  This function is called
 *      iteratively, once for each phase value.
 *   phase - phase value to process.
 *   range - range value to use.
 *   added_value - pointer to offset value used for each call.
 * Returns:  The "wrapped" version of the given phase value.
 */
double wrap_phase(double phase, double range, double *added_value)
{
  phase += *added_value;
  if(phase > range/2)
  {
    *added_value -= range;
    phase -= range;
  }
  else if(phase < -range/2)
  {
    *added_value += range;
    phase += range;
  }
  return phase;
}

