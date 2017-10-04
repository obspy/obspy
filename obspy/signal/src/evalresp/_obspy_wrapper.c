/*
 * ObsPy wrapper functions for evalresp.
 *
 * These are needed because evalresp uses setjmp/longjmp to implement error
 * handling. These cannot be called from Python without crashing due to it
 * messing with the internals of the interpreter.
 *
 * Copyright (C) ObsPy Development Team, 2014.
 *
 * This file is licensed under the terms of the GNU Lesser General Public
 * License, Version 3 (https://www.gnu.org/copyleft/lesser.html).
 *
 */


#include <setjmp.h>
#include "evresp.h"


int _obspy_check_channel(struct channel *chan)
{
  int rc;
  if ((rc = setjmp(jump_buffer)) == 0) {
    /* Direct invocation */
    GblChanPtr = chan;
    check_channel(chan);
    GblChanPtr = NULL;
    return 0;
  } else {
    /* Error called by longjmp */
    GblChanPtr = NULL;
    return rc;
  }
}

int _obspy_norm_resp(struct channel *chan, int start_stage, int stop_stage,
                     int hide_sensitivity_mismatch_warning)
{
  int rc;
  if ((rc = setjmp(jump_buffer)) == 0) {
    /* Direct invocation */
    GblChanPtr = chan;
    norm_resp(chan, start_stage, stop_stage, hide_sensitivity_mismatch_warning);
    GblChanPtr = NULL;
    return 0;
  } else {
    /* Error called by longjmp */
    GblChanPtr = NULL;
    return rc;
  }
}


int _obspy_calc_resp(struct channel *chan, double *freq, int nfreqs,
                     struct complex *output, char *out_units,
                     int start_stage, int stop_stage,
                     int useTotalSensitivityFlag)
{
  int rc;
  if ((rc = setjmp(jump_buffer)) == 0) {
    /* Direct invocation */
    GblChanPtr = chan;
    calc_resp(chan, freq, nfreqs, output, out_units, start_stage, stop_stage,
              useTotalSensitivityFlag);
    GblChanPtr = NULL;
    return 0;
  } else {
    /* Error called by longjmp */
    GblChanPtr = NULL;
    return rc;
  }
}
