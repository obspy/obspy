/***************************************************************************
 * mexMsReadTraces.c
 *
 * This file is part of the library libmseed.
 * 
 *     libmseed is free software; you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation; either version 2 of the License, or
 *     (at your option) any later version.
 * 
 *     libmseed is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with Foobar; if not, write to the Free Software
 *     Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
 *
 * Mex wrapper function for the msReadTraces function in the libmseed library.
 * 
 * mexMsReadTraces takes the following arguments:
 *  filename
 *  [timetolerance]
 *  [sampratetolerance]
 *  [verbosity]
 * 
 * The return value is a Matlab structure similiar to the libmseed structure
 *  MSTrace_s containing the trace header and data.
 *
 * Original written by Stefan Mertl
 * Vienna University of Technology
 * Institute of Geodesy and Geophysics
 * Dept. of Geophysics
 *
 * Further modification by Chad Trabant, IRIS Data Managment Center
 *
 * modified: 2008.171
 ***************************************************************************/

#include "mex.h"
#include "../libmseed.h"

void
mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  MSTraceGroup *mstg = NULL;
  MSTrace *mst = NULL;
  const char **my_fnames = NULL;
  char *filename;
  int buflen;
  double timetol = -1.0;
  double sampratetol = -1.0;
  flag verbose = 0;
  int i, j, nfields;
  mxArray *tmp_val;
  double *tmp_val_ptr;
  int *data;
  
  /* Sanity check input and output */
  if ( nrhs < 1 )
    {
      mexPrintf ("mexMsReadTraces - Read Mini-SEED data into Matlab\n\n");
      mexPrintf ("Usage: mexMsReadTraces (filename, [timetol], [sampratetol], [verbosity])\n");
      mexPrintf ("  filename    - Name of file to read Mini-SEED data from\n");
      mexPrintf ("  timetol     - Time tolerance, default is 1/2 sample period (-1)\n");
      mexPrintf ("  sampratetol - Sample rate tolerance, default is rate depdendent (-1)\n");
      mexPrintf ("  verbosity   - Level of diagnostic messages, default 0\n\n");
      mexErrMsgTxt ("At lease one 1 argument required.");
    }
  else if ( nlhs > 1 )
    {
      mexErrMsgTxt ("Too many output arguments.");
    }
  
  /* Redirect libmseed logging messages to Matlab functions */
  ms_loginit ((void *)&mexPrintf, NULL, (void *)&mexWarnMsgTxt, NULL);
  
  /* Get the length of the input string */
  buflen = (mxGetM (prhs[0]) * mxGetN (prhs[0])) + 1;
  
  /* Allocate memory for input string */
  filename = mxCalloc (buflen, sizeof (char));
  
  /* Assign the input arguments to variables */
  if ( mxGetString (prhs[0], filename, buflen) )
    mexErrMsgTxt ("Not enough space. Filename string is truncated.");
  if ( nrhs >= 2 )
    timetol = mxGetScalar(prhs[1]);
  if ( nrhs >= 3 )
    sampratetol = mxGetScalar(prhs[2]);
  if ( nrhs >= 4 )
    verbose = (flag) mxGetScalar(prhs[3]);
  
  /* Read the file */
  if ( ms_readtraces (&mstg, filename, -1, timetol, sampratetol, 1, 1, 1, verbose) != MS_NOERROR )
    mexErrMsgTxt ("Error reading files");
  
  /* Print some information to the Matlab command prompt */
  mst_printtracelist (mstg, 0, verbose, 1);
  
  /* Create the Matlab output structure */
  mst = mstg->traces;
  for (i=0; i < mstg->numtraces; i++)
    {
      if (i==0)
   	{
	  nfields = 13;
	  my_fnames = mxCalloc (nfields, sizeof (*my_fnames));
	  my_fnames[0] = "network";
	  my_fnames[1] = "station";
	  my_fnames[2] = "location";
	  my_fnames[3] = "channel";
	  my_fnames[4] = "dataquality";
	  my_fnames[5] = "type";
	  my_fnames[6] = "startTime";
	  my_fnames[7] = "endTime";
	  my_fnames[8] = "sampleRate";
	  my_fnames[9] = "sampleCount";
	  my_fnames[10] = "numberOfSamples";
	  my_fnames[11] = "sampleType";
	  my_fnames[12] = "data";
	  plhs[0] = mxCreateStructMatrix(mstg->numtraces, 1, nfields, my_fnames);
	  mxFree(my_fnames);
   	}
      
      /* Copy the data of the mst structure to the matlab output structure. */
      data = (int*)mst->datasamples;
      tmp_val = mxCreateDoubleMatrix(mst->numsamples, 1, mxREAL);
      tmp_val_ptr = mxGetPr(tmp_val);
      for (j = 0; j < mst->numsamples; j++)
	{
	  tmp_val_ptr[j] = data[j];
	}
      
      mxSetFieldByNumber(plhs[0], i, 0, mxCreateString(mst->network));
      mxSetFieldByNumber(plhs[0], i, 1, mxCreateString(mst->station));
      mxSetFieldByNumber(plhs[0], i, 2, mxCreateString(mst->location));
      mxSetFieldByNumber(plhs[0], i, 3, mxCreateString(mst->channel));
      mxSetFieldByNumber(plhs[0], i, 4, mxCreateDoubleScalar((int)mst->dataquality));
      mxSetFieldByNumber(plhs[0], i, 5, mxCreateDoubleScalar((int)mst->type));
      mxSetFieldByNumber(plhs[0], i, 6, mxCreateDoubleScalar(mst->starttime));
      mxSetFieldByNumber(plhs[0], i, 7, mxCreateDoubleScalar(mst->endtime));
      mxSetFieldByNumber(plhs[0], i, 8, mxCreateDoubleScalar(mst->samprate));
      mxSetFieldByNumber(plhs[0], i, 9, mxCreateDoubleScalar(mst->samplecnt));
      mxSetFieldByNumber(plhs[0], i, 10, mxCreateDoubleScalar(mst->numsamples));
      mxSetFieldByNumber(plhs[0], i, 11, mxCreateDoubleScalar((int)mst->sampletype));
      mxSetFieldByNumber(plhs[0], i, 12, tmp_val);
      
      mst = mst->next;
    }
  
  mst_freegroup (&mstg);
}
