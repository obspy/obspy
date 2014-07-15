/* evalresp.c:  Main module for 'evalresp'. */

/*
   10/19/2005 -- [ET]  Added parameters for List-blockette interpolation;
                       added warnings for unrecognized parameters.
    11/3/2005 -- [ET]  Moved 'use_delay()' function from 'calc_fctns.c'
                       to 'evalresp.c'.
     2/6/2006 -- [ET]  Moved 'use_delay()' function from 'evalresp.c'
                       to 'evresp.c'.
    3/28/2006 -- [ET]  Added "free(freqs)" to end of 'main()' function.
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <stdlib.h>
#include "evresp.h"

#define DATIMESIZE 32
#define TODAYSIZE 10



int main(int argc, char *argv[])
{
  char *sta_list, *cha_list, units[MAXFLDLEN], *file, *verbose;
  char *net_code, *locid, *type, rtype[MAXFLDLEN];
  int lin_typ = 0, nfreqs, i, fswidx, tmp_val, fldlen;
  int start_stage = -1, stop_stage = 0, stdio_flag = 0;
  int listinterp_out_flag = 0, listinterp_in_flag = 0;
  int unwrap_flag = 0;
  double listinterp_tension = 1000.0;
  char *t_o_day;
  char *datime;
  double incr, freq_lims[2], temp_val, *freqs;
  double val;
  struct response *first;
  char *minfstr, *maxfstr, *numfstr;
  char param_err_msgstr[] = "%s: missing option to argument '%s'";
  int useTotalSensitivityFlag = 0; /* IGD 01/29/10 */
  

  curr_seq_no = -1;

  myLabel[0]= '\0';

  if (argc < 5) {
    printf("EVALRESP V%s\n", REVNUM);
    printf("\nUSAGE: evalresp STALST CHALST YYYY DAY MINFREQ");
    printf(" MAXFREQ NFREQ [options]\n\n");
    printf("  OPTIONS\n\n");
    printf("    '-f file'              (directory-name|filename)\n");
    printf("    '-u units'             ('dis'|'vel'|'acc'|'def')\n");
    printf("    '-t time-of-day'       (HH:MM:SS)\n");
    printf("    '-s type-of-spacing'   (log|lin)\n");
    printf("    '-n netid'             ('II'|'IU'|'G'|'*'...)\n");
    printf("    '-l locid'             ('01'|'AA,AB,AC'|'A?'|'*'...)\n");
    printf("    '-r resp_type'         ('ap'=amp/pha | 'cs'=complex spectra | 'fap'=freq/amp/pha)\n");
    printf("    '-stage start [stop]'  (start and stop are integer stage numbers)\n");
    printf("    '-stdio'               (take input from stdin, output to stdout)\n");
    printf("    '-use-estimated-delay' (use estimated delay instead of correction applied in computation of ASYM FIR response)\n");
    printf("    '-il'                  (interpolate List blockette output)\n");
    printf("    '-ii'                  (interpolate List blockette input)\n");
    printf("    '-it tension'          (tension for List blockette interpolation)\n");
    printf("    '-unwrap'              (unwrap phase if the output is AP) \n");
    printf("    '-ts'                  (Use total sensitivity from stage 0 instead of computed)\n");
    printf("    '-v'                   (verbose; list");
    printf(" parameters on stdout)\n\n");
    printf("    NOTES:\n");
    printf("      (1) If the 'file' argument is a directory, ");
    printf("that directory will be\n          searced for RESP ");
    printf("files of the form RESP.NETID.STA.CHA.  Files\n");
    printf("          of this type are created by rdseed");
    printf(" when it is run with the\n          ");
    printf("'-R' option or when the '-d' option is used and");
    printf(" responses are\n          requested\n"); 
    printf("      (2) If the 'file' argument is a file, that");
    printf(" file is assumed to be\n          output from a ");
    printf("call to rdseed with the '-R' option\n");
    printf("      (3) If the 'file' argument is missing, the");
    printf(" current directory\n          will be searced for ");
    printf("RESP files of the form RESP.NETID.STA.CHA\n");
    printf("      (4) the directory indicated by the environment ");
    printf("variable SEEDRESP\n          will also be searched for");
    printf(" the requested files");
    printf(" (if it is defined),\n          but if matching station-");
    printf("channel-network values are found in\n");
    printf("          both directories, then the local files");
    printf(" take precedence\n");
    printf("      (5) the NETID (above) indicates a one or two letter network code.\n");
    printf("          If no network code exists in the SEED volume, a value of ''\n");
    printf("          or '*' will match the 'no network' case (since the ?? string\n");
    printf("          in the RESP file output is replaced with an empty string in\n");
    printf("          the program).\n");
    printf("      (6) the '-stage' option allows the user to specify the range of\n");
    printf("          stages that will be included in the calculation.  If only one\n");
    printf("          stage number is given, a response will only be calculated for\n");
    printf("          that stage.  If both a start and stop stage number are given,\n");
    printf("          any stage between (and including) the start and stop stages\n");
    printf("          will be included in the calculation.\n\n");
    printf("  EXAMPLES:\n");
    printf("evalresp AAK,ARU,TLY VHZ 1992 21 0.001 10 100 -f ");
    printf("/sd15/EVRESP/NEW/rdseed.out\n");
    printf("evalresp KONO BHN,BHE 1992 1 0.001 10 100 -f ");
    printf("/sd15/EVRESP/NEW -t 12:31:04 -v\n");
    printf("evalresp FRB BHE,BHZ 1994 31 0.001 10 100 -f ");
    printf("resp.all_stations -n '*' -v\n\n");    exit(1);
  }

    /* find index of first switch parameter */
  fswidx = 0;                /* loop until switch param found (if any) */
  while(++fswidx < argc &&   /*  (check if switch param or real #) */
               (strncmp(argv[fswidx],"-",1) != 0 || is_real(argv[fswidx])));
  if(fswidx < 5) {
    error_exit(USAGE_ERROR,"Not all of the required inputs are \n\t"
            "present (%d missing), type '%s' for usage", 8-fswidx, argv[0]);
  }
    /* setup min,max,num freq values; use defaults if missing */
  minfstr = (fswidx > 5) ? argv[5] : "1.0";
  maxfstr = (fswidx > 6) ? argv[6] : minfstr;
  numfstr = (fswidx > 7) ? argv[7] : "1";

    /* warn about any unexpected non-switch parameters */
  for(i=8; i<fswidx; ++i)
  {
    if(argv[i] != NULL && argv[i][0] != '\0')
      fprintf(stderr,"WARNING:  Unrecognized parameter:  %s\n",argv[i]);
  }

  /* initialize the optional arguments */

  /* If user did not define -use-delay option by default it is FALSE */
  use_delay(FALSE);

  strncpy(units,"",MAXFLDLEN);
  strncpy(rtype,"",MAXFLDLEN);
  net_code = locid = file = verbose = type = (char *)NULL;
  if((t_o_day = (char *)malloc(strlen("00:00:00")+1)) == (char *)NULL) {
  }
  sprintf(t_o_day, "00:00:00");

  /* then get the optional arguments (if any) */

  for (i = fswidx; i < argc; i++) {
    if (0 == strcmp(argv[i], "-use-estimated-delay"))
       use_delay(TRUE);
    else if (0 == strcmp(argv[i], "-use-delay"))  /* IGD 04/29/09 Backward compatibility */
       use_delay(TRUE);
    else if(!strcmp(argv[i], "-u")){
      if((++i) < argc && *argv[i] != '-')
        strncpy(units,argv[i],MAXFLDLEN);
      else
        error_exit(USAGE_ERROR,param_err_msgstr,argv[0],argv[i-1]);
    }
    else if(!strcmp(argv[i], "-r")){
      if((++i) < argc && *argv[i] != '-')
        strncpy(rtype,argv[i],MAXFLDLEN);
      else
        error_exit(USAGE_ERROR,param_err_msgstr,argv[0],argv[i-1]);
    }
    else if(!strcmp(argv[i], "-stdio")){
      stdio_flag = 1;
    }
    else if(!strcmp(argv[i], "-stage")){
      if((++i) < argc && is_int(argv[i]))
        start_stage = atoi(argv[i]);
      else
        error_exit(USAGE_ERROR,param_err_msgstr,argv[0],argv[i-1]);
      if((++i) < argc && is_int(argv[i])) {
        tmp_val = atoi(argv[i]);
        if(tmp_val > start_stage)
          stop_stage = tmp_val;
        else {
          stop_stage = start_stage;
          start_stage = tmp_val;
        }
      }
      else {
        i--;
      }
    }
    else if(!strcmp(argv[i], "-n")){
      if((++i) < argc && *argv[i] != '-')
        net_code = argv[i];
      else
        error_exit(USAGE_ERROR,param_err_msgstr,argv[0],argv[i-1]);
    }
    else if(!strcmp(argv[i], "-l")){
      if((++i) < argc && *argv[i] != '-')
        locid = argv[i];
      else
        error_exit(USAGE_ERROR,param_err_msgstr,argv[0],argv[i-1]);
    }
    else if(!strcmp(argv[i], "-t")){
      if((++i) < argc && *argv[i] != '-') {
        fldlen = strlen(argv[i]);
        if(fldlen >= TODAYSIZE) {
            t_o_day = realloc(t_o_day,(fldlen+1));
        }
        strcpy(t_o_day, argv[i]);
      }
      else
        error_exit(USAGE_ERROR,param_err_msgstr,argv[0],argv[i-1]);
    }
    else if(!strcmp(argv[i], "-f")){
      if((++i) < argc && *argv[i] != '-')
        file = argv[i];
      else
        error_exit(USAGE_ERROR,param_err_msgstr,argv[0],argv[i-1]);
    }
    else if(!strcmp(argv[i], "-s")){
      if((++i) < argc && *argv[i] != '-')
        type = argv[i];
      else
        error_exit(USAGE_ERROR,param_err_msgstr,argv[0],argv[i-1]);
      if(strcmp(type,"lin") & strcmp(type,"log")){
        error_exit(USAGE_ERROR,"%s: option '-s' illegal type '%s'",
                   argv[0], type);
      }
    }
    else if(!strcmp(argv[i], "-unwrap"))
      unwrap_flag = 1;
    else if(!strcmp(argv[i], "-ts"))
      useTotalSensitivityFlag = 1;
    else if(!strcmp(argv[i], "-il"))
      listinterp_out_flag = 1;
    else if(!strcmp(argv[i], "-ii"))
      listinterp_in_flag = 1;
    else if(!strcmp(argv[i], "-it")){
      if((++i) < argc && *argv[i] != '-') {
        if(is_real(argv[i]))
          listinterp_tension = atof(argv[i]);
        else {
          error_exit(USAGE_ERROR,
              "%s:  illegal value for 'it' parameter:  %s",argv[0],argv[i]);
        }
      }
      else
        error_exit(USAGE_ERROR,param_err_msgstr,argv[0],argv[i-1]);
    }
    else if(!strcmp(argv[i], "-v"))
      verbose = argv[i];
    else if(argv[i] != NULL && argv[i][0] != '\0')
      fprintf(stderr,"WARNING:  Unrecognized parameter:  %s\n",argv[i]);
  }

  sta_list = argv[1];
  cha_list = argv[2];

  if(!is_int(argv[3]))
    error_exit(USAGE_ERROR,"year must be an integer, found '%s'", argv[3]);
  if(!is_int(argv[4]))
    error_exit(USAGE_ERROR,"julian day must be an integer, found '%s'", argv[4]);
  if(!is_time(t_o_day))
    error_exit(USAGE_ERROR,"'time of day' must have format 'hh[:mm[:ss[.#####]]]', found '%s'",
                           t_o_day);

  fldlen = strlen(argv[3])+strlen(argv[4])+strlen(t_o_day)+3;
  if((datime = (char *)malloc(fldlen)) == (char *)NULL) {
  }
  sprintf(datime, "%s,%s,%s", argv[3], argv[4], t_o_day);

  if(!is_real(minfstr) || !is_real(maxfstr))
    error_exit(USAGE_ERROR,"freq_lims must be real numbers, found (%s,%s)",
               minfstr,maxfstr);
  freq_lims[0] = atof(minfstr);
  freq_lims[1] = atof(maxfstr);
  if ((freq_lims[0] == 0.0 || freq_lims[1] == 0.0) &&
                                (type == NULL || strcmp(type,"lin") != 0)) {
    error_exit(USAGE_ERROR," freq lims can't equal 0 if log spacing is used");
  }
  if(!is_int(numfstr))
    error_exit(USAGE_ERROR,"nfreqs must be an integer, found '%s'", numfstr);
  nfreqs = atoi(numfstr);

  /* check the frequency input values and range */

  if(freq_lims[1] < freq_lims[0]){
    temp_val = freq_lims[1];
    freq_lims[1] = freq_lims[0];
    freq_lims[0] = temp_val;
  }
  else if(nfreqs <= 0){
    error_exit(USAGE_ERROR,"nfreqs value entered must be positive, entered %d",
               nfreqs);
  }

  /* depending on the type of spacing, set either a step size or a multiplication
     factor for the loop over frequencies */

  if(type != NULL && !strcmp(type,"lin")){
    lin_typ = 1;
    if (nfreqs == 1)
      incr = -1.0;
    else
      incr = (freq_lims[1]-freq_lims[0])/(nfreqs-1);
  }
  else
    if (nfreqs == 1)
      incr = -1.0;
    else
      incr = pow((freq_lims[1]/freq_lims[0]), (1./(nfreqs-1)));

/* if net_code not given, default to "*" value */

  if(net_code == (char *)NULL)
    net_code = "*";

/* if locid not given, default to "*" value */

  if(locid == (char *)NULL)
    locid = "*";

/* convert the units and response type arguments to lower case after
   checking to make sure they match one of the appropriate input values */

  if(strlen(rtype)) {
    for(i = 0; i < (int)strlen(rtype); i++)
      *(rtype+i) = toupper(*(rtype+i));
    if(strcmp(rtype,"CS") == 0)        /* if complex-spectra output then */
      listinterp_out_flag = 0;         /* force List-out-interp flag clear */
    else if((strcmp(rtype,"AP") != 0)) {  /* if invalid value then abort */
      if ((strcmp(rtype,"FAP") != 0))
	error_exit(USAGE_ERROR,"evalresp; rtype entered ('%s') not a recognized string (see usage)", rtype);
    }
  }
  else
    strncpy(rtype,"AP",MAXFLDLEN);

  if(strlen(units)) {
    for(i = 0; i < (int)strlen(units); i++)
      *(units+i) = toupper(*(units+i));
    if(strcmp(units,"DIS") && strcmp(units,"VEL") && strcmp(units,"ACC") && strcmp(units,"DEF"))
      error_exit(USAGE_ERROR,"evalresp; units entered ('%s') not a recognized string (see usage)",
                 units);
    else if(!strcmp(units,"DEF"))
      def_units_flag = 1;
    else
      def_units_flag = 0;
  }
  else
    strncpy(units,"VEL",MAXFLDLEN);

/* allocate space for the frequencies and fill with appropriate values */

  freqs = alloc_double(nfreqs);
  for(i = 0, val = freq_lims[0]; i < nfreqs; i++) {
    freqs[i] = val;
    if(lin_typ)
      val += incr;
    else
      val *= incr;
  }


/* then get the response for each of the requested station-channel pairs at
    each of the requested frequencies */

  first = evresp_itp(sta_list,cha_list,net_code,locid,datime,units,file,
                     freqs,nfreqs,rtype,verbose,start_stage,stop_stage,
                     stdio_flag,listinterp_out_flag,listinterp_in_flag,
                     listinterp_tension, useTotalSensitivityFlag);
  if (!first)
  {
	fprintf(stderr, "EVRESP FAILED\n");
	exit(-1);
  }

/* and print the responses to a set of files */

  print_resp_itp(freqs,nfreqs,first,rtype,stdio_flag,
                                    listinterp_out_flag,listinterp_tension, unwrap_flag);

  free_response(first);
  free(freqs);                    /* added 3/28/2006 -- [ET] */

  exit(0);
  return 0;             /* 'return' statement to avoid compiler warning */
}

