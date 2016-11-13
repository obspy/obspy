/***************************************************************************
 * msview.c
 *
 * A simple example of using libmseed.
 *
 * Opens a user specified file, parses the Mini-SEED records and prints
 * details for each record.
 *
 * Written by Chad Trabant, ORFEUS/EC-Project MEREDIAN
 *
 * modified 2016.233
 ***************************************************************************/

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef WIN32
#include <signal.h>
static void term_handler (int sig);
#endif

#include <libmseed.h>

#define VERSION "[libmseed " LIBMSEED_VERSION " example]"
#define PACKAGE "msview"

static flag verbose = 0;
static flag ppackets = 0;
static flag basicsum = 0;
static int printdata = 0;
static int reclen = -1;
static char *inputfile = 0;

static int parameter_proc (int argcount, char **argvec);
static void usage (void);
static void term_handler (int sig);

int
main (int argc, char **argv)
{
  MSRecord *msr = 0;

  int64_t totalrecs = 0;
  int64_t totalsamps = 0;
  int retcode;

#ifndef WIN32
  /* Signal handling, use POSIX calls with standardized semantics */
  struct sigaction sa;

  sa.sa_flags = SA_RESTART;
  sigemptyset (&sa.sa_mask);

  sa.sa_handler = term_handler;
  sigaction (SIGINT, &sa, NULL);
  sigaction (SIGQUIT, &sa, NULL);
  sigaction (SIGTERM, &sa, NULL);

  sa.sa_handler = SIG_IGN;
  sigaction (SIGHUP, &sa, NULL);
  sigaction (SIGPIPE, &sa, NULL);
#endif

  /* Process given parameters (command line and parameter file) */
  if (parameter_proc (argc, argv) < 0)
    return -1;

  /* Loop over the input file */
  while ((retcode = ms_readmsr (&msr, inputfile, reclen, NULL, NULL, 1,
                                printdata, verbose)) == MS_NOERROR)
  {
    totalrecs++;
    totalsamps += msr->samplecnt;

    msr_print (msr, ppackets);

    if (printdata && msr->numsamples > 0)
    {
      int line, col, cnt, samplesize;
      int lines = (msr->numsamples / 6) + 1;
      void *sptr;

      if ((samplesize = ms_samplesize (msr->sampletype)) == 0)
      {
        ms_log (2, "Unrecognized sample type: '%c'\n", msr->sampletype);
      }

      for (cnt = 0, line = 0; line < lines; line++)
      {
        for (col = 0; col < 6; col++)
        {
          if (cnt < msr->numsamples)
          {
            sptr = (char *)msr->datasamples + (cnt * samplesize);

            if (msr->sampletype == 'i')
              ms_log (0, "%10d  ", *(int32_t *)sptr);

            else if (msr->sampletype == 'f')
              ms_log (0, "%10.8g  ", *(float *)sptr);

            else if (msr->sampletype == 'd')
              ms_log (0, "%10.10g  ", *(double *)sptr);

            cnt++;
          }
        }
        ms_log (0, "\n");

        /* If only printing the first 6 samples break out here */
        if (printdata == 1)
          break;
      }
    }
  }

  if (retcode != MS_ENDOFFILE)
    ms_log (2, "Cannot read %s: %s\n", inputfile, ms_errorstr (retcode));

  /* Make sure everything is cleaned up */
  ms_readmsr (&msr, NULL, 0, NULL, NULL, 0, 0, 0);

  if (basicsum)
    ms_log (1, "Records: %" PRId64 ", Samples: %" PRId64 "\n",
            totalrecs, totalsamps);

  return 0;
} /* End of main() */

/***************************************************************************
 * parameter_proc():
 * Process the command line parameters.
 *
 * Returns 0 on success, and -1 on failure
 ***************************************************************************/
static int
parameter_proc (int argcount, char **argvec)
{
  int optind;

  /* Process all command line arguments */
  for (optind = 1; optind < argcount; optind++)
  {
    if (strcmp (argvec[optind], "-V") == 0)
    {
      ms_log (1, "%s version: %s\n", PACKAGE, VERSION);
      exit (0);
    }
    else if (strcmp (argvec[optind], "-h") == 0)
    {
      usage ();
      exit (0);
    }
    else if (strncmp (argvec[optind], "-v", 2) == 0)
    {
      verbose += strspn (&argvec[optind][1], "v");
    }
    else if (strncmp (argvec[optind], "-p", 2) == 0)
    {
      ppackets += strspn (&argvec[optind][1], "p");
    }
    else if (strncmp (argvec[optind], "-d", 2) == 0)
    {
      printdata = 1;
    }
    else if (strncmp (argvec[optind], "-D", 2) == 0)
    {
      printdata = 2;
    }
    else if (strcmp (argvec[optind], "-s") == 0)
    {
      basicsum = 1;
    }
    else if (strcmp (argvec[optind], "-r") == 0)
    {
      reclen = atoi (argvec[++optind]);
    }
    else if (strncmp (argvec[optind], "-", 1) == 0 &&
             strlen (argvec[optind]) > 1)
    {
      ms_log (2, "Unknown option: %s\n", argvec[optind]);
      exit (1);
    }
    else if (inputfile == 0)
    {
      inputfile = argvec[optind];
    }
    else
    {
      ms_log (2, "Unknown option: %s\n", argvec[optind]);
      exit (1);
    }
  }

  /* Make sure an inputfile was specified */
  if (!inputfile)
  {
    ms_log (2, "No input file was specified\n\n");
    ms_log (1, "%s version %s\n\n", PACKAGE, VERSION);
    ms_log (1, "Try %s -h for usage\n", PACKAGE);
    exit (1);
  }

  /* Report the program version */
  if (verbose)
    ms_log (1, "%s version: %s\n", PACKAGE, VERSION);

  return 0;
} /* End of parameter_proc() */

/***************************************************************************
 * usage():
 * Print the usage message and exit.
 ***************************************************************************/
static void
usage (void)
{
  fprintf (stderr, "%s version: %s\n\n", PACKAGE, VERSION);
  fprintf (stderr, "Usage: %s [options] file\n\n", PACKAGE);
  fprintf (stderr,
           " ## Options ##\n"
           " -V             Report program version\n"
           " -h             Show this usage message\n"
           " -v             Be more verbose, multiple flags can be used\n"
           " -p             Print details of header, multiple flags can be used\n"
           " -d             Print first 6 sample values\n"
           " -D             Print all sample values\n"
           " -s             Print a basic summary after processing a file\n"
           " -r bytes       Specify record length in bytes, required if no Blockette 1000\n"
           "\n"
           " file           File of Mini-SEED records\n"
           "\n");
} /* End of usage() */

#ifndef WIN32
/***************************************************************************
 * term_handler:
 * Signal handler routine. 
 ***************************************************************************/
static void
term_handler (int sig)
{
  exit (0);
}
#endif
