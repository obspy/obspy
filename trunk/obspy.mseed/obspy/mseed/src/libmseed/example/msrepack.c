/***************************************************************************
 * msrepack.c
 *
 * A simple example of using the Mini-SEED record library to pack data.
 *
 * Opens a user specified file, parses the Mini-SEED records and
 * opionally re-packs the data records and saves them to a specified
 * output file.
 *
 * Written by Chad Trabant, ORFEUS/EC-Project MEREDIAN
 *
 * modified 2007.118
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

#ifndef WIN32
  #include <signal.h>
  static void term_handler (int sig);
#endif

#include <libmseed.h>

#define VERSION "[libmseed " LIBMSEED_VERSION " example]"
#define PACKAGE "msrepack"

static short int verbose   = 0;
static short int ppackets  = 0;
static short int tracepack = 1;
static int   reclen        = 0;
static int   packreclen    = -1;
static char *encodingstr   = 0;
static int   packencoding  = -1;
static int   byteorder     = -1;
static char *inputfile     = 0;
static FILE *outfile       = 0;

static int parameter_proc (int argcount, char **argvec);
static void record_handler (char *record, int reclen, void *ptr);
static void usage (void);
static void term_handler (int sig);

int
main (int argc, char **argv)
{
  MSRecord *msr = 0;
  MSTraceGroup *mstg = 0;
  MSTrace *mst;
  int retcode;

  int totalrecs  = 0;
  int totalsamps = 0;
  int packedsamples;
  int packedrecords;
  int lastrecord;
  int iseqnum = 1;
  
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
  
  /* Setup input encoding format if specified */
  if ( encodingstr )
    {
      int inputencoding = strtoul (encodingstr, NULL, 10);
      
      if ( inputencoding == 0 && errno == EINVAL )
	{
	  ms_log (2, "Error parsing input encoding format: %s\n", encodingstr);
	  return -1;
	}
      
      MS_UNPACKENCODINGFORMAT (inputencoding);
    }
  
  /* Init MSTraceGroup */
  mstg = mst_initgroup (mstg);
  
  /* Loop over the input file */
  while ( (retcode = ms_readmsr (&msr, inputfile, reclen, NULL, &lastrecord,
				 1, 1, verbose)) == MS_NOERROR )
    {
      totalrecs++;
      totalsamps += msr->samplecnt;
      
      msr_print (msr, ppackets);
      
      if ( packreclen >= 0 )
	msr->reclen = packreclen;
      else
	packreclen = msr->reclen;
      
      if ( packencoding >= 0 )
	msr->encoding = packencoding;
      else
	packencoding = msr->encoding;
      
      if ( byteorder >= 0 )
	msr->byteorder = byteorder;
      else
	byteorder = msr->byteorder;
      
      /* After unpacking the record, the start time in msr->starttime
	 is a potentially corrected start time, if correction has been
	 applied make sure the correction bit flag is set as it will
	 be used as a packing template. */
      if ( msr->fsdh->time_correct && ! (msr->fsdh->act_flags & 0x02) )
	{
	  ms_log (1, "Setting time correction applied flag for %s_%s_%s_%s\n",
		  msr->network, msr->station, msr->location, msr->channel);
	  msr->fsdh->act_flags |= 0x02;
	}
      
      /* If no samples in the record just pack the header */
      if ( outfile && msr->numsamples == 0 )
	{
	  msr_pack_header (msr, 1, verbose);
	  record_handler (msr->record, msr->reclen, NULL);
	}
      
      /* Pack each record individually */
      else if ( outfile && ! tracepack )
	{
	  msr->sequence_number = iseqnum;
	  
	  packedrecords = msr_pack (msr, &record_handler, NULL, &packedsamples, 1, verbose);
	  
	  if ( packedrecords == -1 )
	    ms_log (2, "Cannot pack records\n"); 
	  else
	    ms_log (1, "Packed %d records\n", packedrecords); 
	  
	  iseqnum = msr->sequence_number;
	}
      
      /* Pack records from a MSTraceGroup */
      else if ( outfile && tracepack )
	{
	  mst = mst_addmsrtogroup (mstg, msr, 0, -1.0, -1.0);
	  
	  if ( ! mst )
	    {
	      ms_log (2, "Error adding MSRecord to MStrace!\n");
	      break;
	    }
	  	  
	  /* Reset sequence number and free previous template */
	  if ( mst->prvtptr )
	    {
	      MSRecord *tmsr = (MSRecord *) mst->prvtptr;
	      
	      /* Retain sequence number from previous template */
	      msr->sequence_number = tmsr->sequence_number;
	      
	      msr_free (&tmsr);
	    }
	  else
	    {
	      msr->sequence_number = 1;
	    }
	  
	  /* Copy MSRecord and store as template */
	  mst->prvtptr = msr_duplicate (msr, 0);
	  
	  if ( ! mst->prvtptr )
	    {
	      ms_log (2, "Error duplicating MSRecord for template!\n");
	      break;
	    }
	  
	  /* Pack traces based on selected method */
	  packedrecords = 0;
	  if ( tracepack == 1 )
	    {
	      mst = mstg->traces;
	      while ( mst )
		{
		  packedrecords += mst_pack (mst, &record_handler, NULL, packreclen,
					     packencoding, byteorder, &packedsamples,
					     lastrecord, verbose, (MSRecord *)mst->prvtptr);
		  mst = mst->next;
		}
	      
	      ms_log (1, "Packed %d records\n", packedrecords);
	    }
	  if ( tracepack == 2 && lastrecord )
	    {
	      mst = mstg->traces;
	      while ( mst )
		{
		  packedrecords += mst_pack (mst, &record_handler, NULL, packreclen,
					     packencoding, byteorder, &packedsamples,
					     lastrecord, verbose, (MSRecord *)mst->prvtptr);
		  mst = mst->next;
		}
	      
	      ms_log (1, "Packed %d records\n", packedrecords);
	    }
	}
    }
  
  if ( retcode != MS_ENDOFFILE )
    ms_log (2, "Error reading %s: %s\n", inputfile, ms_errorstr(retcode));
  
  /* Make sure everything is cleaned up */
  ms_readmsr (&msr, NULL, 0, NULL, NULL, 0, 0, 0);
  mst_freegroup (&mstg);
  
  if ( outfile )
    fclose (outfile);
  
  return 0;
}  /* End of main() */


/***************************************************************************
 * parameter_proc:
 *
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
	  usage();
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
      else if (strcmp (argvec[optind], "-a") == 0)
	{
	  reclen = -1;
	}
      else if (strcmp (argvec[optind], "-i") == 0)
	{
	  tracepack = 0;
	}
      else if (strcmp (argvec[optind], "-t") == 0)
	{
	  tracepack = 2;
	}
      else if (strcmp (argvec[optind], "-r") == 0)
	{
	  reclen = strtol (argvec[++optind], NULL, 10);
	}
      else if (strcmp (argvec[optind], "-e") == 0)
	{
	  encodingstr = argvec[++optind];
	}
      else if (strcmp (argvec[optind], "-R") == 0)
	{
	  packreclen = strtol (argvec[++optind], NULL, 10);
	}
      else if (strcmp (argvec[optind], "-E") == 0)
	{
	  packencoding = strtol (argvec[++optind], NULL, 10);
	}
      else if (strcmp (argvec[optind], "-b") == 0)
	{
	  byteorder = strtol (argvec[++optind], NULL, 10);
	}
      else if (strcmp (argvec[optind], "-o") == 0)
	{
	  if ( (outfile = fopen(argvec[++optind], "wb")) == NULL )
	    {
	      ms_log (2, "Error opening output file: %s\n",
		      argvec[++optind]);
	      exit (0);
	    }	       
	}
      else if (strncmp (argvec[optind], "-", 1) == 0 &&
	       strlen (argvec[optind]) > 1 )
	{
	  ms_log (2, "Unknown option: %s\n", argvec[optind]);
	  exit (1);
	}
      else if ( ! inputfile )
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
  if ( ! inputfile )
    {
      ms_log (2, "No input file was specified\n\n");
      ms_log (1, "%s version %s\n\n", PACKAGE, VERSION);
      ms_log (1, "Try %s -h for usage\n", PACKAGE);
      exit (1);
    }

  /* Make sure an outputfile was specified */
  if ( ! outfile )
    {
      ms_log (2, "No output file was specified\n\n");
      ms_log (1, "Try %s -h for usage\n", PACKAGE);
      exit (1);
    }
  
  /* Report the program version */
  if ( verbose )
    ms_log (1, "%s version: %s\n", PACKAGE, VERSION);
  
  return 0;
}  /* End of parameter_proc() */


/***************************************************************************
 * record_handler:
 * Saves passed records to the output file.
 ***************************************************************************/
static void
record_handler (char *record, int reclen, void *ptr)
{
  if ( fwrite(record, reclen, 1, outfile) != 1 )
    {
      ms_log (2, "Cannot write to output file\n");
    }
}  /* End of record_handler() */


/***************************************************************************
 * usage:
 * Print the usage message and exit.
 ***************************************************************************/
static void
usage (void)
{
  fprintf (stderr, "%s version: %s\n\n", PACKAGE, VERSION);
  fprintf (stderr, "Usage: %s [options] -o outfile infile\n\n", PACKAGE);
  fprintf (stderr,
	   " ## Options ##\n"
	   " -V             Report program version\n"
	   " -h             Show this usage message\n"
	   " -v             Be more verbose, multiple flags can be used\n"
	   " -p             Print details of input headers, multiple flags can be used\n"
	   " -a             Autodetect every input record length, needed with mixed lengths\n"
	   " -r bytes       Specify record length in bytes, required if no Blockette 1000\n"
	   " -e encoding    Specify encoding format for input data samples\n"
	   " -i             Pack data individually for each input record\n"
	   " -t             Pack data from traces after reading all data\n"
	   " -R bytes       Specify record length in bytes for packing\n"
	   " -E encoding    Specify encoding format for packing\n"
	   " -b byteorder   Specify byte order for packing, MSBF: 1, LSBF: 0\n"
	   "\n"
	   " -o outfile     Specify the output file, required\n"
	   "\n"
	   " infile         Input Mini-SEED file\n"
	   "\n"
	   "The default packing method is to use parameters from the input records\n"
	   "(reclen, encoding, byteorder, etc.) and pack records as soon as enough\n"
	   "samples are available.  This method is a good balance between preservation\n"
	   "of blockettes, header values from input records and pack efficiency\n"
	   "compared to the other methods of packing, namely options -i and -t.\n"
	   "In most Mini-SEED repacking schemes some level of header information loss\n"
	   "or time shifting should be expected, especially in the case where the record\n"
	   "length is changed.\n"
	   "\n"
	   "Unless each input record is being packed individually, option -i, it is\n"
	   "not recommended to pack files containing records for different data streams.\n");
}  /* End of usage() */


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
