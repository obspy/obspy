/***************************************************************************
 *
 * Routines to manage files of Mini-SEED.
 *
 * Written by Chad Trabant
 *   ORFEUS/EC-Project MEREDIAN
 *   IRIS Data Management Center
 *
 * modified: 2010.007
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

#include "libmseed.h"

/* Byte stream length for read-ahead header fingerprinting */
#define NEXTHDRLEN 48

static int ms_fread (char *buf, int size, int num, FILE *stream);
static int ms_ateof (FILE *stream);

/* Pack type parameters for the 8 defined types:
 * [type] : [hdrlen] [sizelen] [chksumlen]
 */
int8_t packtypes[9][3] = {
  { 0, 0, 0 },
  { 8, 8, 8 },
  { 11, 8, 8 },
  { 11, 8, 8 },
  { 11, 8, 8 },
  { 11, 8, 8 },
  { 13, 8, 8 },
  { 15, 8, 8 },
  { 22, 15, 10 }};

/*********************************************************************
 * Notes about packed files as read by ms_readmsr_main()
 *
 * In general a packed file includes a pack file identifier at the
 * very beginning, followed by pack header for a data block, followed
 * by the data block, followed by a chksum for the data block.  The
 * pack header, data block and chksum are then repeated for each data
 * block in the file:
 *
 *   ID    HDR     DATA    CHKSUM    HDR     DATA    CHKSUM
 * |----|-------|--....--|--------|-------|--....--|--------| ...
 *
 *      |________ repeats ________|
 *
 * The HDR section contains fixed width ASCII fields identifying the
 * data in the next section and it's length in bytes.  With this
 * information the offset of the next CHKSUM and HDR are completely
 * predictable.
 *
 * packtypes[type][0]: length of pack header length
 * packtypes[type][1]: length of size field in pack header
 * packtypes[type][2]: chksum length following data blocks, skipped
 *
 * Notes from seed_pack.h documenting the PQI and PLS pack types:
 *
 * ___________________________________________________________________
 * There were earlier pack file types numbered 1 through 6.  These have been discontinued.
 * Current file formats can be described as follows:
 *
 * Quality-Indexed Pack - Type 7:
 * _____10_____2__2___3_____8_______mod 256_______8_____2__2___3_____8_______mod 256_______8____ ...
 * |PQI-      |q |lc|chn|  size  | ...data... | chksum |q |lc|chn|  size  | ...data... | chksum  ...
 * parsing guide:
 *      10    |     15 hdr       |     xx     |   8    |    15 hdr        |    xx   
 *            |+0|+2|+4 |+7      |
 * 
 * 
 * Large-Size Pack - Type 8: (for large channel blocks)
 * _____10_____2__2___3_____15_______mod 256_______8____2__2__2___3_____15_______mod 256_______8____ ...
 * |PLS-------|q |lc|chn|  size  | ...data... | chksum |--|q |lc|chn|  size  | ...data... | chksum  ...
 * uniform parsing guide:
 * |    10    |       22         |    xx      |    10     |      22          |       xx   |
 *            |+0|+2|+4 |+7      |
 * (note the use of hyphens after the PLS marker and just after the checksum.  this will serve as a visual
 * aid when scanning between channel blocks and provide consistent 32 byte spacing between data blocks)
 * ___________________________________________________________________
 *
 *********************************************************************/

/* Initialize the global file reading parameters */
MSFileParam gMSFileParam = {NULL, NULL, "", 1, MINRECLEN, 0, 0, 0, 0};


/**********************************************************************
 * ms_readmsr:
 *
 * This routine is a simple wrapper for ms_readmsr_main() that uses
 * the global file reading parameters.  This routine is not thread
 * safe and cannot be used to read more than one file at a time.
 *
 * See the comments with ms_readmsr_main() for return values and
 * further description of arguments.
 *********************************************************************/
int
ms_readmsr (MSRecord **ppmsr, char *msfile, int reclen, off_t *fpos,
	    int *last, flag skipnotdata, flag dataflag, flag verbose)
{
  MSFileParam *msfp = &gMSFileParam;
  
  return ms_readmsr_main (&msfp, ppmsr, msfile, reclen, fpos,
			  last, skipnotdata, dataflag, NULL, verbose);
}  /* End of ms_readmsr() */


/**********************************************************************
 * ms_readmsr_r:
 *
 * This routine is a simple wrapper for ms_readmsr_main() that uses
 * the re-entrant capabilities.  This routine is not thread safe and
 * cannot be used to read more than one file at a time.
 *
 * See the comments with ms_readmsr_main() for return values and
 * further description of arguments.
 *********************************************************************/
int
ms_readmsr_r (MSFileParam **ppmsfp, MSRecord **ppmsr, char *msfile,
	      int reclen, off_t *fpos, int *last, flag skipnotdata,
	      flag dataflag, flag verbose)
{
  return ms_readmsr_main (ppmsfp, ppmsr, msfile, reclen, fpos,
			  last, skipnotdata, dataflag, NULL, verbose);
}  /* End of ms_readmsr_r() */


/**********************************************************************
 * ms_readmsr_main:
 *
 * This routine will open and read, with subsequent calls, all
 * Mini-SEED records in specified file.
 *
 * All static file reading parameters are stored in a MSFileParam
 * struct and returned (via a pointer to a pointer) for the calling
 * routine to use in subsequent calls.  A MSFileParam struct will be
 * allocated if necessary.  This routine is thread safe and can be
 * used to read multiple files in parallel as long as the file reading
 * parameters are managed appropriately.
 *
 * If reclen is 0 the length of the first record is automatically
 * detected, all subsequent records are then expected to have the same
 * length as the first.
 *
 * If reclen is negative the length of every record is automatically
 * detected.
 *
 * For auto detection of record length the record must include a 1000
 * blockette.  This routine will search up to 8192 bytes into the
 * record for the 1000 blockette.
 *
 * If *fpos is not NULL it will be updated to reflect the file
 * position (offset from the beginning in bytes) from where the
 * returned record was read.  As a special case, if *fpos is not NULL
 * and the value it points to is less than 0 this will be interpreted
 * as a (positive) starting offset from which to begin reading data;
 * this feature does not work with packed files.
 *
 * If *last is not NULL it will be set to 1 when the last record in
 * the file is being returned, otherwise it will be 0.
 *
 * If the skipnotdata flag is true any data chunks read that do not
 * have vald data record indicators (D, R, Q, M, etc.) will be skipped.
 *
 * dataflag will be passed directly to msr_unpack().
 *
 * If a Selections list is supplied it will be used to determine when
 * a section of data in a packed file may be skipped, packed files are
 * internal to the IRIS DMC.
 *
 * After reading all the records in a file the controlling program
 * should call it one last time with msfile set to NULL.  This will
 * close the file and free allocated memory.
 *
 * Returns MS_NOERROR and populates an MSRecord struct at *ppmsr on
 * successful read, returns MS_ENDOFFILE on EOF, otherwise returns a
 * libmseed error code (listed in libmseed.h) and *ppmsr is set to
 * NULL.
 *********************************************************************/
int
ms_readmsr_main (MSFileParam **ppmsfp, MSRecord **ppmsr, char *msfile,
		 int reclen, off_t *fpos, int *last, flag skipnotdata,
		 flag dataflag, Selections *selections, flag verbose)
{
  MSFileParam *msfp;
  off_t packdatasize;
  int packskipsize;
  int autodetexp = 8;
  int prevreadlen = 0;
  int detsize;
  int retcode = MS_NOERROR;
  
  if ( ! ppmsr )
    return MS_GENERROR;
  
  if ( ! ppmsfp )
    return MS_GENERROR;
  
  msfp = *ppmsfp;
  
  /* Initialize the file read parameters if needed */
  if ( ! msfp )
    {
      msfp = (MSFileParam *) malloc (sizeof (MSFileParam));
      
      if ( msfp == NULL )
	{
	  ms_log (2, "ms_readmsr_main(): Cannot allocate memory\n");
	  return MS_GENERROR;
	}
      
      /* Redirect the supplied pointer to the allocted params */
      *ppmsfp = msfp;
      
      msfp->fp = NULL;
      msfp->rawrec = NULL;
      msfp->filename[0] = '\0';
      msfp->autodet = 1;
      msfp->readlen = MINRECLEN;
      msfp->packtype = 0;
      msfp->packhdroffset = 0;
      msfp->filepos = 0;
      msfp->recordcount = 0;
    }
  
  /* When cleanup is requested */
  if ( msfile == NULL )
    {
      msr_free (ppmsr);
      
      if ( msfp->fp != NULL )
	fclose (msfp->fp);
      
      if ( msfp->rawrec != NULL )
	free (msfp->rawrec);
      
      /* If the file parameters are the global parameters reset them */
      if ( *ppmsfp == &gMSFileParam )
	{
	  gMSFileParam.fp = NULL;
	  gMSFileParam.rawrec = NULL;
	  gMSFileParam.filename[0] = '\0';
	  gMSFileParam.autodet = 1;
	  gMSFileParam.readlen = MINRECLEN;
	  gMSFileParam.packtype = 0;
	  gMSFileParam.packhdroffset = 0;
	  gMSFileParam.filepos = 0;
	  gMSFileParam.recordcount = 0;
	}
      /* Otherwise free the MSFileParam */
      else
	{
	  free (*ppmsfp);
	  *ppmsfp = NULL;
	}
      
      return MS_NOERROR;
    }
  
  /* Sanity check: track if we are reading the same file */
  if ( msfp->fp && strncmp (msfile, msfp->filename, sizeof(msfp->filename)) )
    {
      ms_log (2, "ms_readmsr_main() called with a different file name without being reset\n");
      
      /* Close previous file and reset needed variables */
      if ( msfp->fp != NULL )
	fclose (msfp->fp);
      
      msfp->fp = NULL;
      msfp->autodet = 1;
      msfp->readlen = MINRECLEN;
      msfp->packtype = 0;
      msfp->packhdroffset = 0;
      msfp->filepos = 0;
      msfp->recordcount = 0;
    }
  
  /* Open the file if needed, redirect to stdin if file is "-" */
  if ( msfp->fp == NULL )
    {
      /* Store the filename for tracking */
      strncpy (msfp->filename, msfile, sizeof(msfp->filename) - 1);
      msfp->filename[sizeof(msfp->filename) - 1] = '\0';
      
      if ( strcmp (msfile, "-") == 0 )
	{
	  msfp->fp = stdin;
	}
      else if ( (msfp->fp = fopen (msfile, "rb")) == NULL )
	{
	  ms_log (2, "Cannot open file: %s (%s)\n", msfile, strerror (errno));
	  
	  msr_free (ppmsr);
	  
	  return MS_GENERROR;
	}
    }
  
  /* Seek to a specified offset if requested */
  if ( fpos != NULL && *fpos < 0 )
    {
      /* Only try to seek in real files, not stdin */
      if ( msfp->fp != stdin )
	{
	  if ( lmp_fseeko (msfp->fp, *fpos * -1, SEEK_SET) )
	    {
	      ms_log (2, "Cannot seek in file: %s (%s)\n", msfile, strerror (errno));
	      
	      if ( msfp->fp )
		{ fclose (msfp->fp); msfp->fp = NULL; }
	      if ( msfp->rawrec )
		{ free (msfp->rawrec); msfp->rawrec = NULL; }
	      
	      return MS_GENERROR;
	    }
	  
	  msfp->filepos = *fpos * -1;
	}
    }
  
  /* Force the record length if specified */
  if ( reclen > 0 && msfp->autodet )
    {
      msfp->readlen = reclen;
      msfp->autodet = 0;
      
      if ( msfp->rawrec )
	free (msfp->rawrec);
      
      msfp->rawrec = (char *) malloc (msfp->readlen);
      
      if ( msfp->rawrec == NULL )
	{
	  ms_log (2, "ms_readmsr_main(): Cannot allocate memory\n");
	  
	  if ( msfp->fp )
	    { fclose (msfp->fp); msfp->fp = NULL; }
	  
	  return MS_GENERROR;
	}
    }
  
  /* If reclen is negative reset readlen for autodetection */
  if ( reclen < 0 )
    msfp->readlen = MINRECLEN;
  
  /* Zero the last record indicator */
  if ( last )
    *last = 0;
  
  /* Autodetect the record length */
  if ( msfp->autodet || reclen < 0 )
    {
      detsize = 0;
      prevreadlen = 0;
      
      while ( detsize <= 0 && msfp->readlen <= 8192 )
	{
	  packdatasize = 0;
	  packskipsize = 0;
	  
	  msfp->rawrec = (char *) realloc (msfp->rawrec, msfp->readlen);
	  
	  if ( msfp->rawrec == NULL )
	    {
	      ms_log (2, "ms_readmsr_main(): Cannot reallocate memory\n");
	      return MS_GENERROR;
	    }
	  
	  /* Read data into record buffer */
	  if ( (ms_fread (msfp->rawrec + prevreadlen, 1, (msfp->readlen - prevreadlen), msfp->fp)) < (msfp->readlen - prevreadlen) )
	    {
	      if ( ! feof (msfp->fp) )
		{
		  ms_log (2, "Short read at %d bytes during length detection\n", msfp->readlen);
		  retcode = MS_GENERROR;
		}
	      else
		{
		  retcode = MS_ENDOFFILE;
		}
	      
	      if ( msfp->recordcount == 0 && ! msfp->packtype )
		{
		  if ( verbose > 0 )
		    ms_log (2, "%s: No data records read, not SEED?\n", msfile);
		  retcode = MS_NOTSEED;
		}
	      
	      if ( msfp->fp )
		{ fclose (msfp->fp); msfp->fp = NULL; }
	      msr_free (ppmsr);
	      if ( msfp->rawrec )
		{ free (msfp->rawrec); msfp->rawrec = NULL; }
	      
	      return retcode;
	    }
	  
	  msfp->filepos = lmp_ftello (msfp->fp);
	  
	  /* Determine record length:
	   * If packed file and we are at the next info, length is implied.
	   * Otherwise use ms_find_reclen() */
	  if ( msfp->packtype && msfp->packhdroffset == msfp->filepos )
	    {
	      detsize = msfp->readlen;
	      break;
	    }
	  else if ( (detsize = ms_find_reclen (msfp->rawrec, msfp->readlen, msfp->fp)) > 0 )
	    {
	      break;
	    }
	  
	  /* Test for packed file signature at the beginning of the file */
	  if ( *(msfp->rawrec) == 'P' && msfp->filepos == MINRECLEN && detsize == -1 )
	    {
	      msfp->packtype = 0;
	      
	      /* Determine pack type, the negative pack type indicates initial header */
	      if ( ! memcmp ("PED", msfp->rawrec, 3) )
		msfp->packtype = -1;
	      else if ( ! memcmp ("PSD", msfp->rawrec, 3) )
		msfp->packtype = -2;
	      else if ( ! memcmp ("PLC", msfp->rawrec, 3) )
		msfp->packtype = -6;
	      else if ( ! memcmp ("PQI", msfp->rawrec, 3) )
		msfp->packtype = -7;
	      else if ( ! memcmp ("PLS", msfp->rawrec, 3) )
		msfp->packtype = -8;
	      
	      if ( verbose > 0 )
		ms_log (1, "Detected packed file (%3.3s: type %d)\n", msfp->rawrec, -msfp->packtype);
	    }
	  
	  /* Read pack header */
	  if ( msfp->packtype && (msfp->packtype < 0 || (msfp->filepos - msfp->readlen) == msfp->packhdroffset) )
	    {
	      char hdrstr[30];
	      long long datasize;
	      
	      /* Determine bytes to skip before header: either initial ID block or type-specific chksum block */
	      packskipsize = ( msfp->packtype < 0 ) ? 10 : packtypes[msfp->packtype][2];
	      
	      if ( msfp->packtype < 0 )
		msfp->packtype = -msfp->packtype;
	      
	      /* Read pack length from pack header accounting for bytes that should be skipped */
	      memset (hdrstr, 0, sizeof(hdrstr));
	      memcpy (hdrstr, msfp->rawrec + (packtypes[msfp->packtype][0] + packskipsize - packtypes[msfp->packtype][1]),
		      packtypes[msfp->packtype][1]);
	      sscanf (hdrstr, " %lld", &datasize);
	      packdatasize = (off_t) datasize;
	      
	      /* Next pack header = File position - read length + skipsize + header size + data size
	       * This offset is actually to the data block chksum which is skipped by the logic above,
	       * the next pack header should directly follow the chksum. */
	      msfp->packhdroffset = msfp->filepos - msfp->readlen + packskipsize + packtypes[msfp->packtype][0] + packdatasize;
	      
	      if ( verbose > 1 )
		ms_log (1, "Read packed file header at offset %lld (%d bytes follow)\n",
			(long long int) (msfp->filepos - msfp->readlen + packskipsize),
			packdatasize);
	    }
	  
	  /* Skip if data record or packed file not detected */
	  if ( detsize == -1 && skipnotdata && ! msfp->packtype )
	    {
	      if ( verbose > 1 )
		{
		  if ( MS_ISVALIDBLANK((char *)msfp->rawrec) )
		    ms_log (1, "Skipped %d bytes of blank/noise record at byte offset %lld\n",
			    msfp->readlen, (long long) msfp->filepos - msfp->readlen);
		  else
		    ms_log (1, "Skipped %d bytes of non-data record at byte offset %lld\n",
			    msfp->readlen, (long long) msfp->filepos - msfp->readlen);
		}
	    }
	  /* Otherwise prepare to read more data */
	  else
	    {
	      /* Compensate for pack header */
	      if ( msfp->packtype && packskipsize )
		{
		  /* Shift first data record to beginning of buffer */
		  memmove (msfp->rawrec, msfp->rawrec + (packtypes[msfp->packtype][0] + packskipsize),
			   msfp->readlen - (packtypes[msfp->packtype][0] + packskipsize));
		  
		  prevreadlen = msfp->readlen - (packtypes[msfp->packtype][0] + packskipsize);
		}
	      /* Increase read length to the next record size up */
	      else
		{
		  prevreadlen = msfp->readlen;
		  autodetexp++;
		  msfp->readlen = (unsigned int) 1 << autodetexp;
		}
	    }
	  
	  /* Check for match if selections are supplied and pack header was read */
	  if ( selections && msfp->packtype && packdatasize )
	    {
	      char srcname[100];
	      
	      /* Minium read length (MINRECLEN) should always have the fixed section */
	      ms_recsrcname (msfp->rawrec, srcname, 1);
	      
	      if ( ! ms_matchselect (selections, srcname, HPTERROR, HPTERROR, NULL) )
		{
		  if ( verbose > 1 )
		    {
		      ms_log (1, "Skipping packed section for %s at byte offset %lld\n",
			      srcname, (long long) msfp->filepos - msfp->readlen);
		    }
		  
		  /* Skip to next pack header */
		  if ( lmp_fseeko (msfp->fp, msfp->packhdroffset, SEEK_SET) )
		    {
		      ms_log (2, "Cannot seek in file: %s (%s)\n", msfile, strerror (errno));
		      
		      if ( msfp->fp )
			{ fclose (msfp->fp); msfp->fp = NULL; }
		      msr_free (ppmsr);
		      if ( msfp->rawrec )
			{ free (msfp->rawrec); msfp->rawrec = NULL; }
		      
		      return MS_GENERROR;
		    }
		  
		  msfp->filepos = msfp->packhdroffset;
		  msfp->readlen = MINRECLEN;
		  detsize = 0;
		  prevreadlen = 0;
		}
	    }
	}  /* End of record length detection */
      
      if ( detsize <= 0 )
	{
	  ms_log (2, "Cannot detect record length at byte offset %lld: %s\n",
		  (long long) msfp->filepos - prevreadlen, msfile);
	  
	  /* Print common errors and raw details if verbose */
	  ms_parse_raw (msfp->rawrec, prevreadlen, verbose, -1);
	  
	  if ( msfp->fp )
	    { fclose (msfp->fp); msfp->fp = NULL; }
	  msr_free (ppmsr);
	  if ( msfp->rawrec )
	    { free (msfp->rawrec); msfp->rawrec = NULL; }
	  
	  return MS_NOTSEED;
	}
      
      msfp->autodet = 0;
      
      if ( verbose > 1 )
	ms_log (1, "Detected record length of %d bytes\n", detsize);
      
      if ( detsize < MINRECLEN || detsize > MAXRECLEN )
	{
	  ms_log (2, "Detected record length is out of range: %d\n", detsize);
	  
	  if ( msfp->fp )
	    { fclose (msfp->fp); msfp->fp = NULL; }
	  msr_free (ppmsr);
	  if ( msfp->rawrec )
	    { free (msfp->rawrec); msfp->rawrec = NULL; }
	  
	  return MS_OUTOFRANGE;
	}
      
      msfp->rawrec = (char *) realloc (msfp->rawrec, detsize);

      if ( msfp->rawrec == NULL )
	{
	  ms_log (2, "ms_readmsr_main(): Cannot allocate memory\n");
	  return MS_GENERROR;
	}
      
      /* Read the rest of the first record */
      if ( (detsize - msfp->readlen) > 0 )
	{
	  if ( (ms_fread (msfp->rawrec+msfp->readlen, 1, detsize-msfp->readlen, msfp->fp)) < (detsize-msfp->readlen) )
	    {
	      if ( ! feof (msfp->fp) )
		{
		  ms_log (2, "Short read at %d bytes during length detection\n", msfp->readlen);
		  retcode = MS_GENERROR;
		}
	      else
		{
		  retcode = MS_ENDOFFILE;
		}
	      
	      if ( msfp->recordcount == 0 && ! msfp->packtype )
		{
		  if ( verbose > 0 )
		    ms_log (2, "%s: No data records read, not SEED?\n", msfile);
		  retcode = MS_NOTSEED;
		}
	      
	      if ( msfp->fp )
		{ fclose (msfp->fp); msfp->fp = NULL; }
	      msr_free (ppmsr);
	      if ( msfp->rawrec )
		{ free (msfp->rawrec); msfp->rawrec = NULL; }
	      
	      return retcode;
	    }
	  
	  msfp->filepos = lmp_ftello (msfp->fp);
	}
      
      /* Set file position offset for beginning of record */
      if ( fpos != NULL )
	*fpos = msfp->filepos - detsize;
      
      /* Test if this is the last record */
      if ( last )
	if ( ms_ateof (msfp->fp) )
	  *last = 1;
      
      msfp->readlen = detsize;
      msr_free (ppmsr);
      
      if ( (retcode = msr_unpack (msfp->rawrec, msfp->readlen, ppmsr, dataflag, verbose)) != MS_NOERROR )
	{
	  if ( msfp->fp )
	    { fclose (msfp->fp); msfp->fp = NULL; }
	  msr_free (ppmsr);
	  if ( msfp->rawrec )
	    { free (msfp->rawrec); msfp->rawrec = NULL; }

	  return retcode;
	}
      
      /* Set record length if it was not already done */
      if ( (*ppmsr)->reclen == 0 )
	(*ppmsr)->reclen = msfp->readlen;
      
      msfp->recordcount++;
      return MS_NOERROR;
    }
  
  /* Read subsequent records or initial forced length record */
  for (;;)
    {
      /* Read data into record buffer */
      if ( (ms_fread (msfp->rawrec, 1, msfp->readlen, msfp->fp)) < msfp->readlen )
	{
	  if ( ! feof (msfp->fp) )
	    {
	      ms_log (2, "Short read at %d bytes during length detection\n", msfp->readlen);
	      retcode = MS_GENERROR;
	    }
	  else
	    {
	      retcode = MS_ENDOFFILE;
	    }
	  
	  if ( msfp->recordcount == 0 && ! msfp->packtype )
	    {
	      if ( verbose > 0 )
		ms_log (2, "%s: No data records read, not SEED?\n", msfile);
	      retcode = MS_NOTSEED;
	    }
	  
	  if ( msfp->fp )
	    { fclose (msfp->fp); msfp->fp = NULL; }
	  msr_free (ppmsr);
	  if ( msfp->rawrec )
	    { free (msfp->rawrec); msfp->rawrec = NULL; }
	  
	  return retcode;
	}
      
      msfp->filepos = lmp_ftello (msfp->fp);
      
      /* Set file position offset for beginning of record */
      if ( fpos != NULL )
	*fpos = msfp->filepos - msfp->readlen;
      
      if ( last )
	if ( ms_ateof (msfp->fp) )
	  *last = 1;
      
      if ( skipnotdata )
	{
	  if ( MS_ISVALIDHEADER(msfp->rawrec) )
	    {
	      break;
	    }
	  else if ( verbose > 1 )
	    {
	      if ( MS_ISVALIDBLANK((char *)msfp->rawrec) )
		ms_log (1, "Skipped %d bytes of blank/noise record at byte offset %lld\n",
			msfp->readlen, (long long) msfp->filepos - msfp->readlen);
	      else
		ms_log (1, "Skipped %d bytes of non-data record at byte offset %lld\n",
			msfp->readlen, (long long) msfp->filepos - msfp->readlen);
	    }
	}
      else if ( ! MS_ISVALIDHEADER(msfp->rawrec) )
	{
	  /* Print common errors and raw details if verbose */
	  ms_parse_raw (msfp->rawrec, msfp->readlen, verbose, -1);
	  
	  return MS_NOTSEED;
	}
      else
	{
	  break;
	}
    }
  
  if ( (retcode = msr_unpack (msfp->rawrec, msfp->readlen, ppmsr, dataflag, verbose)) != MS_NOERROR )
    {
      if ( msfp->fp )
	{ fclose (msfp->fp); msfp->fp = NULL; }
      msr_free (ppmsr);
      if ( msfp->rawrec )
	{ free (msfp->rawrec); msfp->rawrec = NULL; }
      
      return retcode;
    }
  
  /* Set record length if it was not already done */
  if ( (*ppmsr)->reclen == 0 )
    {
      (*ppmsr)->reclen = msfp->readlen;
    }
  /* Test that any detected record length is the same as the read length */
  else if ( (*ppmsr)->reclen != msfp->readlen )
    {
      ms_log (2, "Detected record length (%d) != read length (%d)\n",
	      (*ppmsr)->reclen, msfp->readlen);
      
      return MS_WRONGLENGTH;
    }
  
  msfp->recordcount++;
  return MS_NOERROR;
}  /* End of ms_readmsr_main() */


/*********************************************************************
 * ms_readtraces:
 *
 * This routine will open and read all Mini-SEED records in specified
 * file and populate a trace group.  This routine is thread safe.
 *
 * If reclen is 0 the length of the first record is automatically
 * detected, all subsequent records are then expected to have the same
 * length as the first.
 *
 * If reclen is negative the length of every record is automatically
 * detected.
 *
 * Returns MS_NOERROR and populates an MSTraceGroup struct at *ppmstg
 * on successful read, otherwise returns a libmseed error code (listed
 * in libmseed.h).
 *********************************************************************/
int
ms_readtraces (MSTraceGroup **ppmstg, char *msfile, int reclen,
	       double timetol, double sampratetol, flag dataquality,
	       flag skipnotdata, flag dataflag, flag verbose)
{
  MSRecord *msr = 0;
  MSFileParam *msfp = 0;
  int retcode;
  
  if ( ! ppmstg )
    return MS_GENERROR;
  
  /* Initialize MSTraceGroup if needed */
  if ( ! *ppmstg )
    {
      *ppmstg = mst_initgroup (*ppmstg);
      
      if ( ! *ppmstg )
	return MS_GENERROR;
    }
  
  /* Loop over the input file */
  while ( (retcode = ms_readmsr_main (&msfp, &msr, msfile, reclen, NULL, NULL,
				      skipnotdata, dataflag, NULL, verbose)) == MS_NOERROR)
    {
      mst_addmsrtogroup (*ppmstg, msr, dataquality, timetol, sampratetol);
    }
  
  /* Reset return code to MS_NOERROR on successful read by ms_readmsr() */
  if ( retcode == MS_ENDOFFILE )
    retcode = MS_NOERROR;
  
  ms_readmsr_main (&msfp, &msr, NULL, 0, NULL, NULL, 0, 0, NULL, 0);
  
  return retcode;
}  /* End of ms_readtraces() */


/*********************************************************************
 * ms_readtracelist:
 *
 * This routine will open and read all Mini-SEED records in specified
 * file and populate a trace list.  This routine is thread safe.
 *
 * If reclen is 0 the length of the first record is automatically
 * detected, all subsequent records are then expected to have the same
 * length as the first.
 *
 * If reclen is negative the length of every record is automatically
 * detected.
 *
 * Returns MS_NOERROR and populates an MSTraceList struct at *ppmstl
 * on successful read, otherwise returns a libmseed error code (listed
 * in libmseed.h).
 *********************************************************************/
int
ms_readtracelist (MSTraceList **ppmstl, char *msfile, int reclen,
		  double timetol, double sampratetol, flag dataquality,
		  flag skipnotdata, flag dataflag, flag verbose)
{
  MSRecord *msr = 0;
  MSFileParam *msfp = 0;
  int retcode;
  
  if ( ! ppmstl )
    return MS_GENERROR;
  
  /* Initialize MSTraceList if needed */
  if ( ! *ppmstl )
    {
      *ppmstl = mstl_init (*ppmstl);
      
      if ( ! *ppmstl )
	return MS_GENERROR;
    }
  
  /* Loop over the input file */
  while ( (retcode = ms_readmsr_main (&msfp, &msr, msfile, reclen, NULL, NULL,
				      skipnotdata, dataflag, NULL, verbose)) == MS_NOERROR)
    {
      mstl_addmsr (*ppmstl, msr, dataquality, 1, timetol, sampratetol);
    }
  
  /* Reset return code to MS_NOERROR on successful read by ms_readmsr() */
  if ( retcode == MS_ENDOFFILE )
    retcode = MS_NOERROR;
  
  ms_readmsr_main (&msfp, &msr, NULL, 0, NULL, NULL, 0, 0, NULL, 0);
  
  return retcode;
}  /* End of ms_readtracelist() */


/********************************************************************
 * ms_find_reclen:
 *
 * Determine SEED data record length with the following steps:
 *
 * 1) determine that the buffer contains a SEED data record by
 * verifying known signatures (fields with known limited values)
 *
 * 2) search the record up to recbuflen bytes for a 1000 blockette.
 *
 * 3) If no blockette 1000 is found and fileptr is not NULL, read the
 * next 48 bytes from the file and determine if it is the fixed second
 * of another record or blank/noise record, thereby implying the
 * record length is recbuflen.  The original read position of the file
 * is restored.
 *
 * Returns:
 * -1 : data record not detected or error
 *  0 : data record detected but could not determine length
 * >0 : size of the record in bytes
 *********************************************************************/
int
ms_find_reclen ( const char *recbuf, int recbuflen, FILE *fileptr )
{
  uint16_t blkt_offset;    /* Byte offset for next blockette */
  uint8_t swapflag  = 0;   /* Byte swapping flag */
  uint8_t foundlen = 0;    /* Found record length */
  int32_t reclen = -1;     /* Size of record in bytes */
  
  uint16_t blkt_type;
  uint16_t next_blkt;
  
  struct fsdh_s *fsdh;
  struct blkt_1000_s *blkt_1000;
  char nextfsdh[NEXTHDRLEN];
  
  /* Check for valid fixed section of header */
  if ( ! MS_ISVALIDHEADER(recbuf) )
    return -1;
  
  fsdh = (struct fsdh_s *) recbuf;
  
  /* Check to see if byte swapping is needed (bogus year makes good test) */
  if ( (fsdh->start_time.year < 1900) ||
       (fsdh->start_time.year > 2050) )
    swapflag = 1;
  
  blkt_offset = fsdh->blockette_offset;
  
  /* Swap order of blkt_offset if needed */
  if ( swapflag ) ms_gswap2 (&blkt_offset);
  
  /* Loop through blockettes as long as number is non-zero and viable */
  while ( blkt_offset != 0 &&
	  blkt_offset <= recbuflen )
    {
      memcpy (&blkt_type, recbuf + blkt_offset, 2);
      memcpy (&next_blkt, recbuf + blkt_offset + 2, 2);
      
      if ( swapflag )
	{
	  ms_gswap2 (&blkt_type);
	  ms_gswap2 (&next_blkt);
	}
      
      /* Found a 1000 blockette, not truncated */
      if ( blkt_type == 1000  &&
	   (blkt_offset + 4 + sizeof(struct blkt_1000_s)) <= recbuflen )
	{
          blkt_1000 = (struct blkt_1000_s *) (recbuf + blkt_offset + 4);
	  
          foundlen = 1;
	  
          /* Calculate record size in bytes as 2^(blkt_1000->reclen) */
	  reclen = (unsigned int) 1 << blkt_1000->reclen;
	  
	  break;
        }
      
      blkt_offset = next_blkt;
    }
  
  /* If record length was not determined by a 1000 blockette scan the file
   * and search for the next record */
  if ( reclen == -1 && fileptr )
    {
      /* Read data into record buffer */
      if ( (ms_fread (nextfsdh, 1, NEXTHDRLEN, fileptr)) < NEXTHDRLEN )
	{
	  /* If no the EOF an error occured (short read) */
	  if ( ! feof (fileptr) )
	    {
	      ms_log (2, "ms_find_reclen(): Error reading file\n");
	      return -1;
	    }
	  /* If EOF the record length is recbuflen */
	  else
	    {
	      foundlen = 1;
	      reclen = recbuflen;
	    }
	}
      else
	{
	  /* Rewind file read pointer */
	  if ( lmp_fseeko (fileptr, -NEXTHDRLEN, SEEK_CUR) )
	    {
	      ms_log (2, "ms_find_reclen(): %s\n", strerror(errno));
	      return -1;
	    }
	  
	  /* Check for next fixed header */
	  if ( MS_ISVALIDHEADER((char *)nextfsdh) || MS_ISVALIDBLANK((char *)nextfsdh) )
	    {
	      foundlen = 1;
	      reclen = recbuflen;
	    }
	}
    }
  
  if ( ! foundlen )
    return 0;
  else
    return reclen;
}  /* End of ms_find_reclen() */


/*********************************************************************
 * ms_fread:
 *
 * A wrapper for fread that handles EOF and error conditions.
 *
 * Returns the return value from fread.
 *********************************************************************/
static int
ms_fread (char *buf, int size, int num, FILE *stream)
{
  int read = 0;
  
  read = fread (buf, size, num, stream);
  
  if ( read <= 0 && size && num )
    {
      if ( ferror (stream) )
	ms_log (2, "ms_fread(): Cannot read input file\n");
      
      else if ( ! feof (stream) )
	ms_log (2, "ms_fread(): Unknown return from fread()\n");
    }
  
  return read;
}  /* End of ms_fread() */


/*********************************************************************
 * ms_ateof:
 *
 * Check if stream is at the end-of-file by reading a single character
 * and unreading it if necessary.
 *
 * Returns 1 if stream is at EOF otherwise 0.
 *********************************************************************/
static int
ms_ateof (FILE *stream)
{
  int c;
  
  c = getc (stream);
  
  if ( c == EOF )
    {
      if ( ferror (stream) )
	ms_log (2, "ms_ateof(): Error reading next character from stream\n");
      
      else if ( feof (stream) )
	return 1;
      
      else
	ms_log (2, "ms_ateof(): Unknown error reading next character from stream\n");
    }
  else
    {
      if ( ungetc (c, stream) == EOF )
	ms_log (2, "ms_ateof(): Error ungetting character\n");
    }
  
  return 0;
}  /* End of ms_ateof() */
