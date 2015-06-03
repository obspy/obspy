/****************************************************************************
 *
 * Routines to manage files of Mini-SEED.
 *
 * Written by Chad Trabant
 *   IRIS Data Management Center
 *
 * modified: 2015.108
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "libmseed.h"

static int ms_fread (char *buf, int size, int num, FILE *stream);

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
MSFileParam gMSFileParam = {NULL, "", NULL, 0, 0, 0, 0, 0, 0, 0};


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
ms_readmsr (MSRecord **ppmsr, const char *msfile, int reclen, off_t *fpos,
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
 * the re-entrant capabilities.  This routine is thread safe and can
 * be used to read more than one file at a time as long as separate
 * MSFileParam structures are used for each file.
 *
 * See the comments with ms_readmsr_main() for return values and
 * further description of arguments.
 *********************************************************************/
int
ms_readmsr_r (MSFileParam **ppmsfp, MSRecord **ppmsr, const char *msfile,
	      int reclen, off_t *fpos, int *last, flag skipnotdata,
	      flag dataflag, flag verbose)
{
  return ms_readmsr_main (ppmsfp, ppmsr, msfile, reclen, fpos,
			  last, skipnotdata, dataflag, NULL, verbose);
}  /* End of ms_readmsr_r() */


/**********************************************************************
 * ms_shift_msfp:
 *
 * A helper routine to shift (remove bytes from the beginning of) the
 * file reading buffer for a MSFP.  The buffer length, reading offset
 * and file position indicators are all updated as necessary.
 *
 *********************************************************************/
static void
ms_shift_msfp (MSFileParam *msfp, int shift)
{
  if ( ! msfp )
    return;
  
  if ( shift <= 0 && shift > msfp->readlen )
    {
      ms_log (2, "ms_shift_msfp(): Cannot shift buffer, shift: %d, readlen: %d, readoffset: %d\n",
	      shift, msfp->readlen, msfp->readoffset);
      return;
    }
  
  memmove (msfp->rawrec, msfp->rawrec + shift, msfp->readlen - shift);
  msfp->readlen -= shift;
  
  if ( shift < msfp->readoffset )
    {
      msfp->readoffset -= shift;
    }
  else
    {
      msfp->filepos += (shift - msfp->readoffset);
      msfp->readoffset = 0;
    }
  
  return;
}  /* End of ms_shift_msfp() */


/* Macro to calculate length of unprocessed buffer */
#define MSFPBUFLEN(MSFP) (MSFP->readlen - MSFP->readoffset)

/* Macro to return current reading position */
#define MSFPREADPTR(MSFP) (MSFP->rawrec + MSFP->readoffset)

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
 * If reclen is 0 or negative the length of every record is
 * automatically detected.  For auto detection of record length the
 * record must include a 1000 blockette or be followed by a valid
 * record header or end of file.
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
 * have valid data record indicators (D, R, Q, M, etc.) will be skipped.
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
ms_readmsr_main (MSFileParam **ppmsfp, MSRecord **ppmsr, const char *msfile,
		 int reclen, off_t *fpos, int *last, flag skipnotdata,
		 flag dataflag, Selections *selections, flag verbose)
{
  MSFileParam *msfp;
  off_t packdatasize = 0;
  int packskipsize;
  int parseval = 0;
  int readsize = 0;
  int readcount = 0;
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
	  ms_log (2, "ms_readmsr_main(): Cannot allocate memory for MSFP\n");
	  return MS_GENERROR;
	}
      
      /* Redirect the supplied pointer to the allocated params */
      *ppmsfp = msfp;
      
      msfp->fp = NULL;
      msfp->filename[0] = '\0';
      msfp->rawrec = NULL;
      msfp->readlen = 0;
      msfp->readoffset = 0;
      msfp->packtype = 0;
      msfp->packhdroffset = 0;
      msfp->filepos = 0;
      msfp->filesize = 0;
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
	  gMSFileParam.filename[0] = '\0';
          gMSFileParam.rawrec = NULL;
	  gMSFileParam.readlen = 0;
	  gMSFileParam.readoffset = 0;
	  gMSFileParam.packtype = 0;
	  gMSFileParam.packhdroffset = 0;
	  gMSFileParam.filepos = 0;
	  gMSFileParam.filesize = 0;
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
  
  /* Allocate reading buffer */
  if ( msfp->rawrec == NULL )
    {
      if ( ! (msfp->rawrec = (char *) malloc (MAXRECLEN)) )
	{
	  ms_log (2, "ms_readmsr_main(): Cannot allocate memory for read buffer\n");
	  return MS_GENERROR;
	}
    }
  
  /* Sanity check: track if we are reading the same file */
  if ( msfp->fp && strncmp (msfile, msfp->filename, sizeof(msfp->filename)) )
    {
      ms_log (2, "ms_readmsr_main() called with a different file name without being reset\n");
      
      /* Close previous file and reset needed variables */
      if ( msfp->fp != NULL )
	fclose (msfp->fp);
      
      msfp->fp = NULL;
      msfp->readlen = 0;
      msfp->readoffset = 0;
      msfp->packtype = 0;
      msfp->packhdroffset = 0;
      msfp->filepos = 0;
      msfp->filesize = 0;
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
      else
	{
	  if ( (msfp->fp = fopen (msfile, "rb")) == NULL )
	    {
	      ms_log (2, "Cannot open file: %s (%s)\n", msfile, strerror (errno));
	      msr_free (ppmsr);
	      
	      return MS_GENERROR;
	    }
	  else
	    {
	      /* Determine file size */
	      struct stat sbuf;
	      
	      if ( fstat (fileno(msfp->fp), &sbuf) )
		{
		  ms_log (2, "Cannot open file: %s (%s)\n", msfile, strerror (errno));
		  msr_free (ppmsr);
		  
		  return MS_GENERROR;
		}
	      
	      msfp->filesize = sbuf.st_size;
	    }
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
	      
	      return MS_GENERROR;
	    }
	  
	  msfp->filepos = *fpos * -1;
	  msfp->readlen = 0;
	  msfp->readoffset = 0;
	}
    }
  
  /* Zero the last record indicator */
  if ( last )
    *last = 0;
  
  /* Read data and search for records */
  for (;;)
    {
      /* Read more data into buffer if not at EOF and buffer has less than MINRECLEN
       * or more data is needed for the current record detected in buffer. */
      if ( ! feof(msfp->fp) && (MSFPBUFLEN(msfp) < MINRECLEN || parseval > 0) )
	{
	  /* Reset offsets if no unprocessed data in buffer */
	  if ( MSFPBUFLEN(msfp) <= 0 )
	    {
	      msfp->readlen = 0;
	      msfp->readoffset = 0;
	    }
	  /* Otherwise shift existing data to beginning of buffer */
	  else if ( msfp->readoffset > 0 )
	    {
	      ms_shift_msfp (msfp, msfp->readoffset);
	    }
	  
	  /* Determine read size */
	  readsize = (MAXRECLEN - msfp->readlen);
	  
	  /* Read data into record buffer */
	  readcount = ms_fread (msfp->rawrec + msfp->readlen, 1, readsize, msfp->fp);
	  
	  if ( readcount != readsize )
	    {
	      if ( ! feof (msfp->fp) )
		{
		  ms_log (2, "Short read of %d bytes starting from %"PRId64"\n",
			  readsize, msfp->filepos);
		  retcode = MS_GENERROR;
		  break;
		}
	    }
	  
	  /* Update read buffer length */
	  msfp->readlen += readcount;
	  
	  /* File position corresponding to start of buffer; not strictly necessary */
	  if ( msfp->fp != stdin )
	    msfp->filepos = lmp_ftello (msfp->fp) - msfp->readlen;
	}
      
      /* Test for packed file signature at the beginning of the file */
      if ( msfp->filepos == 0 && *(MSFPREADPTR(msfp)) == 'P' && MSFPBUFLEN(msfp) >= 48 )
	{
	  msfp->packtype = 0;
	  
	  /* Determine pack type, the negative pack type indicates initial header */
	  if ( ! memcmp ("PED", MSFPREADPTR(msfp), 3) )
	    msfp->packtype = -1;
	  else if ( ! memcmp ("PSD", MSFPREADPTR(msfp), 3) )
	    msfp->packtype = -2;
	  else if ( ! memcmp ("PLC", MSFPREADPTR(msfp), 3) )
	    msfp->packtype = -6;
	  else if ( ! memcmp ("PQI", MSFPREADPTR(msfp), 3) )
	    msfp->packtype = -7;
	  else if ( ! memcmp ("PLS", MSFPREADPTR(msfp), 3) )
	    msfp->packtype = -8;
	  
	  if ( verbose > 0 )
	    ms_log (1, "Detected packed file (%3.3s: type %d)\n", MSFPREADPTR(msfp), -msfp->packtype);
	}
      
      /* Read pack headers, initial and subsequent headers including (ignored) chksum values */
      if ( msfp->packtype && (msfp->packtype < 0 || msfp->filepos == msfp->packhdroffset) && MSFPBUFLEN(msfp) >= 48 )
	{
	  char hdrstr[30];
	  int64_t datasize;
	  
	  /* Determine bytes to skip before header: either initial ID block or type-specific chksum block */
	  packskipsize = ( msfp->packtype < 0 ) ? 10 : packtypes[msfp->packtype][2];
	  
	  if ( msfp->packtype < 0 )
	    msfp->packtype = -msfp->packtype;
	  
	  /* Read pack length from pack header accounting for bytes that should be skipped */
	  memset (hdrstr, 0, sizeof(hdrstr));
	  memcpy (hdrstr, MSFPREADPTR(msfp) + (packtypes[msfp->packtype][0] + packskipsize - packtypes[msfp->packtype][1]),
		  packtypes[msfp->packtype][1]);
	  sscanf (hdrstr, " %"SCNd64, &datasize);
	  packdatasize = (off_t) datasize;
	  
	  /* Next pack header = File position + skipsize + header size + data size
	   * This offset is actually to the data block chksum which is skipped by the logic above,
	   * the next pack header should directly follow the chksum. */
	  msfp->packhdroffset = msfp->filepos + packskipsize + packtypes[msfp->packtype][0] + packdatasize;
	  
	  if ( verbose > 1 )
	    ms_log (1, "Read packed file header at offset %"PRId64" (%d bytes follow), chksum offset: %"PRId64"\n",
		    (msfp->filepos + packskipsize), packdatasize,
		    msfp->packhdroffset);

	  /* Shift buffer to new reading offset (aligns records in buffer) */
	  ms_shift_msfp (msfp, msfp->readoffset + (packskipsize + packtypes[msfp->packtype][0]));
	} /* End of packed header processing */
      
      /* Check for match if selections are supplied and pack header was read, */
      /* only when enough data is in buffer and not reading from stdin pipe */
      if ( selections && msfp->packtype && packdatasize && MSFPBUFLEN(msfp) >= 48 && msfp->fp != stdin )
	{
	  char srcname[100];
	  
	  ms_recsrcname (MSFPREADPTR(msfp), srcname, 1);
	  
	  if ( ! ms_matchselect (selections, srcname, HPTERROR, HPTERROR, NULL) )
	    {
	      /* Update read position if next section is in buffer */
	      if ( MSFPBUFLEN(msfp) >= (msfp->packhdroffset - msfp->filepos) )
		{
		  if ( verbose > 1 )
		    {
		      ms_log (1, "Skipping (jump) packed section for %s (%d bytes) starting at offset %"PRId64"\n",
			      srcname, (msfp->packhdroffset - msfp->filepos), msfp->filepos);
		    }
		  
		  msfp->readoffset += (msfp->packhdroffset - msfp->filepos);
		  msfp->filepos = msfp->packhdroffset;
		  packdatasize = 0;
		}
	      
	      /* Otherwise seek to next pack header and reset reading position */
	      else
		{
		  if ( verbose > 1 )
		    {
		      ms_log (1, "Skipping (seek) packed section for %s (%d bytes) starting at offset %"PRId64"\n",
			      srcname, (msfp->packhdroffset - msfp->filepos), msfp->filepos);
		    }

		  if ( lmp_fseeko (msfp->fp, msfp->packhdroffset, SEEK_SET) )
		    {
		      ms_log (2, "Cannot seek in file: %s (%s)\n", msfile, strerror (errno));
		      
		      return MS_GENERROR;
		      break;
		    }
		  
		  msfp->filepos = msfp->packhdroffset;
		  msfp->readlen = 0;
		  msfp->readoffset = 0;
		  packdatasize = 0;
		}
	      
	      /* Return to top of loop for proper pack header handling */
	      continue;
	    }
	} /* End of selection processing */
      
      /* Attempt to parse record from buffer */
      if ( MSFPBUFLEN(msfp) >= MINRECLEN )
	{
	  int parselen = MSFPBUFLEN(msfp);
	  
	  /* Limit the parse length to offset of pack header if present in the buffer */
	  if ( msfp->packhdroffset && msfp->packhdroffset < (msfp->filepos + MSFPBUFLEN(msfp)) )
	    parselen = msfp->packhdroffset - msfp->filepos;
	  
 	  parseval = msr_parse (MSFPREADPTR(msfp), parselen, ppmsr, reclen, dataflag, verbose);
	  
	  /* Record detected and parsed */
	  if ( parseval == 0 )
	    {
	      if ( verbose > 1 )
		ms_log (1, "Read record length of %d bytes\n", (*ppmsr)->reclen);
	      
	      /* Test if this is the last record if file size is known (not pipe) */
	      if ( last && msfp->filesize )
		if ( (msfp->filesize - (msfp->filepos + (*ppmsr)->reclen)) < MINRECLEN )
		  *last = 1;
	      
	      /* Return file position for this record */
	      if ( fpos )
		*fpos = msfp->filepos;
	      
	      /* Update reading offset, file position and record count */
	      msfp->readoffset += (*ppmsr)->reclen;
	      msfp->filepos += (*ppmsr)->reclen;
	      msfp->recordcount++;
	      
	      retcode = MS_NOERROR;
	      break;
	    }
	  else if ( parseval < 0 )
	    {
	      /* Skip non-data if requested */ 
	      if ( skipnotdata )
		{
		  if ( verbose > 1 )
		    {
		      if ( MS_ISVALIDBLANK((char *)MSFPREADPTR(msfp)) )
			ms_log (1, "Skipped %d bytes of blank/noise record at byte offset %"PRId64"\n",
				MINRECLEN, msfp->filepos);
		      else
			ms_log (1, "Skipped %d bytes of non-data record at byte offset %"PRId64"\n",
				MINRECLEN, msfp->filepos);
		    }
		  
		  /* Skip MINRECLEN bytes, update reading offset and file position */
		  msfp->readoffset += MINRECLEN;
		  msfp->filepos += MINRECLEN;
		}
	      /* Parsing errors */ 
	      else
		{
		  ms_log (2, "Cannot detect record at byte offset %"PRId64": %s\n",
			  msfp->filepos, msfile);
		  
		  /* Print common errors and raw details if verbose */
		  ms_parse_raw (MSFPREADPTR(msfp), MSFPBUFLEN(msfp), verbose, -1);
		  
		  retcode = parseval;
		  break;
		}
	    }
	  else /* parseval > 0 (found record but need more data) */
	    {
	      /* Determine implied record length if needed */
	      int32_t impreclen = reclen;
	      
	      /* Check for parse hints that are larger than MAXRECLEN */
	      if ( (MSFPBUFLEN(msfp) + parseval) > MAXRECLEN )
		{
		  if ( skipnotdata )
		    {
		      /* Skip MINRECLEN bytes, update reading offset and file position */
		      msfp->readoffset += MINRECLEN;
		      msfp->filepos += MINRECLEN;
		    }
		  else
		    {
		      retcode = MS_OUTOFRANGE;
		      break;
		    }
		}
	      
	      /* Pack header check, if pack header offset is within buffer */
	      else if ( impreclen <= 0 && msfp->packhdroffset &&
			msfp->packhdroffset < (msfp->filepos + MSFPBUFLEN(msfp)) )
		{
		  impreclen = msfp->packhdroffset - msfp->filepos;
		  
		  /* Check that record length is within range and a power of 2.
		   * Power of two if (X & (X - 1)) == 0 */
		  if ( impreclen >= MINRECLEN && impreclen <= MAXRECLEN &&
		       (impreclen & (impreclen - 1)) == 0 )
		    {
		      /* Set the record length implied by the next pack header */
		      reclen = impreclen;
		    }
		  else
		    {
		      ms_log (1, "Implied record length (%d) is invalid\n", impreclen);
		      
		      retcode = MS_NOTSEED;
		      break;
		    }
		}
	      
	      /* End of file check */
	      else if ( impreclen <= 0 && feof (msfp->fp) )
		{
		  impreclen = msfp->filesize - msfp->filepos;
		  
		  /* Check that record length is within range and a power of 2.
		   * Power of two if (X & (X - 1)) == 0 */
		  if ( impreclen >= MINRECLEN && impreclen <= MAXRECLEN &&
		       (impreclen & (impreclen - 1)) == 0 )
		    {
		      /* Set the record length implied by the end of the file */
		      reclen = impreclen;
		    }
		  /* Otherwise a trucated record */
		  else
		    {
		      if ( verbose )
			{
			  if ( msfp->filesize )
			    ms_log (1, "Truncated record at byte offset %"PRId64", filesize %d: %s\n",
				    msfp->filepos, msfp->filesize, msfile);
			  else
			    ms_log (1, "Truncated record at byte offset %"PRId64"\n",
				    msfp->filepos);
			}
		      
		      retcode = MS_ENDOFFILE;
		      break;
		    }
		}
	    }
	}  /* End of record detection */
      
      /* Finished when within MINRECLEN from EOF and buffer less than MINRECLEN */
      if ( (msfp->filesize - msfp->filepos) < MINRECLEN && MSFPBUFLEN(msfp) < MINRECLEN )
	{
	  if ( msfp->recordcount == 0 && msfp->packtype == 0 )
	    {
	      if ( verbose > 0 )
		ms_log (2, "%s: No data records read, not SEED?\n", msfile);
	      retcode = MS_NOTSEED;
	    }
	  else
	    {
	      retcode = MS_ENDOFFILE;
	    }
	  
	  break;
	}
    }  /* End of reading, record detection and parsing loop */
  
  /* Cleanup target MSRecord if returning an error */
  if ( retcode != MS_NOERROR )
    {
      msr_free (ppmsr);
    }
  
  return retcode;
}  /* End of ms_readmsr_main() */


/*********************************************************************
 * ms_readtraces:
 *
 * This is a simple wrapper for ms_readtraces_selection() that uses no
 * selections.
 *
 * See the comments with ms_readtraces_selection() for return values
 * and further description of arguments.
 *********************************************************************/
int
ms_readtraces (MSTraceGroup **ppmstg, const char *msfile, int reclen,
	       double timetol, double sampratetol, flag dataquality,
	       flag skipnotdata, flag dataflag, flag verbose)
{
  return ms_readtraces_selection (ppmstg, msfile, reclen,
				  timetol, sampratetol, NULL,
				  dataquality, skipnotdata,
				  dataflag, verbose);
}  /* End of ms_readtraces() */


/*********************************************************************
 * ms_readtraces_timewin:
 *
 * This is a wrapper for ms_readtraces_selection() that creates a
 * simple selection for a specified time window.
 *
 * See the comments with ms_readtraces_selection() for return values
 * and further description of arguments.
 *********************************************************************/
int
ms_readtraces_timewin (MSTraceGroup **ppmstg, const char *msfile, int reclen,
		       double timetol, double sampratetol,
		       hptime_t starttime, hptime_t endtime, flag dataquality,
		       flag skipnotdata, flag dataflag, flag verbose)
{
  Selections selection;
  SelectTime selecttime;
  
  selection.srcname[0] = '*';
  selection.srcname[1] = '\0';
  selection.timewindows = &selecttime;
  selection.next = NULL;
  
  selecttime.starttime = starttime;
  selecttime.endtime = endtime;
  selecttime.next = NULL;
  
  return ms_readtraces_selection (ppmstg, msfile, reclen,
				  timetol, sampratetol, &selection,
				  dataquality, skipnotdata,
				  dataflag, verbose);
}  /* End of ms_readtraces_timewin() */


/*********************************************************************
 * ms_readtraces_selection:
 *
 * This routine will open and read all Mini-SEED records in specified
 * file and populate a trace group.  This routine is thread safe.
 *
 * If reclen is <= 0 the length of every record is automatically
 * detected.
 *
 * If a Selections list is supplied it will be used to limit which
 * records are added to the trace group.
 *
 * Returns MS_NOERROR and populates an MSTraceGroup struct at *ppmstg
 * on successful read, otherwise returns a libmseed error code (listed
 * in libmseed.h).
 *********************************************************************/
int
ms_readtraces_selection (MSTraceGroup **ppmstg, const char *msfile,
			 int reclen, double timetol, double sampratetol,
			 Selections *selections, flag dataquality,
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
      /* Test against selections if supplied */
      if ( selections )
	{
	  char srcname[50];
	  hptime_t endtime;
	  
	  msr_srcname (msr, srcname, 1);
	  endtime = msr_endtime (msr);
	  
	  if ( ms_matchselect (selections, srcname, msr->starttime, endtime, NULL) == NULL )
	    {
	      continue;
	    }
	}
      
      /* Add to trace group */
      mst_addmsrtogroup (*ppmstg, msr, dataquality, timetol, sampratetol);
    }
  
  /* Reset return code to MS_NOERROR on successful read by ms_readmsr() */
  if ( retcode == MS_ENDOFFILE )
    retcode = MS_NOERROR;
  
  ms_readmsr_main (&msfp, &msr, NULL, 0, NULL, NULL, 0, 0, NULL, 0);
  
  return retcode;
}  /* End of ms_readtraces_selection() */


/*********************************************************************
 * ms_readtracelist:
 *
 * This is a simple wrapper for ms_readtracelist_selection() that uses
 * no selections.
 *
 * See the comments with ms_readtracelist_selection() for return
 * values and further description of arguments.
 *********************************************************************/
int
ms_readtracelist (MSTraceList **ppmstl, const char *msfile, int reclen,
		  double timetol, double sampratetol, flag dataquality,
		  flag skipnotdata, flag dataflag, flag verbose)
{
  return ms_readtracelist_selection (ppmstl, msfile, reclen,
				     timetol, sampratetol, NULL,
				     dataquality, skipnotdata,
				     dataflag, verbose);
}  /* End of ms_readtracelist() */


/*********************************************************************
 * ms_readtracelist_timewin:
 *
 * This is a wrapper for ms_readtraces_selection() that creates a
 * simple selection for a specified time window.
 *
 * See the comments with ms_readtraces_selection() for return values
 * and further description of arguments.
 *********************************************************************/
int
ms_readtracelist_timewin (MSTraceList **ppmstl, const char *msfile,
			  int reclen, double timetol, double sampratetol,
			  hptime_t starttime, hptime_t endtime, flag dataquality,
			  flag skipnotdata, flag dataflag, flag verbose)
{
  Selections selection;
  SelectTime selecttime;
  
  selection.srcname[0] = '*';
  selection.srcname[1] = '\0';
  selection.timewindows = &selecttime;
  selection.next = NULL;
  
  selecttime.starttime = starttime;
  selecttime.endtime = endtime;
  selecttime.next = NULL;
  
  return ms_readtracelist_selection (ppmstl, msfile, reclen,
				     timetol, sampratetol, &selection,
				     dataquality, skipnotdata,
				     dataflag, verbose);
}  /* End of ms_readtracelist_timewin() */


/*********************************************************************
 * ms_readtracelist_selection:
 *
 * This routine will open and read all Mini-SEED records in specified
 * file and populate a trace list.  This routine is thread safe.
 *
 * If reclen is <= 0 the length of every record is automatically
 * detected.
 *
 * If a Selections list is supplied it will be used to limit which
 * records are added to the trace list.
 *
 * Returns MS_NOERROR and populates an MSTraceList struct at *ppmstl
 * on successful read, otherwise returns a libmseed error code (listed
 * in libmseed.h).
 *********************************************************************/
int
ms_readtracelist_selection (MSTraceList **ppmstl, const char *msfile,
			    int reclen, double timetol, double sampratetol,
			    Selections *selections, flag dataquality,
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
      /* Test against selections if supplied */
      if ( selections )
	{
	  char srcname[50];
	  hptime_t endtime;
	  
	  msr_srcname (msr, srcname, 1);
	  endtime = msr_endtime (msr);
	  
	  if ( ms_matchselect (selections, srcname, msr->starttime, endtime, NULL) == NULL )
	    {
	      continue;
	    }
	}
      
      /* Add to trace list */
      mstl_addmsr (*ppmstl, msr, dataquality, 1, timetol, sampratetol);
    }
  
  /* Reset return code to MS_NOERROR on successful read by ms_readmsr() */
  if ( retcode == MS_ENDOFFILE )
    retcode = MS_NOERROR;
  
  ms_readmsr_main (&msfp, &msr, NULL, 0, NULL, NULL, 0, 0, NULL, 0);
  
  return retcode;
}  /* End of ms_readtracelist_selection() */


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
  
  read = (int) fread (buf, size, num, stream);
  
  if ( read <= 0 && size && num )
    {
      if ( ferror (stream) )
	ms_log (2, "ms_fread(): Cannot read input file\n");
      
      else if ( ! feof (stream) )
	ms_log (2, "ms_fread(): Unknown return from fread()\n");
    }
  
  return read;
}  /* End of ms_fread() */


/***************************************************************************
 * ms_record_handler_int:
 *
 * Internal record handler.  The handler data should be a pointer to
 * an open file descriptor to which records will be written.
 *
 ***************************************************************************/
static void
ms_record_handler_int (char *record, int reclen, void *ofp)
{
  if ( fwrite(record, reclen, 1, (FILE *)ofp) != 1 )
    {
      ms_log (2, "Error writing to output file\n");
    }
}  /* End of ms_record_handler_int() */


/***************************************************************************
 * msr_writemseed:
 *
 * Pack MSRecord data into Mini-SEED record(s) by calling msr_pack() and
 * write to a specified file.
 *
 * Returns the number of records written on success and -1 on error.
 ***************************************************************************/
int
msr_writemseed ( MSRecord *msr, const char *msfile, flag overwrite,
		 int reclen, flag encoding, flag byteorder, flag verbose )
{
  FILE *ofp;
  char srcname[50];
  char *perms = (overwrite) ? "wb":"ab";
  int packedrecords = 0;
  
  if ( ! msr || ! msfile )
    return -1;
  
  /* Open output file or use stdout */
  if ( strcmp (msfile, "-") == 0 )
    {
      ofp = stdout;
    }
  else if ( (ofp = fopen (msfile, perms)) == NULL )
    {
      ms_log (1, "Cannot open output file %s: %s\n", msfile, strerror(errno));
      
      return -1;
    }
  
  /* Pack the MSRecord */
  if ( msr->numsamples > 0 )
    {
      msr->encoding = encoding;
      msr->reclen = reclen;
      msr->byteorder = byteorder;
      
      packedrecords = msr_pack (msr, &ms_record_handler_int, ofp, NULL, 1, verbose-1);
      
      if ( packedrecords < 0 )
        {
	  msr_srcname (msr, srcname, 1);
          ms_log (1, "Cannot write Mini-SEED for %s\n", srcname);
        }
    }
  
  /* Close file and return record count */
  fclose (ofp);
  
  return (packedrecords >= 0) ? packedrecords : -1;
}  /* End of msr_writemseed() */


/***************************************************************************
 * mst_writemseed:
 *
 * Pack MSTrace data into Mini-SEED records by calling mst_pack() and
 * write to a specified file.
 *
 * Returns the number of records written on success and -1 on error.
 ***************************************************************************/
int
mst_writemseed ( MSTrace *mst, const char *msfile, flag overwrite,
		 int reclen, flag encoding, flag byteorder, flag verbose )
{
  FILE *ofp;
  char srcname[50];
  char *perms = (overwrite) ? "wb":"ab";
  int packedrecords = 0;
  
  if ( ! mst || ! msfile )
    return -1;
  
  /* Open output file or use stdout */
  if ( strcmp (msfile, "-") == 0 )
    {
      ofp = stdout;
    }
  else if ( (ofp = fopen (msfile, perms)) == NULL )
    {
      ms_log (1, "Cannot open output file %s: %s\n", msfile, strerror(errno));
      
      return -1;
    }
  
  /* Pack the MSTrace */
  if ( mst->numsamples > 0 )
    {
      packedrecords = mst_pack (mst, &ms_record_handler_int, ofp, reclen, encoding,
				byteorder, NULL, 1, verbose-1, NULL);
      
      if ( packedrecords < 0 )
        {
	  mst_srcname (mst, srcname, 1);
          ms_log (1, "Cannot write Mini-SEED for %s\n", srcname);
        }
    }
  
  /* Close file and return record count */
  fclose (ofp);
  
  return (packedrecords >= 0) ? packedrecords : -1;
}  /* End of mst_writemseed() */


/***************************************************************************
 * mst_writemseedgroup:
 *
 * Pack MSTraceGroup data into Mini-SEED records by calling mst_pack()
 * for each MSTrace in the group and write to a specified file.
 *
 * Returns the number of records written on success and -1 on error.
 ***************************************************************************/
int
mst_writemseedgroup ( MSTraceGroup *mstg, const char *msfile, flag overwrite,
		      int reclen, flag encoding, flag byteorder, flag verbose )
{
  MSTrace *mst;
  FILE *ofp;
  char srcname[50];
  char *perms = (overwrite) ? "wb":"ab";
  int trpackedrecords;
  int packedrecords = 0;
  
  if ( ! mstg || ! msfile )
    return -1;
  
  /* Open output file or use stdout */
  if ( strcmp (msfile, "-") == 0 )
    {
      ofp = stdout;
    }
  else if ( (ofp = fopen (msfile, perms)) == NULL )
    {
      ms_log (1, "Cannot open output file %s: %s\n", msfile, strerror(errno));
      
      return -1;
    }
  
  /* Pack each MSTrace in the group */
  mst = mstg->traces;
  while ( mst )
    {
      if ( mst->numsamples <= 0 )
        {
          mst = mst->next;
          continue;
        }
      
      trpackedrecords = mst_pack (mst, &ms_record_handler_int, ofp, reclen, encoding,
                                  byteorder, NULL, 1, verbose-1, NULL);
      
      if ( trpackedrecords < 0 )
        {
	  mst_srcname (mst, srcname, 1);
          ms_log (1, "Cannot write Mini-SEED for %s\n", srcname);
        }
      else
        {
          packedrecords += trpackedrecords;
        }
      
      mst = mst->next;
    }
  
  /* Close file and return record count */
  fclose (ofp);
  
  return packedrecords;
}  /* End of mst_writemseedgroup() */

