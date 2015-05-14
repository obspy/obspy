/***************************************************************************
 *
 * Routines to parse Mini-SEED.
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

#include "libmseed.h"


/**********************************************************************
 * msr_parse:
 *
 * This routine will attempt to parse (detect and unpack) a Mini-SEED
 * record from a specified memory buffer and populate a supplied
 * MSRecord structure.
 *
 * If reclen is less than or equal to 0 the length of record is
 * automatically detected otherwise reclen should be the correct
 * record length.
 *
 * For auto detection of record length the record should include a
 * 1000 blockette or be followed by another record header in the
 * buffer.
 *
 * dataflag will be passed directly to msr_unpack().
 *
 * Return values:
 *   0 : Success, populates the supplied MSRecord.
 *  >0 : Data record detected but not enough data is present, the
 *       return value is a hint of how many more bytes are needed.
 *  <0 : libmseed error code (listed in libmseed.h) is returned.
 *********************************************************************/
int
msr_parse ( char *record, int recbuflen, MSRecord **ppmsr, int reclen,
	    flag dataflag, flag verbose )
{
  int detlen = 0;
  int retcode = 0;
  
  if ( ! ppmsr )
    return MS_GENERROR;
  
  if ( ! record )
    return MS_GENERROR;
  
  /* Sanity check: record length cannot be larger than buffer */
  if ( reclen > 0 && reclen > recbuflen )
    {
      ms_log (2, "ms_parse() Record length (%d) cannot be larger than buffer (%d)\n",
	      reclen, recbuflen);
      return MS_GENERROR;
    }
  
  /* Autodetect the record length */
  if ( reclen <= 0 )
    {
      detlen = ms_detect (record, recbuflen);
      
      /* No data record detected */
      if ( detlen < 0 )
	{
	  return MS_NOTSEED;
	}
      
      /* Found record but could not determine length */
      if ( detlen == 0 )
	{
	  return MINRECLEN;
	}
      
      if ( verbose > 2 )
	{
	  ms_log (1, "Detected record length of %d bytes\n", detlen);
	}
      
      reclen = detlen;
    }
  
  /* Check that record length is in supported range */
  if ( reclen < MINRECLEN || reclen > MAXRECLEN )
    {
      ms_log (2, "Record length is out of range: %d (allowed: %d to %d)\n",
	      reclen, MINRECLEN, MAXRECLEN);
      
      return MS_OUTOFRANGE;
    }
  
  /* Check if more data is required, return hint */
  if ( reclen > recbuflen )
    {
      if ( verbose > 2 )
	ms_log (1, "Detected %d byte record, need %d more bytes\n",
		reclen, (reclen - recbuflen));
      
      return (reclen - recbuflen);
    }
  
  /* Unpack record */
  if ( (retcode = msr_unpack (record, reclen, ppmsr, dataflag, verbose)) != MS_NOERROR )
    {
      msr_free (ppmsr);
      
      return retcode;
    }
  
  return MS_NOERROR;
}  /* End of msr_parse() */


/**********************************************************************
 * msr_parse_selection:
 *
 * This routine wraps msr_parse() to parse and return the first record
 * from a memory buffer that matches optional Selections.  If the
 * selections pointer is NULL the effect is to search the buffer for
 * the first parsable record.
 *
 * The offset value specifies the starting offset in the buffer and,
 * on success, the offset in the buffer to record parsed.
 *
 * The caller should manage the value of the offset in two ways:
 * 
 * 1) on subsequent calls after a record has been parsed the caller
 * should increment the offset by the record length returned or
 * properly manipulate the record buffer pointer, buffer length and
 * offset to the same effect.
 *
 * 2) when the end of the buffer is reached MS_GENERROR (-1) is
 * returned, the caller should check the offset value against the
 * record buffer length to determine when the entire buffer has been
 * searched.
 * 
 * Return values: same as msr_parse() except that MS_GENERROR is
 * returned when end-of-buffer is reached.
 *********************************************************************/
int
msr_parse_selection ( char *recbuf, int recbuflen, int64_t *offset,
		      MSRecord **ppmsr, int reclen,
		      Selections *selections, flag dataflag, flag verbose )
{
  int retval = MS_GENERROR;
  int unpackretval;
  flag dataswapflag = 0;
  flag bigendianhost = ms_bigendianhost();
  
  if ( ! ppmsr )
    return MS_GENERROR;
  
  if ( ! recbuf )
    return MS_GENERROR;
  
  if ( ! offset )
    return MS_GENERROR;
  
  while ( *offset < recbuflen )
    {
      retval = msr_parse (recbuf+*offset, (int)(recbuflen-*offset), ppmsr, reclen, 0, verbose);
      
      if ( retval )
        {
          if ( verbose )
            ms_log (2, "Error parsing record at offset %"PRId64"\n", *offset);
	  
          *offset += MINRECLEN;
        }
      else
        {
	  if ( selections && ! msr_matchselect (selections, *ppmsr, NULL) )
	    {
	      *offset += (*ppmsr)->reclen;
	      retval = MS_GENERROR;
	    }
	  else
	    {
	      if ( dataflag )
		{
		  /* If BE host and LE data need swapping */
		  if ( bigendianhost && (*ppmsr)->byteorder == 0 )
		    dataswapflag = 1;
		  /* If LE host and BE data (or bad byte order value) need swapping */
		  else if ( !bigendianhost && (*ppmsr)->byteorder > 0 )
		    dataswapflag = 1;
		  
		  unpackretval = msr_unpack_data (*ppmsr, dataswapflag, verbose);
		  
		  if ( unpackretval < 0 )
		    return unpackretval;
		  else
		    (*ppmsr)->numsamples = unpackretval;
		}
	      
	      break;
	    }
        }
    }
  
  return retval;
}  /* End of msr_parse_selection() */


/********************************************************************
 * ms_detect:
 *
 * Determine SEED data record length with the following steps:
 *
 * 1) determine that the buffer contains a SEED data record by
 * verifying known signatures (fields with known limited values)
 *
 * 2) search the record up to recbuflen bytes for a 1000 blockette.
 *
 * 3) If no blockette 1000 is found search at MINRECLEN-byte offsets
 * for the fixed section of the next header or blank/noise record,
 * thereby implying the record length.
 *
 * Returns:
 * -1 : data record not detected or error
 *  0 : data record detected but could not determine length
 * >0 : size of the record in bytes
 *********************************************************************/
int
ms_detect ( const char *record, int recbuflen )
{
  uint16_t blkt_offset;    /* Byte offset for next blockette */
  uint8_t swapflag  = 0;   /* Byte swapping flag */
  uint8_t foundlen = 0;    /* Found record length */
  int32_t reclen = -1;     /* Size of record in bytes */
  
  uint16_t blkt_type;
  uint16_t next_blkt;
  
  struct fsdh_s *fsdh;
  struct blkt_1000_s *blkt_1000;
  const char *nextfsdh;
  
  /* Buffer must be at least 48 bytes (the fixed section) */
  if ( recbuflen < 48 )
    return -1;

  /* Check for valid fixed section of header */
  if ( ! MS_ISVALIDHEADER(record) )
    return -1;
  
  fsdh = (struct fsdh_s *) record;
  
  /* Check to see if byte swapping is needed by checking for sane year and day */
  if ( ! MS_ISVALIDYEARDAY(fsdh->start_time.year, fsdh->start_time.day) )
    swapflag = 1;
  
  blkt_offset = fsdh->blockette_offset;
  
  /* Swap order of blkt_offset if needed */
  if ( swapflag ) ms_gswap2 (&blkt_offset);
  
  /* Loop through blockettes as long as number is non-zero and viable */
  while ( blkt_offset != 0 &&
	  blkt_offset <= recbuflen )
    {
      memcpy (&blkt_type, record + blkt_offset, 2);
      memcpy (&next_blkt, record + blkt_offset + 2, 2);
      
      if ( swapflag )
	{
	  ms_gswap2 (&blkt_type);
	  ms_gswap2 (&next_blkt);
	}
      
      /* Found a 1000 blockette, not truncated */
      if ( blkt_type == 1000  &&
	   (int)(blkt_offset + 4 + sizeof(struct blkt_1000_s)) <= recbuflen )
	{
          blkt_1000 = (struct blkt_1000_s *) (record + blkt_offset + 4);
	  
          foundlen = 1;
	  
          /* Calculate record size in bytes as 2^(blkt_1000->reclen) */
	  reclen = (unsigned int) 1 << blkt_1000->reclen;
          
	  break;
        }
      
      /* Safety check for invalid offset */
      if ( next_blkt != 0 && ( next_blkt < 4 || (next_blkt - 4) <= blkt_offset ) )
	{
	  ms_log (2, "Invalid blockette offset (%d) less than or equal to current offset (%d)\n",
		  next_blkt, blkt_offset);
	  return -1;
	}
      
      blkt_offset = next_blkt;
    }
  
  /* If record length was not determined by a 1000 blockette scan the buffer
   * and search for the next record */
  if ( reclen == -1 )
    {
      nextfsdh = record + MINRECLEN;
      
      /* Check for record header or blank/noise record at MINRECLEN byte offsets */
      while ( ((nextfsdh - record) + 48) < recbuflen )
	{
	  if ( MS_ISVALIDHEADER(nextfsdh) || MS_ISVALIDBLANK(nextfsdh) )
            {
	      foundlen = 1;
	      reclen = nextfsdh - record;
	      break;
	    }
	  
	  nextfsdh += MINRECLEN;
	}
    }
  
  if ( ! foundlen )
    return 0;
  else
    return reclen;
}  /* End of ms_detect() */


/***************************************************************************
 * ms_parse_raw:
 *
 * Parse and verify a SEED data record header (fixed section and
 * blockettes) at the lowest level, printing error messages for
 * invalid header values and optionally print raw header values.  The
 * memory at 'record' is assumed to be a Mini-SEED record.  Not every
 * possible test is performed, common errors and those causing
 * libmseed parsing to fail should be detected.
 *
 * The 'details' argument is interpreted as follows:
 *
 * details:
 *  0 = only print error messages for invalid header fields
 *  1 = print basic fields in addition to invalid field errors
 *  2 = print all fields in addition to invalid field errors
 *
 * The 'swapflag' argument is interpreted as follows:
 *
 * swapflag:
 *  1 = swap multibyte quantities
 *  0 = do no swapping
 * -1 = autodetect byte order using year test, swap if needed
 *
 * Any byte swapping performed by this routine is applied directly to
 * the memory reference by the record pointer.
 *
 * This routine is primarily intended to diagnose invalid Mini-SEED headers.
 *
 * Returns 0 when no errors were detected or a positive count of
 * errors detected.
 ***************************************************************************/
int
ms_parse_raw ( char *record, int maxreclen, flag details, flag swapflag )
{
  struct fsdh_s *fsdh;
  double nomsamprate;
  char srcname[50];
  char *X;
  char b;
  int retval = 0;
  int b1000encoding = -1;
  int b1000reclen = -1;
  int endofblockettes = -1;
  int idx;
  
  if ( ! record )
    return 1;
  
  /* Generate a source name string */
  srcname[0] = '\0';
  ms_recsrcname (record, srcname, 1);
  
  fsdh = (struct fsdh_s *) record;
  
  /* Check to see if byte swapping is needed by testing the year and day */
  if ( swapflag == -1 && ! MS_ISVALIDYEARDAY(fsdh->start_time.year, fsdh->start_time.day) )
    swapflag = 1;
  else
    swapflag = 0;
  
  if ( details > 1 )
    {
      if ( swapflag == 1 )
	ms_log (0, "Swapping multi-byte quantities in header\n");
      else
	ms_log (0, "Not swapping multi-byte quantities in header\n");
    }
  
  /* Swap byte order */
  if ( swapflag )
    {
      MS_SWAPBTIME (&fsdh->start_time);
      ms_gswap2a (&fsdh->numsamples);
      ms_gswap2a (&fsdh->samprate_fact);
      ms_gswap2a (&fsdh->samprate_mult);
      ms_gswap4a (&fsdh->time_correct);
      ms_gswap2a (&fsdh->data_offset);
      ms_gswap2a (&fsdh->blockette_offset);
    }
  
  /* Validate fixed section header fields */
  X = record;  /* Pointer of convenience */
  
  /* Check record sequence number, 6 ASCII digits */
  if ( ! isdigit((int) *(X))   || ! isdigit ((int) *(X+1)) ||
       ! isdigit((int) *(X+2)) || ! isdigit ((int) *(X+3)) ||
       ! isdigit((int) *(X+4)) || ! isdigit ((int) *(X+5)) )
    {
      ms_log (2, "%s: Invalid sequence number: '%c%c%c%c%c%c'\n", srcname, X, X+1, X+2, X+3, X+4, X+5);
      retval++;
    }
  
  /* Check header/quality indicator */
  if ( ! MS_ISDATAINDICATOR(*(X+6)) )
    {
      ms_log (2, "%s: Invalid header indicator (DRQM): '%c'\n", srcname, X+6);
      retval++;
    }
  
  /* Check reserved byte, space or NULL */
  if ( ! (*(X+7) == ' ' || *(X+7) == '\0') )
    {
      ms_log (2, "%s: Invalid fixed section reserved byte (Space): '%c'\n", srcname, X+7);
      retval++;
    }
  
  /* Check station code, 5 alphanumerics or spaces */
  if ( ! (isalnum((unsigned char) *(X+8)) || *(X+8) == ' ') ||
       ! (isalnum((unsigned char) *(X+9)) || *(X+9) == ' ') ||
       ! (isalnum((unsigned char) *(X+10)) || *(X+10) == ' ') ||
       ! (isalnum((unsigned char) *(X+11)) || *(X+11) == ' ') ||
       ! (isalnum((unsigned char) *(X+12)) || *(X+12) == ' ') )
    {
      ms_log (2, "%s: Invalid station code: '%c%c%c%c%c'\n", srcname, X+8, X+9, X+10, X+11, X+12);
      retval++;
    }
  
  /* Check location ID, 2 alphanumerics or spaces */
  if ( ! (isalnum((unsigned char) *(X+13)) || *(X+13) == ' ') ||
       ! (isalnum((unsigned char) *(X+14)) || *(X+14) == ' ') )
    {
      ms_log (2, "%s: Invalid location ID: '%c%c'\n", srcname, X+13, X+14);
      retval++;
    }
  
  /* Check channel codes, 3 alphanumerics or spaces */
  if ( ! (isalnum((unsigned char) *(X+15)) || *(X+15) == ' ') ||
       ! (isalnum((unsigned char) *(X+16)) || *(X+16) == ' ') ||
       ! (isalnum((unsigned char) *(X+17)) || *(X+17) == ' ') )
    {
      ms_log (2, "%s: Invalid channel codes: '%c%c%c'\n", srcname, X+15, X+16, X+17);
      retval++;
    }
  
  /* Check network code, 2 alphanumerics or spaces */
  if ( ! (isalnum((unsigned char) *(X+18)) || *(X+18) == ' ') ||
       ! (isalnum((unsigned char) *(X+19)) || *(X+19) == ' ') )
    {
      ms_log (2, "%s: Invalid network code: '%c%c'\n", srcname, X+18, X+19);
      retval++;
    }
  
  /* Check start time fields */
  if ( fsdh->start_time.year < 1900 || fsdh->start_time.year > 2100 )
    {
      ms_log (2, "%s: Unlikely start year (1900-2100): '%d'\n", srcname, fsdh->start_time.year);
      retval++;
    }
  if ( fsdh->start_time.day < 1 || fsdh->start_time.day > 366 )
    {
      ms_log (2, "%s: Invalid start day (1-366): '%d'\n", srcname, fsdh->start_time.day);
      retval++;
    }
  if ( fsdh->start_time.hour > 23 )
    {
      ms_log (2, "%s: Invalid start hour (0-23): '%d'\n", srcname, fsdh->start_time.hour);
      retval++;
    }
  if ( fsdh->start_time.min > 59 )
    {
      ms_log (2, "%s: Invalid start minute (0-59): '%d'\n", srcname, fsdh->start_time.min);
      retval++;
    }
  if ( fsdh->start_time.sec > 60 )
    {
      ms_log (2, "%s: Invalid start second (0-60): '%d'\n", srcname, fsdh->start_time.sec);
      retval++;
    }
  if ( fsdh->start_time.fract > 9999 )
    {
      ms_log (2, "%s: Invalid start fractional seconds (0-9999): '%d'\n", srcname, fsdh->start_time.fract);
      retval++;
    }
  
  /* Check number of samples, max samples in 4096-byte Steim-2 encoded record: 6601 */
  if ( fsdh->numsamples > 20000 )
    {
      ms_log (2, "%s: Unlikely number of samples (>20000): '%d'\n", srcname, fsdh->numsamples);
      retval++;
    }
  
  /* Sanity check that there is space for blockettes when both data and blockettes are present */
  if ( fsdh->numsamples > 0 && fsdh->numblockettes > 0 && fsdh->data_offset <= fsdh->blockette_offset )
    {
      ms_log (2, "%s: No space for %d blockettes, data offset: %d, blockette offset: %d\n", srcname,
	      fsdh->numblockettes, fsdh->data_offset, fsdh->blockette_offset);
      retval++;
    }
  
  
  /* Print raw header details */
  if ( details >= 1 )
    {
      /* Determine nominal sample rate */
      nomsamprate = ms_nomsamprate (fsdh->samprate_fact, fsdh->samprate_mult);
  
      /* Print header values */
      ms_log (0, "RECORD -- %s\n", srcname);
      ms_log (0, "        sequence number: '%c%c%c%c%c%c'\n", fsdh->sequence_number[0], fsdh->sequence_number[1], fsdh->sequence_number[2],
	      fsdh->sequence_number[3], fsdh->sequence_number[4], fsdh->sequence_number[5]);
      ms_log (0, " data quality indicator: '%c'\n", fsdh->dataquality);
      if ( details > 0 )
        ms_log (0, "               reserved: '%c'\n", fsdh->reserved);
      ms_log (0, "           station code: '%c%c%c%c%c'\n", fsdh->station[0], fsdh->station[1], fsdh->station[2], fsdh->station[3], fsdh->station[4]);
      ms_log (0, "            location ID: '%c%c'\n", fsdh->location[0], fsdh->location[1]);
      ms_log (0, "          channel codes: '%c%c%c'\n", fsdh->channel[0], fsdh->channel[1], fsdh->channel[2]);
      ms_log (0, "           network code: '%c%c'\n", fsdh->network[0], fsdh->network[1]);
      ms_log (0, "             start time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", fsdh->start_time.year, fsdh->start_time.day,
	      fsdh->start_time.hour, fsdh->start_time.min, fsdh->start_time.sec, fsdh->start_time.fract, fsdh->start_time.unused);
      ms_log (0, "      number of samples: %d\n", fsdh->numsamples);
      ms_log (0, "     sample rate factor: %d  (%.10g samples per second)\n",
              fsdh->samprate_fact, nomsamprate);
      ms_log (0, " sample rate multiplier: %d\n", fsdh->samprate_mult);
      
      /* Print flag details if requested */
      if ( details > 1 )
        {
          /* Activity flags */
	  b = fsdh->act_flags;
	  ms_log (0, "         activity flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
		  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
		  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
	  if ( b & 0x01 ) ms_log (0, "                         [Bit 0] Calibration signals present\n");
	  if ( b & 0x02 ) ms_log (0, "                         [Bit 1] Time correction applied\n");
	  if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Beginning of an event, station trigger\n");
	  if ( b & 0x08 ) ms_log (0, "                         [Bit 3] End of an event, station detrigger\n");
	  if ( b & 0x10 ) ms_log (0, "                         [Bit 4] A positive leap second happened in this record\n");
	  if ( b & 0x20 ) ms_log (0, "                         [Bit 5] A negative leap second happened in this record\n");
	  if ( b & 0x40 ) ms_log (0, "                         [Bit 6] Event in progress\n");
	  if ( b & 0x80 ) ms_log (0, "                         [Bit 7] Undefined bit set\n");
	  
	  /* I/O and clock flags */
	  b = fsdh->io_flags;
	  ms_log (0, "    I/O and clock flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
		  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
		  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
	  if ( b & 0x01 ) ms_log (0, "                         [Bit 0] Station volume parity error possibly present\n");
	  if ( b & 0x02 ) ms_log (0, "                         [Bit 1] Long record read (possibly no problem)\n");
	  if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Short record read (record padded)\n");
	  if ( b & 0x08 ) ms_log (0, "                         [Bit 3] Start of time series\n");
	  if ( b & 0x10 ) ms_log (0, "                         [Bit 4] End of time series\n");
	  if ( b & 0x20 ) ms_log (0, "                         [Bit 5] Clock locked\n");
	  if ( b & 0x40 ) ms_log (0, "                         [Bit 6] Undefined bit set\n");
	  if ( b & 0x80 ) ms_log (0, "                         [Bit 7] Undefined bit set\n");
	  
	  /* Data quality flags */
	  b = fsdh->dq_flags;
	  ms_log (0, "     data quality flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
		  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
		  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
	  if ( b & 0x01 ) ms_log (0, "                         [Bit 0] Amplifier saturation detected\n");
	  if ( b & 0x02 ) ms_log (0, "                         [Bit 1] Digitizer clipping detected\n");
	  if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Spikes detected\n");
	  if ( b & 0x08 ) ms_log (0, "                         [Bit 3] Glitches detected\n");
	  if ( b & 0x10 ) ms_log (0, "                         [Bit 4] Missing/padded data present\n");
	  if ( b & 0x20 ) ms_log (0, "                         [Bit 5] Telemetry synchronization error\n");
	  if ( b & 0x40 ) ms_log (0, "                         [Bit 6] A digital filter may be charging\n");
	  if ( b & 0x80 ) ms_log (0, "                         [Bit 7] Time tag is questionable\n");
	}
      
      ms_log (0, "   number of blockettes: %d\n", fsdh->numblockettes);
      ms_log (0, "        time correction: %ld\n", (long int) fsdh->time_correct);
      ms_log (0, "            data offset: %d\n", fsdh->data_offset);
      ms_log (0, " first blockette offset: %d\n", fsdh->blockette_offset);
    } /* Done printing raw header details */
  
  
  /* Validate and report information in the blockette chain */
  if ( fsdh->blockette_offset > 46 && fsdh->blockette_offset < maxreclen )
    {
      int blkt_offset = fsdh->blockette_offset;
      int blkt_count = 0;
      int blkt_length;
      uint16_t blkt_type;
      uint16_t next_blkt;
      char *blkt_desc;
      
      /* Traverse blockette chain */
      while ( blkt_offset != 0 && blkt_offset < maxreclen )
	{
	  /* Every blockette has a similar 4 byte header: type and next */
	  memcpy (&blkt_type, record + blkt_offset, 2);
	  memcpy (&next_blkt, record + blkt_offset+2, 2);
	  
	  if ( swapflag )
	    {
	      ms_gswap2 (&blkt_type);
	      ms_gswap2 (&next_blkt);
	    }
	  
	  /* Print common header fields */
	  if ( details >= 1 )
	    {
	      blkt_desc =  ms_blktdesc(blkt_type);
	      ms_log (0, "          BLOCKETTE %u: (%s)\n", blkt_type, (blkt_desc) ? blkt_desc : "Unknown");
	      ms_log (0, "              next blockette: %u\n", next_blkt);
	    }
	  
	  blkt_length = ms_blktlen (blkt_type, record + blkt_offset, swapflag);
	  if ( blkt_length == 0 )
	    {
	      ms_log (2, "%s: Unknown blockette length for type %d\n", srcname, blkt_type);
	      retval++;
	    }
	  
	  /* Track end of blockette chain */
	  endofblockettes = blkt_offset + blkt_length - 1;
	  
	  /* Sanity check that the blockette is contained in the record */
	  if ( endofblockettes > maxreclen )
	    {
	      ms_log (2, "%s: Blockette type %d at offset %d with length %d does not fix in record (%d)\n",
		      srcname, blkt_type, blkt_offset, blkt_length, maxreclen);
	      retval++;
	      break;
	    }
	  
	  if ( blkt_type == 100 )
	    {
	      struct blkt_100_s *blkt_100 = (struct blkt_100_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		ms_gswap4 (&blkt_100->samprate);
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "          actual sample rate: %.10g\n", blkt_100->samprate);
		  
		  if ( details > 1 )
		    {
		      b = blkt_100->flags;
		      ms_log (0, "             undefined flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			      bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			      bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		      
		      ms_log (0, "          reserved bytes (3): %u,%u,%u\n",
			      blkt_100->reserved[0], blkt_100->reserved[1], blkt_100->reserved[2]);
		    }
		}
	    }
	  
	  else if ( blkt_type == 200 )
	    {
	      struct blkt_200_s *blkt_200 = (struct blkt_200_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  ms_gswap4 (&blkt_200->amplitude);
		  ms_gswap4 (&blkt_200->period);
		  ms_gswap4 (&blkt_200->background_estimate);
		  MS_SWAPBTIME (&blkt_200->time);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "            signal amplitude: %g\n", blkt_200->amplitude);
		  ms_log (0, "               signal period: %g\n", blkt_200->period);
		  ms_log (0, "         background estimate: %g\n", blkt_200->background_estimate);
		  
		  if ( details > 1 )
		    {
		      b = blkt_200->flags;
		      ms_log (0, "       event detection flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			      bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			      bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		      if ( b & 0x01 ) ms_log (0, "                         [Bit 0] 1: Dilatation wave\n");
		      else            ms_log (0, "                         [Bit 0] 0: Compression wave\n");
		      if ( b & 0x02 ) ms_log (0, "                         [Bit 1] 1: Units after deconvolution\n");
		      else            ms_log (0, "                         [Bit 1] 0: Units are digital counts\n");
		      if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Bit 0 is undetermined\n");
		      ms_log (0, "               reserved byte: %u\n", blkt_200->reserved);
		    }
		  
		  ms_log (0, "           signal onset time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_200->time.year, blkt_200->time.day,
			  blkt_200->time.hour, blkt_200->time.min, blkt_200->time.sec, blkt_200->time.fract, blkt_200->time.unused);
		  ms_log (0, "               detector name: %.24s\n", blkt_200->detector);
		}
	    }
	  
	  else if ( blkt_type == 201 )
	    {
	      struct blkt_201_s *blkt_201 = (struct blkt_201_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  ms_gswap4 (&blkt_201->amplitude);
		  ms_gswap4 (&blkt_201->period);
		  ms_gswap4 (&blkt_201->background_estimate);
		  MS_SWAPBTIME (&blkt_201->time);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "            signal amplitude: %g\n", blkt_201->amplitude);
		  ms_log (0, "               signal period: %g\n", blkt_201->period);
		  ms_log (0, "         background estimate: %g\n", blkt_201->background_estimate);
		  
		  b = blkt_201->flags;
		  ms_log (0, "       event detection flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		  if ( b & 0x01 ) ms_log (0, "                         [Bit 0] 1: Dilation wave\n");
		  else            ms_log (0, "                         [Bit 0] 0: Compression wave\n");
		  
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_201->reserved);
		  ms_log (0, "           signal onset time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_201->time.year, blkt_201->time.day,
			  blkt_201->time.hour, blkt_201->time.min, blkt_201->time.sec, blkt_201->time.fract, blkt_201->time.unused);
		  ms_log (0, "                  SNR values: ");
		  for (idx=0; idx < 6; idx++) ms_log (0, "%u  ", blkt_201->snr_values[idx]);
		  ms_log (0, "\n");
		  ms_log (0, "              loopback value: %u\n", blkt_201->loopback);
		  ms_log (0, "              pick algorithm: %u\n", blkt_201->pick_algorithm);
		  ms_log (0, "               detector name: %.24s\n", blkt_201->detector);
		}
	    }
	  
	  else if ( blkt_type == 300 )
	    {
	      struct blkt_300_s *blkt_300 = (struct blkt_300_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  MS_SWAPBTIME (&blkt_300->time);
		  ms_gswap4 (&blkt_300->step_duration);
		  ms_gswap4 (&blkt_300->interval_duration);
		  ms_gswap4 (&blkt_300->amplitude);
		  ms_gswap4 (&blkt_300->reference_amplitude);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "      calibration start time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_300->time.year, blkt_300->time.day,
			  blkt_300->time.hour, blkt_300->time.min, blkt_300->time.sec, blkt_300->time.fract, blkt_300->time.unused);
		  ms_log (0, "      number of calibrations: %u\n", blkt_300->numcalibrations);
		  
		  b = blkt_300->flags;
		  ms_log (0, "           calibration flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		  if ( b & 0x01 ) ms_log (0, "                         [Bit 0] First pulse is positive\n");
		  if ( b & 0x02 ) ms_log (0, "                         [Bit 1] Calibration's alternate sign\n");
		  if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Calibration was automatic\n");
		  if ( b & 0x08 ) ms_log (0, "                         [Bit 3] Calibration continued from previous record(s)\n");
		  
		  ms_log (0, "               step duration: %u\n", blkt_300->step_duration);
		  ms_log (0, "           interval duration: %u\n", blkt_300->interval_duration);
		  ms_log (0, "            signal amplitude: %g\n", blkt_300->amplitude);
		  ms_log (0, "        input signal channel: %.3s", blkt_300->input_channel);
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_300->reserved);
		  ms_log (0, "         reference amplitude: %u\n", blkt_300->reference_amplitude);
		  ms_log (0, "                    coupling: %.12s\n", blkt_300->coupling);
		  ms_log (0, "                     rolloff: %.12s\n", blkt_300->rolloff);
		}
	    }
	  
	  else if ( blkt_type == 310 )
	    {
	      struct blkt_310_s *blkt_310 = (struct blkt_310_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  MS_SWAPBTIME (&blkt_310->time);
		  ms_gswap4 (&blkt_310->duration);
		  ms_gswap4 (&blkt_310->period);
		  ms_gswap4 (&blkt_310->amplitude);
		  ms_gswap4 (&blkt_310->reference_amplitude);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "      calibration start time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_310->time.year, blkt_310->time.day,
			  blkt_310->time.hour, blkt_310->time.min, blkt_310->time.sec, blkt_310->time.fract, blkt_310->time.unused);
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_310->reserved1);
		  
		  b = blkt_310->flags;
		  ms_log (0, "           calibration flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		  if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Calibration was automatic\n");
		  if ( b & 0x08 ) ms_log (0, "                         [Bit 3] Calibration continued from previous record(s)\n");
		  if ( b & 0x10 ) ms_log (0, "                         [Bit 4] Peak-to-peak amplitude\n");
		  if ( b & 0x20 ) ms_log (0, "                         [Bit 5] Zero-to-peak amplitude\n");
		  if ( b & 0x40 ) ms_log (0, "                         [Bit 6] RMS amplitude\n");
		  
		  ms_log (0, "        calibration duration: %u\n", blkt_310->duration);
		  ms_log (0, "               signal period: %g\n", blkt_310->period);
		  ms_log (0, "            signal amplitude: %g\n", blkt_310->amplitude);
		  ms_log (0, "        input signal channel: %.3s", blkt_310->input_channel);
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_310->reserved2);	      
		  ms_log (0, "         reference amplitude: %u\n", blkt_310->reference_amplitude);
		  ms_log (0, "                    coupling: %.12s\n", blkt_310->coupling);
		  ms_log (0, "                     rolloff: %.12s\n", blkt_310->rolloff);
		}
	    }
	  
	  else if ( blkt_type == 320 )
	    {
	      struct blkt_320_s *blkt_320 = (struct blkt_320_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  MS_SWAPBTIME (&blkt_320->time);
		  ms_gswap4 (&blkt_320->duration);
		  ms_gswap4 (&blkt_320->ptp_amplitude);
		  ms_gswap4 (&blkt_320->reference_amplitude);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "      calibration start time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_320->time.year, blkt_320->time.day,
			  blkt_320->time.hour, blkt_320->time.min, blkt_320->time.sec, blkt_320->time.fract, blkt_320->time.unused);
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_320->reserved1);
		  
		  b = blkt_320->flags;
		  ms_log (0, "           calibration flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		  if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Calibration was automatic\n");
		  if ( b & 0x08 ) ms_log (0, "                         [Bit 3] Calibration continued from previous record(s)\n");
		  if ( b & 0x10 ) ms_log (0, "                         [Bit 4] Random amplitudes\n");
		  
		  ms_log (0, "        calibration duration: %u\n", blkt_320->duration);
		  ms_log (0, "      peak-to-peak amplitude: %g\n", blkt_320->ptp_amplitude);
		  ms_log (0, "        input signal channel: %.3s", blkt_320->input_channel);
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_320->reserved2);
		  ms_log (0, "         reference amplitude: %u\n", blkt_320->reference_amplitude);
		  ms_log (0, "                    coupling: %.12s\n", blkt_320->coupling);
		  ms_log (0, "                     rolloff: %.12s\n", blkt_320->rolloff);
		  ms_log (0, "                  noise type: %.8s\n", blkt_320->noise_type);
		}
	    }
	  
	  else if ( blkt_type == 390 )
	    {
	      struct blkt_390_s *blkt_390 = (struct blkt_390_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  MS_SWAPBTIME (&blkt_390->time);
		  ms_gswap4 (&blkt_390->duration);
		  ms_gswap4 (&blkt_390->amplitude);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "      calibration start time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_390->time.year, blkt_390->time.day,
			  blkt_390->time.hour, blkt_390->time.min, blkt_390->time.sec, blkt_390->time.fract, blkt_390->time.unused);
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_390->reserved1);
		  
		  b = blkt_390->flags;
		  ms_log (0, "           calibration flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		  if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Calibration was automatic\n");
		  if ( b & 0x08 ) ms_log (0, "                         [Bit 3] Calibration continued from previous record(s)\n");
		  
		  ms_log (0, "        calibration duration: %u\n", blkt_390->duration);
		  ms_log (0, "            signal amplitude: %g\n", blkt_390->amplitude);
		  ms_log (0, "        input signal channel: %.3s", blkt_390->input_channel);
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_390->reserved2);
		}
	    }

	  else if ( blkt_type == 395 )
	    {
	      struct blkt_395_s *blkt_395 = (struct blkt_395_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		MS_SWAPBTIME (&blkt_395->time);
	      
	      if ( details >= 1 )
		{ 
		  ms_log (0, "        calibration end time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_395->time.year, blkt_395->time.day,
			  blkt_395->time.hour, blkt_395->time.min, blkt_395->time.sec, blkt_395->time.fract, blkt_395->time.unused);
		  if ( details > 1 )
		    ms_log (0, "          reserved bytes (2): %u,%u\n",
			    blkt_395->reserved[0], blkt_395->reserved[1]);
		}
	    }
	  
	  else if ( blkt_type == 400 )
	    {
	      struct blkt_400_s *blkt_400 = (struct blkt_400_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  ms_gswap4 (&blkt_400->azimuth);
		  ms_gswap4 (&blkt_400->slowness);
		  ms_gswap4 (&blkt_400->configuration);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "      beam azimuth (degrees): %g\n", blkt_400->azimuth);
		  ms_log (0, "  beam slowness (sec/degree): %g\n", blkt_400->slowness);
		  ms_log (0, "               configuration: %u\n", blkt_400->configuration);
		  if ( details > 1 )
		    ms_log (0, "          reserved bytes (2): %u,%u\n",
			    blkt_400->reserved[0], blkt_400->reserved[1]);
		}
	    }

	  else if ( blkt_type == 405 )
	    {
	      struct blkt_405_s *blkt_405 = (struct blkt_405_s *) (record + blkt_offset + 4);
	      uint16_t firstvalue = blkt_405->delay_values[0];  /* Work on a private copy */
	      
	      if ( swapflag )
		ms_gswap2 (&firstvalue);
	      
	      if ( details >= 1 )
		ms_log (0, "           first delay value: %u\n", firstvalue);
	    }
	  
	  else if ( blkt_type == 500 )
	    {
	      struct blkt_500_s *blkt_500 = (struct blkt_500_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  ms_gswap4 (&blkt_500->vco_correction);
		  MS_SWAPBTIME (&blkt_500->time);
		  ms_gswap4 (&blkt_500->exception_count);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "              VCO correction: %g%%\n", blkt_500->vco_correction);
		  ms_log (0, "           time of exception: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_500->time.year, blkt_500->time.day,
			  blkt_500->time.hour, blkt_500->time.min, blkt_500->time.sec, blkt_500->time.fract, blkt_500->time.unused);
		  ms_log (0, "                        usec: %d\n", blkt_500->usec);
		  ms_log (0, "           reception quality: %u%%\n", blkt_500->reception_qual);
		  ms_log (0, "             exception count: %u\n", blkt_500->exception_count);
		  ms_log (0, "              exception type: %.16s\n", blkt_500->exception_type);
		  ms_log (0, "                 clock model: %.32s\n", blkt_500->clock_model);
		  ms_log (0, "                clock status: %.128s\n", blkt_500->clock_status);
		}
	    }
	  
	  else if ( blkt_type == 1000 )
	    {
	      struct blkt_1000_s *blkt_1000 = (struct blkt_1000_s *) (record + blkt_offset + 4);
	      char order[40];
	      
	      /* Calculate record size in bytes as 2^(blkt_1000->rec_len) */
	      b1000reclen = (unsigned int) 1 << blkt_1000->reclen;
	      
	      /* Big or little endian? */
	      if (blkt_1000->byteorder == 0)
		strncpy (order, "Little endian", sizeof(order)-1);
	      else if (blkt_1000->byteorder == 1)
		strncpy (order, "Big endian", sizeof(order)-1);
	      else
		strncpy (order, "Unknown value", sizeof(order)-1);
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "                    encoding: %s (val:%u)\n",
			  (char *) ms_encodingstr (blkt_1000->encoding), blkt_1000->encoding);
		  ms_log (0, "                  byte order: %s (val:%u)\n",
			  order, blkt_1000->byteorder);
		  ms_log (0, "               record length: %d (val:%u)\n",
			  b1000reclen, blkt_1000->reclen);
		  
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_1000->reserved);
		}
	      
	      /* Save encoding format */
	      b1000encoding = blkt_1000->encoding;
	      
	      /* Sanity check encoding format */
	      if ( ! (b1000encoding >= 0 && b1000encoding <= 5) &&
		   ! (b1000encoding >= 10 && b1000encoding <= 19) &&
		   ! (b1000encoding >= 30 && b1000encoding <= 33) )
		{
		  ms_log (2, "%s: Blockette 1000 encoding format invalid (0-5,10-19,30-33): %d\n", srcname, b1000encoding);
		  retval++;
		}
	      
	      /* Sanity check byte order flag */
	      if ( blkt_1000->byteorder != 0 && blkt_1000->byteorder != 1 )
		{
		  ms_log (2, "%s: Blockette 1000 byte order flag invalid (0 or 1): %d\n", srcname, blkt_1000->byteorder);
		  retval++;
		}
	    }
	  
	  else if ( blkt_type == 1001 )
	    {
	      struct blkt_1001_s *blkt_1001 = (struct blkt_1001_s *) (record + blkt_offset + 4);
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "              timing quality: %u%%\n", blkt_1001->timing_qual);
		  ms_log (0, "                micro second: %d\n", blkt_1001->usec);
		  
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_1001->reserved);
		  
		  ms_log (0, "                 frame count: %u\n", blkt_1001->framecnt);
		}
	    }
	  
	  else if ( blkt_type == 2000 )
	    {
	      struct blkt_2000_s *blkt_2000 = (struct blkt_2000_s *) (record + blkt_offset + 4);
	      char order[40];
	      
	      if ( swapflag )
		{
		  ms_gswap2 (&blkt_2000->length);
		  ms_gswap2 (&blkt_2000->data_offset);
		  ms_gswap4 (&blkt_2000->recnum);
		}
	      
	      /* Big or little endian? */
	      if (blkt_2000->byteorder == 0)
		strncpy (order, "Little endian", sizeof(order)-1);
	      else if (blkt_2000->byteorder == 1)
		strncpy (order, "Big endian", sizeof(order)-1);
	      else
		strncpy (order, "Unknown value", sizeof(order)-1);
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "            blockette length: %u\n", blkt_2000->length);
		  ms_log (0, "                 data offset: %u\n", blkt_2000->data_offset);
		  ms_log (0, "               record number: %u\n", blkt_2000->recnum);
		  ms_log (0, "                  byte order: %s (val:%u)\n",
			  order, blkt_2000->byteorder);
		  b = blkt_2000->flags;
		  ms_log (0, "                  data flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		  
		  if ( details > 1 )
		    {
		      if ( b & 0x01 ) ms_log (0, "                         [Bit 0] 1: Stream oriented\n");
		      else            ms_log (0, "                         [Bit 0] 0: Record oriented\n");
		      if ( b & 0x02 ) ms_log (0, "                         [Bit 1] 1: Blockette 2000s may NOT be packaged\n");
		      else            ms_log (0, "                         [Bit 1] 0: Blockette 2000s may be packaged\n");
		      if ( ! (b & 0x04) && ! (b & 0x08) )
			ms_log (0, "                      [Bits 2-3] 00: Complete blockette\n");
		      else if ( ! (b & 0x04) && (b & 0x08) )
			ms_log (0, "                      [Bits 2-3] 01: First blockette in span\n");
		      else if ( (b & 0x04) && (b & 0x08) )
			ms_log (0, "                      [Bits 2-3] 11: Continuation blockette in span\n");
		      else if ( (b & 0x04) && ! (b & 0x08) )
			ms_log (0, "                      [Bits 2-3] 10: Final blockette in span\n");
		      if ( ! (b & 0x10) && ! (b & 0x20) )
			ms_log (0, "                      [Bits 4-5] 00: Not file oriented\n");
		      else if ( ! (b & 0x10) && (b & 0x20) )
			ms_log (0, "                      [Bits 4-5] 01: First blockette of file\n");
		      else if ( (b & 0x10) && ! (b & 0x20) )
			ms_log (0, "                      [Bits 4-5] 10: Continuation of file\n");
		      else if ( (b & 0x10) && (b & 0x20) )
			ms_log (0, "                      [Bits 4-5] 11: Last blockette of file\n");
		    }
		  
		  ms_log (0, "           number of headers: %u\n", blkt_2000->numheaders);
		  
		  /* Crude display of the opaque data headers */
		  if ( details > 1 )
		    ms_log (0, "                     headers: %.*s\n",
			    (blkt_2000->data_offset - 15), blkt_2000->payload);
		}
	    }
	  
	  else
	    {
	      ms_log (2, "%s: Unrecognized blockette type: %d\n", srcname, blkt_type);
	      retval++;
	    }
	  
	  /* Sanity check the next blockette offset */
	  if ( next_blkt && next_blkt <= endofblockettes )
	    {
	      ms_log (2, "%s: Next blockette offset (%d) is within current blockette ending at byte %d\n",
		      srcname, next_blkt, endofblockettes);
	      blkt_offset = 0;
	    }
	  else
	    {
	      blkt_offset = next_blkt;
	    }
	  
	  blkt_count++;
	} /* End of looping through blockettes */
      
      /* Check that the blockette offset is within the maximum record size */
      if ( blkt_offset > maxreclen )
	{
	  ms_log (2, "%s: Blockette offset (%d) beyond maximum record length (%d)\n", srcname, blkt_offset, maxreclen);
	  retval++;
	}
      
      /* Check that the data and blockette offsets are within the record */
      if ( b1000reclen && fsdh->data_offset > b1000reclen )
	{
	  ms_log (2, "%s: Data offset (%d) beyond record length (%d)\n", srcname, fsdh->data_offset, b1000reclen);
	  retval++;
	}
      if ( b1000reclen && fsdh->blockette_offset > b1000reclen )
	{
	  ms_log (2, "%s: Blockette offset (%d) beyond record length (%d)\n", srcname, fsdh->blockette_offset, b1000reclen);
	  retval++;
	}
      
      /* Check that the data offset is beyond the end of the blockettes */
      if ( fsdh->numsamples && fsdh->data_offset <= endofblockettes )
	{
	  ms_log (2, "%s: Data offset (%d) is within blockette chain (end of blockettes: %d)\n", srcname, fsdh->data_offset, endofblockettes);
	  retval++;
	}
      
      /* Check that the correct number of blockettes were parsed */
      if ( fsdh->numblockettes != blkt_count )
	{
	  ms_log (2, "%s: Specified number of blockettes (%d) not equal to those parsed (%d)\n", srcname, fsdh->numblockettes, blkt_count);
	  retval++;
	}
    }
  
  return retval;
} /* End of ms_parse_raw() */
