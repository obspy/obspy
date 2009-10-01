/***************************************************************************
 * msrutils.c:
 *
 * Generic routines to operate on Mini-SEED records.
 *
 * Written by Chad Trabant
 *   ORFEUS/EC-Project MEREDIAN
 *   IRIS Data Management Center
 *
 * modified: 2009.175
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "libmseed.h"


/***************************************************************************
 * msr_init:
 *
 * Initialize and return an MSRecord struct, allocating memory if
 * needed.  If memory for the fsdh and datasamples fields has been
 * allocated the pointers will be retained for reuse.  If a blockette
 * chain is present all associated memory will be released.
 *
 * Returns a pointer to a MSRecord struct on success or NULL on error.
 ***************************************************************************/
MSRecord *
msr_init ( MSRecord *msr )
{
  void *fsdh = 0;
  void *datasamples = 0;
  
  if ( ! msr )
    {
      msr = (MSRecord *) malloc (sizeof(MSRecord));
    }
  else
    {
      fsdh = msr->fsdh;
      datasamples = msr->datasamples;
      
      if ( msr->blkts )
        msr_free_blktchain (msr);

      if ( msr->ststate )
	free (msr->ststate);
    }
  
  if ( msr == NULL )
    {
      ms_log (2, "msr_init(): Cannot allocate memory\n");
      return NULL;
    }
  
  memset (msr, 0, sizeof (MSRecord));
  
  msr->fsdh = fsdh;
  msr->datasamples = datasamples;
  
  msr->reclen = -1;
  msr->samplecnt = -1;
  msr->byteorder = -1;
  msr->encoding = -1;
  
  return msr;
} /* End of msr_init() */


/***************************************************************************
 * msr_free:
 *
 * Free all memory associated with a MSRecord struct.
 ***************************************************************************/
void
msr_free ( MSRecord **ppmsr )
{
  if ( ppmsr != NULL && *ppmsr != 0 )
    {      
      /* Free fixed section header if populated */
      if ( (*ppmsr)->fsdh )
        free ((*ppmsr)->fsdh);
      
      /* Free blockette chain if populated */
      if ( (*ppmsr)->blkts )
        msr_free_blktchain (*ppmsr);
      
      /* Free datasamples if present */
      if ( (*ppmsr)->datasamples )
	free ((*ppmsr)->datasamples);
      
      /* Free stream processing state if present */
      if ( (*ppmsr)->ststate )
        free ((*ppmsr)->ststate);

      free (*ppmsr);
      
      *ppmsr = NULL;
    }
} /* End of msr_free() */


/***************************************************************************
 * msr_free_blktchain:
 *
 * Free all memory associated with a blockette chain in a MSRecord
 * struct and set MSRecord->blkts to NULL.  Also reset the shortcut
 * blockette pointers.
 ***************************************************************************/
void
msr_free_blktchain ( MSRecord *msr )
{
  if ( msr )
    {
      if ( msr->blkts )
        {
          BlktLink *bc = msr->blkts;
          BlktLink *nb = NULL;
          
          while ( bc )
	    {
	      nb = bc->next;
	      
	      if ( bc->blktdata )
		free (bc->blktdata);
	      
	      free (bc);
	      
	      bc = nb;
	    }
          
          msr->blkts = 0;
        }

      msr->Blkt100  = 0;
      msr->Blkt1000 = 0;
      msr->Blkt1001 = 0;      
    }
} /* End of msr_free_blktchain() */


/***************************************************************************
 * msr_addblockette:
 *
 * Add a blockette to the blockette chain of an MSRecord.  'blktdata'
 * should be the body of the blockette type 'blkttype' of 'length'
 * bytes without the blockette header (type and next offsets).  The
 * 'chainpos' value controls which end of the chain the blockette is
 * added to.  If 'chainpos' is 0 the blockette will be added to the
 * end of the chain (last blockette), other wise it will be added to
 * the beginning of the chain (first blockette).
 *
 * Returns a pointer to the BlktLink added to the chain on success and
 * NULL on error.
 ***************************************************************************/
BlktLink *
msr_addblockette (MSRecord *msr, char *blktdata, int length, int blkttype,
		  int chainpos)
{
  BlktLink *blkt;
  
  if ( ! msr )
    return NULL;
  
  blkt = msr->blkts;
  
  if ( blkt )
    {
      if ( chainpos != 0 )
	{
	  blkt = (BlktLink *) malloc (sizeof(BlktLink));
	  
	  blkt->next = msr->blkts;
	  msr->blkts = blkt;
	}
      else
	{
	  /* Find the last blockette */
	  while ( blkt && blkt->next )
	    {
	      blkt = blkt->next;
	    }
	  
	  blkt->next = (BlktLink *) malloc (sizeof(BlktLink));
	  
	  blkt = blkt->next;
	  blkt->next = 0;
	}
      
      if ( blkt == NULL )
	{
	  ms_log (2, "msr_addblockette(): Cannot allocate memory\n");
	  return NULL;
	}
    }
  else
    {
      msr->blkts = (BlktLink *) malloc (sizeof(BlktLink));
      
      if ( msr->blkts == NULL )
	{
	  ms_log (2, "msr_addblockette(): Cannot allocate memory\n");
	  return NULL;
	}
      
      blkt = msr->blkts;
      blkt->next = 0;
    }
  
  blkt->blktoffset = 0;
  blkt->blkt_type = blkttype;
  blkt->next_blkt = 0;
  
  blkt->blktdata = (char *) malloc (length);
  
  if ( blkt->blktdata == NULL )
    {
      ms_log (2, "msr_addblockette(): Cannot allocate memory\n");
      return NULL;
    }
  
  memcpy (blkt->blktdata, blktdata, length);
  blkt->blktdatalen = length;
  
  /* Setup the shortcut pointer for common blockettes */
  switch ( blkttype )
    {
    case 100:
      msr->Blkt100 = blkt->blktdata;
      break;
    case 1000:
      msr->Blkt1000 = blkt->blktdata;
      break;
    case 1001:
      msr->Blkt1001 = blkt->blktdata;
      break;
    }
  
  return blkt;
} /* End of msr_addblockette() */


/***************************************************************************
 * msr_normalize_header:
 *
 * Normalize header values between the MSRecord struct and the
 * associated fixed-section of the header and blockettes.  Essentially
 * this updates the SEED structured data in the MSRecord.fsdh struct
 * and MSRecord.blkts chain with values stored at the MSRecord level.
 *
 * Returns the header length in bytes on success or -1 on error.
 ***************************************************************************/
int
msr_normalize_header ( MSRecord *msr, flag verbose )
{
  struct blkt_link_s *cur_blkt;
  char seqnum[7];
  int offset = 0;
  int blktcnt = 0;
  int reclenexp = 0;
  int reclenfind;
  
  if ( ! msr )
    return -1;
  
  /* Update values in fixed section of data header */
  if ( msr->fsdh )
    {
      if ( verbose > 2 )
	ms_log (1, "Normalizing fixed section of data header\n");
      
      /* Roll-over sequence number if necessary */
      if ( msr->sequence_number > 999999 )
	msr->sequence_number = 1;
      
      /* Update values in the MSRecord.fsdh struct */
      snprintf (seqnum, 7, "%06d", msr->sequence_number);
      memcpy (msr->fsdh->sequence_number, seqnum, 6);
      msr->fsdh->dataquality = msr->dataquality;
      msr->fsdh->reserved = ' ';
      ms_strncpopen (msr->fsdh->network, msr->network, 2);
      ms_strncpopen (msr->fsdh->station, msr->station, 5);
      ms_strncpopen (msr->fsdh->location, msr->location, 2);
      ms_strncpopen (msr->fsdh->channel, msr->channel, 3);
      ms_hptime2btime (msr->starttime, &(msr->fsdh->start_time));
      ms_genfactmult (msr->samprate, &(msr->fsdh->samprate_fact), &(msr->fsdh->samprate_mult));
      
      offset += 48;
      
      if ( msr->blkts )
	msr->fsdh->blockette_offset = offset;
      else
	msr->fsdh->blockette_offset = 0;
    }
  
  /* Traverse blockette chain and performs necessary updates*/
  cur_blkt = msr->blkts;
  
  if ( cur_blkt && verbose > 2 )
    ms_log (1, "Normalizing blockette chain\n");
  
  while ( cur_blkt )
    {
      offset += 4;
      
      if ( cur_blkt->blkt_type == 100 && msr->Blkt100 )
	{
	  msr->Blkt100->samprate = msr->samprate;
	  offset += sizeof (struct blkt_100_s);
	}
      else if ( cur_blkt->blkt_type == 1000 && msr->Blkt1000 )
	{
	  msr->Blkt1000->byteorder = msr->byteorder;
	  msr->Blkt1000->encoding = msr->encoding;
	  
	  /* Calculate the record length as an exponent of 2 */
	  for (reclenfind=1, reclenexp=1; reclenfind <= MAXRECLEN; reclenexp++)
	    {
	      reclenfind *= 2;
	      if ( reclenfind == msr->reclen ) break;
	    }
	  
	  if ( reclenfind != msr->reclen )
	    {
	      ms_log (2, "msr_normalize_header(): Record length %d is not a power of 2\n",
		      msr->reclen);
	      return -1;
	    }
	  
	  msr->Blkt1000->reclen = reclenexp;
	  
	  offset += sizeof (struct blkt_1000_s);
	}
      
      else if ( cur_blkt->blkt_type == 1001 )
	{
	  hptime_t sec, usec;
	  
	  /* Insert microseconds offset */
	  sec = msr->starttime / (HPTMODULUS / 10000);
	  usec = msr->starttime - (sec * (HPTMODULUS / 10000));
	  usec /= (HPTMODULUS / 1000000);
	  
	  msr->Blkt1001->usec = (int8_t) usec;
	  offset += sizeof (struct blkt_1001_s);
	}
      
      blktcnt++;
      cur_blkt = cur_blkt->next;
    }

  if ( msr->fsdh )
    msr->fsdh->numblockettes = blktcnt;
  
  return offset;
} /* End of msr_normalize_header() */


/***************************************************************************
 * msr_duplicate:
 *
 * Duplicate an MSRecord struct
 * including the fixed-section data
 * header and blockette chain.  If
 * the datadup flag is true and the
 * source MSRecord has associated
 * data samples copy them as well.
 *
 * Returns a pointer to a new MSRecord on success and NULL on error.
 ***************************************************************************/
MSRecord *
msr_duplicate (MSRecord *msr, flag datadup)
{
  MSRecord *dupmsr = 0;
  int samplesize = 0;
  
  if ( ! msr )
    return NULL;
  
  /* Allocate target MSRecord structure */
  if ( (dupmsr = msr_init (NULL)) == NULL )
    return NULL;
  
  /* Copy MSRecord structure */
  memcpy (dupmsr, msr, sizeof(MSRecord));
  
  /* Copy fixed-section data header structure */
  if ( msr->fsdh )
    {
      /* Allocate memory for new FSDH structure */
      if ( (dupmsr->fsdh = (struct fsdh_s *) malloc (sizeof(struct fsdh_s))) == NULL )
	{
	  ms_log (2, "msr_duplicate(): Error allocating memory\n");
	  free (dupmsr);
	  return NULL;
	}
      
      /* Copy the contents */
      memcpy (dupmsr->fsdh, msr->fsdh, sizeof(struct fsdh_s));
    }
  
  /* Copy the blockette chain */
  if ( msr->blkts )
    {
      BlktLink *blkt = msr->blkts;
      BlktLink *next = NULL;
      
      dupmsr->blkts = 0;
      while ( blkt )
	{
	  next = blkt->next;
	  
	  /* Add blockette to chain of new MSRecord */
	  if ( msr_addblockette (dupmsr, blkt->blktdata, blkt->blktdatalen,
				 blkt->blkt_type, 0) == NULL )
	    {
	      ms_log (2, "msr_duplicate(): Error adding blockettes\n");
	      msr_free (&dupmsr);
	      return NULL;
	    }
	  
	  blkt = next;
	}
    }
  
  /* Copy data samples if requested and available */
  if ( datadup && msr->datasamples )
    {
      /* Determine size of samples in bytes */
      samplesize = ms_samplesize (msr->sampletype);
      
      if ( samplesize == 0 )
	{
	  ms_log (2, "msr_duplicate(): unrecognized sample type: '%c'\n",
		  msr->sampletype);
	  free (dupmsr);
	  return NULL;
	}
      
      /* Allocate memory for new data array */
      if ( (dupmsr->datasamples = (void *) malloc (msr->numsamples * samplesize)) == NULL )
	{
	  ms_log (2, "msr_duplicate(): Error allocating memory\n");
	  free (dupmsr);
	  return NULL;	  
	}
      
      /* Copy the data array */
      memcpy (dupmsr->datasamples, msr->datasamples, (msr->numsamples * samplesize));
    }
  /* Otherwise make sure the sample array and count are zero */
  else
    {
      dupmsr->datasamples = 0;
      dupmsr->numsamples = 0;
    }
  
  return dupmsr;
} /* End of msr_duplicate() */


/***************************************************************************
 * msr_samprate:
 *
 * Calculate and return a double precision sample rate for the
 * specified MSRecord.  If a Blockette 100 was included and parsed,
 * the "Actual sample rate" (field 3) will be returned, otherwise a
 * nominal sample rate will be calculated from the sample rate factor
 * and multiplier in the fixed section data header.
 *
 * Returns the positive sample rate on success and -1.0 on error.
 ***************************************************************************/
double
msr_samprate (MSRecord *msr)
{
  if ( ! msr )
    return -1.0;
  
  if ( msr->Blkt100 )
    return (double) msr->Blkt100->samprate;
  else
    return msr_nomsamprate (msr);  
} /* End of msr_samprate() */


/***************************************************************************
 * msr_nomsamprate:
 *
 * Calculate a double precision nominal sample rate from the sample
 * rate factor and multiplier in the FSDH struct of the specified
 * MSRecord.
 *
 * Returns the positive sample rate on success and -1.0 on error.
 ***************************************************************************/
double
msr_nomsamprate (MSRecord *msr)
{
  if ( ! msr )
    return -1.0;
  
  return ms_nomsamprate (msr->fsdh->samprate_fact, msr->fsdh->samprate_mult);
} /* End of msr_nomsamprate() */


/***************************************************************************
 * msr_starttime:
 *
 * Convert a btime struct of a FSDH struct of a MSRecord (the record
 * start time) into a high precision epoch time and apply time
 * corrections if any are specified in the header and bit 1 of the
 * activity flags indicates that it has not already been applied.  If
 * a Blockette 1001 is included and has been parsed the microseconds
 * of field 4 are also applied.
 *
 * Returns a high precision epoch time on success and HPTERROR on
 * error.
 ***************************************************************************/
hptime_t
msr_starttime (MSRecord *msr)
{
  hptime_t starttime = msr_starttime_uc (msr);
  
  if ( ! msr || starttime == HPTERROR )
    return HPTERROR;
  
  /* Check if a correction is included and if it has been applied,
     bit 1 of activity flags indicates if it has been appiled */
  
  if ( msr->fsdh->time_correct != 0 &&
       ! (msr->fsdh->act_flags & 0x02) )
    {
      starttime += (hptime_t) msr->fsdh->time_correct * (HPTMODULUS / 10000);
    }
  
  /* Apply microsecond precision in a parsed Blockette 1001 */
  if ( msr->Blkt1001 )
    {
      starttime += (hptime_t) msr->Blkt1001->usec * (HPTMODULUS / 1000000);
    }
  
  return starttime;
} /* End of msr_starttime() */


/***************************************************************************
 * msr_starttime_uc:
 *
 * Convert a btime struct of a FSDH struct of a MSRecord (the record
 * start time) into a high precision epoch time.  This time has no
 * correction(s) applied to it.
 *
 * Returns a high precision epoch time on success and HPTERROR on
 * error.
 ***************************************************************************/
hptime_t
msr_starttime_uc (MSRecord *msr)
{
  if ( ! msr )
    return HPTERROR;

  if ( ! msr->fsdh )
    return HPTERROR;
  
  return ms_btime2hptime (&msr->fsdh->start_time);
} /* End of msr_starttime_uc() */


/***************************************************************************
 * msr_endtime:
 *
 * Calculate the time of the last sample in the record; this is the
 * actual last sample time and *not* the time "covered" by the last
 * sample.
 *
 * Returns the time of the last sample as a high precision epoch time
 * on success and HPTERROR on error.
 ***************************************************************************/
hptime_t
msr_endtime (MSRecord *msr)
{
  hptime_t span = 0;
  
  if ( ! msr )
    return HPTERROR;

  if ( msr->samprate > 0.0 && msr->samplecnt > 0 )
    span = ((double) (msr->samplecnt - 1) / msr->samprate * HPTMODULUS) + 0.5;
  
  return (msr->starttime + span);
} /* End of msr_endtime() */


/***************************************************************************
 * msr_srcname:
 *
 * Generate a source name string for a specified MSRecord in the
 * format: 'NET_STA_LOC_CHAN' or, if the quality flag is true:
 * 'NET_STA_LOC_CHAN_QUAL'.  The passed srcname must have enough room
 * for the resulting string.
 *
 * Returns a pointer to the resulting string or NULL on error.
 ***************************************************************************/
char *
msr_srcname (MSRecord *msr, char *srcname, flag quality)
{
  char *src = srcname;
  char *cp = srcname;
  
  if ( ! msr || ! srcname )
    return NULL;
  
  /* Build the source name string */
  cp = msr->network;
  while ( *cp ) { *src++ = *cp++; }
  *src++ = '_';
  cp = msr->station;
  while ( *cp ) { *src++ = *cp++; }  
  *src++ = '_';
  cp = msr->location;
  while ( *cp ) { *src++ = *cp++; }  
  *src++ = '_';
  cp = msr->channel;
  while ( *cp ) { *src++ = *cp++; }  
  
  if ( quality )
    {
      *src++ = '_';
      *src++ = msr->dataquality;
    }
  
  *src = '\0';
  
  return srcname;
} /* End of msr_srcname() */


/***************************************************************************
 * msr_print:
 *
 * Prints header values in an MSRecord struct, if 'details' is greater
 * than 0 then detailed information about each blockette is printed.
 * If 'details' is greater than 1 very detailed information is
 * printed.  If no FSDH (msr->fsdh) is present only a single line with
 * basic information is printed.
 ***************************************************************************/
void
msr_print (MSRecord *msr, flag details)
{
  double nomsamprate;
  char srcname[50];
  char time[25];
  char b;
  int idx;
  
  if ( ! msr )
    return;
  
  /* Generate a source name string */
  srcname[0] = '\0';
  msr_srcname (msr, srcname, 0);
  
  /* Generate a start time string */
  ms_hptime2seedtimestr (msr->starttime, time, 1);
  
  /* Report information in the fixed header */
  if ( details > 0 && msr->fsdh )
    {
      nomsamprate = msr_nomsamprate (msr);
      
      ms_log (0, "%s, %06d, %c\n", srcname, msr->sequence_number, msr->dataquality);
      ms_log (0, "             start time: %s\n", time);
      ms_log (0, "      number of samples: %d\n", msr->fsdh->numsamples);
      ms_log (0, "     sample rate factor: %d  (%.10g samples per second)\n",
	      msr->fsdh->samprate_fact, nomsamprate);
      ms_log (0, " sample rate multiplier: %d\n", msr->fsdh->samprate_mult);
      
      if ( details > 1 )
	{
	  /* Activity flags */
	  b = msr->fsdh->act_flags;
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
	  b = msr->fsdh->io_flags;
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
	  b = msr->fsdh->dq_flags;
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

      ms_log (0, "   number of blockettes: %d\n", msr->fsdh->numblockettes);
      ms_log (0, "        time correction: %ld\n", (long int) msr->fsdh->time_correct);
      ms_log (0, "            data offset: %d\n", msr->fsdh->data_offset);
      ms_log (0, " first blockette offset: %d\n", msr->fsdh->blockette_offset);
    }
  else
    {
      ms_log (0, "%s, %06d, %c, %d, %d samples, %-.10g Hz, %s\n",
	      srcname, msr->sequence_number, msr->dataquality,
	      msr->reclen, msr->samplecnt, msr->samprate, time);
    }

  /* Report information in the blockette chain */
  if ( details > 0 && msr->blkts )
    {
      BlktLink *cur_blkt = msr->blkts;
      
      while ( cur_blkt )
	{
	  if ( cur_blkt->blkt_type == 100 )
	    {
	      struct blkt_100_s *blkt_100 = (struct blkt_100_s *) cur_blkt->blktdata;
	      
	      ms_log (0, "          BLOCKETTE %u: (%s)\n", cur_blkt->blkt_type,
		      ms_blktdesc(cur_blkt->blkt_type));
	      ms_log (0, "              next blockette: %u\n", cur_blkt->next_blkt);
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

	  else if ( cur_blkt->blkt_type == 200 )
	    {
	      struct blkt_200_s *blkt_200 = (struct blkt_200_s *) cur_blkt->blktdata;
	      
	      ms_log (0, "          BLOCKETTE %u: (%s)\n", cur_blkt->blkt_type,
		      ms_blktdesc(cur_blkt->blkt_type));
	      ms_log (0, "              next blockette: %u\n", cur_blkt->next_blkt);
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
	      
	      ms_btime2seedtimestr (&blkt_200->time, time);
	      ms_log (0, "           signal onset time: %s\n", time);
	      ms_log (0, "               detector name: %.24s\n", blkt_200->detector);
	    }

	  else if ( cur_blkt->blkt_type == 201 )
	    {
	      struct blkt_201_s *blkt_201 = (struct blkt_201_s *) cur_blkt->blktdata;
	      
	      ms_log (0, "          BLOCKETTE %u: (%s)\n", cur_blkt->blkt_type,
		      ms_blktdesc(cur_blkt->blkt_type));
	      ms_log (0, "              next blockette: %u\n", cur_blkt->next_blkt);
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
	      ms_btime2seedtimestr (&blkt_201->time, time);
	      ms_log (0, "           signal onset time: %s\n", time);
	      ms_log (0, "                  SNR values: ");
	      for (idx=0; idx < 6; idx++) ms_log (0, "%u  ", blkt_201->snr_values[idx]);
	      ms_log (0, "\n");
	      ms_log (0, "              loopback value: %u\n", blkt_201->loopback);
	      ms_log (0, "              pick algorithm: %u\n", blkt_201->pick_algorithm);
	      ms_log (0, "               detector name: %.24s\n", blkt_201->detector);
	    }

	  else if ( cur_blkt->blkt_type == 300 )
	    {
	      struct blkt_300_s *blkt_300 = (struct blkt_300_s *) cur_blkt->blktdata;
	      
	      ms_log (0, "          BLOCKETTE %u: (%s)\n", cur_blkt->blkt_type,
		      ms_blktdesc(cur_blkt->blkt_type));
	      ms_log (0, "              next blockette: %u\n", cur_blkt->next_blkt);
	      ms_btime2seedtimestr (&blkt_300->time, time);
	      ms_log (0, "      calibration start time: %s\n", time);
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
	  
	  else if ( cur_blkt->blkt_type == 310 )
	    {
	      struct blkt_310_s *blkt_310 = (struct blkt_310_s *) cur_blkt->blktdata;
	      
	      ms_log (0, "          BLOCKETTE %u: (%s)\n", cur_blkt->blkt_type,
		      ms_blktdesc(cur_blkt->blkt_type));
	      ms_log (0, "              next blockette: %u\n", cur_blkt->next_blkt);
	      ms_btime2seedtimestr (&blkt_310->time, time);
	      ms_log (0, "      calibration start time: %s\n", time);
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

	  else if ( cur_blkt->blkt_type == 320 )
	    {
	      struct blkt_320_s *blkt_320 = (struct blkt_320_s *) cur_blkt->blktdata;
	      
	      ms_log (0, "          BLOCKETTE %u: (%s)\n", cur_blkt->blkt_type,
		      ms_blktdesc(cur_blkt->blkt_type));
	      ms_log (0, "              next blockette: %u\n", cur_blkt->next_blkt);
	      ms_btime2seedtimestr (&blkt_320->time, time);
	      ms_log (0, "      calibration start time: %s\n", time);
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
	  
	  else if ( cur_blkt->blkt_type == 390 )
	    {
	      struct blkt_390_s *blkt_390 = (struct blkt_390_s *) cur_blkt->blktdata;
	      
	      ms_log (0, "          BLOCKETTE %u: (%s)\n", cur_blkt->blkt_type,
		      ms_blktdesc(cur_blkt->blkt_type));
	      ms_log (0, "              next blockette: %u\n", cur_blkt->next_blkt);
	      ms_btime2seedtimestr (&blkt_390->time, time);
	      ms_log (0, "      calibration start time: %s\n", time);
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

	  else if ( cur_blkt->blkt_type == 395 )
	    {
	      struct blkt_395_s *blkt_395 = (struct blkt_395_s *) cur_blkt->blktdata;
	      
	      ms_log (0, "          BLOCKETTE %u: (%s)\n", cur_blkt->blkt_type,
		      ms_blktdesc(cur_blkt->blkt_type));
	      ms_log (0, "              next blockette: %u\n", cur_blkt->next_blkt);
	      ms_btime2seedtimestr (&blkt_395->time, time);
	      ms_log (0, "        calibration end time: %s\n", time);
	      if ( details > 1 )
		ms_log (0, "          reserved bytes (2): %u,%u\n",
			blkt_395->reserved[0], blkt_395->reserved[1]);
	    }

	  else if ( cur_blkt->blkt_type == 400 )
	    {
	      struct blkt_400_s *blkt_400 = (struct blkt_400_s *) cur_blkt->blktdata;
	      
	      ms_log (0, "          BLOCKETTE %u: (%s)\n", cur_blkt->blkt_type,
		      ms_blktdesc(cur_blkt->blkt_type));
	      ms_log (0, "              next blockette: %u\n", cur_blkt->next_blkt);
	      ms_log (0, "      beam azimuth (degrees): %g\n", blkt_400->azimuth);
	      ms_log (0, "  beam slowness (sec/degree): %g\n", blkt_400->slowness);
	      ms_log (0, "               configuration: %u\n", blkt_400->configuration);
	      if ( details > 1 )
		ms_log (0, "          reserved bytes (2): %u,%u\n",
			blkt_400->reserved[0], blkt_400->reserved[1]);
	    }

	  else if ( cur_blkt->blkt_type == 405 )
	    {
	      struct blkt_405_s *blkt_405 = (struct blkt_405_s *) cur_blkt->blktdata;
	      
	      ms_log (0, "          BLOCKETTE %u: (%s, incomplete)\n", cur_blkt->blkt_type,
		      ms_blktdesc(cur_blkt->blkt_type));
	      ms_log (0, "              next blockette: %u\n", cur_blkt->next_blkt);
	      ms_log (0, "           first delay value: %u\n", blkt_405->delay_values[0]);
	    }

	  else if ( cur_blkt->blkt_type == 500 )
	    {
	      struct blkt_500_s *blkt_500 = (struct blkt_500_s *) cur_blkt->blktdata;
	      
	      ms_log (0, "          BLOCKETTE %u: (%s)\n", cur_blkt->blkt_type,
		      ms_blktdesc(cur_blkt->blkt_type));
	      ms_log (0, "              next blockette: %u\n", cur_blkt->next_blkt);
	      ms_log (0, "              VCO correction: %g%%\n", blkt_500->vco_correction);
	      ms_btime2seedtimestr (&blkt_500->time, time);
	      ms_log (0, "           time of exception: %s\n", time);
	      ms_log (0, "                        usec: %d\n", blkt_500->usec);
	      ms_log (0, "           reception quality: %u%%\n", blkt_500->reception_qual);
	      ms_log (0, "             exception count: %u\n", blkt_500->exception_count);
	      ms_log (0, "              exception type: %.16s\n", blkt_500->exception_type);
	      ms_log (0, "                 clock model: %.32s\n", blkt_500->clock_model);
	      ms_log (0, "                clock status: %.128s\n", blkt_500->clock_status);
	    }
	  
	  else if ( cur_blkt->blkt_type == 1000 )
	    {
	      struct blkt_1000_s *blkt_1000 = (struct blkt_1000_s *) cur_blkt->blktdata;
	      int recsize;
	      char order[40];
	      
	      /* Calculate record size in bytes as 2^(blkt_1000->rec_len) */
	      recsize = (unsigned int) 1 << blkt_1000->reclen;
	      
	      /* Big or little endian? */
	      if (blkt_1000->byteorder == 0)
		strncpy (order, "Little endian", sizeof(order)-1);
	      else if (blkt_1000->byteorder == 1)
		strncpy (order, "Big endian", sizeof(order)-1);
	      else
		strncpy (order, "Unknown value", sizeof(order)-1);
	      
	      ms_log (0, "         BLOCKETTE %u: (%s)\n", cur_blkt->blkt_type,
		      ms_blktdesc(cur_blkt->blkt_type));
	      ms_log (0, "              next blockette: %u\n", cur_blkt->next_blkt);
	      ms_log (0, "                    encoding: %s (val:%u)\n",
		      (char *) ms_encodingstr (blkt_1000->encoding), blkt_1000->encoding);
	      ms_log (0, "                  byte order: %s (val:%u)\n",
		      order, blkt_1000->byteorder);
	      ms_log (0, "               record length: %d (val:%u)\n",
		      recsize, blkt_1000->reclen);
	      
	      if ( details > 1 )
		ms_log (0, "               reserved byte: %u\n", blkt_1000->reserved);
	    }
	  
	  else if ( cur_blkt->blkt_type == 1001 )
	    {
	      struct blkt_1001_s *blkt_1001 = (struct blkt_1001_s *) cur_blkt->blktdata;
	      
	      ms_log (0, "         BLOCKETTE %u: (%s)\n", cur_blkt->blkt_type,
		      ms_blktdesc(cur_blkt->blkt_type));
	      ms_log (0, "              next blockette: %u\n", cur_blkt->next_blkt);
	      ms_log (0, "              timing quality: %u%%\n", blkt_1001->timing_qual);
	      ms_log (0, "                micro second: %d\n", blkt_1001->usec);
	      
	      if ( details > 1 )
		ms_log (0, "               reserved byte: %u\n", blkt_1001->reserved);
	      
	      ms_log (0, "                 frame count: %u\n", blkt_1001->framecnt);
	    }

	  else if ( cur_blkt->blkt_type == 2000 )
	    {
	      struct blkt_2000_s *blkt_2000 = (struct blkt_2000_s *) cur_blkt->blktdata;
	      char order[40];
	      
	      /* Big or little endian? */
	      if (blkt_2000->byteorder == 0)
		strncpy (order, "Little endian", sizeof(order)-1);
	      else if (blkt_2000->byteorder == 1)
		strncpy (order, "Big endian", sizeof(order)-1);
	      else
		strncpy (order, "Unknown value", sizeof(order)-1);
	      
	      ms_log (0, "         BLOCKETTE %u: (%s)\n", cur_blkt->blkt_type,
		      ms_blktdesc(cur_blkt->blkt_type));
	      ms_log (0, "              next blockette: %u\n", cur_blkt->next_blkt);
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
	  
	  else
	    {
	      ms_log (0, "         BLOCKETTE %u: (%s, not parsed)\n", cur_blkt->blkt_type,
		      ms_blktdesc(cur_blkt->blkt_type));
	      ms_log (0, "              next blockette: %u\n", cur_blkt->next_blkt);
	    }
	  
	  cur_blkt = cur_blkt->next;
	}
    }
} /* End of msr_print() */


/***************************************************************************
 * msr_host_latency:
 *
 * Calculate the latency based on the host time in UTC accounting for
 * the time covered using the number of samples and sample rate; in
 * other words, the difference between the host time and the time of
 * the last sample in the specified Mini-SEED record.
 *
 * Double precision is returned, but the true precision is dependent
 * on the accuracy of the host system clock among other things.
 *
 * Returns seconds of latency or 0.0 on error (indistinguishable from
 * 0.0 latency).
 ***************************************************************************/
double
msr_host_latency (MSRecord *msr)
{
  double span = 0.0;            /* Time covered by the samples */
  double epoch;                 /* Current epoch time */
  double latency = 0.0;
  time_t tv;

  if ( msr == NULL )
    return 0.0;
  
  /* Calculate the time covered by the samples */
  if ( msr->samprate > 0.0 && msr->samplecnt > 0 )
    span = (1.0 / msr->samprate) * (msr->samplecnt - 1);
  
  /* Grab UTC time according to the system clock */
  epoch = (double) time(&tv);
  
  /* Now calculate the latency */
  latency = epoch - ((double) msr->starttime / HPTMODULUS) - span;
  
  return latency;
} /* End of msr_host_latency() */
