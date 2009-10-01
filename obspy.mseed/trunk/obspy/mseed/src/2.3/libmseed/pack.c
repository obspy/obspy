/***************************************************************************
 * pack.c:
 *
 * Generic routines to pack Mini-SEED records using an MSrecord as a
 * header template and data source.
 *
 * Written by Chad Trabant,
 *   IRIS Data Management Center
 *
 * modified: 2008.220
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "libmseed.h"
#include "packdata.h"

/* Function(s) internal to this file */
static int msr_pack_header_raw (MSRecord *msr, char *rawrec, int maxheaderlen,
				flag swapflag, flag normalize, flag verbose);
static int msr_update_header (MSRecord * msr, char *rawrec, flag swapflag,
			      flag verbose);
static int msr_pack_data (void *dest, void *src, int maxsamples, int maxdatabytes,
			  int *packsamples, int32_t *lastintsample, flag comphistory,
			  char sampletype, flag encoding, flag swapflag,
			  flag verbose);

/* Header and data byte order flags controlled by environment variables */
/* -2 = not checked, -1 = checked but not set, or 0 = LE and 1 = BE */
flag packheaderbyteorder = -2;
flag packdatabyteorder = -2;

/* A pointer to the srcname of the record being packed */
char *PACK_SRCNAME = NULL;


/***************************************************************************
 * msr_pack:
 *
 * Pack data into SEED data records.  Using the record header values
 * in the MSRecord as a template the common header fields are packed
 * into the record header, blockettes in the blockettes chain are
 * packed and data samples are packed in the encoding format indicated
 * by the MSRecord->encoding field.  A Blockette 1000 will be added if
 * one is not present.
 *
 * The MSRecord->datasamples array and MSRecord->numsamples value will
 * not be changed by this routine.  It is the responsibility of the
 * calling routine to adjust the data buffer if desired.
 *
 * As each record is filled and finished they are passed to
 * record_handler which expects 1) a char * to the record, 2) the
 * length of the record and 3) a pointer supplied by the original
 * caller containing optional private data (handlerdata).  It is the
 * responsibility of record_handler to process the record, the memory
 * will be re-used or freed when record_handler returns.
 *
 * If the flush flag != 0 all of the data will be packed into data
 * records even though the last one will probably not be filled.
 *
 * Default values are: data record & quality indicator = 'D', record
 * length = 4096, encoding = 11 (Steim2) and byteorder = 1 (MSBF).
 * The defaults are triggered when the the msr->dataquality is 0 or
 * msr->reclen, msr->encoding and msr->byteorder are -1 respectively.
 *
 * Returns the number of records created on success and -1 on error.
 ***************************************************************************/
int
msr_pack ( MSRecord * msr, void (*record_handler) (char *, int, void *),
	   void *handlerdata, int *packedsamples, flag flush, flag verbose )
{
  uint16_t *HPnumsamples;
  uint16_t *HPdataoffset;
  char *rawrec;
  char *envvariable;
  char srcname[50];
  
  flag headerswapflag = 0;
  flag dataswapflag = 0;
  flag packret;
  
  int samplesize;
  int headerlen;
  int dataoffset;
  int maxdatabytes;
  int maxsamples;
  int recordcnt = 0;
  int totalpackedsamples;
  int packsamples, packoffset;
  
  if ( ! msr )
    return -1;
  
  if ( ! record_handler )
    {
      ms_log (2, "msr_pack(): record_handler() function pointer not set!\n");
      return -1;
    }

  /* Allocate stream processing state space if needed */
  if ( ! msr->ststate )
    {
      msr->ststate = (StreamState *) malloc (sizeof(StreamState));
      if ( ! msr->ststate )
        {
          ms_log (2, "msr_pack(): Could not allocate memory for StreamState\n");
          return -1;
        }
      memset (msr->ststate, 0, sizeof(StreamState));
    }

  /* Generate source name for MSRecord */
  if ( msr_srcname (msr, srcname, 1) == NULL )
    {
      ms_log (2, "msr_unpack_data(): Cannot generate srcname\n");
      return MS_GENERROR;
    }
  
  /* Set shared srcname pointer to source name */
  PACK_SRCNAME = &srcname[0];
  
  /* Read possible environmental variables that force byteorder */
  if ( packheaderbyteorder == -2 )
    {
      if ( (envvariable = getenv("PACK_HEADER_BYTEORDER")) )
	{
	  if ( *envvariable != '0' && *envvariable != '1' )
	    {
	      ms_log (2, "Environment variable PACK_HEADER_BYTEORDER must be set to '0' or '1'\n");
	      return -1;
	    }
	  else if ( *envvariable == '0' )
	    {
	      packheaderbyteorder = 0;
	      if ( verbose > 2 )
		ms_log (1, "PACK_HEADER_BYTEORDER=0, packing little-endian header\n");
	    }
	  else
	    {
	      packheaderbyteorder = 1;
	      if ( verbose > 2 )
		ms_log (1, "PACK_HEADER_BYTEORDER=1, packing big-endian header\n");
	    }
	}
      else
	{
	  packheaderbyteorder = -1;
	}
    }
  if ( packdatabyteorder == -2 )
    {
      if ( (envvariable = getenv("PACK_DATA_BYTEORDER")) )
	{
	  if ( *envvariable != '0' && *envvariable != '1' )
	    {
	      ms_log (2, "Environment variable PACK_DATA_BYTEORDER must be set to '0' or '1'\n");
	      return -1;
	    }
	  else if ( *envvariable == '0' )
	    {
	      packdatabyteorder = 0;
	      if ( verbose > 2 )
		ms_log (1, "PACK_DATA_BYTEORDER=0, packing little-endian data samples\n");
	    }
	  else
	    {
	      packdatabyteorder = 1;
	      if ( verbose > 2 )
		ms_log (1, "PACK_DATA_BYTEORDER=1, packing big-endian data samples\n");
	    }
	}
      else
	{
	  packdatabyteorder = -1;
	}
    }

  /* Set default indicator, record length, byte order and encoding if needed */
  if ( msr->dataquality == 0 ) msr->dataquality = 'D';
  if ( msr->reclen == -1 ) msr->reclen = 4096;
  if ( msr->byteorder == -1 )  msr->byteorder = 1;
  if ( msr->encoding == -1 ) msr->encoding = DE_STEIM2;
  
  /* Cleanup/reset sequence number */
  if ( msr->sequence_number <= 0 || msr->sequence_number > 999999)
    msr->sequence_number = 1;
  
  if ( msr->reclen < MINRECLEN || msr->reclen > MAXRECLEN )
    {
      ms_log (2, "msr_pack(%s): Record length is out of range: %d\n",
	      PACK_SRCNAME, msr->reclen);
      return -1;
    }
  
  if ( msr->numsamples <= 0 )
    {
      ms_log (2, "msr_pack(%s): No samples to pack\n", PACK_SRCNAME);
      return -1;
    }
  
  samplesize = ms_samplesize (msr->sampletype);
  
  if ( ! samplesize )
    {
      ms_log (2, "msr_pack(%s): Unknown sample type '%c'\n",
	      PACK_SRCNAME, msr->sampletype);
      return -1;
    }
  
  /* Sanity check for msr/quality indicator */
  if ( ! MS_ISDATAINDICATOR(msr->dataquality) )
    {
      ms_log (2, "msr_pack(%s): Record header & quality indicator unrecognized: '%c'\n",
	      PACK_SRCNAME, msr->dataquality);
      ms_log (2, "msr_pack(%s): Packing failed.\n", PACK_SRCNAME);
      return -1;
    }
  
  /* Allocate space for data record */
  rawrec = (char *) malloc (msr->reclen);
  
  if ( rawrec == NULL )
    {
      ms_log (2, "msr_pack(%s): Cannot allocate memory\n", PACK_SRCNAME);
      return -1;
    }
  
  /* Set header pointers to known offsets into FSDH */
  HPnumsamples = (uint16_t *) (rawrec + 30);
  HPdataoffset = (uint16_t *) (rawrec + 44);
  
  /* Check to see if byte swapping is needed */
  if ( msr->byteorder != ms_bigendianhost() )
    headerswapflag = dataswapflag = 1;
  
  /* Check if byte order is forced */
  if ( packheaderbyteorder >= 0 )
    {
      headerswapflag = ( msr->byteorder != packheaderbyteorder ) ? 1 : 0;
    }
  
  if ( packdatabyteorder >= 0 )
    {
      dataswapflag = ( msr->byteorder != packdatabyteorder ) ? 1 : 0;
    }
  
  if ( verbose > 2 )
    {
      if ( headerswapflag && dataswapflag )
	ms_log (1, "%s: Byte swapping needed for packing of header and data samples\n", PACK_SRCNAME);
      else if ( headerswapflag )
	ms_log (1, "%s: Byte swapping needed for packing of header\n", PACK_SRCNAME);
      else if ( dataswapflag )
	ms_log (1, "%s: Byte swapping needed for packing of data samples\n", PACK_SRCNAME);
      else
	ms_log (1, "%s: Byte swapping NOT needed for packing\n", PACK_SRCNAME);
    }
  
  /* Add a blank 1000 Blockette if one is not present, the blockette values
     will be populated in msr_pack_header_raw()/msr_normalize_header() */
  if ( ! msr->Blkt1000 )
    {
      struct blkt_1000_s blkt1000;
      memset (&blkt1000, 0, sizeof (struct blkt_1000_s));
      
      if ( verbose > 2 )
	ms_log (1, "%s: Adding 1000 Blockette\n", PACK_SRCNAME);
      
      if ( ! msr_addblockette (msr, (char *) &blkt1000, sizeof(struct blkt_1000_s), 1000, 0) )
	{
	  ms_log (2, "msr_pack(%s): Error adding 1000 Blockette\n", PACK_SRCNAME);
	  return -1;
	}
    }
  
  headerlen = msr_pack_header_raw (msr, rawrec, msr->reclen, headerswapflag, 1, verbose);
  
  if ( headerlen == -1 )
    {
      ms_log (2, "msr_pack(%s): Error packing header\n", PACK_SRCNAME);
      return -1;
    }

  /* Determine offset to encoded data */
  if ( msr->encoding == DE_STEIM1 || msr->encoding == DE_STEIM2 )
    {
      dataoffset = 64;
      while ( dataoffset < headerlen )
	dataoffset += 64;
      
      /* Zero memory between blockettes and data if any */
      memset (rawrec + headerlen, 0, dataoffset - headerlen);
    }
  else
    {
      dataoffset = headerlen;
    }
  
  *HPdataoffset = (uint16_t) dataoffset;
  if ( headerswapflag ) ms_gswap2 (HPdataoffset);
  
  /* Determine the max data bytes and sample count */
  maxdatabytes = msr->reclen - dataoffset;
  
  if ( msr->encoding == DE_STEIM1 )
    {
      maxsamples = (int) (maxdatabytes/64) * STEIM1_FRAME_MAX_SAMPLES;
    }
  else if ( msr->encoding == DE_STEIM2 )
    {
      maxsamples = (int) (maxdatabytes/64) * STEIM2_FRAME_MAX_SAMPLES;
    }
  else
    {
      maxsamples = maxdatabytes / samplesize;
    }
  
  /* Pack samples into records */
  *HPnumsamples = 0;
  totalpackedsamples = 0;
  if ( packedsamples ) *packedsamples = 0;
  packoffset = 0;
  
  while ( (msr->numsamples - totalpackedsamples) > maxsamples || flush )
    {
      packret = msr_pack_data (rawrec + dataoffset,
			       (char *) msr->datasamples + packoffset,
			       (msr->numsamples - totalpackedsamples), maxdatabytes,
			       &packsamples, &msr->ststate->lastintsample, msr->ststate->comphistory,
			       msr->sampletype, msr->encoding, dataswapflag, verbose);
      
      if ( packret )
	{
	  ms_log (2, "msr_pack(%s): Error packing record\n", PACK_SRCNAME);
	  return -1;
	}
      
      packoffset += packsamples * samplesize;
      
      /* Update number of samples */
      *HPnumsamples = (uint16_t) packsamples;
      if ( headerswapflag ) ms_gswap2 (HPnumsamples);
      
      if ( verbose > 0 )
	ms_log (1, "%s: Packed %d samples\n", PACK_SRCNAME, packsamples);
      
      /* Send record to handler */
      record_handler (rawrec, msr->reclen, handlerdata);
      
      totalpackedsamples += packsamples;
      if ( packedsamples ) *packedsamples = totalpackedsamples;
      msr->ststate->packedsamples += packsamples;
      
      /* Update record header for next record */
      msr->sequence_number = ( msr->sequence_number >= 999999 ) ? 1 : msr->sequence_number + 1;
      if ( msr->samprate > 0 )
        msr->starttime += (double) packsamples / msr->samprate * HPTMODULUS;
      msr_update_header (msr, rawrec, headerswapflag, verbose);
      
      recordcnt++;
      msr->ststate->packedrecords++;

      /* Set compression history flag for subsequent records (Steim encodings) */
      if ( ! msr->ststate->comphistory )
        msr->ststate->comphistory = 1;
     
      if ( totalpackedsamples >= msr->numsamples )
	break;
    }
  
  if ( verbose > 2 )
    ms_log (1, "%s: Packed %d total samples\n", PACK_SRCNAME, totalpackedsamples);
  
  free (rawrec);
  
  return recordcnt;
} /* End of msr_pack() */


/***************************************************************************
 * msr_pack_header:
 *
 * Pack data header/blockettes into the SEED record at
 * MSRecord->record.  Unlike msr_pack no default values are applied,
 * the header structures are expected to be self describing and no
 * Blockette 1000 will be added.  This routine is only useful for
 * re-packing a record header.
 *
 * Returns the header length in bytes on success and -1 on error.
 ***************************************************************************/
int
msr_pack_header ( MSRecord *msr, flag normalize, flag verbose )
{
  char srcname[50];
  char *envvariable;
  flag headerswapflag = 0;
  int headerlen;
  int maxheaderlen;
  
  if ( ! msr )
    return -1;
  
  /* Generate source name for MSRecord */
  if ( msr_srcname (msr, srcname, 1) == NULL )
    {
      ms_log (2, "msr_unpack_data(): Cannot generate srcname\n");
      return MS_GENERROR;
    }
  
  /* Set shared srcname pointer to source name */
  PACK_SRCNAME = &srcname[0];

  /* Read possible environmental variables that force byteorder */
  if ( packheaderbyteorder == -2 )
    {
      if ( (envvariable = getenv("PACK_HEADER_BYTEORDER")) )
	{
	  if ( *envvariable != '0' && *envvariable != '1' )
	    {
	      ms_log (2, "Environment variable PACK_HEADER_BYTEORDER must be set to '0' or '1'\n");
	      return -1;
	    }
	  else if ( *envvariable == '0' )
	    {
	      packheaderbyteorder = 0;
	      if ( verbose > 2 )
		ms_log (1, "PACK_HEADER_BYTEORDER=0, packing little-endian header\n");
	    }
	  else
	    {
	      packheaderbyteorder = 1;
	      if ( verbose > 2 )
		ms_log (1, "PACK_HEADER_BYTEORDER=1, packing big-endian header\n");
	    }
	}
      else
	{
	  packheaderbyteorder = -1;
	}
    }

  if ( msr->reclen < MINRECLEN || msr->reclen > MAXRECLEN )
    {
      ms_log (2, "msr_pack_header(%s): record length is out of range: %d\n",
	      PACK_SRCNAME, msr->reclen);
      return -1;
    }
  
  if ( msr->byteorder != 0 && msr->byteorder != 1 )
    {
      ms_log (2, "msr_pack_header(%s): byte order is not defined correctly: %d\n",
	      PACK_SRCNAME, msr->byteorder);
      return -1;
    }
    
  if ( msr->fsdh )
    {
      maxheaderlen = (msr->fsdh->data_offset > 0) ?
	msr->fsdh->data_offset :
	msr->reclen;
    }
  else
    {
      maxheaderlen = msr->reclen;
    }
    
  /* Check to see if byte swapping is needed */
  if ( msr->byteorder != ms_bigendianhost() )
    headerswapflag = 1;
  
  /* Check if byte order is forced */
  if ( packheaderbyteorder >= 0 )
    {
      headerswapflag = ( msr->byteorder != packheaderbyteorder ) ? 1: 0;
    }
  
  if ( verbose > 2 )
    {
      if ( headerswapflag )
	ms_log (1, "%s: Byte swapping needed for packing of header\n", PACK_SRCNAME);
      else
	ms_log (1, "%s: Byte swapping NOT needed for packing of header\n", PACK_SRCNAME);
    }
  
  headerlen = msr_pack_header_raw (msr, msr->record, maxheaderlen,
				   headerswapflag, normalize, verbose);
  
  return headerlen;
}  /* End of msr_pack_header() */


/***************************************************************************
 * msr_pack_header_raw:
 *
 * Pack data header/blockettes into the specified SEED data record.
 *
 * Returns the header length in bytes on success or -1 on error.
 ***************************************************************************/
static int
msr_pack_header_raw ( MSRecord *msr, char *rawrec, int maxheaderlen,
		      flag swapflag, flag normalize, flag verbose )
{
  struct blkt_link_s *cur_blkt;
  struct fsdh_s *fsdh;
  int16_t offset;
  int blktcnt = 0;
  int nextoffset;

  if ( ! msr || ! rawrec )
    return -1;
  
  /* Make sure a fixed section of data header is available */
  if ( ! msr->fsdh )
    {
      msr->fsdh = (struct fsdh_s *) calloc (1, sizeof (struct fsdh_s));
      
      if ( msr->fsdh == NULL )
	{
	  ms_log (2, "msr_pack_header_raw(%s): Cannot allocate memory\n",
		  PACK_SRCNAME);
	  return -1;
	}
    }
  
  /* Update the SEED structures associated with the MSRecord */
  if ( normalize )
    if ( msr_normalize_header (msr, verbose) < 0 )
      {
	ms_log (2, "msr_pack_header_raw(%s): error normalizing header values\n",
		PACK_SRCNAME);
	return -1;
      }
  
  if ( verbose > 2 )
    ms_log (1, "%s: Packing fixed section of data header\n", PACK_SRCNAME);
  
  if ( maxheaderlen > msr->reclen )
    {
      ms_log (2, "msr_pack_header_raw(%s): maxheaderlen of %d is beyond record length of %d\n",
	      PACK_SRCNAME, maxheaderlen, msr->reclen);
      return -1;
    }
  
  if ( maxheaderlen < sizeof(struct fsdh_s) )
    {
      ms_log (2, "msr_pack_header_raw(%s): maxheaderlen of %d is too small, must be >= %d\n",
	      PACK_SRCNAME, maxheaderlen, sizeof(struct fsdh_s));
      return -1;
    }
  
  fsdh = (struct fsdh_s *) rawrec;
  offset = 48;
  
  /* Roll-over sequence number if necessary */
  if ( msr->sequence_number > 999999 )
    msr->sequence_number = 1;
  
  /* Copy FSDH associated with the MSRecord into the record */  
  memcpy (fsdh, msr->fsdh, sizeof(struct fsdh_s));
  
  /* Swap byte order? */
  if ( swapflag )
    {
      MS_SWAPBTIME (&fsdh->start_time);
      ms_gswap2 (&fsdh->numsamples);
      ms_gswap2 (&fsdh->samprate_fact);
      ms_gswap2 (&fsdh->samprate_mult);
      ms_gswap4 (&fsdh->time_correct);
      ms_gswap2 (&fsdh->data_offset);
      ms_gswap2 (&fsdh->blockette_offset);
    }
  
  /* Traverse blockette chain and pack blockettes at 'offset' */
  cur_blkt = msr->blkts;
  
  while ( cur_blkt && offset < maxheaderlen )
    {
      /* Check that the blockette fits */
      if ( (offset + 4 + cur_blkt->blktdatalen) > maxheaderlen )
	{
	  ms_log (2, "msr_pack_header_raw(%s): header exceeds maxheaderlen of %d\n",
		  PACK_SRCNAME, maxheaderlen);
	  break;
	}
      
      /* Pack blockette type and leave space for next offset */
      memcpy (rawrec + offset, &cur_blkt->blkt_type, 2);
      if ( swapflag ) ms_gswap2 (rawrec + offset);
      nextoffset = offset + 2;
      offset += 4;
      
      if ( cur_blkt->blkt_type == 100 )
	{
	  struct blkt_100_s *blkt_100 = (struct blkt_100_s *) (rawrec + offset);
	  memcpy (blkt_100, cur_blkt->blktdata, sizeof (struct blkt_100_s));
	  offset += sizeof (struct blkt_100_s);
	  
	  if ( swapflag )
	    {
	      ms_gswap4 (&blkt_100->samprate);
	    }
	}
      
      else if ( cur_blkt->blkt_type == 200 )
	{
	  struct blkt_200_s *blkt_200 = (struct blkt_200_s *) (rawrec + offset);
	  memcpy (blkt_200, cur_blkt->blktdata, sizeof (struct blkt_200_s));
	  offset += sizeof (struct blkt_200_s);
	  
	  if ( swapflag )
	    {
	      ms_gswap4 (&blkt_200->amplitude);
	      ms_gswap4 (&blkt_200->period);
	      ms_gswap4 (&blkt_200->background_estimate);
	      MS_SWAPBTIME (&blkt_200->time);
	    }
	}
      
      else if ( cur_blkt->blkt_type == 201 )
	{
	  struct blkt_201_s *blkt_201 = (struct blkt_201_s *) (rawrec + offset);
	  memcpy (blkt_201, cur_blkt->blktdata, sizeof (struct blkt_201_s));
	  offset += sizeof (struct blkt_201_s);
	  
	  if ( swapflag )
	    {
	      ms_gswap4 (&blkt_201->amplitude);
	      ms_gswap4 (&blkt_201->period);
	      ms_gswap4 (&blkt_201->background_estimate);
	      MS_SWAPBTIME (&blkt_201->time);
	    }
	}

      else if ( cur_blkt->blkt_type == 300 )
	{
	  struct blkt_300_s *blkt_300 = (struct blkt_300_s *) (rawrec + offset);
	  memcpy (blkt_300, cur_blkt->blktdata, sizeof (struct blkt_300_s));
	  offset += sizeof (struct blkt_300_s);
	  
	  if ( swapflag )
	    {
	      MS_SWAPBTIME (&blkt_300->time);
	      ms_gswap4 (&blkt_300->step_duration);
	      ms_gswap4 (&blkt_300->interval_duration);
	      ms_gswap4 (&blkt_300->amplitude);
	      ms_gswap4 (&blkt_300->reference_amplitude);
	    }
	}

      else if ( cur_blkt->blkt_type == 310 )
	{
	  struct blkt_310_s *blkt_310 = (struct blkt_310_s *) (rawrec + offset);
	  memcpy (blkt_310, cur_blkt->blktdata, sizeof (struct blkt_310_s));
	  offset += sizeof (struct blkt_310_s);
	  
	  if ( swapflag )
	    {
	      MS_SWAPBTIME (&blkt_310->time);
	      ms_gswap4 (&blkt_310->duration);
	      ms_gswap4 (&blkt_310->period);
	      ms_gswap4 (&blkt_310->amplitude);
	      ms_gswap4 (&blkt_310->reference_amplitude);
	    }
	}
      
      else if ( cur_blkt->blkt_type == 320 )
	{
	  struct blkt_320_s *blkt_320 = (struct blkt_320_s *) (rawrec + offset);
	  memcpy (blkt_320, cur_blkt->blktdata, sizeof (struct blkt_320_s));
	  offset += sizeof (struct blkt_320_s);
	  
	  if ( swapflag )
	    {
	      MS_SWAPBTIME (&blkt_320->time);
	      ms_gswap4 (&blkt_320->duration);
	      ms_gswap4 (&blkt_320->ptp_amplitude);
	      ms_gswap4 (&blkt_320->reference_amplitude);
	    }
	}

      else if ( cur_blkt->blkt_type == 390 )
	{
	  struct blkt_390_s *blkt_390 = (struct blkt_390_s *) (rawrec + offset);
	  memcpy (blkt_390, cur_blkt->blktdata, sizeof (struct blkt_390_s));
	  offset += sizeof (struct blkt_390_s);
	  
	  if ( swapflag )
	    {
	      MS_SWAPBTIME (&blkt_390->time);
	      ms_gswap4 (&blkt_390->duration);
	      ms_gswap4 (&blkt_390->amplitude);
	    }
	}
      
      else if ( cur_blkt->blkt_type == 395 )
	{
	  struct blkt_395_s *blkt_395 = (struct blkt_395_s *) (rawrec + offset);
	  memcpy (blkt_395, cur_blkt->blktdata, sizeof (struct blkt_395_s));
	  offset += sizeof (struct blkt_395_s);
	  
	  if ( swapflag )
	    {
	      MS_SWAPBTIME (&blkt_395->time);
	    }
	}

      else if ( cur_blkt->blkt_type == 400 )
	{
	  struct blkt_400_s *blkt_400 = (struct blkt_400_s *) (rawrec + offset);
	  memcpy (blkt_400, cur_blkt->blktdata, sizeof (struct blkt_400_s));
	  offset += sizeof (struct blkt_400_s);
	  
	  if ( swapflag )
	    {
	      ms_gswap4 (&blkt_400->azimuth);
	      ms_gswap4 (&blkt_400->slowness);
	      ms_gswap2 (&blkt_400->configuration);
	    }
	}

      else if ( cur_blkt->blkt_type == 405 )
	{
	  struct blkt_405_s *blkt_405 = (struct blkt_405_s *) (rawrec + offset);
	  memcpy (blkt_405, cur_blkt->blktdata, sizeof (struct blkt_405_s));
	  offset += sizeof (struct blkt_405_s);
	  
	  if ( swapflag )
	    {
	      ms_gswap2 (&blkt_405->delay_values);
	    }

	  if ( verbose > 0 )
	    {
	      ms_log (1, "msr_pack_header_raw(%s): WARNING Blockette 405 cannot be fully supported\n",
		      PACK_SRCNAME);
	    }
	}

      else if ( cur_blkt->blkt_type == 500 )
	{
	  struct blkt_500_s *blkt_500 = (struct blkt_500_s *) (rawrec + offset);
	  memcpy (blkt_500, cur_blkt->blktdata, sizeof (struct blkt_500_s));
	  offset += sizeof (struct blkt_500_s);
	  
	  if ( swapflag )
	    {
	      ms_gswap4 (&blkt_500->vco_correction);
	      MS_SWAPBTIME (&blkt_500->time);
	      ms_gswap4 (&blkt_500->exception_count);
	    }
	}
      
      else if ( cur_blkt->blkt_type == 1000 )
	{
	  struct blkt_1000_s *blkt_1000 = (struct blkt_1000_s *) (rawrec + offset);
	  memcpy (blkt_1000, cur_blkt->blktdata, sizeof (struct blkt_1000_s));
	  offset += sizeof (struct blkt_1000_s);
	  
	  /* This guarantees that the byte order is in sync with msr_pack() */
	  if ( packdatabyteorder >= 0 )
	    blkt_1000->byteorder = packdatabyteorder;
	}
      
      else if ( cur_blkt->blkt_type == 1001 )
	{
	  struct blkt_1001_s *blkt_1001 = (struct blkt_1001_s *) (rawrec + offset);
	  memcpy (blkt_1001, cur_blkt->blktdata, sizeof (struct blkt_1001_s));
	  offset += sizeof (struct blkt_1001_s);
	}

      else if ( cur_blkt->blkt_type == 2000 )
	{
	  struct blkt_2000_s *blkt_2000 = (struct blkt_2000_s *) (rawrec + offset);
	  memcpy (blkt_2000, cur_blkt->blktdata, cur_blkt->blktdatalen);
	  offset += cur_blkt->blktdatalen;
	  
	  if ( swapflag )
	    {
	      ms_gswap2 (&blkt_2000->length);
	      ms_gswap2 (&blkt_2000->data_offset);
	      ms_gswap4 (&blkt_2000->recnum);
	    }
	  
	  /* Nothing done to pack the opaque headers and data, they should already
	     be packed into the blockette payload */
	}
      
      else
	{
	  memcpy (rawrec + offset, cur_blkt->blktdata, cur_blkt->blktdatalen);
	  offset += cur_blkt->blktdatalen;
	}
      
      /* Pack the offset to the next blockette */
      if ( cur_blkt->next )
	{
	  memcpy (rawrec + nextoffset, &offset, 2);
	  if ( swapflag ) ms_gswap2 (rawrec + nextoffset);
	}
      else
	{
	  memset (rawrec + nextoffset, 0, 2);
	}
      
      blktcnt++;
      cur_blkt = cur_blkt->next;
    }
  
  fsdh->numblockettes = blktcnt;
  
  if ( verbose > 2 )
    ms_log (1, "%s: Packed %d blockettes\n", PACK_SRCNAME, blktcnt);
  
  return offset;
}  /* End of msr_pack_header_raw() */


/***************************************************************************
 * msr_update_header:
 *
 * Update the header values that change between records: start time,
 * sequence number, etc.
 *
 * Returns 0 on success or -1 on error.
 ***************************************************************************/
static int
msr_update_header ( MSRecord *msr, char *rawrec, flag swapflag,
		    flag verbose )
{
  struct fsdh_s *fsdh;
  char seqnum[7];
  
  if ( ! msr || ! rawrec )
    return -1;
  
  if ( verbose > 2 )
    ms_log (1, "%s: Updating fixed section of data header\n", PACK_SRCNAME);
  
  fsdh = (struct fsdh_s *) rawrec;
  
  /* Pack values into the fixed section of header */
  snprintf (seqnum, 7, "%06d", msr->sequence_number);
  memcpy (fsdh->sequence_number, seqnum, 6);

  ms_hptime2btime (msr->starttime, &(fsdh->start_time));
  
  /* Swap byte order? */
  if ( swapflag )
    {
      MS_SWAPBTIME (&fsdh->start_time);
    }
  
  return 0;
}  /* End of msr_update_header() */


/************************************************************************
 *  msr_pack_data:
 *
 *  Pack Mini-SEED data samples.  The input data samples specified as
 *  'src' will be packed with 'encoding' format and placed in 'dest'.
 *  
 *  If a pointer to a 32-bit integer sample is provided in the
 *  argument 'lastintsample' and 'comphistory' is true the sample
 *  value will be used to seed the difference buffer for Steim1/2
 *  encoding and provide a compression history.  It will also be
 *  updated with the last sample packed in order to be used with a
 *  subsequent call to this routine.
 *
 *  The number of samples packed will be placed in 'packsamples' and
 *  the number of bytes packed will be placed in 'packbytes'.
 *
 *  Return 0 on success and a negative number on error.
 ************************************************************************/
static int
msr_pack_data (void *dest, void *src, int maxsamples, int maxdatabytes,
	       int *packsamples, int32_t *lastintsample, flag comphistory,
	       char sampletype, flag encoding, flag swapflag, flag verbose)
{
  int retval;
  int nframes;
  int npacked;
  int32_t *intbuff;
  int32_t d0;
  
  /* Decide if this is a format that we can decode */
  switch (encoding)
    {
      
    case DE_ASCII:
      if ( sampletype != 'a' )
	{
	  ms_log (2, "%s: Sample type must be ascii (a) for ASCII encoding not '%c'\n",
		  PACK_SRCNAME, sampletype);
	  return -1;
	}
      
      if ( verbose > 1 )
	ms_log (1, "%s: Packing ASCII data\n", PACK_SRCNAME);
      
      retval = msr_pack_text (dest, src, maxsamples, maxdatabytes, 1,
			      &npacked, packsamples);
      
      break;
      
    case DE_INT16:
      if ( sampletype != 'i' )
	{
	  ms_log (2, "%s: Sample type must be integer (i) for integer-16 encoding not '%c'\n",
		  PACK_SRCNAME, sampletype);
	  return -1;
	}
      
      if ( verbose > 1 )
	ms_log (1, "%s: Packing INT-16 data samples\n", PACK_SRCNAME);
      
      retval = msr_pack_int_16 (dest, src, maxsamples, maxdatabytes, 1,
				&npacked, packsamples, swapflag);
      
      break;
      
    case DE_INT32:
      if ( sampletype != 'i' )
	{
	  ms_log (2, "%s: Sample type must be integer (i) for integer-32 encoding not '%c'\n",
		  PACK_SRCNAME, sampletype);
	  return -1;
	}

      if ( verbose > 1 )
	ms_log (1, "%s: Packing INT-32 data samples\n", PACK_SRCNAME);
      
      retval = msr_pack_int_32 (dest, src, maxsamples, maxdatabytes, 1,
				&npacked, packsamples, swapflag);
      
      break;
      
    case DE_FLOAT32:
      if ( sampletype != 'f' )
	{
	  ms_log (2, "%s: Sample type must be float (f) for float-32 encoding not '%c'\n",
		  PACK_SRCNAME, sampletype);
	  return -1;
	}

      if ( verbose > 1 )
	ms_log (1, "%s: Packing FLOAT-32 data samples\n", PACK_SRCNAME);
      
      retval = msr_pack_float_32 (dest, src, maxsamples, maxdatabytes, 1,
				  &npacked, packsamples, swapflag);

      break;
      
    case DE_FLOAT64:
      if ( sampletype != 'd' )
	{
	  ms_log (2, "%s: Sample type must be double (d) for float-64 encoding not '%c'\n",
		  PACK_SRCNAME, sampletype);
	  return -1;
	}
      
      if ( verbose > 1 )
	ms_log (1, "%s: Packing FLOAT-64 data samples\n", PACK_SRCNAME);
      
      retval = msr_pack_float_64 (dest, src, maxsamples, maxdatabytes, 1,
				  &npacked, packsamples, swapflag);
      
      break;
      
    case DE_STEIM1:
      if ( sampletype != 'i' )
	{
	  ms_log (2, "%s: Sample type must be integer (i) for Steim-1 compression not '%c'\n",
		  PACK_SRCNAME, sampletype);
	  return -1;
	}
      
      intbuff = (int32_t *) src;
      
      /* If a previous sample is supplied use it for compression history otherwise cold-start */
      d0 = ( lastintsample && comphistory ) ? (intbuff[0] - *lastintsample) : 0;
      
      if ( verbose > 1 )
	ms_log (1, "%s: Packing Steim-1 data frames\n", PACK_SRCNAME);
      
      nframes = maxdatabytes / 64;
      
      retval = msr_pack_steim1 (dest, src, d0, maxsamples, nframes, 1,
				&npacked, packsamples, swapflag);
      
      /* If a previous sample is supplied update it with the last sample value */
      if ( lastintsample && retval == 0 )
	*lastintsample = intbuff[*packsamples-1];
      
      break;
      
    case DE_STEIM2:
      if ( sampletype != 'i' )
	{
	  ms_log (2, "%s: Sample type must be integer (i) for Steim-2 compression not '%c'\n",
		  PACK_SRCNAME, sampletype);
	  return -1;
	}
      
      intbuff = (int32_t *) src;
      
      /* If a previous sample is supplied use it for compression history otherwise cold-start */
      d0 = ( lastintsample && comphistory ) ? (intbuff[0] - *lastintsample) : 0;
      
      if ( verbose > 1 )
	ms_log (1, "%s: Packing Steim-2 data frames\n", PACK_SRCNAME);
      
      nframes = maxdatabytes / 64;
      
      retval = msr_pack_steim2 (dest, src, d0, maxsamples, nframes, 1,
				&npacked, packsamples, swapflag);
      
      /* If a previous sample is supplied update it with the last sample value */
      if ( lastintsample && retval == 0 )
	*lastintsample = intbuff[*packsamples-1];
      
      break;
      
    default:
      ms_log (2, "%s: Unable to pack format %d\n", PACK_SRCNAME, encoding);
      
      return -1;
    }
    
  return retval;
} /* End of msr_pack_data() */
