/***************************************************************************
 * unpack.c:
 *
 * Generic routines to unpack Mini-SEED records.
 *
 * Appropriate values from the record header will be byte-swapped to
 * the host order.  The purpose of this code is to provide a portable
 * way of accessing common SEED data record header information.  All
 * data structures in SEED 2.4 data records are supported.  The data
 * samples are optionally decompressed/unpacked.
 *
 * Written by Chad Trabant,
 *   ORFEUS/EC-Project MEREDIAN
 *   IRIS Data Management Center
 *
 * modified: 2016.273
 ***************************************************************************/
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "libmseed.h"
#include "unpackdata.h"

/* Function(s) internal to this file */
static int check_environment (int verbose);

/* Header and data byte order flags controlled by environment variables */
/* -2 = not checked, -1 = checked but not set, or 0 = LE and 1 = BE */
flag unpackheaderbyteorder = -2;
flag unpackdatabyteorder   = -2;

/* Data encoding format/fallback controlled by environment variable */
/* -2 = not checked, -1 = checked but not set, or = encoding */
int unpackencodingformat   = -2;
int unpackencodingfallback = -2;

/***************************************************************************
 * msr_unpack:
 *
 * Unpack a SEED data record header/blockettes and populate a MSRecord
 * struct. All approriate fields are byteswapped, if needed, and
 * pointers to structured data are setup in addition to setting the
 * common header fields.
 *
 * If 'dataflag' is true the data samples are unpacked/decompressed
 * and the MSRecord->datasamples pointer is set appropriately.  The
 * data samples will be either 32-bit integers, 32-bit floats or
 * 64-bit floats (doubles) with the same byte order as the host
 * machine.  The MSRecord->numsamples will be set to the actual number
 * of samples unpacked/decompressed and MSRecord->sampletype will
 * indicated the sample type.
 *
 * All appropriate values will be byte-swapped to the host order,
 * including the data samples.
 *
 * All header values, blockette values and data samples will be
 * overwritten by subsequent calls to this function.
 *
 * If the msr struct is NULL it will be allocated.
 *
 * Returns MS_NOERROR and populates the MSRecord struct at *ppmsr on
 * success, otherwise returns a libmseed error code (listed in
 * libmseed.h).
 ***************************************************************************/
int
msr_unpack (char *record, int reclen, MSRecord **ppmsr,
            flag dataflag, flag verbose)
{
  flag headerswapflag = 0;
  flag dataswapflag   = 0;
  int retval;

  MSRecord *msr = NULL;
  char sequence_number[7];
  char srcname[50];

  /* For blockette parsing */
  BlktLink *blkt_link = 0;
  uint16_t blkt_type;
  uint16_t next_blkt;
  uint32_t blkt_offset;
  uint32_t blkt_length;
  int blkt_count = 0;

  if (!ppmsr)
  {
    ms_log (2, "msr_unpack(): ppmsr argument cannot be NULL\n");
    return MS_GENERROR;
  }

  /* Verify that record includes a valid header */
  if (!MS_ISVALIDHEADER (record))
  {
    ms_recsrcname (record, srcname, 1);
    ms_log (2, "msr_unpack(%s) Record header & quality indicator unrecognized: '%c'\n", srcname);
    ms_log (2, "msr_unpack(%s) This is not a valid Mini-SEED record\n", srcname);

    return MS_NOTSEED;
  }

  /* Verify that passed record length is within supported range */
  if (reclen < MINRECLEN || reclen > MAXRECLEN)
  {
    ms_recsrcname (record, srcname, 1);
    ms_log (2, "msr_unpack(%s): Record length is out of range: %d\n", srcname, reclen);
    return MS_OUTOFRANGE;
  }

  /* Initialize the MSRecord */
  if (!(*ppmsr = msr_init (*ppmsr)))
    return MS_GENERROR;

  /* Shortcut pointer, historical and help readability */
  msr = *ppmsr;

  /* Set raw record pointer and record length */
  msr->record = record;
  msr->reclen = reclen;

  /* Check environment variables if necessary */
  if (unpackheaderbyteorder == -2 ||
      unpackdatabyteorder == -2 ||
      unpackencodingformat == -2 ||
      unpackencodingfallback == -2)
    if (check_environment (verbose))
      return MS_GENERROR;

  /* Allocate and copy fixed section of data header */
  msr->fsdh = realloc (msr->fsdh, sizeof (struct fsdh_s));

  if (msr->fsdh == NULL)
  {
    ms_log (2, "msr_unpack(): Cannot allocate memory\n");
    return MS_GENERROR;
  }

  memcpy (msr->fsdh, record, sizeof (struct fsdh_s));

  /* Check to see if byte swapping is needed by testing the year and day */
  if (!MS_ISVALIDYEARDAY (msr->fsdh->start_time.year, msr->fsdh->start_time.day))
    headerswapflag = dataswapflag = 1;

  /* Check if byte order is forced */
  if (unpackheaderbyteorder >= 0)
  {
    headerswapflag = (ms_bigendianhost () != unpackheaderbyteorder) ? 1 : 0;
  }

  if (unpackdatabyteorder >= 0)
  {
    dataswapflag = (ms_bigendianhost () != unpackdatabyteorder) ? 1 : 0;
  }

  /* Swap byte order? */
  if (headerswapflag)
  {
    MS_SWAPBTIME (&msr->fsdh->start_time);
    ms_gswap2a (&msr->fsdh->numsamples);
    ms_gswap2a (&msr->fsdh->samprate_fact);
    ms_gswap2a (&msr->fsdh->samprate_mult);
    ms_gswap4a (&msr->fsdh->time_correct);
    ms_gswap2a (&msr->fsdh->data_offset);
    ms_gswap2a (&msr->fsdh->blockette_offset);
  }

  /* Populate some of the common header fields */
  strncpy (sequence_number, msr->fsdh->sequence_number, 6);
  sequence_number[6]   = '\0';
  msr->sequence_number = (int32_t)strtol (sequence_number, NULL, 10);
  msr->dataquality     = msr->fsdh->dataquality;
  ms_strncpcleantail (msr->network, msr->fsdh->network, 2);
  ms_strncpcleantail (msr->station, msr->fsdh->station, 5);
  ms_strncpcleantail (msr->location, msr->fsdh->location, 2);
  ms_strncpcleantail (msr->channel, msr->fsdh->channel, 3);
  msr->samplecnt = msr->fsdh->numsamples;

  /* Generate source name for MSRecord */
  if (msr_srcname (msr, srcname, 1) == NULL)
  {
    ms_log (2, "msr_unpack(): Cannot generate srcname\n");
    return MS_GENERROR;
  }

  /* Report byte swapping status */
  if (verbose > 2)
  {
    if (headerswapflag)
      ms_log (1, "%s: Byte swapping needed for unpacking of header\n", srcname);
    else
      ms_log (1, "%s: Byte swapping NOT needed for unpacking of header\n", srcname);
  }

  /* Traverse the blockettes */
  blkt_offset = msr->fsdh->blockette_offset;

  while ((blkt_offset != 0) &&
         ((int)blkt_offset < reclen) &&
         (blkt_offset < MAXRECLEN))
  {
    /* Every blockette has a similar 4 byte header: type and next */
    memcpy (&blkt_type, record + blkt_offset, 2);
    blkt_offset += 2;
    memcpy (&next_blkt, record + blkt_offset, 2);
    blkt_offset += 2;

    if (headerswapflag)
    {
      ms_gswap2 (&blkt_type);
      ms_gswap2 (&next_blkt);
    }

    /* Get blockette length */
    blkt_length = ms_blktlen (blkt_type,
                              record + blkt_offset - 4,
                              headerswapflag);

    if (blkt_length == 0)
    {
      ms_log (2, "msr_unpack(%s): Unknown blockette length for type %d\n",
              srcname, blkt_type);
      break;
    }

    /* Make sure blockette is contained within the msrecord buffer */
    if ((int)(blkt_offset - 4 + blkt_length) > reclen)
    {
      ms_log (2, "msr_unpack(%s): Blockette %d extends beyond record size, truncated?\n",
              srcname, blkt_type);
      break;
    }

    if (blkt_type == 100)
    { /* Found a Blockette 100 */
      struct blkt_100_s *blkt_100;

      blkt_link = msr_addblockette (msr, record + blkt_offset,
                                    sizeof (struct blkt_100_s),
                                    blkt_type, 0);
      if (!blkt_link)
        break;

      blkt_link->blktoffset = blkt_offset - 4;
      blkt_link->next_blkt  = next_blkt;

      blkt_100 = (struct blkt_100_s *)blkt_link->blktdata;

      if (headerswapflag)
      {
        ms_gswap4 (&blkt_100->samprate);
      }

      msr->samprate = msr->Blkt100->samprate;
    }

    else if (blkt_type == 200)
    { /* Found a Blockette 200 */
      struct blkt_200_s *blkt_200;

      blkt_link = msr_addblockette (msr, record + blkt_offset,
                                    sizeof (struct blkt_200_s),
                                    blkt_type, 0);
      if (!blkt_link)
        break;

      blkt_link->blktoffset = blkt_offset - 4;
      blkt_link->next_blkt  = next_blkt;

      blkt_200 = (struct blkt_200_s *)blkt_link->blktdata;

      if (headerswapflag)
      {
        ms_gswap4 (&blkt_200->amplitude);
        ms_gswap4 (&blkt_200->period);
        ms_gswap4 (&blkt_200->background_estimate);
        MS_SWAPBTIME (&blkt_200->time);
      }
    }

    else if (blkt_type == 201)
    { /* Found a Blockette 201 */
      struct blkt_201_s *blkt_201;

      blkt_link = msr_addblockette (msr, record + blkt_offset,
                                    sizeof (struct blkt_201_s),
                                    blkt_type, 0);
      if (!blkt_link)
        break;

      blkt_link->blktoffset = blkt_offset - 4;
      blkt_link->next_blkt  = next_blkt;

      blkt_201 = (struct blkt_201_s *)blkt_link->blktdata;

      if (headerswapflag)
      {
        ms_gswap4 (&blkt_201->amplitude);
        ms_gswap4 (&blkt_201->period);
        ms_gswap4 (&blkt_201->background_estimate);
        MS_SWAPBTIME (&blkt_201->time);
      }
    }

    else if (blkt_type == 300)
    { /* Found a Blockette 300 */
      struct blkt_300_s *blkt_300;

      blkt_link = msr_addblockette (msr, record + blkt_offset,
                                    sizeof (struct blkt_300_s),
                                    blkt_type, 0);
      if (!blkt_link)
        break;

      blkt_link->blktoffset = blkt_offset - 4;
      blkt_link->next_blkt  = next_blkt;

      blkt_300 = (struct blkt_300_s *)blkt_link->blktdata;

      if (headerswapflag)
      {
        MS_SWAPBTIME (&blkt_300->time);
        ms_gswap4 (&blkt_300->step_duration);
        ms_gswap4 (&blkt_300->interval_duration);
        ms_gswap4 (&blkt_300->amplitude);
        ms_gswap4 (&blkt_300->reference_amplitude);
      }
    }

    else if (blkt_type == 310)
    { /* Found a Blockette 310 */
      struct blkt_310_s *blkt_310;

      blkt_link = msr_addblockette (msr, record + blkt_offset,
                                    sizeof (struct blkt_310_s),
                                    blkt_type, 0);
      if (!blkt_link)
        break;

      blkt_link->blktoffset = blkt_offset - 4;
      blkt_link->next_blkt  = next_blkt;

      blkt_310 = (struct blkt_310_s *)blkt_link->blktdata;

      if (headerswapflag)
      {
        MS_SWAPBTIME (&blkt_310->time);
        ms_gswap4 (&blkt_310->duration);
        ms_gswap4 (&blkt_310->period);
        ms_gswap4 (&blkt_310->amplitude);
        ms_gswap4 (&blkt_310->reference_amplitude);
      }
    }

    else if (blkt_type == 320)
    { /* Found a Blockette 320 */
      struct blkt_320_s *blkt_320;

      blkt_link = msr_addblockette (msr, record + blkt_offset,
                                    sizeof (struct blkt_320_s),
                                    blkt_type, 0);
      if (!blkt_link)
        break;

      blkt_link->blktoffset = blkt_offset - 4;
      blkt_link->next_blkt  = next_blkt;

      blkt_320 = (struct blkt_320_s *)blkt_link->blktdata;

      if (headerswapflag)
      {
        MS_SWAPBTIME (&blkt_320->time);
        ms_gswap4 (&blkt_320->duration);
        ms_gswap4 (&blkt_320->ptp_amplitude);
        ms_gswap4 (&blkt_320->reference_amplitude);
      }
    }

    else if (blkt_type == 390)
    { /* Found a Blockette 390 */
      struct blkt_390_s *blkt_390;

      blkt_link = msr_addblockette (msr, record + blkt_offset,
                                    sizeof (struct blkt_390_s),
                                    blkt_type, 0);
      if (!blkt_link)
        break;

      blkt_link->blktoffset = blkt_offset - 4;
      blkt_link->next_blkt  = next_blkt;

      blkt_390 = (struct blkt_390_s *)blkt_link->blktdata;

      if (headerswapflag)
      {
        MS_SWAPBTIME (&blkt_390->time);
        ms_gswap4 (&blkt_390->duration);
        ms_gswap4 (&blkt_390->amplitude);
      }
    }

    else if (blkt_type == 395)
    { /* Found a Blockette 395 */
      struct blkt_395_s *blkt_395;

      blkt_link = msr_addblockette (msr, record + blkt_offset,
                                    sizeof (struct blkt_395_s),
                                    blkt_type, 0);
      if (!blkt_link)
        break;

      blkt_link->blktoffset = blkt_offset - 4;
      blkt_link->next_blkt  = next_blkt;

      blkt_395 = (struct blkt_395_s *)blkt_link->blktdata;

      if (headerswapflag)
      {
        MS_SWAPBTIME (&blkt_395->time);
      }
    }

    else if (blkt_type == 400)
    { /* Found a Blockette 400 */
      struct blkt_400_s *blkt_400;

      blkt_link = msr_addblockette (msr, record + blkt_offset,
                                    sizeof (struct blkt_400_s),
                                    blkt_type, 0);
      if (!blkt_link)
        break;

      blkt_link->blktoffset = blkt_offset - 4;
      blkt_link->next_blkt  = next_blkt;

      blkt_400 = (struct blkt_400_s *)blkt_link->blktdata;

      if (headerswapflag)
      {
        ms_gswap4 (&blkt_400->azimuth);
        ms_gswap4 (&blkt_400->slowness);
        ms_gswap2 (&blkt_400->configuration);
      }
    }

    else if (blkt_type == 405)
    { /* Found a Blockette 405 */
      struct blkt_405_s *blkt_405;

      blkt_link = msr_addblockette (msr, record + blkt_offset,
                                    sizeof (struct blkt_405_s),
                                    blkt_type, 0);
      if (!blkt_link)
        break;

      blkt_link->blktoffset = blkt_offset - 4;
      blkt_link->next_blkt  = next_blkt;

      blkt_405 = (struct blkt_405_s *)blkt_link->blktdata;

      if (headerswapflag)
      {
        ms_gswap2 (&blkt_405->delay_values);
      }

      if (verbose > 0)
      {
        ms_log (1, "msr_unpack(%s): WARNING Blockette 405 cannot be fully supported\n",
                srcname);
      }
    }

    else if (blkt_type == 500)
    { /* Found a Blockette 500 */
      struct blkt_500_s *blkt_500;

      blkt_link = msr_addblockette (msr, record + blkt_offset,
                                    sizeof (struct blkt_500_s),
                                    blkt_type, 0);
      if (!blkt_link)
        break;

      blkt_link->blktoffset = blkt_offset - 4;
      blkt_link->next_blkt  = next_blkt;

      blkt_500 = (struct blkt_500_s *)blkt_link->blktdata;

      if (headerswapflag)
      {
        ms_gswap4 (&blkt_500->vco_correction);
        MS_SWAPBTIME (&blkt_500->time);
        ms_gswap4 (&blkt_500->exception_count);
      }
    }

    else if (blkt_type == 1000)
    { /* Found a Blockette 1000 */
      struct blkt_1000_s *blkt_1000;

      blkt_link = msr_addblockette (msr, record + blkt_offset,
                                    sizeof (struct blkt_1000_s),
                                    blkt_type, 0);
      if (!blkt_link)
        break;

      blkt_link->blktoffset = blkt_offset - 4;
      blkt_link->next_blkt  = next_blkt;

      blkt_1000 = (struct blkt_1000_s *)blkt_link->blktdata;

      /* Calculate record length in bytes as 2^(blkt_1000->reclen) */
      msr->reclen = (uint32_t)1 << blkt_1000->reclen;

      /* Compare against the specified length */
      if (msr->reclen != reclen && verbose)
      {
        ms_log (2, "msr_unpack(%s): Record length in Blockette 1000 (%d) != specified length (%d)\n",
                srcname, msr->reclen, reclen);
      }

      msr->encoding  = blkt_1000->encoding;
      msr->byteorder = blkt_1000->byteorder;
    }

    else if (blkt_type == 1001)
    { /* Found a Blockette 1001 */
      blkt_link = msr_addblockette (msr, record + blkt_offset,
                                    sizeof (struct blkt_1001_s),
                                    blkt_type, 0);
      if (!blkt_link)
        break;

      blkt_link->blktoffset = blkt_offset - 4;
      blkt_link->next_blkt  = next_blkt;
    }

    else if (blkt_type == 2000)
    { /* Found a Blockette 2000 */
      struct blkt_2000_s *blkt_2000;
      uint16_t b2klen;

      /* Read the blockette length from blockette */
      memcpy (&b2klen, record + blkt_offset, 2);
      if (headerswapflag)
        ms_gswap2 (&b2klen);

      /* Minus four bytes for the blockette type and next fields */
      b2klen -= 4;

      blkt_link = msr_addblockette (msr, record + blkt_offset,
                                    b2klen, blkt_type, 0);
      if (!blkt_link)
        break;

      blkt_link->blktoffset = blkt_offset - 4;
      blkt_link->next_blkt  = next_blkt;

      blkt_2000 = (struct blkt_2000_s *)blkt_link->blktdata;

      if (headerswapflag)
      {
        ms_gswap2 (&blkt_2000->length);
        ms_gswap2 (&blkt_2000->data_offset);
        ms_gswap4 (&blkt_2000->recnum);
      }
    }

    else
    { /* Unknown blockette type */
      if (blkt_length >= 4)
      {
        blkt_link = msr_addblockette (msr, record + blkt_offset,
                                      blkt_length - 4,
                                      blkt_type, 0);

        if (!blkt_link)
          break;

        blkt_link->blktoffset = blkt_offset - 4;
        blkt_link->next_blkt  = next_blkt;
      }
    }

    /* Check that the next blockette offset is beyond the current blockette */
    if (next_blkt && next_blkt < (blkt_offset + blkt_length - 4))
    {
      ms_log (2, "msr_unpack(%s): Offset to next blockette (%d) is within current blockette ending at byte %d\n",
              srcname, next_blkt, (blkt_offset + blkt_length - 4));

      blkt_offset = 0;
    }
    /* Check that the offset is within record length */
    else if (next_blkt && next_blkt > reclen)
    {
      ms_log (2, "msr_unpack(%s): Offset to next blockette (%d) from type %d is beyond record length\n",
              srcname, next_blkt, blkt_type);

      blkt_offset = 0;
    }
    else
    {
      blkt_offset = next_blkt;
    }

    blkt_count++;
  } /* End of while looping through blockettes */

  /* Check for a Blockette 1000 */
  if (msr->Blkt1000 == 0)
  {
    if (verbose > 1)
    {
      ms_log (1, "%s: Warning: No Blockette 1000 found\n", srcname);
    }
  }

  /* Check that the data offset is after the blockette chain */
  if (blkt_link && msr->fsdh->numsamples && msr->fsdh->data_offset < (blkt_link->blktoffset + blkt_link->blktdatalen + 4))
  {
    ms_log (1, "%s: Warning: Data offset in fixed header (%d) is within the blockette chain ending at %d\n",
            srcname, msr->fsdh->data_offset, (blkt_link->blktoffset + blkt_link->blktdatalen + 4));
  }

  /* Check that the blockette count matches the number parsed */
  if (msr->fsdh->numblockettes != blkt_count)
  {
    ms_log (1, "%s: Warning: Number of blockettes in fixed header (%d) does not match the number parsed (%d)\n",
            srcname, msr->fsdh->numblockettes, blkt_count);
  }

  /* Populate remaining common header fields */
  msr->starttime = msr_starttime (msr);
  msr->samprate  = msr_samprate (msr);

  /* Set MSRecord->byteorder if data byte order is forced */
  if (unpackdatabyteorder >= 0)
  {
    msr->byteorder = unpackdatabyteorder;
  }

  /* Check if encoding format is forced */
  if (unpackencodingformat >= 0)
  {
    msr->encoding = unpackencodingformat;
  }

  /* Use encoding format fallback if defined and no encoding is set,
     also make sure the byteorder is set by default to big endian */
  if (unpackencodingfallback >= 0 && msr->encoding == -1)
  {
    msr->encoding = unpackencodingfallback;

    if (msr->byteorder == -1)
    {
      msr->byteorder = 1;
    }
  }

  /* Unpack the data samples if requested */
  if (dataflag && msr->samplecnt > 0)
  {
    flag dswapflag     = headerswapflag;
    flag bigendianhost = ms_bigendianhost ();

    /* Determine byte order of the data and set the dswapflag as
       needed; if no Blkt1000 or UNPACK_DATA_BYTEORDER environment
       variable setting assume the order is the same as the header */
    if (msr->Blkt1000 != 0 && unpackdatabyteorder < 0)
    {
      dswapflag = 0;

      /* If BE host and LE data need swapping */
      if (bigendianhost && msr->byteorder == 0)
        dswapflag = 1;
      /* If LE host and BE data (or bad byte order value) need swapping */
      else if (!bigendianhost && msr->byteorder > 0)
        dswapflag = 1;
    }
    else if (unpackdatabyteorder >= 0)
    {
      dswapflag = dataswapflag;
    }

    if (verbose > 2 && dswapflag)
      ms_log (1, "%s: Byte swapping needed for unpacking of data samples\n", srcname);
    else if (verbose > 2)
      ms_log (1, "%s: Byte swapping NOT needed for unpacking of data samples\n", srcname);

    retval = msr_unpack_data (msr, dswapflag, verbose);

    if (retval < 0)
      return retval;
    else
      msr->numsamples = retval;
  }
  else
  {
    if (msr->datasamples)
      free (msr->datasamples);

    msr->datasamples = 0;
    msr->numsamples  = 0;
  }

  return MS_NOERROR;
} /* End of msr_unpack() */

/************************************************************************
 *  msr_unpack_data:
 *
 *  Unpack Mini-SEED data samples for a given MSRecord.  The packed
 *  data is accessed in the record indicated by MSRecord->record and
 *  the unpacked samples are placed in MSRecord->datasamples.  The
 *  resulting data samples are either 32-bit integers, 32-bit floats
 *  or 64-bit floats in host byte order.
 *
 *  Return number of samples unpacked or negative libmseed error code.
 ************************************************************************/
int
msr_unpack_data (MSRecord *msr, int swapflag, flag verbose)
{
  int datasize;       /* byte size of data samples in record */
  int nsamples;       /* number of samples unpacked	     */
  int unpacksize;     /* byte size of unpacked samples	     */
  int samplesize = 0; /* size of the data samples in bytes   */
  char srcname[50];
  const char *dbuf;

  if (!msr)
    return MS_GENERROR;

  /* Check for decode debugging environment variable */
  if (getenv ("DECODE_DEBUG"))
    decodedebug = 1;

  /* Generate source name for MSRecord */
  if (msr_srcname (msr, srcname, 1) == NULL)
  {
    ms_log (2, "msr_unpack(): Cannot generate srcname\n");
    return MS_GENERROR;
  }

  /* Sanity record length */
  if (msr->reclen == -1)
  {
    ms_log (2, "msr_unpack_data(%s): Record size unknown\n", srcname);
    return MS_NOTSEED;
  }
  else if (msr->reclen < MINRECLEN || msr->reclen > MAXRECLEN)
  {
    ms_log (2, "msr_unpack_data(%s): Unsupported record length: %d\n",
            srcname, msr->reclen);
    return MS_OUTOFRANGE;
  }

  /* Sanity check data offset before creating a pointer based on the value */
  if (msr->fsdh->data_offset < 48 || msr->fsdh->data_offset >= msr->reclen)
  {
    ms_log (2, "msr_unpack_data(%s): data offset value is not valid: %d\n",
            srcname, msr->fsdh->data_offset);
    return MS_GENERROR;
  }

  datasize = msr->reclen - msr->fsdh->data_offset;
  dbuf     = msr->record + msr->fsdh->data_offset;

  switch (msr->encoding)
  {
  case DE_ASCII:
    samplesize = 1;
    break;
  case DE_INT16:
  case DE_INT32:
  case DE_FLOAT32:
  case DE_STEIM1:
  case DE_STEIM2:
  case DE_GEOSCOPE24:
  case DE_GEOSCOPE163:
  case DE_GEOSCOPE164:
  case DE_CDSN:
  case DE_SRO:
  case DE_DWWSSN:
    samplesize = 4;
    break;
  case DE_FLOAT64:
    samplesize = 8;
    break;
  default:
    samplesize = 0;
    break;
  }

  /* Calculate buffer size needed for unpacked samples */
  unpacksize = (int)msr->samplecnt * samplesize;

  /* (Re)Allocate space for the unpacked data */
  if (unpacksize > 0)
  {
    msr->datasamples = realloc (msr->datasamples, unpacksize);

    if (msr->datasamples == NULL)
    {
      ms_log (2, "msr_unpack_data(%s): Cannot (re)allocate memory\n", srcname);
      return MS_GENERROR;
    }
  }
  else
  {
    if (msr->datasamples)
      free (msr->datasamples);
    msr->datasamples = 0;
    msr->numsamples  = 0;
  }

  if (verbose > 2)
    ms_log (1, "%s: Unpacking %" PRId64 " samples\n", srcname, msr->samplecnt);

  /* Decode data samples according to encoding */
  switch (msr->encoding)
  {
  case DE_ASCII:
    if (verbose > 1)
      ms_log (1, "%s: Found ASCII data\n", srcname);

    nsamples = (int)msr->samplecnt;
    memcpy (msr->datasamples, dbuf, nsamples);
    msr->sampletype = 'a';
    break;

  case DE_INT16:
    if (verbose > 1)
      ms_log (1, "%s: Unpacking INT16 data samples\n", srcname);

    nsamples = msr_decode_int16 ((int16_t *)dbuf, (int)msr->samplecnt,
                                 msr->datasamples, unpacksize, swapflag);

    msr->sampletype = 'i';
    break;

  case DE_INT32:
    if (verbose > 1)
      ms_log (1, "%s: Unpacking INT32 data samples\n", srcname);

    nsamples = msr_decode_int32 ((int32_t *)dbuf, (int)msr->samplecnt,
                                 msr->datasamples, unpacksize, swapflag);

    msr->sampletype = 'i';
    break;

  case DE_FLOAT32:
    if (verbose > 1)
      ms_log (1, "%s: Unpacking FLOAT32 data samples\n", srcname);

    nsamples = msr_decode_float32 ((float *)dbuf, (int)msr->samplecnt,
                                   msr->datasamples, unpacksize, swapflag);

    msr->sampletype = 'f';
    break;

  case DE_FLOAT64:
    if (verbose > 1)
      ms_log (1, "%s: Unpacking FLOAT64 data samples\n", srcname);

    nsamples = msr_decode_float64 ((double *)dbuf, (int)msr->samplecnt,
                                   msr->datasamples, unpacksize, swapflag);

    msr->sampletype = 'd';
    break;

  case DE_STEIM1:
    if (verbose > 1)
      ms_log (1, "%s: Unpacking Steim1 data frames\n", srcname);

    nsamples = msr_decode_steim1 ((int32_t *)dbuf, datasize, (int)msr->samplecnt,
                                  msr->datasamples, unpacksize, srcname, swapflag);

    if (nsamples < 0)
      return MS_GENERROR;

    msr->sampletype = 'i';
    break;

  case DE_STEIM2:
    if (verbose > 1)
      ms_log (1, "%s: Unpacking Steim2 data frames\n", srcname);

    nsamples = msr_decode_steim2 ((int32_t *)dbuf, datasize, (int)msr->samplecnt,
                                  msr->datasamples, unpacksize, srcname, swapflag);

    if (nsamples < 0)
      return MS_GENERROR;

    msr->sampletype = 'i';
    break;

  case DE_GEOSCOPE24:
  case DE_GEOSCOPE163:
  case DE_GEOSCOPE164:
    if (verbose > 1)
    {
      if (msr->encoding == DE_GEOSCOPE24)
        ms_log (1, "%s: Unpacking GEOSCOPE 24bit integer data samples\n",
                srcname);
      if (msr->encoding == DE_GEOSCOPE163)
        ms_log (1, "%s: Unpacking GEOSCOPE 16bit gain ranged/3bit exponent data samples\n",
                srcname);
      if (msr->encoding == DE_GEOSCOPE164)
        ms_log (1, "%s: Unpacking GEOSCOPE 16bit gain ranged/4bit exponent data samples\n",
                srcname);
    }

    nsamples = msr_decode_geoscope ((char *)dbuf, (int)msr->samplecnt, msr->datasamples,
                                    unpacksize, msr->encoding, srcname, swapflag);

    msr->sampletype = 'f';
    break;

  case DE_CDSN:
    if (verbose > 1)
      ms_log (1, "%s: Unpacking CDSN encoded data samples\n", srcname);

    nsamples = msr_decode_cdsn ((int16_t *)dbuf, (int)msr->samplecnt, msr->datasamples,
                                unpacksize, swapflag);

    msr->sampletype = 'i';
    break;

  case DE_SRO:
    if (verbose > 1)
      ms_log (1, "%s: Unpacking SRO encoded data samples\n", srcname);

    nsamples = msr_decode_sro ((int16_t *)dbuf, (int)msr->samplecnt, msr->datasamples,
                               unpacksize, srcname, swapflag);

    msr->sampletype = 'i';
    break;

  case DE_DWWSSN:
    if (verbose > 1)
      ms_log (1, "%s: Unpacking DWWSSN encoded data samples\n", srcname);

    nsamples = msr_decode_dwwssn ((int16_t *)dbuf, (int)msr->samplecnt, msr->datasamples,
                                  unpacksize, swapflag);

    msr->sampletype = 'i';
    break;

  default:
    ms_log (2, "%s: Unsupported encoding format %d (%s)\n",
            srcname, msr->encoding, (char *)ms_encodingstr (msr->encoding));

    return MS_UNKNOWNFORMAT;
  }

  if (nsamples != msr->samplecnt)
  {
    ms_log (2, "msr_unpack_data(%s): only decoded %d samples of %d expected\n",
            srcname, nsamples, msr->samplecnt);
    return MS_GENERROR;
  }

  return nsamples;
} /* End of msr_unpack_data() */

/************************************************************************
 *  check_environment:
 *
 *  Check environment variables and set global variables appropriately.
 *
 *  Return 0 on success and -1 on error.
 ************************************************************************/
static int
check_environment (int verbose)
{
  char *envvariable;

  /* Read possible environmental variables that force byteorder */
  if (unpackheaderbyteorder == -2)
  {
    if ((envvariable = getenv ("UNPACK_HEADER_BYTEORDER")))
    {
      if (*envvariable != '0' && *envvariable != '1')
      {
        ms_log (2, "Environment variable UNPACK_HEADER_BYTEORDER must be set to '0' or '1'\n");
        return -1;
      }
      else if (*envvariable == '0')
      {
        unpackheaderbyteorder = 0;
        if (verbose > 2)
          ms_log (1, "UNPACK_HEADER_BYTEORDER=0, unpacking little-endian header\n");
      }
      else
      {
        unpackheaderbyteorder = 1;
        if (verbose > 2)
          ms_log (1, "UNPACK_HEADER_BYTEORDER=1, unpacking big-endian header\n");
      }
    }
    else
    {
      unpackheaderbyteorder = -1;
    }
  }

  if (unpackdatabyteorder == -2)
  {
    if ((envvariable = getenv ("UNPACK_DATA_BYTEORDER")))
    {
      if (*envvariable != '0' && *envvariable != '1')
      {
        ms_log (2, "Environment variable UNPACK_DATA_BYTEORDER must be set to '0' or '1'\n");
        return -1;
      }
      else if (*envvariable == '0')
      {
        unpackdatabyteorder = 0;
        if (verbose > 2)
          ms_log (1, "UNPACK_DATA_BYTEORDER=0, unpacking little-endian data samples\n");
      }
      else
      {
        unpackdatabyteorder = 1;
        if (verbose > 2)
          ms_log (1, "UNPACK_DATA_BYTEORDER=1, unpacking big-endian data samples\n");
      }
    }
    else
    {
      unpackdatabyteorder = -1;
    }
  }

  /* Read possible environmental variable that forces encoding format */
  if (unpackencodingformat == -2)
  {
    if ((envvariable = getenv ("UNPACK_DATA_FORMAT")))
    {
      unpackencodingformat = (int)strtol (envvariable, NULL, 10);

      if (unpackencodingformat < 0 || unpackencodingformat > 33)
      {
        ms_log (2, "Environment variable UNPACK_DATA_FORMAT set to invalid value: '%d'\n", unpackencodingformat);
        return -1;
      }
      else if (verbose > 2)
        ms_log (1, "UNPACK_DATA_FORMAT, unpacking data in encoding format %d\n", unpackencodingformat);
    }
    else
    {
      unpackencodingformat = -1;
    }
  }

  /* Read possible environmental variable to be used as a fallback encoding format */
  if (unpackencodingfallback == -2)
  {
    if ((envvariable = getenv ("UNPACK_DATA_FORMAT_FALLBACK")))
    {
      unpackencodingfallback = (int)strtol (envvariable, NULL, 10);

      if (unpackencodingfallback < 0 || unpackencodingfallback > 33)
      {
        ms_log (2, "Environment variable UNPACK_DATA_FORMAT_FALLBACK set to invalid value: '%d'\n",
                unpackencodingfallback);
        return -1;
      }
      else if (verbose > 2)
        ms_log (1, "UNPACK_DATA_FORMAT_FALLBACK, fallback data unpacking encoding format %d\n",
                unpackencodingfallback);
    }
    else
    {
      unpackencodingfallback = 10; /* Default fallback is Steim-1 encoding */
    }
  }

  return 0;
} /* End of check_environment() */
