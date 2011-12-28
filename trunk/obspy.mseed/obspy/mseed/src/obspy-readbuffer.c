/***************************************************************************
 * obspy-readbuffer.c:
 *
 * Reads a memory buffer to a MSTraceList structure and parses Selections.
 *
 * Parts are copied from tracelist.c and unpack.c from libmseed by Chad
 * Trabant
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

#include "libmseed/libmseed.h"
#include "libmseed/unpackdata.h"

// Linkable container of MSRecords
typedef struct LinkedRecordList_s {
    struct MSRecord_s      *record;       // This record
    struct LinkedRecordList_s  *previous; // The previous record container
    struct LinkedRecordList_s  *next;     // The next record container
}
LinkedRecordList;

// Container for a continuous linked list of records.
typedef struct ContinuousSegment_s {
    hptime_t starttime;                     // Time of the first sample
    hptime_t endtime;                       // Time of the last sample
    double samprate;                        // Sample rate
    char sampletype;                        // Sampletype
    hptime_t hpdelta;                       // High precission sample period
    int64_t samplecnt;                         // Total sample count
    void *datasamples;                      // Actual data samples
    struct LinkedRecordList_s *firstRecord; // First item
    struct LinkedRecordList_s *lastRecord;  // Last item
    struct ContinuousSegment_s *next;       // Next segment
    struct ContinuousSegment_s *previous;   // Previous segment
}
ContinuousSegment;

// A container for continuous segments with the same id
typedef struct LinkedIDList_s {
    char network[11];                         // Network designation, NULL terminated
    char station[11];                         // Station designation, NULL terminated
    char location[11];                        // Location designation, NULL terminated
    char channel[11];                         // Channel designation, NULL terminated
    char dataquality;                         // Data quality indicator */
    struct ContinuousSegment_s *firstSegment; // Pointer to first of list of segments
    struct ContinuousSegment_s *lastSegment;  // Pointer to last of list of segments
    struct LinkedIDList_s *next;              // Pointer to next id
    struct LinkedIDList_s *previous;          // Pointer to previous id
}
LinkedIDList;


// Forward declarations
static int msr_unpack_data (MSRecord * msr, int swapflag, int verbose);
LinkedIDList * lil_init(void);
LinkedRecordList * lrl_init (void);
ContinuousSegment * seg_init(void);
void lrl_free(LinkedRecordList * lrl);
void seg_free(ContinuousSegment * seg);
void lil_free(LinkedIDList * lil);


// Init function for the LinkedIDList
LinkedIDList *
lil_init(void)
{
    // Allocate 0 initialized memory.
    LinkedIDList *lil = (LinkedIDList *) malloc (sizeof(LinkedIDList));
    if ( lil == NULL ) {
        ms_log (2, "lil_init(): Cannot allocate memory\n");
        return NULL;
    }
    memset (lil, 0, sizeof (LinkedIDList));
    return lil;
}

// Init function for the LinkedRecordList
LinkedRecordList *
lrl_init (void)
{
    // Allocate 0 initialized memory.
    LinkedRecordList *lrl = (LinkedRecordList *) malloc (sizeof(LinkedRecordList));
    if ( lrl == NULL ) {
        ms_log (2, "lrl_init(): Cannot allocate memory\n");
        return NULL;
    }
    memset (lrl, 0, sizeof (LinkedRecordList));
    return lrl;
}

// Init a Segment with a linked record list.
ContinuousSegment *
seg_init(void)
{
    ContinuousSegment *seg = (ContinuousSegment *) malloc (sizeof(ContinuousSegment));
    if ( seg == NULL ) {
        ms_log (2, "seg_init(): Cannot allocate memory\n");
        return NULL;
    }
    memset (seg, 0, sizeof (ContinuousSegment));
    return seg;
}

// Frees a LinkedRecordList. The given Record is assumed to be the head of the
// list.
void
lrl_free(LinkedRecordList * lrl)
{
    LinkedRecordList * next;
    while ( lrl != NULL) {
        next = lrl->next;
        msr_free(&lrl->record);
        free(lrl);
        if (next == NULL) {
            break;
        }
        lrl = next;
    }
    lrl = NULL;
}

// Frees a ContinuousSegment and all structures associated with it.
// The given segment is supposed to be the head of the linked list.
void
seg_free(ContinuousSegment * seg)
{
    ContinuousSegment * next;
    while (seg != NULL) {
        next = seg->next;
        // free(seg->datasamples);
        if (seg->firstRecord != NULL) {
            lrl_free(seg->firstRecord);
        }
        free(seg);
        if (next == NULL) {
            break;
        }
        seg = next;
    }
    seg = NULL;
}

// Free a LinkedIDList and all structures associated with it.
void
lil_free(LinkedIDList * lil)
{
    LinkedIDList * next;
    while ( lil != NULL) {
        next = lil->next;
        if (lil->firstSegment != NULL) {
            seg_free(lil->firstSegment);
        }
        free(lil);
        if (next == NULL) {
            break;
        }
        lil = next;
    }
    lil = NULL;
}


// Function that reads from a MiniSEED binary file from a char buffer and
// returns a LinkedIDList.
LinkedIDList *
readMSEEDBuffer (char *mseed, int buflen, Selections *selections, flag
                 unpack_data, int reclen, flag verbose,
                 long (*allocData) (int, char))
{
    int retcode = 0;
    int retval = 0;
    flag swapflag = 0;

    // current offset of mseed char pointer
    int offset = 0;

    // Unpack without reading the data first
    flag dataflag = 0;

    // Init all the pointers to NULL. Most compilers should do this anyway.
    LinkedIDList * idListHead = NULL;
    LinkedIDList * idListCurrent = NULL;
    LinkedIDList * idListLast = NULL;
    ContinuousSegment * segmentCurrent = NULL;
    hptime_t lastgap;
    hptime_t hptimetol;
    hptime_t nhptimetol;
    long data_offset;
    LinkedRecordList *recordHead = NULL;
    LinkedRecordList *recordPrevious = NULL;
    LinkedRecordList *recordCurrent = NULL;
    int datasize;


    //
    // Read all records and save them in a linked list.
    //
    int record_count = 0;
    while (offset < buflen) {
        MSRecord *msr = msr_init(NULL);
        retcode = msr_parse ( (mseed+offset), buflen, &msr, reclen, dataflag, verbose);
        if ( ! (retcode == MS_NOERROR)) {
            break;
        }

        // Test against selections if supplied
        if ( selections ) {
            char srcname[50];
            hptime_t endtime;
            msr_srcname (msr, srcname, 1);
            endtime = msr_endtime (msr);
            if ( ms_matchselect (selections, srcname, msr->starttime, endtime, NULL) == NULL ) {
                // Add the record length for the next iteration
                offset += msr->reclen;
                // Free record.
                msr_free(&msr);
                continue;
            }
        }
        record_count += 1;

        recordCurrent = lrl_init ();
        // Append to linked record list if one exists.
        if ( recordHead != NULL ) {
            recordPrevious->next = recordCurrent;
            recordCurrent->previous = recordPrevious;
            recordCurrent->next = NULL;
            recordPrevious = recordCurrent;
        }
        // Otherwise create a new one.
        else {
            recordHead = recordCurrent;
            recordCurrent->previous = NULL;
            recordPrevious = recordCurrent;
        }
        recordCurrent->record = msr;

        // Determine the byteorder swapflag only for the very first record. The byteorder
        // should not change within the file.
        // XXX: Maybe check for every record?
        if (swapflag <= 0) {
            // Returns 0 if the host is little endian, otherwise 1.
            flag bigendianhost = ms_bigendianhost();
            // Set the swapbyteflag if it is needed.
            if ( msr->Blkt1000 != 0) {
                /* If BE host and LE data need swapping */
                if ( bigendianhost && msr->byteorder == 0 ) {
                    swapflag = 1;
                }
                /* If LE host and BE data (or bad byte order value) need swapping */
                if ( !bigendianhost && msr->byteorder > 0 ) {
                    swapflag = 1;
                }
            }
        }

        // Actually unpack the data if the flag is not set.
        if (unpack_data != 0) {
            retval = msr_unpack_data (msr, swapflag, verbose);
        }
     
        if ( retval > 0 ) {
            msr->numsamples = retval;
        }

        // Add the record length for the next iteration
        offset += msr->reclen;
    }
    // Return empty id list if no records could be found.
    if (record_count == 0) {
        idListHead = lil_init();
        return idListHead;
    }


    // All records that match the selection are now stored in a LinkedRecordList
    // that starts at recordHead. The next step is to sort them by matching ids
    // and then by time.
    recordCurrent = recordHead;
    while (recordCurrent != NULL) {
        // Check if the ID of the record is already available and if not create a
        // new one.
        // Start with the last id as it is most likely to be the correct one.
        idListCurrent = idListLast;
        while (idListCurrent != NULL) {
            if (strcmp(idListCurrent->network, recordCurrent->record->network) == 0 && 
                strcmp(idListCurrent->station, recordCurrent->record->station) == 0 && 
                strcmp(idListCurrent->location, recordCurrent->record->location) == 0 && 
                strcmp(idListCurrent->channel, recordCurrent->record->channel) == 0 && 
                idListCurrent->dataquality == recordCurrent->record->dataquality) {
                break;
            } 
            else {
                idListCurrent = idListCurrent->previous;
            }
        }

        // Create a new id list if one is needed.
        if (idListCurrent == NULL) {
            idListCurrent = lil_init();
            idListCurrent->previous = idListLast;
            if (idListLast != NULL) {
                idListLast->next = idListCurrent;
            }
            idListLast = idListCurrent;
            if (idListHead == NULL) {
                idListHead = idListCurrent;
            }

            // Set the IdList attributes.
            strcpy(idListCurrent->network, recordCurrent->record->network);
            strcpy(idListCurrent->station, recordCurrent->record->station);
            strcpy(idListCurrent->location, recordCurrent->record->location);
            strcpy(idListCurrent->channel, recordCurrent->record->channel);
            idListCurrent->dataquality = recordCurrent->record->dataquality;
        }

        // Now check if the current record fits exactly to the end of the last
        // segment of the current id. If not create a new segment. Therefore
        // if records with the same id are in wrong order a new segment will be
        // created. This is on purpose.
        segmentCurrent = idListCurrent->lastSegment;
        if (segmentCurrent != NULL) { 
            hptimetol = (hptime_t) (0.5 * segmentCurrent->hpdelta);
            nhptimetol = ( hptimetol ) ? -hptimetol : 0;
            lastgap = recordCurrent->record->starttime - segmentCurrent->endtime - segmentCurrent->hpdelta;
        } 
        if ( segmentCurrent != NULL &&
             segmentCurrent->sampletype == recordCurrent->record->sampletype &&
             // Test the default sample rate tolerance: abs(1-sr1/sr2) < 0.0001
             MS_ISRATETOLERABLE (segmentCurrent->samprate, recordCurrent->record->samprate) &&
             // Check if the times are within the time tolerance
             lastgap <= hptimetol && lastgap >= nhptimetol) {
            recordCurrent->previous = segmentCurrent->lastRecord;
            segmentCurrent->lastRecord = segmentCurrent->lastRecord->next = recordCurrent;
            segmentCurrent->samplecnt += recordCurrent->record->samplecnt;
            segmentCurrent->endtime = msr_endtime(recordCurrent->record);
        }
        // Otherwise create a new segment and add the current record.
        else {
            segmentCurrent = seg_init();
            segmentCurrent->previous = idListCurrent->lastSegment;
            if (idListCurrent->lastSegment != NULL) {
                idListCurrent->lastSegment->next = segmentCurrent;
            }
            else {
                idListCurrent->firstSegment = segmentCurrent;
            }
            idListCurrent->lastSegment = segmentCurrent;

            segmentCurrent->starttime = recordCurrent->record->starttime;
            segmentCurrent->endtime = msr_endtime(recordCurrent->record);
            segmentCurrent->samprate = recordCurrent->record->samprate;
            segmentCurrent->sampletype = recordCurrent->record->sampletype;
            segmentCurrent->samplecnt = recordCurrent->record->samplecnt;
            // Calculate high-precision sample period
            segmentCurrent->hpdelta = (hptime_t) (( recordCurrent->record->samprate ) ?
                           (HPTMODULUS / recordCurrent->record->samprate) : 0.0);
            segmentCurrent->firstRecord = segmentCurrent->lastRecord = recordCurrent;
            recordCurrent->previous = NULL;
        }
        recordPrevious = recordCurrent->next;
        recordCurrent->next = NULL;
        recordCurrent = recordPrevious;
    }


    // Now loop over all segments, combine the records and free the msr
    // structures.
    idListCurrent = idListHead;
    while (idListCurrent != NULL)
    {
        segmentCurrent = idListCurrent->firstSegment;

        while (segmentCurrent != NULL) {
            if (segmentCurrent->datasamples) {
                free(segmentCurrent->datasamples);
            }
            // Allocate data via a callback function.
            if (unpack_data != 0) {
                segmentCurrent->datasamples = (void *) allocData(segmentCurrent->samplecnt, segmentCurrent->sampletype);
            }

            // Loop over all records, write the data to the buffer and free the msr structures.
            recordCurrent = segmentCurrent->firstRecord;
            data_offset = (long)(segmentCurrent->datasamples);
            while (recordCurrent != NULL) {
                datasize = recordCurrent->record->samplecnt * ms_samplesize(recordCurrent->record->sampletype);
                memcpy((void *)data_offset, recordCurrent->record->datasamples, datasize);
                // Free the record.
                msr_free(&(recordCurrent->record));
                // Increase the data_offset and the record.
                data_offset += (long)datasize;
                recordCurrent = recordCurrent->next;
            }
      
            segmentCurrent = segmentCurrent->next;
        }
        idListCurrent = idListCurrent->next;
    }
    return idListHead;
}



// The following is a copy of msr_unpack_data from unpack.c because it
// unfortunately is a static function.

// Some declarations for the following msr_unpack_data
flag unpackheaderbyteorder_2 = -2;
flag unpackdatabyteorder_2   = -2;
int unpackencodingformat_2   = -2;
int unpackencodingfallback_2 = -2;
char *UNPACK_SRCNAME_2 = NULL;

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
static int
msr_unpack_data ( MSRecord *msr, int swapflag, int verbose )
{
  int     datasize;             /* byte size of data samples in record  */
  int     nsamples;             /* number of samples unpacked           */
  int     unpacksize;           /* byte size of unpacked samples        */
  int     samplesize = 0;       /* size of the data samples in bytes    */
  const char *dbuf;
  int32_t    *diffbuff;
  int32_t     x0, xn;
  
  /* Sanity record length */
  if ( msr->reclen == -1 )
  {
      ms_log (2, "msr_unpack_data(%s): Record size unknown\n",
              UNPACK_SRCNAME_2);
      return MS_NOTSEED;
  }
  switch (msr->encoding)
  {
    case DE_ASCII:
      samplesize = 1; break;
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
      samplesize = 4; break;
    case DE_FLOAT64:
      samplesize = 8; break;
    default:
      samplesize = 0; break;
  }
  
  /* Calculate buffer size needed for unpacked samples */
  unpacksize = msr->samplecnt * samplesize;
  
  /* (Re)Allocate space for the unpacked data */
  if ( unpacksize > 0 )
    {
      msr->datasamples = realloc (msr->datasamples, unpacksize);
      
      if ( msr->datasamples == NULL )
        {
          ms_log (2, "msr_unpack_data(%s): Cannot (re)allocate memory\n",
                  UNPACK_SRCNAME_2);
          return MS_GENERROR;
        }
    }
  else
    {
      if ( msr->datasamples )
        free (msr->datasamples);
      msr->datasamples = 0;
      msr->numsamples = 0;
    }
  
  datasize = msr->reclen - msr->fsdh->data_offset;
  dbuf = msr->record + msr->fsdh->data_offset;
  
  if ( verbose > 2 )
    ms_log (1, "%s: Unpacking %d samples\n",
            UNPACK_SRCNAME_2, msr->samplecnt);
  
  /* Decide if this is a encoding that we can decode */
  switch (msr->encoding)
    {
      
    case DE_ASCII:
      if ( verbose > 1 )
        ms_log (1, "%s: Found ASCII data\n", UNPACK_SRCNAME_2);
      
      nsamples = msr->samplecnt;
      memcpy (msr->datasamples, dbuf, nsamples);
      msr->sampletype = 'a';      
      break;
      
    case DE_INT16:
      if ( verbose > 1 )
        ms_log (1, "%s: Unpacking INT-16 data samples\n", UNPACK_SRCNAME_2);
      
      nsamples = msr_unpack_int_16 ((int16_t *)dbuf, msr->samplecnt,
                                    msr->samplecnt, msr->datasamples,
                                    swapflag);
      msr->sampletype = 'i';
      break;
      
    case DE_INT32:
      if ( verbose > 1 )
        ms_log (1, "%s: Unpacking INT-32 data samples\n", UNPACK_SRCNAME_2);
      
      nsamples = msr_unpack_int_32 ((int32_t *)dbuf, msr->samplecnt,
                                    msr->samplecnt, msr->datasamples,
                                    swapflag);
      msr->sampletype = 'i';
      break;
      
    case DE_FLOAT32:
      if ( verbose > 1 )
        ms_log (1, "%s: Unpacking FLOAT-32 data samples\n", UNPACK_SRCNAME_2);
      
      nsamples = msr_unpack_float_32 ((float *)dbuf, msr->samplecnt,
                                      msr->samplecnt, msr->datasamples,
                                      swapflag);
      msr->sampletype = 'f';
      break;
      
    case DE_FLOAT64:
      if ( verbose > 1 )
        ms_log (1, "%s: Unpacking FLOAT-64 data samples\n", UNPACK_SRCNAME_2);
      
      nsamples = msr_unpack_float_64 ((double *)dbuf, msr->samplecnt,
                                      msr->samplecnt, msr->datasamples,
                                      swapflag);
      msr->sampletype = 'd';
      break;
      
    case DE_STEIM1:
      diffbuff = (int32_t *) malloc(unpacksize);
      if ( diffbuff == NULL )
        {
          ms_log (2, "msr_unpack_data(%s): Cannot allocate diff buffer\n",
                  UNPACK_SRCNAME_2);
          return MS_GENERROR;
        }
      
      if ( verbose > 1 )
        ms_log (1, "%s: Unpacking Steim-1 data frames\n", UNPACK_SRCNAME_2);
      
      nsamples = msr_unpack_steim1 ((FRAME *)dbuf, datasize, msr->samplecnt,
                                    msr->samplecnt, msr->datasamples, diffbuff, 
                                    &x0, &xn, swapflag, verbose);
      msr->sampletype = 'i';
      free (diffbuff);
      break;
      
    case DE_STEIM2:
      diffbuff = (int32_t *) malloc(unpacksize);
      if ( diffbuff == NULL )
        {
          ms_log (2, "msr_unpack_data(%s): Cannot allocate diff buffer\n",
                  UNPACK_SRCNAME_2);
          return MS_GENERROR;
        }
      
      if ( verbose > 1 )
        ms_log (1, "%s: Unpacking Steim-2 data frames\n", UNPACK_SRCNAME_2);
      
      nsamples = msr_unpack_steim2 ((FRAME *)dbuf, datasize, msr->samplecnt,
                                    msr->samplecnt, msr->datasamples, diffbuff,
                                    &x0, &xn, swapflag, verbose);
      msr->sampletype = 'i';
      free (diffbuff);
      break;
      
    case DE_GEOSCOPE24:
    case DE_GEOSCOPE163:
    case DE_GEOSCOPE164:
      if ( verbose > 1 )
        {
          if ( msr->encoding == DE_GEOSCOPE24 )
            ms_log (1, "%s: Unpacking GEOSCOPE 24bit integer data samples\n",
                    UNPACK_SRCNAME_2);
          if ( msr->encoding == DE_GEOSCOPE163 )
            ms_log (1, "%s: Unpacking GEOSCOPE 16bit gain ranged/3bit exponent data samples\n",
                    UNPACK_SRCNAME_2);
          if ( msr->encoding == DE_GEOSCOPE164 )
            ms_log (1, "%s: Unpacking GEOSCOPE 16bit gain ranged/4bit exponent data samples\n",
                    UNPACK_SRCNAME_2);
        }
      
      nsamples = msr_unpack_geoscope (dbuf, msr->samplecnt, msr->samplecnt,
                                      msr->datasamples, msr->encoding, swapflag);
      msr->sampletype = 'f';
      break;
      
    case DE_CDSN:
      if ( verbose > 1 )
        ms_log (1, "%s: Unpacking CDSN encoded data samples\n", UNPACK_SRCNAME_2);
      
      nsamples = msr_unpack_cdsn ((int16_t *)dbuf, msr->samplecnt, msr->samplecnt,
                                  msr->datasamples, swapflag);
      msr->sampletype = 'i';
      break;
      
    case DE_SRO:
      if ( verbose > 1 )
        ms_log (1, "%s: Unpacking SRO encoded data samples\n", UNPACK_SRCNAME_2);
      
      nsamples = msr_unpack_sro ((int16_t *)dbuf, msr->samplecnt, msr->samplecnt,
                                 msr->datasamples, swapflag);
      msr->sampletype = 'i';
      break;
      
    case DE_DWWSSN:
      if ( verbose > 1 )
        ms_log (1, "%s: Unpacking DWWSSN encoded data samples\n", UNPACK_SRCNAME_2);
      
      nsamples = msr_unpack_dwwssn ((int16_t *)dbuf, msr->samplecnt, msr->samplecnt,
                                    msr->datasamples, swapflag);
      msr->sampletype = 'i';
      break;
      
    default:
      ms_log (2, "%s: Unsupported encoding format %d (%s)\n",
              UNPACK_SRCNAME_2, msr->encoding, (char *) ms_encodingstr(msr->encoding));
      
      return MS_UNKNOWNFORMAT;
    }
  
  return nsamples;
} /* End of msr_unpack_data() */
