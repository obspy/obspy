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
#include <math.h>
#include <time.h>
#include <ctype.h>

#include "libmseed/libmseed.h"
#include "libmseed/unpackdata.h"


// Similar to MS_ISVALIDBLANK but also works for blocks consisting only of
// spaces.
#define OBSPY_ISVALIDBLANK(X) (                            \
  (isdigit ((int) *(X))   || !*(X)   || *(X) == ' ') &&    \
  (isdigit ((int) *(X+1)) || !*(X+1) || *(X+1) == ' ') &&  \
  (isdigit ((int) *(X+2)) || !*(X+2) || *(X+2) == ' ') &&  \
  (isdigit ((int) *(X+3)) || !*(X+3) || *(X+3) == ' ') &&  \
  (isdigit ((int) *(X+4)) || !*(X+4) || *(X+4) == ' ') &&  \
  (isdigit ((int) *(X+5)) || !*(X+5) || *(X+5) == ' ') &&  \
  (*(X+6) ==' ') && (*(X+7) ==' ') && (*(X+8) ==' ') &&  \
  (*(X+9) ==' ') && (*(X+10)==' ') && (*(X+11)==' ') &&  \
  (*(X+12)==' ') && (*(X+13)==' ') && (*(X+14)==' ') &&  \
  (*(X+15)==' ') && (*(X+16)==' ') && (*(X+17)==' ') &&  \
  (*(X+18)==' ') && (*(X+19)==' ') && (*(X+20)==' ') &&  \
  (*(X+21)==' ') && (*(X+22)==' ') && (*(X+23)==' ') &&  \
  (*(X+24)==' ') && (*(X+25)==' ') && (*(X+26)==' ') &&  \
  (*(X+27)==' ') && (*(X+28)==' ') && (*(X+29)==' ') &&  \
  (*(X+30)==' ') && (*(X+31)==' ') && (*(X+32)==' ') &&  \
  (*(X+33)==' ') && (*(X+34)==' ') && (*(X+35)==' ') &&  \
  (*(X+36)==' ') && (*(X+37)==' ') && (*(X+38)==' ') &&  \
  (*(X+39)==' ') && (*(X+40)==' ') && (*(X+41)==' ') &&  \
  (*(X+42)==' ') && (*(X+43)==' ') && (*(X+44)==' ') &&  \
  (*(X+45)==' ') && (*(X+46)==' ') && (*(X+47)==' ') )


float roundf_(float x)
{
   return x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f);
}


// Dummy wrapper around malloc.
void * allocate_bytes(int count) {
    return malloc(count);
}


// Linkable container of MSRecords
typedef struct LinkedRecordList_s {
    struct MSRecord_s      *record;       // This record
    struct LinkedRecordList_s  *previous; // The previous record container
    struct LinkedRecordList_s  *next;     // The next record container
}
LinkedRecordList;

// Container for a continuous linked list of records.
typedef struct ContinuousSegment_s {
    hptime_t starttime;           // Time of the first sample
    hptime_t endtime;             // Time of the last sample
    double samprate;              // Sample rate
    char sampletype;              // Sampletype
    hptime_t hpdelta;             // High precission sample period
    int64_t recordcnt;            // Record count for segment.
    int64_t samplecnt;            // Total sample count
    int8_t encoding;              // Encoding of the first record.
    int8_t byteorder;             // Byteorder of the first record.
    int32_t reclen;               // Record length of the first record.
    /* Timing quality is a vendor specific value from 0 to 100% of maximum
     * accuracy, taking into account both clock quality and data flags. */
    uint8_t timing_qual;
    /* type of calibration available, BLK 300 = 1, BLK 310 = 2, BLK 320 = 3
     * BLK 390 = 4, BLK 395 = -2 */
    int8_t calibration_type;
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

// Print function that does nothing.
void empty_print(char *string) {}


// Simple function logging nice error messages.
void log_error(int errcode, int offset) {
    switch ( errcode ) {
        case MS_ENDOFFILE:
            ms_log(1, "readMSEEDBuffer(): Unexpected end of file when "
                      "parsing record starting at offset %d. The rest "
                      "of the file will not be read.\n", offset);
            break;
        case MS_GENERROR:
            ms_log(1, "readMSEEDBuffer(): Generic error when parsing "
                      "record starting at offset %d. The rest of the "
                      "file will not be read.\n", offset);
            break;
        case MS_NOTSEED:
            // This is likely not called at all as non-SEED records will
            // verbosely be skipped and ObsPy keeps trying until it finds a
            // record.
            ms_log(1, "readMSEEDBuffer(): Record starting at offset "
                      "%d is not valid SEED. The rest of the file "
                      "will not be read.\n", offset);
            break;
        case MS_WRONGLENGTH:
            ms_log(1, "readMSEEDBuffer(): Length of data read was not "
                      "correct when parsing record starting at "
                      "offset %d. The rest of the file will not be "
                      "read.\n", offset);
            break;
        case MS_OUTOFRANGE:
            ms_log(1, "readMSEEDBuffer(): SEED record length out of "
                      "range for record starting at offset %d. The "
                      "rest of the file will not be read.\n", offset);
            break;
        case MS_UNKNOWNFORMAT:
            ms_log(1, "readMSEEDBuffer(): Unknown data encoding "
                      "format for record starting at offset %d. The "
                      "rest of the file will not be read.\n", offset);
            break;
        case MS_STBADCOMPFLAG:
            ms_log(1, "readMSEEDBuffer(): Invalid STEIM compression "
                      "flag(s) in record starting at offset %d. The "
                      "rest of the file will not be read.\n", offset);
            break;
        default:
            ms_log(1, "readMSEEDBuffer(): Unknown error '%d' in "
                      "record starting at offset %d. The rest of the "
                      "file will not be read.\n", errcode, offset);
            break;
    }
}


// Helper function to connect libmseed's logging and error messaging to Python
// functions.
void setupLogging(void (*diag_print) (char*),
                  void (*log_print) (char*)) {
    ms_loginit(log_print, "INFO: ", diag_print, "ERROR: ");
}


// Function that reads from a MiniSEED binary file from a char buffer and
// returns a LinkedIDList.
LinkedIDList *
readMSEEDBuffer (char *mseed, int buflen, Selections *selections, flag
                 unpack_data, int reclen, flag verbose, flag details,
                 int header_byteorder, long long (*allocData) (int, char))
{
    int retcode = 0;
    int retval = 0;
    flag swapflag = 0;
    flag bigendianhost = ms_bigendianhost();

    // current offset of mseed char pointer
    int offset = 0;

    // Unpack without reading the data first
    flag dataflag = 0;

    // the timing_qual of BLK 1001
    uint8_t timing_qual = 0xFF;

    // the calibration type, availability of BLK 300, 310, 320, 390, 395
    int8_t calibration_type = -1;

    // Init all the pointers to NULL. Most compilers should do this anyway.
    LinkedIDList * idListHead = NULL;
    LinkedIDList * idListCurrent = NULL;
    LinkedIDList * idListLast = NULL;
    MSRecord *msr = NULL;
    ContinuousSegment * segmentCurrent = NULL;
    hptime_t lastgap = 0;
    hptime_t hptimetol = 0;
    hptime_t nhptimetol = 0;
    long long data_offset;
    LinkedRecordList *recordHead = NULL;
    LinkedRecordList *recordPrevious = NULL;
    LinkedRecordList *recordCurrent = NULL;
    int datasize;
    int record_count = 0;

    if (header_byteorder >= 0) {
        // Enforce little endian.
        if (header_byteorder == 0) {
            MS_UNPACKHEADERBYTEORDER(0);
        }
        // Enforce big endian.
        else {
            MS_UNPACKHEADERBYTEORDER(1);
        }
    }
    else {
        MS_UNPACKHEADERBYTEORDER(-1);
    }

    // Read all records and save them in a linked list.
    while (offset < buflen) {
        msr = msr_init(NULL);
        if ( msr == NULL ) {
            ms_log (2, "readMSEEDBuffer(): Error initializing msr\n");
            return NULL;
        }
        if (verbose > 1) {
            ms_log(0, "readMSEEDBuffer(): calling msr_parse with "
                      "mseed+offset=%d+%d, buflen=%d, reclen=%d, dataflag=%d, verbose=%d\n",
                      mseed, offset, buflen, reclen, dataflag, verbose);
        }

        // If the record length is given, make sure at least that amount of data is available.
        if (reclen != -1) {
            if (offset + reclen > buflen) {
                ms_log(1, "readMSEEDBuffer(): Last reclen exceeds buflen, skipping.\n");
                msr_free(&msr);
                break;
            }
        }
        // Otherwise assume the smallest possible record length and assure that enough
        // data is present.
        else {
            if (offset + MINRECLEN > buflen) {
                ms_log(1, "readMSEEDBuffer(): Last record only has %i byte(s) which "
                          "is not enough to constitute a full SEED record. Corrupt data? "
                          "Record will be skipped.\n", buflen - offset);
                msr_free(&msr);
                break;
            }
        }

        // Skip empty or noise records.
        if (OBSPY_ISVALIDBLANK(mseed + offset)) {
            offset += MINRECLEN;
            continue;
        }

        // Pass (buflen - offset) because msr_parse() expects only a single record. This
        // way libmseed can take care to not overstep bounds.
        // Return values:
        //   0 : Success, populates the supplied MSRecord.
        //  >0 : Data record detected but not enough data is present, the
        //       return value is a hint of how many more bytes are needed.
        //  <0 : libmseed error code (listed in libmseed.h) is returned.
        retcode = msr_parse ((mseed+offset), buflen - offset, &msr, reclen, dataflag, verbose);
        // If its not a record, skip MINRECLEN bytes and try again.
        if (retcode == MS_NOTSEED) {
            ms_log(1,
                   "readMSEEDBuffer(): Not a SEED record. Will skip bytes "
                   "%i to %i.\n", offset, offset + MINRECLEN - 1);
            msr_free(&msr);
            offset += MINRECLEN;
            continue;
        }
        // Handle all other error.
        else if (retcode < 0) {
            log_error(retcode, offset);
            msr_free(&msr);
            break;
        }
        // Data missing at the end.
        else if (retcode > 0 && retcode >= (buflen - offset)) {
            log_error(MS_ENDOFFILE, offset);
            msr_free(&msr);
            break;
        }
        // Lacking Blockette 1000.
        else if ( retcode > 0 && retcode < (buflen - offset)) {
            // Check if the remaining bytes can exactly make up a record length.
            int r_bytes = buflen - offset;
            float exp = log10((float)r_bytes) / log10(2.0);
            if ((fmodf(exp, 1.0) < 0.0000001) && ((int)roundf_(exp) >= 7) && ((int)roundf_(exp) <= 256)) {

                retcode = msr_parse((mseed + offset), buflen - offset, &msr, r_bytes, dataflag, verbose);

                if ( retcode != 0 ) {
                    log_error(retcode, offset);
                    msr_free(&msr);
                    break;
                }

            }
            else {
                msr_free(&msr);
                break;
            }
        }

        if (offset + msr->reclen > buflen) {
            ms_log(1, "readMSEEDBuffer(): Last msr->reclen exceeds buflen, skipping.\n");
            msr_free(&msr);
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


        // Figure out if the byte-order of the data has to be swapped.
        swapflag = 0;
        // If blockette 1000 is present, use it.
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
        // Otherwise assume the data has the same byte order as the header.
        // This needs to be done on the raw header bytes as libmseed only returns
        // header fields in the native byte order.
        else {
            unsigned char* _t = (unsigned char*)mseed + offset + 20;
            unsigned int year = _t[0] | _t[1] << 8;
            unsigned int day = _t[2] | _t[3] << 8;
            // Swap data if header needs to be swapped.
            if (!MS_ISVALIDYEARDAY(year, day)) {
                swapflag = 1;
            }
        }

        // Actually unpack the data if the flag is not set and if the data
        // offset is valid.
        if ((unpack_data != 0) && (msr->fsdh->data_offset >= 48) &&
            (msr->fsdh->data_offset < msr->reclen) &&
            (msr->samplecnt > 0)) {
            retval = msr_unpack_data (msr, swapflag, verbose);
        }

        if ( retval > 0 ) {
            msr->numsamples = retval;
        }

        if ( msr->fsdh->start_time.fract > 9999 ) {
            ms_log(1, "readMSEEDBuffer(): Record with offset=%d has a "
                      "fractional second (.0001 seconds) of %d. This is not "
                      "strictly valid but will be interpreted as one or more "
                      "additional seconds.",
                      offset, msr->fsdh->start_time.fract);
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
        if (details == 1) {
            /* extract information on calibration BLKs */
            calibration_type = -1;
            if (recordCurrent->record->blkts) {
                BlktLink *cur_blkt = recordCurrent->record->blkts;
                while (cur_blkt) {
                    switch (cur_blkt->blkt_type) {
                    case 300:
                        calibration_type = 1;
                        break;
                    case 310:
                        calibration_type = 2;
                        break;
                    case 320:
                        calibration_type = 3;
                        break;
                    case 390:
                        calibration_type = 4;
                        break;
                    case 395:
                        calibration_type = -2;
                        break;
                    default:
                        break;
                    }
                    cur_blkt = cur_blkt->next;
                }
            }
            /* extract information based on timing quality */
            timing_qual = 0xFF;
            if (recordCurrent->record->Blkt1001 != 0) {
                timing_qual = recordCurrent->record->Blkt1001->timing_qual;
            }
        }
        if ( segmentCurrent != NULL &&

             // This is important for zero data record coupled with not unpacking
             // the data. It needs to be split in two places: Before the zero data
             // record and after it.
             recordCurrent->record->samplecnt > 0 && segmentCurrent->samplecnt > 0 &&

             segmentCurrent->sampletype == recordCurrent->record->sampletype &&
             // Test the default sample rate tolerance: abs(1-sr1/sr2) < 0.0001
             MS_ISRATETOLERABLE (segmentCurrent->samprate, recordCurrent->record->samprate) &&
             // Check if the times are within the time tolerance
             lastgap <= hptimetol && lastgap >= nhptimetol &&
             segmentCurrent->timing_qual == timing_qual &&
             segmentCurrent->calibration_type == calibration_type) {
            recordCurrent->previous = segmentCurrent->lastRecord;
            segmentCurrent->lastRecord = segmentCurrent->lastRecord->next = recordCurrent;
            segmentCurrent->recordcnt += 1;
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

            // These will be set to the value of the first record. They are
            // not used anywhere but just serve informational purposes.
            segmentCurrent->encoding = recordCurrent->record->encoding;
            segmentCurrent->byteorder = recordCurrent->record->byteorder;
            segmentCurrent->reclen = recordCurrent->record->reclen;

            segmentCurrent->starttime = recordCurrent->record->starttime;
            segmentCurrent->endtime = msr_endtime(recordCurrent->record);
            segmentCurrent->samprate = recordCurrent->record->samprate;
            segmentCurrent->sampletype = recordCurrent->record->sampletype;
            segmentCurrent->recordcnt = 1;
            segmentCurrent->samplecnt = recordCurrent->record->samplecnt;
            // Calculate high-precision sample period
            segmentCurrent->hpdelta = (hptime_t) (( recordCurrent->record->samprate ) ?
                           (HPTMODULUS / recordCurrent->record->samprate) : 0.0);
            segmentCurrent->timing_qual = timing_qual;
            segmentCurrent->calibration_type = calibration_type;
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
            data_offset = (long long)(segmentCurrent->datasamples);
            while (recordCurrent != NULL) {
                datasize = recordCurrent->record->samplecnt * ms_samplesize(recordCurrent->record->sampletype);
                memcpy((void *)data_offset, recordCurrent->record->datasamples, datasize);
                // Free the record.
                msr_free(&(recordCurrent->record));
                // Increase the data_offset and the record.
                data_offset += (long long)datasize;
                recordCurrent = recordCurrent->next;
            }

            segmentCurrent = segmentCurrent->next;
        }
        idListCurrent = idListCurrent->next;
    }
    return idListHead;
}
