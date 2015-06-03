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
    hptime_t starttime;                     // Time of the first sample
    hptime_t endtime;                       // Time of the last sample
    double samprate;                        // Sample rate
    char sampletype;                        // Sampletype
    hptime_t hpdelta;                       // High precission sample period
    int64_t samplecnt;                         // Total sample count
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


// Function that reads from a MiniSEED binary file from a char buffer and
// returns a LinkedIDList.
LinkedIDList *
readMSEEDBuffer (char *mseed, int buflen, Selections *selections, flag
                 unpack_data, int reclen, flag verbose, flag details,
                 int header_byteorder, long (*allocData) (int, char),
                 void (*diag_print) (char*), void (*log_print) (char*))
{
    int retcode = 0;
    int retval = 0;
    flag swapflag = 0;

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
    long data_offset;
    LinkedRecordList *recordHead = NULL;
    LinkedRecordList *recordPrevious = NULL;
    LinkedRecordList *recordCurrent = NULL;
    int datasize;
    int record_count = 0;

    // A negative verbosity suppressed as much as possible.
    if (verbose < 0) {
        ms_loginit(&empty_print, NULL, &empty_print, NULL);
    }
    else {
        ms_loginit(log_print, "INFO: ", diag_print, "ERROR: ");
    }

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

    //
    // Read all records and save them in a linked list.
    //
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
            if (offset + 256 > buflen) {
                ms_log(1, "readMSEEDBuffer(): Last record only has %i byte(s) which "
                          "is not enough to constitute a full SEED record. Corrupt data? "
                          "Record will be skipped.\n", buflen - offset);
                msr_free(&msr);
                break;
            }
        }

        // Pass (buflen - offset) because msr_parse() expects only a single record. This
        // way libmseed can take care to not overstep bounds.
        retcode = msr_parse ( (mseed+offset), buflen - offset, &msr, reclen, dataflag, verbose);
        if (retcode != MS_NOERROR) {
            switch ( retcode ) {
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
                              "file will not be read.\n", retcode, offset);
                    break;
            }
            msr_free(&msr);
            break;
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

        // Determine the byte order swapflag only for the very first record.
        // The byte order should not change within the file.
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
             segmentCurrent->sampletype == recordCurrent->record->sampletype &&
             // Test the default sample rate tolerance: abs(1-sr1/sr2) < 0.0001
             MS_ISRATETOLERABLE (segmentCurrent->samprate, recordCurrent->record->samprate) &&
             // Check if the times are within the time tolerance
             lastgap <= hptimetol && lastgap >= nhptimetol &&
             segmentCurrent->timing_qual == timing_qual &&
             segmentCurrent->calibration_type == calibration_type) {
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
