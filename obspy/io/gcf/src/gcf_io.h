//
//  LOG:
//    2017-07-12  Adjusted meaning of GcfSeg.blk to be oldest block number rather
//                  than lowest (which was not set proper anyway), CPS
//    2020-02-29  Added attributes FIC and RIC and new error code (11) to 
//                  struct GcfSeg, CPS
//    2020-03-01  Added attribute sysType to struct GcfSeg, CPS
//    2020-03-02  Added error code 21
//    2020-03-05  Minor adjustments to silence compiler warnings, cps
//    2020-03-20  Added declaration on merge_GcfFile(), cps
//    2020-03-21  Minor changes in comments/docs
//    2020-04-15  Added inclusion of time.h
//    2022-03-10  Removed use of time.h by changing use of time_t to derrived arithmetic type gtime,
//                 Added platform specific dependenscies
//    2022-03-17  Added mode=3 to read_gcf
//
// (C) Peter Schmidt 2017-2022

#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#define UNKNOWN -2147483647
#define MAX_DATA_BLOCK 1004


/* Set platform specific defines */
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  #include <windows.h>
  #include <sys/types.h>
  #include <io.h>
  #include <BaseTsd.h>
  typedef SSIZE_T ssize_t; 
  
  #ifndef O_RDONLY
    #define O_RDONLY _O_RDONLY
  #endif 
  
  #ifndef O_WRONLY
    #define O_WRONLY _O_WRONLY
  #endif  
  
  #ifndef O_CREAT
    #define O_CREAT _O_CREAT
  #endif  
  
  #ifndef O_TRUNC
    #define O_TRUNC _O_TRUNC
  #endif 
  
  #ifndef close
    #define close(fd) _close(fd)
  #endif
  
  #ifndef write
    #define write(fd, buffer, count) _write(fd, buffer, count)
  #endif
  
  #ifndef read
    #define read(fd, buffer, count) _read(fd, buffer, count)
  #endif
  
  #define open_w(path, flags, mode) _open(path, flags, mode)
  #define open_r(path, flags) _open(path, flags)
  #define ORFLAG _O_RDONLY | _O_BINARY
  #define OWFLAG _O_WRONLY | _O_CREAT | _O_TRUNC  | _O_BINARY
  #define FPERM  _S_IWRITE

  /* For MSVC 2012 and earlier define standard int types, otherwise use inttypes.h */
  #if defined(_MSC_VER) && _MSC_VER <= 1700
    typedef unsigned char uint8_t;
    typedef signed short int int16_t;
    typedef unsigned short int uint16_t;
    typedef signed int int32_t;
    typedef unsigned int uint32_t;
  #else
    #include <inttypes.h>
  #endif
#else
  #include <unistd.h>
  #define open_w(path, flags, mode) open(path, flags, mode)
  #define open_r(path, flags) open(path, flags)
  #define ORFLAG O_RDONLY 
  #define OWFLAG O_WRONLY | O_CREAT | O_TRUNC
  #define FPERM  S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH
#endif


typedef uint8_t uint8;
typedef int16_t int16;
typedef uint16_t uint16;
typedef int32_t int32;
typedef uint32_t uint32;
typedef uint32_t gtime;    // gcf time use at present 15 bits to represent days since 1989-11-17 and 17 bits to represent 
                           // seconds since midnight, hence maximum date is 2079-08-04. Represented as seconds only 32 bits
                           // are required even if time is converted to seconds since 1970-01-01 00:00:00


/* Struct to hold header of a data block */
typedef struct BH {
   uint32 systemID;  // system ID, 3 formats depending on value in bits 30 and 31
   uint32 streamID;  // stream ID, bit 31 reserved and should be unset (0)
   uint32 time;      // time days since 1989-11-17 of first data sample in block in bits 17-31 and seconds since midnight (UTC) (86400 indicates addition of leap second) in bits 0-16
   uint8 ttl;        // info on decimation filters 
   uint8 sps;        // sampling rate (1/second) - see GCF reference for handling of sps < 1 and sps > 250
   uint8 comp;       // compression: 1 - 32-bit; 2 - 16-bit; 4 - 8-bit (all signed first difference) and numerator of start time for sps > 250 Hz
   uint8 nrec;       // Number of 4-byte data records in block (max 251)
} BH;


/* Container of continous data segment */
typedef struct {
   char streamID[7];    // streamID
   char systemID[7];    // systemID (max 4 chars if sysType == 2, max 5 chars if sysType == 1, else max 6 chars)
   gtime start;        // start time of data, NOTE: must not pre-date 1989-11-17 00:00:00Z (unixtime: 627264000)
   int t_numerator;     // numerator of start fractional offset (only for sps > 250), NOTE must not be > t_denominator, see
                        //  Guralps GCF documentation (version E) for valid numbers
   int t_denominator;   // denominator of start fractional offset (only for sps > 250), NOTE: Note used in write_gcf(), see
                        //  Guralps GCF documentation (version E) for valid numbers
   int t_leap;          // 1 if block starts at or contains a leap second, else 0 
   int gain;            // variable gain setting, 0 means that it is not used
   int sysType;         // SystemID type, 0 - regular, 1 - extended, 2 - double extended
   int type;            // Instrument type, if sysType > 0 and gain != -1, digitizer can be deduced from:
                        //  sysType:  type:  Digitizer
                        //    0        -     Unknown - probably DM24 Mk2
                        //    1        0     DM24
                        //    1        1     CD24
                        //    2        0     Affinity
                        //    2        1     Minimus 
   int ttl;             // Info on sequence of decimation filters used
   int blk;             // block number for oldest blocks in segment numbering starts at 0 for first block infile
   int err;             // Error code in case something went wrong:
                        //  -1  - Not a data block - this could be expanded to indicate block type
                        //   0  - No errors
                        //   1  - stream ID uses extended system ID format
                        //   2  - stream ID uses double-extended system ID format
                        //   3  - Unknown compression 
                        //   4  - To few/many data samples indicated in header
                        //   5  - Start of first data sample is negative, error may be over-written by codes > 9
                        //   9  - failure decoding some header value
                        //   10 - failure decoding some data value (last data != RIC)
                        //   11 - first difference not 0 (NOTE! this may not be a critical error, more of a warning)
                        //   21 - errors 10+11
   int sps;             // sampling rate (1/second)
   int sps_denom;       // denominator for fractional sampling rate
   int compr;           // compression used, will be that of first block if merged
   int32 FIC;           // Forward integration constant
   int32 RIC;           // Reverse integration constant
   int32 n_data;        // samples in segment and in data vector if allocated 
   int32 n_alloc;       // actual allocated size of data vector, may be 0 if only header have been read
   int32 *data;         // data vector
} GcfSeg;


/* Container of content in a file */
typedef struct {
   int n_blk;       // Number of blocks read from file 
   int n_seg;       // Number of continous segments formed from blocks in file 
   int n_alloc;     // Number of segments for which space have been allocated
   int n_errHead;   // Number of headers with errors
   int n_errData;   // Number of blocks where data could not be properly parsed
   GcfSeg *seg; 
} GcfFile;


int is_LittleEndian_gcf(void);          /* return True (1) if on little endian machine */


/* functions init_Gcf<Seg|File>() initiates values in a GcfSeg|GcfFile struct 
 *  but do not allocate (see functions realloc_Gcf<Seg|File>() for allocation)
 * 
 *  ARGUMENT
 *   obj          struct to initiate/reset
 *   skip_data    [GcfSeg only] if 1 attributes rellated to data vector will
 *                  not be initiated/reset. Use 1 whenever GcfSeg have been 
 *                  allocated but not free'd, else you will have a memory leak
 */
void init_GcfSeg(GcfSeg *obj, int skip_data);
void init_GcfFile(GcfFile *obj); 


/* functions realloc_Gcf<Seg|File>() reallocates space for Data|GcfSeg in a GcfSeg|GcfFile
 * struct and updates associated attributes proper
 * 
 * functions will pass silently for size < 0 and size equal to already allocated size.
 * 
 * ARGUMENTS
 *  obj       struct to reallocate
 *  size      size needed 
 */
void realloc_GcfSeg(GcfSeg *obj, int32 size);
void realloc_GcfFile(GcfFile *obj, int size);


void free_GcfSeg(GcfSeg *obj);         /* Function to free memory allocated in a GcfSeg struct*/
void free_GcfFile(GcfFile *obj);       /* Function to free memory allocated in a GcfFile struct*/


/* function gcfSegSps() returns the sampling rate of a GcfSeg */
double gcfSegSps(GcfSeg *seg);


/* function gcfSegStart() returns the start time of a GcfSeg
 * in unix time. If segment start on leap second start will be
 * shifted one second back in time */
double gcfSegStart(GcfSeg *seg);


/* function gcfSegEnd() returns the end time of a GcfSeg in unix time */
double gcfSegEnd(GcfSeg *seg);


/* function cmpSegHead() returns True (non-zero integer) if the attributes:
 *   systemID, streamID, gain, type, sysType, ttl, sps, sps_denom 
 *  are equal in two GcfSeg, else function returns False (0) */
int cmpSegHead(GcfSeg *s1, GcfSeg *s2);


/* function CheckSegAligned() checks if the data vector in a GcfSeg s2 is a 
 * continuation in time of the data vector in GcfSeg s1
 *
 * NOTE: sampling rate must be same for both GcfSeg and if one segment is not a 
 *        data block the other must neither be
 *       if s1 and s2 are aligned but s2 starts with a leap second function will 
 *        report overlap        
 * 
 * ARGUMENT
 *   s1       GcfSeg to check end of data
 *   s2       GcfSeg to check start of data
 *   tol      misalignment tolerance as fraction of a sample (<0.5)
 * 
 * RETURN
 *  if successfull function returns -1, 0, 1 if s1.end < s2.start (gap), s1.end == s2.start (aligned),
 *  se.end > s2.start (potential overlap) respectively, else function returns -2 (e.g. sampling rate is not same
 *  or one but not both segments are not data blocks)
 */
int CheckSegAligned(GcfSeg *s1, GcfSeg *s2, double tol);


/* function add_GcfSeg() copies a GcfSeg struct to a GcfFile struct
 *
 * NOTE: Function will not merge segment starting on leap second with older segment
 *        nor will it merge segments with errors set
 * 
 * ARGUMENTS
 *   obj        GcfFile to add to
 *   seg        segment to add
 *   mode       if < 2 function will merge segment if aligned with 
 *                segment already in GcfFile. NOTE: function will not handle overlaps
 *   tol        tolerance expressed as fraction of a sample (<0.5) when checking if
 *                segments are aligned
 */
void add_GcfSeg(GcfFile *obj, GcfSeg seg, int mode, double tol);


/* function merge_GcfFile() merges the segments in a GcfFile object if aligned
 * and metadata agrees
 *
 * NOTE: Function will not merge segment starting on leap second with older segment
 *        nor will it merge segments with errors set
 *       Function will not remove overlap by merging if input GcfFile contains 
 *        header only
 *
 *  ARGUMENTS:
 *   obj       GcfFile to merge segments in
 *   mode      if zero and segments overlap in time function will remove overlap by
 *              merging if overlapping data is same.
 *   tol       tolerance expressed as fraction of a sample (<0.5) when checking if
 *               segments are aligned
 */
void merge_GcfFile(GcfFile *obj, int mode, double tol);


/* function parse_gcf_block() parses a 1024 byte gcf data block
 * 
 *  ARGUMENTS
 *   buffer      1024 byte buffer holding the data block
 *   seg         GcfSeg struct to hold parsed data block, should be
 *                properly preallocated to at least 1004 bytes prior
 *                to input if argmunet mode >= 0
 *   mode        if < 0 only header values will be decoded
 *   endian      0 if current machine is Big endian else non-zero (True)
 * 
 *  RETURN
 *   function returns 0 if all went well else the error code set in seg.err,
 *   NOTE: if return code is 3 or 4 no data have been decoded
 */
int parse_gcf_block(unsigned char buffer[1024], GcfSeg *seg, int mode, int endian);


/* function read_gcf() parses a gcf data file
 * 
 *  ARGUMENTS
 *   f      path to file to parse
 *   obj    GcfFile struct, should be properly initiated on input
 *           and will hold content of file, NOTE: if obj.n_seg > 1 
 *           data contains gaps or inconsitencies in streamID and/or
 *           sampling rate.
 *   mode   -2   read headers only but do not merge blocks
 *          -1   read headers only and merge blocks if possible 
 *           0   read headers and data and merge blocks and remove overlaps if possible
 *           1   read headers and data and merge blocks but keep overlaps
 *           2   read headers and data but do not merge blocks
 *           3   read headers and data of first data block only
 *          NOTE: if blocks are not merged they may not be ordered in time but will
 *                be ordered by presence in file, further blocks with errors in header
 *                or where decoding data failed will not be merged.
 *                
 *  RETURN:
 *   function returns 0 if all went well, -1 if file could not be opened, 1 if file is
 *   not data file, Note, even if 0 is returned file may still contain errors in individual
 *   segments.
 */
int read_gcf(const char *f, GcfFile *obj, int mode);


/* function write_gcf() writes a gcf data file
 * 
 * ARGUMETS
 *  f       file to write
 *  obj     data to write
 * 
 * RETURN:
 *  -2 Failed to write to disc
 *  -1 Failed to open file to write to
 *  0 if successful 
 *  1 if no data or inconsistent segment info
 *  2 if unsupported samplingrate
 *  3 erronous fractional start time
 *  4 unsupported gain
 *  5 erronous instrument type
 *  6 to many characters in systemID
 */
int write_gcf(const char *f, GcfFile *obj);



