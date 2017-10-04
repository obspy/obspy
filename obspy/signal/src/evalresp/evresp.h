/*==================================================================
 *                    evresp.h
 *=================================================================*/

/*
    8/28/2001 -- [ET]  Added 'WIN32' directives for Windows compiler
                       compatibility; added 'extern' to variable
                       declarations at end of file.
     8/2/2001 -- [ET]  Version 3.2.28:  Modified to allow command-line
                       parameters for frequency values to be missing
                       (default values used).
   10/21/2005 -- [ET]  Version 3.2.29:  Implemented interpolation of
                       amplitude/phase values from responses containing
                       List blockettes (55); modified message shown when
                       List blockette encountered; modified so as not to
                       require characters after 'units' specifiers like
                       "M" and "COUNTS"; modified to handle case where
                       file contains "B052F03 Location:" and nothing
                       after it on line; added warnings for unrecognized
                       parameters; fixed issue where program would crash
                       under Linux/UNIX ("segmentation fault") if input
                       response file contained Windows-style "CR/LF"
                       line ends; renamed 'strncasecmp()' function
                       to 'strnicmp()' when compiling under Windows.
    11/3/2005 -- [ET]  Version 3.2.30:  Modified to unwrap phase values
                       before interpolation and re-wrap after (if needed);
                       modified to detect when all requested interpolated
                       frequencies are out of range.
    1/18/2006 -- [ET]  Version 3.2.31:  Renamed 'regexp' functions to
                       prevent name clashes with other libraries.
    2/13/2006 -- [ET]  Version 3.2.32:  Moved 'use_delay()' function from
                       'evalresp.c' to 'evresp.c'; modified to close input
                       file when a single response file is specified.
    3/27/2006 -- [ID]  Version 3.2.33:  Added include_HEADERS target
                       "evr_spline.h" to "Makefile.am"; upgraded "missing"
                       script.
    3/28/2006 -- [ET]  Version 3.2.34:  Added 'free()' calls to "evresp.c"
                       to fix memory leaks when program functions used
                       in external applications.
     4/4/2006 -- [ET]  Version 3.2.35:  Modified to support channel-IDs with
                       location codes equal to an empty string.
    8/21/2006 -- [IGD] Version 3.2.36:  Added support for TESLA units.
   10/16/2006 -- [ET]  Version 3.2.37:  Added two more 'free()' calls to
                       "evresp.c" to fix memory leaks when program functions
                       are used in external applications.
    8/11/2006 -- [IGD] Libtoolized evresp library
    4/03/2007 -- [IGD] Added myLabel global variable which is used to add NSLC
                        labels in evalresp logging if --enable-log-label config
                        option is used
    5/14/2010 -- [ET]  Version 3.3.3:  Added "#define strcasecmp stricmp"
                       if Windows.
 */

#ifndef EVRESP_H
#define EVRESP_H

#include <stdio.h>
#include <math.h>
#include <ctype.h>
#include <setjmp.h>
#include <stdarg.h>

/* IGD 10/16/04 This is for Windows which does not use Makefile.am */
#ifdef VERSION
#define REVNUM VERSION
#else
#define REVNUM "3.3.3"
#endif

#define TRUE 1
#define FALSE 0
#define QUERY_DELAY -1
#define STALEN 64
#define NETLEN 64
#define LOCIDLEN 64
#define CHALEN 64
#define OUTPUTLEN 256
#define TMPSTRLEN 64
#define UNITS_STR_LEN 16
#define DATIMLEN 23
#define UNITSLEN 20
#define BLKTSTRLEN 4
#define FLDSTRLEN 3
#define MAXFLDLEN 50
#define MAXLINELEN 256
#define FIR_NORM_TOL 0.02


/* enumeration representing the types of units encountered (Note: if default,
   then the response is just given in input units to output units, no
   interpretation is made of the units used) */

/* IGD 02/03/01 New unit pressure  added */
/* IGD 08/21/06 New units TESLA added */
enum units { UNDEF_UNITS, DIS, VEL, ACC, COUNTS, VOLTS, DEFAULT, PRESSURE, TESLA
};

/*  enumeration representing the types of filters encountered  */

enum filt_types { UNDEF_FILT, LAPLACE_PZ, ANALOG_PZ, IIR_PZ,
                  FIR_SYM_1, FIR_SYM_2, FIR_ASYM, LIST, GENERIC, DECIMATION,
                  GAIN, REFERENCE, FIR_COEFFS, IIR_COEFFS
};

/* enumeration representing the types of stages that are recognized */
/* IGD 05/15/02 Added GENERIC_TYPE */
enum stage_types { UNDEF_STAGE, PZ_TYPE, IIR_TYPE, FIR_TYPE,
		GAIN_TYPE, LIST_TYPE, IIR_COEFFS_TYPE, GENERIC_TYPE
};

/* enumeration representing the types of error codes possible */

enum error_codes { NON_EXIST_FLD = -2, ILLEGAL_RESP_FORMAT = -5,
                   PARSE_ERROR = -4, UNDEF_PREFIX = -3, UNDEF_SEPSTR = -6,
                   OUT_OF_MEMORY = -1, UNRECOG_FILTYPE = -7,
                   UNEXPECTED_EOF = -8, ARRAY_BOUNDS_EXCEEDED = -9,
                   OPEN_FILE_ERROR = 2, RE_COMP_FAILED = 3,
                   MERGE_ERROR = 4, SWAP_FAILED = 5, USAGE_ERROR = 6,
                   BAD_OUT_UNITS = 7, IMPROP_DATA_TYPE = -10,
                   UNSUPPORT_FILTYPE = -11, ILLEGAL_FILT_SPEC = -12,
                   NO_STAGE_MATCHED = -13, UNRECOG_UNITS = -14
};

/* define structures for the compound data types used in evalesp */

/* if Windows compiler then redefine 'complex' to */
/*  differentiate it from the existing struct,    */
/*  and rename 'strcasecmp' functions:            */
#ifdef WIN32
#ifdef complex
#undef complex
#endif
#define complex evr_complex
#define strcasecmp stricmp
#define strncasecmp strnicmp
#endif

struct complex {
  double real;
  double imag;
};

struct string_array {
  int nstrings;
  char **strings;
};

struct scn {
  char *station;
  char *network;
  char *locid;
  char *channel;
  int found;
};

struct response {
  char station[STALEN];
  char network[NETLEN];
  char locid[LOCIDLEN];
  char channel[CHALEN];
  struct complex *rvec;
  int nfreqs;	/*Add by I.Dricker IGD to  support blockette 55 */
  double *freqs; /*Add by I.Dricker IGD to  support blockette 55 */
  struct response *next;
};

struct file_list {
  char *name;
  struct file_list *next_file;
};

struct matched_files {
  int nfiles;
  struct file_list *first_list;
  struct matched_files *ptr_next;
};

struct scn_list {
  int nscn;
  struct scn **scn_vec;
};

/* define structures for the various types of filters defined in seed */

struct pole_zeroType {       /* a Response (Poles & Zeros) blockette */
  int nzeros;                /* (blockettes [43] or [53]) */
  int npoles;
  double a0;
  double a0_freq;
  struct complex *zeros;
  struct complex *poles;
};

struct coeffType {             /* a Response (Coefficients) blockette */
  int nnumer;                  /* (blockettes [44] or [54]) */
  int ndenom;
  double *numer;
  double *denom;
  double h0; /*IGD this field is new v 3.2.17 */
};

struct firType {             /* a FIR Response blockette */
  int ncoeffs;               /* (blockettes [41] or [61])*/
  double *coeffs;
  double h0;
};

struct listType{            /* a Response (List) blockette */
  int nresp;                /* (blockettes [45] or [55]) */
  double *freq;
  double *amp;
  double *phase;
};

struct genericType {         /* a Generic Response blockette */
  int ncorners;              /* (blockettes [46] or [56]) */
  double *corner_freq;
  double *corner_slope;
};

struct decimationType {      /* a Decimation blockette */
  double sample_int;         /* (blockettes [47] or [57]) */
  int deci_fact;
  int deci_offset;
  double estim_delay;
  double applied_corr;
};

struct gainType {            /* a Channel Sensitivity/Gain blockette */
  double gain;               /* (blockettes [48] or [58]) */
  double gain_freq;
};

struct referType {           /* a Response Reference blockette */
  int num_stages;
  int stage_num;
  int num_responses;
};

/* define a blkt as a stucture containing the blockette type, a union
   (blkt_info) containing the blockette info, and a pointer to the next
   blockette in the filter sequence.  The structures will be assembled to
   form a linked list of blockettes that make up a filter, with the last
   blockette containing a '(struct blkt *)NULL' pointer in the 'next_blkt'
   position */

struct blkt {
  int type;
  union {
    struct pole_zeroType pole_zero;
    struct coeffType coeff;
    struct firType fir;
    struct listType list;
    struct genericType generic;
    struct decimationType decimation;
    struct gainType gain;
    struct referType reference;
  } blkt_info;
  struct blkt *next_blkt;
};

/* define a stage as a structure that contains the sequence number, the
   input and output units, a pointer to the first blockette of the filter,
   and a pointer to the next stage in the response.  Again, the last
   stage in the response will be indicated by a '(struct stage *)NULL pointer
   in the 'next_stage' position */

struct stage {
  int sequence_no;
  int input_units;
  int output_units;
  struct blkt *first_blkt;
  struct stage *next_stage;
};

/* and define a channel as a stucture containing a pointer to the head of a
   linked list of stages.  Will access the pieces one stages at a time in the
   same order that they were read from the input file, so a linked list is the
   easiest way to do this (since we don't have to worry about the time penalty
   inherent in following the chain).  As an example, if the first stage read
   was a pole-zero stage, then the parts of that stage contained in a channel
   structure called "ch" would be accessed as:

   struct stage *stage_ptr;
   struct blkt *blkt_ptr;

   stage_ptr = ch.first_stage;
   blkt_ptr = stage_ptr->first_blkt;
   if(blkt_ptr->type == LAPLACE_PZ || blkt_ptr->type == ANALOG_PZ ||
      blkt_ptr->type == IIR_PZ){
       nzeros = blkt_ptr->blkt_info.poles_zeros.nzeros;
       ........
   }

*/

struct channel {
  char staname[STALEN];
  char network[NETLEN];
  char locid[LOCIDLEN];
  char chaname[CHALEN];
  char beg_t[DATIMLEN];
  char end_t[DATIMLEN];
  char first_units[MAXLINELEN];
  char last_units[MAXLINELEN];
  double sensit;
  double sensfreq;
  double calc_sensit;
  double calc_delay;
  double estim_delay;
  double applied_corr;
  double sint;
  int nstages;
  struct stage *first_stage;
};

/* structure used for time comparisons */

struct dateTime {
  int year;
  int jday;
  int hour;
  int min;
  float sec;
};

/* IGD 2007/02/27 */
extern char myLabel[20];
//char myLabel[20] = "aa";

/* utility routines that are used to parse the input file line by line and
   convert the input to what the user wants */

struct string_array *ev_parse_line(char *);
struct string_array *parse_delim_line(char *, char *);
int get_field(FILE *, char *, int, int, char *, int);
int test_field(FILE *, char *, int *, int *, char *, int);
int get_line(FILE *, char *, int, int, char *);           /* checks blkt & fld nos */
int next_line(FILE *, char *, int *, int *, char *);      /* returns blkt & fld nos */
int count_fields(char *);
int count_delim_fields(char *, char *);
int parse_field(char *, int, char *);
int parse_delim_field(char *, int, char *, char *);
int check_line(FILE *, int *, int *, char *);
int get_int(char *);
double get_double(char *);
int check_units(char *);
int string_match(const char *, char *, char *);
int is_int(const char *);
int is_real(const char *);
int is_IIR_coeffs (FILE *, int);   /*IGD */

/* routines used to load a channel's response information into a linked
   list of filter stages, each containing a linked list of blockettes */

int find_resp(FILE *, struct scn_list *, char *, struct channel *);
int get_resp(FILE *, struct scn *, char *, struct channel *);
int get_channel(FILE *, struct channel *);
int next_resp(FILE *);

/* routines used to create a list of files matching the users request */

struct matched_files *find_files(char *, struct scn_list *, int *);
int get_names(char *, struct matched_files *);
int start_child(char *, FILE **, FILE **, FILE **);

/* routines used to allocate vectors of the basic data types used in the
   filter stages */

struct complex *alloc_complex(int);
struct response *alloc_response(int);
struct string_array *alloc_string_array(int);
struct scn *alloc_scn(void);
struct scn_list *alloc_scn_list(int);
struct file_list *alloc_file_list(void);
struct matched_files *alloc_matched_files(void);
double *alloc_double(int);
char *alloc_char(int);
char **alloc_char_ptr(int);

/* allocation routines for the various types of filters */

struct blkt *alloc_pz(void);
struct blkt *alloc_coeff(void);
struct blkt *alloc_fir(void);
struct blkt *alloc_ref(void);
struct blkt *alloc_gain(void);
struct blkt *alloc_list(void);
struct blkt *alloc_generic(void);
struct blkt *alloc_deci(void);
struct stage *alloc_stage(void);

/* routines to free up space associated with dynamically allocated
   structure members */

void free_string_array(struct string_array *);
void free_scn(struct scn *);
void free_scn_list(struct scn_list *);
void free_matched_files(struct matched_files *);
void free_file_list(struct file_list *);
void free_pz(struct blkt *);
void free_coeff(struct blkt *);
void free_fir(struct blkt *);
void free_list(struct blkt *);
void free_generic(struct blkt *);
void free_gain(struct blkt *);
void free_deci(struct blkt *);
void free_ref(struct blkt *);
void free_stages(struct stage *);
void free_channel(struct channel *);
void free_response(struct response *);

/* simple error handling routines to standardize the output error values and
   allow for control to return to 'evresp' if a recoverable error occurs */

void error_exit(int, char *, ...);
void error_return(int, char *, ...);

/* a simple routine that parses the station information from the input file */

int parse_channel(FILE *, struct channel *);

/* parsing routines for various types of filters */

int parse_pref(int *, int *, char *);
void parse_pz(FILE *, struct blkt *, struct stage *);        /* pole-zero */
void parse_coeff(FILE *, struct blkt *, struct stage *);     /* fir */
void parse_iir_coeff(FILE *, struct blkt *, struct stage *);     /*I.Dricker IGD for 2.3.17 iir */
void parse_list(FILE *, struct blkt *, struct stage *);      /* list */
void parse_generic(FILE *, struct blkt *, struct stage *);   /* generic */
int parse_deci(FILE *, struct blkt *);                       /* decimation */
int parse_gain(FILE *, struct blkt *);                       /* gain/sensitivity */
void parse_fir(FILE *, struct blkt *, struct stage *);       /* fir */
void parse_ref(FILE *, struct blkt *, struct stage *);       /* response reference */

/* remove trailing white space from (if last arg is 'a') and add null character to
   end of (if last arg is 'a' or 'e') input (FORTRAN) string */

int add_null(char *, int, char);

/* run a sanity check on the channel's filter sequence */

void merge_coeffs(struct blkt *, struct blkt **);
void merge_lists(struct blkt *, struct blkt **);  /* Added by I.Dricker IGD for v 3.2.17 of evalresp */
void check_channel(struct channel *);
void check_sym(struct blkt *, struct channel *);

/* routines used to calculate the instrument responses */

/*void calc_resp(struct channel *, double *, int, struct complex *,char *, int, int);*/
void calc_resp(struct channel *chan, double *freq, int nfreqs, struct complex *output,
          char *out_units, int start_stage, int stop_stage, int useTotalSensitivityFlag);
void convert_to_units(int, char *, struct complex *, double);
void analog_trans(struct blkt *, double, struct complex *);
void fir_sym_trans(struct blkt *, double, struct complex *);
void fir_asym_trans(struct blkt *, double, struct complex *);
void iir_pz_trans(struct blkt *, double, struct complex *);
void calc_time_shift(double, double, struct complex *);
void zmul(struct complex *, struct complex *);
void norm_resp(struct channel *, int, int, int);
void calc_list(struct blkt *, int , struct complex *); /*IGD i.dricker@isti.com for version 3.2.17 */
void iir_trans(struct blkt *, double , struct complex *); /* IGD for version 3.2.17 */
int is_time (const char * );

/* compare two times and determine if the first is greater, equal to, or less than the second */

int timecmp(struct dateTime *dt1, struct dateTime *dt2);

/* print the channel info, followed by the list of filters */

void print_chan(struct channel *, int, int, int, int, int, int);

/* print the response information to the output files */

void print_resp(double *, int, struct response *, char *, int);
void print_resp_itp(double *, int, struct response *, char *, int,
                                                               int, double, int);

/* evaluate responses for user requested station/channel/network tuple at the
   frequencies requested by the user */

struct response *evresp(char *, char *, char *, char *, char *, char *, char *,
                        double *, int, char *, char *, int, int, int, int);
struct response *evresp_itp(char *, char *, char *, char *, char *, char *,
                            char *, double *, int, char *, char *, int,
                            int, int, int, int, double, int);

 /* Interpolates amplitude and phase values from the set of frequencies
    in the List blockette to the requested set of frequencies. */
void interpolate_list_blockette(double **freq, double **amp, double **phase,
                                 int *p_number_points, double *req_freq_arr,
                                         int req_num_freqs, double tension);


extern char SEEDUNITS[][UNITS_STR_LEN];

/* define the "first line" and "first field" arguments that are used by the
   parsing routines (they are used to compensate for the fact that in
   parsing the RESP files, one extra line is always read because the
   "end" of a filter sequence cannot be determined until the first line
   of the "next" filter sequence or station-channel pair is read from the
   file */

extern char FirstLine[];
extern int FirstField;

/* values for use in calculating responses (defined in 'evresp_') */

extern double Pi;
extern double twoPi;

/* define a global flag to use if using "default" units */

extern int def_units_flag;

/* define a pointer to a channel structure to use in determining the input and
   output units if using "default" units and for use in error output*/

extern struct channel *GblChanPtr;
extern float unitScaleFact;

/* define global variables for use in printing error messages */

extern char *curr_file;
extern int curr_seq_no;

/* and set a global variable to contain the environment for the setjmp/longjmp
   combination for error handling */

extern jmp_buf jump_buffer;

double unwrap_phase(double phase, double prev_phase, double range,
                                                       double *added_value);
double wrap_phase(double phase, double range, double *added_value);
int use_delay(int flag);
#endif

