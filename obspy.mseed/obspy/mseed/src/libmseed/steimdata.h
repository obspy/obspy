/***************************************************************************
 * steimdata.h:
 * 
 * Declarations for Steim compression routines.
 *
 * modified: 2004.278
 ***************************************************************************/

#ifndef STEIMDATA_H
#define STEIMDATA_H 1

#ifdef __cplusplus
extern "C" {
#endif

#define STEIM1_FRAME_MAX_SAMPLES  60
#define STEIM2_FRAME_MAX_SAMPLES  105

#define VALS_PER_FRAME  15      /* # of ints for data per frame.*/

#define STEIM1_SPECIAL_MASK     0
#define STEIM1_BYTE_MASK        1
#define STEIM1_HALFWORD_MASK    2
#define STEIM1_FULLWORD_MASK    3

#define STEIM2_SPECIAL_MASK     0
#define STEIM2_BYTE_MASK        1
#define STEIM2_123_MASK         2
#define STEIM2_567_MASK         3

typedef union u_diff {          /* union for Steim objects.     */
  int8_t          byte[4];      /* 4 1-byte differences.        */
  int16_t         hw[2];        /* 2 halfword differences.      */
  int32_t         fw;           /* 1 fullword difference.       */
} U_DIFF;

typedef struct frame {          /* frame in a seed data record. */
  uint32_t        ctrl;         /* control word for frame.      */
  U_DIFF          w[15];        /* compressed data.             */
} FRAME;

typedef struct dframes {        /* seed data frames.            */
    FRAME f[1];                 /* data record header frames.   */
} DFRAMES;


#ifdef __cplusplus
}
#endif

#endif /* STEIMDATA_H */
