/***************************************************************************
 * unpack.h:
 * 
 * Interface declarations for the Mini-SEED unpacking routines in
 * unpackdata.c
 *
 * modified: 2009.111
 ***************************************************************************/


#ifndef	UNPACKDATA_H
#define	UNPACKDATA_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include "steimdata.h"

/* Pointer to srcname of record being unpacked, declared in unpack.c */
extern char *UNPACK_SRCNAME;
  
extern int msr_unpack_int_16 (int16_t*, int, int, int32_t*, int);
extern int msr_unpack_int_32 (int32_t*, int, int, int32_t*, int);
extern int msr_unpack_float_32 (float*, int, int, float*, int);
extern int msr_unpack_float_64 (double*, int, int, double*, int);
extern int msr_unpack_steim1 (FRAME*, int, int, int, int32_t*, int32_t*,
			      int32_t*, int32_t*, int, int);
extern int msr_unpack_steim2 (FRAME*, int, int, int, int32_t*, int32_t*,
			      int32_t*, int32_t*, int, int);
extern int msr_unpack_geoscope (const char*, int, int, float*, int, int);
extern int msr_unpack_cdsn (int16_t*, int, int, int32_t*, int);
extern int msr_unpack_sro (int16_t*, int, int, int32_t*, int);
extern int msr_unpack_dwwssn (int16_t*, int, int, int32_t*, int);

#ifdef __cplusplus
}
#endif

#endif
