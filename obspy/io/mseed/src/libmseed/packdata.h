/***************************************************************************
 * packdata.h:
 * 
 * Interface declarations for the Mini-SEED packing routines in
 * packdata.c
 *
 * modified: 2008.220
 ***************************************************************************/


#ifndef	PACKDATA_H
#define	PACKDATA_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include "steimdata.h"

/* Pointer to srcname of record being packed, declared in pack.c */
extern char *PACK_SRCNAME;

extern int msr_pack_int_16 (int16_t*, int32_t*, int, int, int, int*, int*, int);
extern int msr_pack_int_32 (int32_t*, int32_t*, int, int, int, int*, int*, int);
extern int msr_pack_float_32 (float*, float*, int, int, int, int*, int*, int);
extern int msr_pack_float_64 (double*, double*, int, int, int, int*, int*, int);
extern int msr_pack_steim1 (DFRAMES*, int32_t*, int32_t, int, int, int, int*, int*, int);
extern int msr_pack_steim2 (DFRAMES*, int32_t*, int32_t, int, int, int, int*, int*, int);
extern int msr_pack_text (char *, char *, int, int, int, int*, int*);

#ifdef __cplusplus
}
#endif

#endif
