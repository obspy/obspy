/***************************************************************************
 * unpackdata.h:
 *
 * Interface declarations for the Mini-SEED unpacking routines in
 * unpackdata.c
 *
 * modified: 2016.273
 ***************************************************************************/

#ifndef UNPACKDATA_H
#define UNPACKDATA_H 1

#ifdef __cplusplus
extern "C" {
#endif

/* Control for printing debugging information, declared in unpackdata.c */
extern int decodedebug;

extern int msr_decode_int16 (int16_t *input, int samplecount, int32_t *output,
                             int outputlength, int swapflag);
extern int msr_decode_int32 (int32_t *input, int samplecount, int32_t *output,
                             int outputlength, int swapflag);
extern int msr_decode_float32 (float *input, int samplecount, float *output,
                               int outputlength, int swapflag);
extern int msr_decode_float64 (double *input, int samplecount, double *output,
                               int outputlength, int swapflag);
extern int msr_decode_steim1 (int32_t *input, int inputlength, int samplecount,
                              int32_t *output, int outputlength, char *srcname,
                              int swapflag);
extern int msr_decode_steim2 (int32_t *input, int inputlength, int samplecount,
                              int32_t *output, int outputlength, char *srcname,
                              int swapflag);
extern int msr_decode_geoscope (char *input, int samplecount, float *output,
                                int outputlength, int encoding, char *srcname,
                                int swapflag);
extern int msr_decode_cdsn (int16_t *input, int samplecount, int32_t *output,
                            int outputlength, int swapflag);
extern int msr_decode_sro (int16_t *input, int samplecount, int32_t *output,
                           int outputlength, char *srcname, int swapflag);
extern int msr_decode_dwwssn (int16_t *input, int samplecount, int32_t *output,
                              int outputlength, int swapflag);

#ifdef __cplusplus
}
#endif

#endif
