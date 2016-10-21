/***************************************************************************
 * packdata.h:
 *
 * Interface declarations for the Mini-SEED packing routines in
 * packdata.c
 *
 * modified: 2016.273
 ***************************************************************************/

#ifndef PACKDATA_H
#define PACKDATA_H 1

#ifdef __cplusplus
extern "C" {
#endif

#define STEIM1_FRAME_MAX_SAMPLES 60
#define STEIM2_FRAME_MAX_SAMPLES 105

/* Control for printing debugging information, declared in packdata.c */
extern int encodedebug;

extern int msr_encode_text (char *input, int samplecount, char *output,
                            int outputlength);
extern int msr_encode_int16 (int32_t *input, int samplecount, int16_t *output,
                             int outputlength, int swapflag);
extern int msr_encode_int32 (int32_t *input, int samplecount, int32_t *output,
                             int outputlength, int swapflag);
extern int msr_encode_float32 (float *input, int samplecount, float *output,
                               int outputlength, int swapflag);
extern int msr_encode_float64 (double *input, int samplecount, double *output,
                               int outputlength, int swapflag);
extern int msr_encode_steim1 (int32_t *input, int samplecount, int32_t *output,
                              int outputlength, int32_t diff0, int swapflag);
extern int msr_encode_steim2 (int32_t *input, int samplecount, int32_t *output,
                              int outputlength, int32_t diff0, char *srcname,
                              int swapflag);

#ifdef __cplusplus
}
#endif

#endif
