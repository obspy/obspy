/* This file is modified by I.Dricker I.dricker@isti.com for version 3.2.17 of evalresp*/
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif

#include "./evresp.h"

#include <stdlib.h>

#include <string.h>


/* alloc_complex:  allocates space for an array of complex numbers, returns a pointer to that
                   array (exits with error if fails) */

struct complex *alloc_complex(int npts)
{
  struct complex *cptr;

  if(npts) {
    if((cptr = (struct complex *) malloc(npts*sizeof(struct complex)))
       == (struct complex *)NULL) {
      error_exit(OUT_OF_MEMORY,"alloc_complex; malloc() failed for (complex) vector");
    }
  }
  else
    cptr = (struct complex *)NULL;

  return(cptr);
}

/* alloc_string_array:  allocates space for an array of strings, returns a
                        pointer to that array (exits with error if fails) */

struct string_array *alloc_string_array(int nstrings)
{
  struct string_array *sl_ptr;
  int i;

  if(nstrings) {
    if((sl_ptr = (struct string_array *) malloc(sizeof(struct string_array)))
       == (struct string_array *)NULL) {
      error_exit(OUT_OF_MEMORY,"alloc_string_array; malloc() failed for (string_array)");
    }
    if((sl_ptr->strings = (char **) malloc(nstrings*sizeof(char *))) == (char **)NULL) {
      error_exit(OUT_OF_MEMORY,"alloc_string_array; malloc() failed for (char *) vector");
    }
    for(i = 0; i < nstrings; i++)
      sl_ptr->strings[i] = (char *)NULL;
    sl_ptr->nstrings = nstrings;
  }
  else
    sl_ptr = (struct string_array *)NULL;

  return(sl_ptr);
}

/* alloc_scn:  allocates space for a station-channel structure, returns a
               pointer to that structure (exits with error if fails) */

struct scn *alloc_scn()
{
  struct scn *scn_ptr;
  

  if((scn_ptr = (struct scn *) malloc(sizeof(struct scn)))
     == (struct scn *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_scn; malloc() failed for (scn)");
  }
  if((scn_ptr->station = (char *) malloc(STALEN*sizeof(char))) == (char *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_scn; malloc() failed for (station)");
  }
  if((scn_ptr->network = (char *) malloc(NETLEN*sizeof(char))) == (char *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_scn; malloc() failed for (station)");
  }
  if((scn_ptr->locid = (char *) malloc(LOCIDLEN*sizeof(char))) == (char *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_scn; malloc() failed for (channel)");
  }
  if((scn_ptr->channel = (char *) malloc(CHALEN*sizeof(char))) == (char *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_scn; malloc() failed for (channel)");
  }

  strncpy(scn_ptr->station,"",STALEN);
  strncpy(scn_ptr->network,"",NETLEN);
  strncpy(scn_ptr->locid,"",LOCIDLEN);
  strncpy(scn_ptr->channel,"",CHALEN);
  scn_ptr->found = 0;

  return(scn_ptr);
}

/* alloc_response:  allocates space for an array of responses, returns a pointer to that
                    array (exits with error if fails).  A 'response' is a combination
                    of a complex array, a station-channel-network, and a pointer to the
                    next 'response' in the list */

struct response *alloc_response(int npts)
{
  struct response *rptr;
  struct complex *cvec;
  int k;

  if(npts) {
    if((rptr = (struct response *) malloc(sizeof(struct response)))
       == (struct response *)NULL) {
      error_exit(OUT_OF_MEMORY,"alloc_response; malloc() failed for (response) vector");
    }
    strncpy(rptr->station,"",STALEN);
    strncpy(rptr->locid,"",LOCIDLEN);
    strncpy(rptr->channel,"",NETLEN);
    strncpy(rptr->network,"",CHALEN);
    rptr->rvec = alloc_complex(npts);
    cvec = rptr->rvec;
    for(k = 0; k < npts; k++) {
      cvec[k].real = 0.0;
      cvec[k].imag = 0.0;
    }
    rptr->next = (struct response *)NULL;
/*IGD add freqs to this structure to process blockette 55 */
    rptr->nfreqs = 0;
    rptr->freqs = (double *) NULL;
  }
  else
    rptr = (struct response *)NULL;

  return(rptr);
}

/* alloc_scn_list:  allocates space for an array of station/channel pairs,
                    returns a pointer to that array (exits with error if
                    fails) */

struct scn_list *alloc_scn_list(int nscn)
{
  struct scn_list *sc_ptr;
  int i;

  if(nscn) {
    if((sc_ptr = (struct scn_list *) malloc(sizeof(struct scn_list)))
       == (struct scn_list *)NULL) {
      error_exit(OUT_OF_MEMORY,"alloc_scn_list; malloc() failed for (scn_list)");
    }
    if((sc_ptr->scn_vec = (struct scn **) malloc(nscn*sizeof(struct scn *)))
       == (struct scn **)NULL) {
      error_exit(OUT_OF_MEMORY,"alloc_scn_list; malloc() failed for (scn_vec)");
    }
    for(i = 0; i < nscn; i++)
      sc_ptr->scn_vec[i] = alloc_scn();
    sc_ptr->nscn = nscn;
  }
  else
    sc_ptr = (struct scn_list *)NULL;

  return(sc_ptr);
}

/* alloc_file_list:  allocates space for an element of a linked list of
                     filenames, returns a pointer to that structure
                     (exits with error if fails) */

struct file_list *alloc_file_list()
{
  struct file_list *flst_ptr;
  

  if((flst_ptr = (struct file_list *) malloc(sizeof(struct file_list)))
     == (struct file_list *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_file_list; malloc() failed for (file_list)");
  }
  flst_ptr->name = (char *)NULL;
  flst_ptr->next_file = (struct file_list *)NULL;

  return(flst_ptr);
}

/* alloc_matched_files:  allocates space for an element of a linked list of
                         matching files, returns a pointer to that structure
                         (exits with error if fails) */

struct matched_files *alloc_matched_files()
{
  struct matched_files *flst_ptr;
  

  if((flst_ptr = (struct matched_files *) malloc(sizeof(struct matched_files)))
     == (struct matched_files *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_matched_files; malloc() failed for (matched_files)");
  }
  flst_ptr->nfiles = 0;
  flst_ptr->first_list = (struct file_list *)NULL;
  flst_ptr->ptr_next = (struct matched_files *)NULL;

  return(flst_ptr);
}

/* alloc_double:  allocates space for an array of double precision numbers, returns a pointer to
                  that array (exits with error if fails) */

double *alloc_double(int npts)
{
  double *dptr;

  if(npts) {
    if((dptr = (double *) malloc(npts*sizeof(double))) == (double *)NULL) {
      error_exit(OUT_OF_MEMORY,"alloc_double; malloc() failed for (double) vector");
    }
  }
  else
    dptr = (double *)NULL;

  return(dptr);
}

/* alloc_char:  allocates space for an array of characters, returns a pointer to
                that array (exits with error if fails) */

char *alloc_char(int len)
{
  char *cptr;

  if(len) {
    if((cptr = (char *) malloc(len*sizeof(char))) == (char *)NULL) {
      error_exit(OUT_OF_MEMORY,"alloc_char; malloc() failed for (char) vector");
    }
  }
  else
    cptr = (char *)NULL;

  return(cptr);
}

/* alloc_char_ptr:  allocates space for an array of char pointers, returns a
                    pointer to that array (exits with error if fails) */

char **alloc_char_ptr(int len)
{
  char **cptr;

  if(len) {
    if((cptr = (char **) malloc(len*sizeof(char *))) == (char **)NULL) {
      error_exit(OUT_OF_MEMORY,"alloc_char_ptr; malloc() failed for (char *) vector");
    }
  }
  else
    cptr = (char **)NULL;

  return(cptr);
}

/* alloc_pz:  allocates space for a pole-zero type filter structure and returns a pointer to that
              structure.
              Note: the space for the complex poles and zeros is not allocated here, the space
                    for these vectors must be allocated as they are read, since the number of
                    poles and zeros is unknown until the blockette is partially parsed. */

struct blkt *alloc_pz()
{
  struct blkt *blkt_ptr;

  if((blkt_ptr = (struct blkt *) malloc(sizeof(struct blkt)))
     == (struct blkt *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_pz; malloc() failed for (Poles & Zeros) blkt structure");
  }

  blkt_ptr->type = 0;
  blkt_ptr->next_blkt = (struct blkt *)NULL;
  blkt_ptr->blkt_info.pole_zero.zeros = (struct complex *)NULL;
  blkt_ptr->blkt_info.pole_zero.poles = (struct complex *)NULL;
  blkt_ptr->blkt_info.pole_zero.nzeros = 0;
  blkt_ptr->blkt_info.pole_zero.npoles = 0;

  return(blkt_ptr);
}

/* alloc_coeff:  allocates space for a coefficients-type filter 
                 Note:  see alloc_pz for details (like alloc_pz, this does not allocate space for
                        the numerators and denominators, that is left until parse_fir()) */

struct blkt *alloc_coeff()
{
  struct blkt *blkt_ptr;

  if((blkt_ptr = (struct blkt *) malloc(sizeof(struct blkt)))
     == (struct blkt *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_coeff; malloc() failed for (FIR) blkt structure");
  }

  blkt_ptr->type = 0;
  blkt_ptr->next_blkt = (struct blkt *)NULL;
  blkt_ptr->blkt_info.coeff.numer = (double *)NULL;
  blkt_ptr->blkt_info.coeff.denom = (double *)NULL;
  blkt_ptr->blkt_info.coeff.nnumer = 0;
  blkt_ptr->blkt_info.coeff.ndenom = 0;
  blkt_ptr->blkt_info.coeff.h0 = 1.0; /*IGD this field is new for v 3.2.17*/  

  return(blkt_ptr);
}

/* alloc_fir:  allocates space for a fir-type filter 
               Note:  see alloc_pz for details (like alloc_pz, this does not allocate space for
                      the numerators and denominators, that is left until parse_fir()) */

struct blkt *alloc_fir()
{
  struct blkt *blkt_ptr;

  if((blkt_ptr = (struct blkt *) malloc(sizeof(struct blkt)))
     == (struct blkt *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_fir; malloc() failed for (FIR) blkt structure");
  }

  blkt_ptr->type = 0;
  blkt_ptr->next_blkt = (struct blkt *)NULL;
  blkt_ptr->blkt_info.fir.coeffs = (double *)NULL;
  blkt_ptr->blkt_info.fir.ncoeffs = 0;
  blkt_ptr->blkt_info.fir.h0 = 1.0;

  return(blkt_ptr);
}

/* alloc_ref:  allocates space for a response reference type filter structure and returns a pointer
               to that structure. */

struct blkt *alloc_ref()
{
  struct blkt *blkt_ptr;

  if((blkt_ptr = (struct blkt *) malloc(sizeof(struct blkt)))
     == (struct blkt *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_ref; malloc() failed for (Resp. Ref.) blkt structure");
  }

  blkt_ptr->type = REFERENCE;
  blkt_ptr->next_blkt = (struct blkt *)NULL;
  blkt_ptr->blkt_info.reference.num_stages = 0;
  blkt_ptr->blkt_info.reference.stage_num = 0;
  blkt_ptr->blkt_info.reference.num_responses = 0;

  return(blkt_ptr);
}

/* alloc_gain:  allocates space for a gain type filter structure and returns a pointer to that
              structure.
              Note: the space for the calibration vectors is not allocated here, the space
                    for these vectors must be allocated as they are read, since the number of
                    calibration points is unknown until the blockette is partially parsed. */

struct blkt *alloc_gain()
{
  struct blkt *blkt_ptr;

  if((blkt_ptr = (struct blkt *) malloc(sizeof(struct blkt)))
     == (struct blkt *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_gain; malloc() failed for (Gain) blkt structure");
  }

  blkt_ptr->type = GAIN;
  blkt_ptr->next_blkt = (struct blkt *)NULL;
  blkt_ptr->blkt_info.gain.gain = 0;
  blkt_ptr->blkt_info.gain.gain_freq = 0;

  return(blkt_ptr);
}

/* alloc_list:  allocates space for a list type filter structure and returns a pointer to that
                structure.
                Note: the space for the amplitude, phase and frequency vectors is not allocated
                      here the user must allocate space for these parameters once the number of
                      frequencies is known */

struct blkt *alloc_list()
{
  struct blkt *blkt_ptr;

  if((blkt_ptr = (struct blkt *) malloc(sizeof(struct blkt)))
     == (struct blkt *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_list; malloc() failed for (List) blkt structure");
  }

  blkt_ptr->type = LIST;
  blkt_ptr->next_blkt = (struct blkt *)NULL;
  blkt_ptr->blkt_info.list.freq = (double *)NULL;
  blkt_ptr->blkt_info.list.amp = (double *)NULL;
  blkt_ptr->blkt_info.list.phase = (double *)NULL;
  blkt_ptr->blkt_info.list.nresp = 0;

  return(blkt_ptr);
}

/* alloc_generic  allocates space for a generic type filter structure and returns a pointer to that
                  structure.
                  Note: the space for the corner_freq, and corner_slope vectors is not allocated
                        here the user must allocate space for these parameters once the number of
                        frequencies is known */

struct blkt *alloc_generic()
{
  struct blkt *blkt_ptr;

  if((blkt_ptr = (struct blkt *) malloc(sizeof(struct blkt)))
     == (struct blkt *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_generic; malloc() failed for (Generic) blkt structure");
  }

  blkt_ptr->type = GENERIC;
  blkt_ptr->next_blkt = (struct blkt *)NULL;
  blkt_ptr->blkt_info.generic.corner_slope = (double *)NULL;
  blkt_ptr->blkt_info.generic.corner_freq = (double *)NULL;
  blkt_ptr->blkt_info.generic.ncorners = 0;

  return(blkt_ptr);
}

/* alloc_deci:  allocates space for a decimation type filter structure and returns a pointer to that
                structure. */

struct blkt *alloc_deci()
{
  struct blkt *blkt_ptr;

  if((blkt_ptr = (struct blkt *) malloc(sizeof(struct blkt)))
     == (struct blkt *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_deci; malloc() failed for (Decimation) blkt structure");
  }

  blkt_ptr->type = DECIMATION;
  blkt_ptr->next_blkt = (struct blkt *)NULL;
  blkt_ptr->blkt_info.decimation.sample_int = 0;
  blkt_ptr->blkt_info.decimation.deci_fact = 0;
  blkt_ptr->blkt_info.decimation.deci_offset = 0;
  blkt_ptr->blkt_info.decimation.estim_delay = 0;
  blkt_ptr->blkt_info.decimation.applied_corr = 0;

  return(blkt_ptr);
}

/* alloc_stage:  allocates space for a decimation type filter structure and returns a pointer to that
                structure. */

struct stage *alloc_stage()
{
  struct stage *stage_ptr;

  if((stage_ptr = (struct stage *) malloc(sizeof(struct stage)))
     == (struct stage *)NULL) {
    error_exit(OUT_OF_MEMORY,"alloc_stage; malloc() failed for stage structure");
  }

  stage_ptr->sequence_no = 0;
  stage_ptr->output_units = 0;
  stage_ptr->input_units = 0;
  stage_ptr->first_blkt = (struct blkt *)NULL;
  stage_ptr->next_stage = (struct stage *)NULL;

  return(stage_ptr);
}

/* free_string_array: a routine that frees up the space associated with a
                     string list type structure */

void free_string_array(struct string_array *lst) {
  int i;

  for(i = 0; i < lst->nstrings; i++) {
    free(lst->strings[i]);
  }
  free(lst->strings);
  free(lst);
}

/* free_scn: a routine that frees up the space associated with a
                       station-channel type structure */

void free_scn(struct scn *ptr) {
 

  free(ptr->station);
  free(ptr->network);
  free(ptr->locid);
  free(ptr->channel);

}

/* free_scn_list: a routine that frees up the space associated with a
                       station-channel list type structure */

void free_scn_list(struct scn_list *lst) {
  int i;

  for(i = 0; i < lst->nscn; i++) {
    free_scn(lst->scn_vec[i]);
    free(lst->scn_vec[i]);
  }
  free(lst->scn_vec);
  free(lst);
}

/* free_matched_files: a routine that frees up the space associated with a
                       matched files type structure */

void free_matched_files(struct matched_files *lst) {
  if(lst != (struct matched_files *)NULL) {
    free_matched_files(lst->ptr_next);
    if(lst->nfiles) {
      free_file_list(lst->first_list);
      free(lst->first_list);
    }
    free(lst);
    lst = (struct matched_files *)NULL;
  }
}

/* free_file_list: a routine that frees up the space associated with a
                   file list type structure */

void free_file_list(struct file_list *lst) {

  if(lst != (struct file_list *)NULL) {
    free_file_list(lst->next_file);
    if(lst->name != (char *)NULL)
      free(lst->name);
    if(lst->next_file != (struct file_list *)NULL)
      free(lst->next_file);
  }

}

/* free_pz: a routine that frees up the space associated with a pole-zero
            type filter */

void free_pz(struct blkt *blkt_ptr) {
  if(blkt_ptr != (struct blkt *)NULL) {
    if(blkt_ptr->blkt_info.pole_zero.zeros != (struct complex *)NULL)
      free(blkt_ptr->blkt_info.pole_zero.zeros);
    if(blkt_ptr->blkt_info.pole_zero.poles != (struct complex *)NULL)
      free(blkt_ptr->blkt_info.pole_zero.poles);
    free(blkt_ptr);
  }
}

/* free_coeff: a routine that frees up the space associated with a coefficients
               type filter */

void free_coeff(struct blkt *blkt_ptr) {
  if(blkt_ptr != (struct blkt *)NULL) {
    if(blkt_ptr->blkt_info.coeff.numer != (double *)NULL)
      free(blkt_ptr->blkt_info.coeff.numer);
    if(blkt_ptr->blkt_info.coeff.denom != (double *)NULL)
      free(blkt_ptr->blkt_info.coeff.denom);
    free(blkt_ptr);
  }
}

/* free_fir: a routine that frees up the space associated with a fir
             type filter */

void free_fir(struct blkt *blkt_ptr) {
  if(blkt_ptr != (struct blkt *)NULL) {
    if(blkt_ptr->blkt_info.fir.coeffs != (double *)NULL)
      free(blkt_ptr->blkt_info.fir.coeffs);
    free(blkt_ptr);
  }
}

/* free_list: a routine that frees up the space associated with a list
              type filter */

void free_list(struct blkt *blkt_ptr) {
  if(blkt_ptr != (struct blkt *)NULL) {
    if(blkt_ptr->blkt_info.list.freq != (double *)NULL)
      free(blkt_ptr->blkt_info.list.freq);
    if(blkt_ptr->blkt_info.list.amp != (double *)NULL)
      free(blkt_ptr->blkt_info.list.amp);
    if(blkt_ptr->blkt_info.list.phase != (double *)NULL)
      free(blkt_ptr->blkt_info.list.phase);
    free(blkt_ptr);
  }
}

/* free_generic: a routine that frees up the space associated with a generic
                 type filter */

void free_generic(struct blkt *blkt_ptr) {
  if(blkt_ptr != (struct blkt *)NULL) {
    if(blkt_ptr->blkt_info.generic.corner_slope != (double *)NULL)
      free(blkt_ptr->blkt_info.generic.corner_slope);
    if(blkt_ptr->blkt_info.generic.corner_freq != (double *)NULL)
      free(blkt_ptr->blkt_info.generic.corner_freq);
    free(blkt_ptr);
  }
}

/* free_gain: a routine that frees up the space associated with a gain
              type filter */

void free_gain(struct blkt *blkt_ptr) {
  if(blkt_ptr != (struct blkt *)NULL) {
    free(blkt_ptr);
  }
}

/* free_deci: a routine that frees up the space associated with a decimation
              type filter */

void free_deci(struct blkt *blkt_ptr) {
  if(blkt_ptr != (struct blkt *)NULL) {
    free(blkt_ptr);
  }
}

/* free_ref: a routine that frees up the space associated with a response
            reference type filter */

void free_ref(struct blkt *blkt_ptr) {
  

  if(blkt_ptr != (struct blkt *)NULL) {
    free(blkt_ptr);
  }
}

/* free_stages: a routine that frees up the space associated with a stages in
                 a channel's response */

void free_stages(struct stage *stage_ptr) {
  struct blkt *this_blkt, *next_blkt;

  if(stage_ptr != (struct stage *)NULL) {
    free_stages(stage_ptr->next_stage);
    this_blkt = stage_ptr->first_blkt;
    while(this_blkt != (struct blkt *)NULL) {
      next_blkt = this_blkt->next_blkt;
      switch (this_blkt->type) {
      case LAPLACE_PZ:
      case ANALOG_PZ:
      case IIR_PZ:
        free_pz(this_blkt);
        break;
      case FIR_SYM_1:
      case FIR_SYM_2:
      case FIR_ASYM:
        free_fir(this_blkt);
        break;
      case FIR_COEFFS:
        free_coeff(this_blkt);
        break;
      case LIST:
        free_list(this_blkt);
        break;
      case GENERIC:
        free_generic(this_blkt);
        break;
      case DECIMATION:
        free_deci(this_blkt);
        break;
      case GAIN:
        free_gain(this_blkt);
        break;
      case REFERENCE:
        free_ref(this_blkt);
        break;
      default:
        break;
      }
      this_blkt = next_blkt;
    }
    free(stage_ptr);
  }
}

/* free_channel: a routine that frees up the space associated with a channel's
                 filter sequence */

void free_channel(struct channel *chan_ptr) {

  free_stages(chan_ptr->first_stage);
  strncpy(chan_ptr->staname,"",STALEN);
  strncpy(chan_ptr->network,"",NETLEN);
  strncpy(chan_ptr->locid,"",LOCIDLEN);
  strncpy(chan_ptr->chaname,"",CHALEN);
  strncpy(chan_ptr->beg_t,"",DATIMLEN);
  strncpy(chan_ptr->end_t,"",DATIMLEN);
  strncpy(chan_ptr->first_units,"",MAXLINELEN);
  strncpy(chan_ptr->last_units,"",MAXLINELEN);
}

/* free_response: a routine that frees up the space associated with a linked
                 list of response information */

void free_response(struct response *resp_ptr) {
  struct response *this_resp, *next_resp;

  this_resp = resp_ptr;
  while(this_resp != (struct response *)NULL) {
    next_resp = this_resp->next;
    free(this_resp->rvec);
    free(this_resp->freqs); /*IGD for v 3.2.17 */
    free(this_resp);
    this_resp = next_resp;
  }
}
