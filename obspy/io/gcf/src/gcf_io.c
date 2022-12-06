//  
//  C-library to read and write gcf files
//
//  compile shared library with: gcc gcf_io.c -Wall -shared -fPIC -g -o gcf_io.so -lm
//  
//  LOG:
//    2017-03-10   Lifted library out of GcfPy.c to stand alone library
//                   added function parse_gcf_block(), cps
//    2017-05-03   Added functions:
//                   gcfSegSps(), gcfSegStart(), gcfSegEnd(), cmpSegHead()
//                  updated merging algorithm in add_GcfSeg() and merge_GcfFile()
//                  to merge the segments with smallest jump in data if more than 
//                  two segments are aligned, cps
//    2017-05-04   Removed BUG causing segfault if no data were decoded due to unknown
//                  compression code or to many (>1004) data samples set inblock header, cps
//    2017-05-08   Removed BUG in new merging algorithm in add_GcfSeg that triggered when
//                  blocks in gcf file are not ordered in time (they usually are), cps
//    2017-05-23   Removed BUG in new merging algorithm in add_GcfSeg that triggered when
//                  data vectors had not been decoded --- Note, this diables new scheme
//                  to merge segments with least data jump if more than two segments are
//                  aligned, cps
//    2017-07-12  Removed BUG in merging algorithm triggering when merging segments with
//                  decoded headers only, further adjusted algorithm to respect not merging
//                  if leap second is set. Adjusted meaning of GcfSeg.blk to be oldest block
//                  number rather than lowest (which was not set proper anyway), cps
//    2019-11-07  Removed BUG in write_gcf() triggering when first difference between data
//                  samples 0 and 1 requires 32bit, cps
//    2020-02-29 --- 2020-03-03
//                Updated to latest (2018) revision of gcf protocol and added use of new 
//                  attributes FIC, RIC, sysType in struct GcfSeg, cps
//    2020-03-01  Removed BUG in merge_GcfFile() that did not update number of samples when 
//                  merging overlapping segments (noted bug while reading code, never seen 
//                  it trigger), cps
//    2020-03-05  Minor adjustments to silence compiler warnings, cps
//    2020-03-14  Moved main() to separate file GcfTool.c, cps
//    2020-03-15  Added check that input systemID is does not contain to many characters, cps
//    2020-03-20  Moved declaration of merge_GcfFile() to header file, cps
//    2020-03-21  Adjusted BUG in CheckSegAligned() (was not a problem for sampling rates < 250 Hz)
//    2020-03-24  Adjusted BUG in write_gcf() computing wrong start time when slicing a 
//                  segment into blocks if segment starts at fractional time
//    2020-03-28  Adjusted BUG in setting error code while decoding data vector as well as BUG in
//                  merging segments in case one of the segments had an error set
//    2020-04-15  Moved inclusion of time.h to header file
//    2022-03-10  Removed use of time.h by changing use of time_t to derrived arithmetic type gtime
//                 moved inclusion of unistd.h and to header file
//    2022-03-15  Added sampling rate 800 Hz as an allowed sampling rate
//    
//  TODO:
//  
//  (C) Reynir Bodvarsson and Peter Schmidt, 2017-2022

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "gcf_io.h"


/*  function init_GcfSeg() initiates values in a GcfSeg struct */
void init_GcfSeg(GcfSeg *obj, int skip_data) {
   memset(obj->streamID,0,sizeof(obj->streamID));
   memset(obj->systemID,0,sizeof(obj->systemID));
   obj->start = 0;
   obj->t_numerator = 0;
   obj->t_denominator = 1;
   obj->t_leap = 0;
   obj->gain = 0;
   obj->type = 0;
   obj->sysType = 0;
   obj->ttl = 0;
   obj->blk = 0;
   obj->err = 0;
   obj->sps = 0;
   obj->sps_denom = 1;
   obj->n_data = 0;
   obj->compr=0;
   obj->FIC = 0;
   obj->RIC = 0;
   if (!skip_data) {
      obj->n_alloc = 0;
      obj->data = (int32 *)NULL;
   }
}


/*  function init_GcfFile() initiates values in a GcfFile struct */
void init_GcfFile(GcfFile *obj) {
   obj->n_blk = 0;
   obj->n_seg = 0;
   obj->n_alloc = 0;
   obj->n_errHead = 0;
   obj->n_errData = 0;
   obj->seg = (GcfSeg *)NULL;
}


/* function realloc_GcfFile() reallocates space for GcfSeg in a GcfFile struct
 * and updates associated attributes proper
 * 
 * NOTE: function will pass silently for size < 0 and size equal to already
 *  allocated size.
 * 
 * ARGUMENTS
 *  obj       struct to reallocate
 *  size      size needed 
 */
void realloc_GcfFile(GcfFile *obj, int size) {
   int i;
   if (size != obj->n_alloc && size > 0) {
      if (size < obj->n_alloc) {
         for (i=obj->n_alloc-1; i>=size; i--) {
            if (obj->seg[i].data) free(obj->seg[i].data);
         }
      }
      obj->seg = (GcfSeg *)realloc(obj->seg,size*sizeof(GcfSeg));
      if (size > obj->n_alloc) {
         for (i=obj->n_alloc; i<size; i++) {
            init_GcfSeg(&obj->seg[i],0);
         }
      }
      obj->n_alloc = size;
      if (size < obj->n_seg) obj->n_seg = size;
   }
}
void realloc_GcfSeg(GcfSeg *obj, int32 size) {
   if (size != obj->n_alloc && size > 0) {
      obj->data = (int32 *)realloc(obj->data,size*sizeof(int32));
      if (size < obj->n_data) obj->n_data = size;
      if (size > obj->n_alloc) memset(obj->data+obj->n_alloc,UNKNOWN,(size-obj->n_alloc)*sizeof(int32));
      obj->n_alloc = size;
   }
}
void free_GcfSeg(GcfSeg *obj) {
   if (obj->data) {
      free(obj->data);
   }
   init_GcfSeg(obj,0);
}
void free_GcfFile(GcfFile *obj) {
   int i;
   if(obj->seg) {
      for (i=0;i<obj->n_alloc;i++) free_GcfSeg(&obj->seg[i]);
      free(obj->seg);
   }
   init_GcfFile(obj);
}


/* function gcfSegSps() returns the sampling rate of a GcfSeg */
double gcfSegSps(GcfSeg *seg) {return seg->sps*1.0/seg->sps_denom;}

/* function gcfSegStart() returns the start time of a GcfSeg
 * in unix time. If segment start on leap second start will be
 * shifted one second back in time */
double gcfSegStart(GcfSeg *seg) {return (double)seg->start-seg->t_leap+seg->t_numerator*1.0/seg->t_denominator;}

/* function gcfSegEnd() returns the end time of a GcfSeg in unix time */
double gcfSegEnd(GcfSeg *seg) {return gcfSegStart(seg)+seg->n_data/gcfSegSps(seg);}

/* function cmpSegHead() returns True (non-zero integer) if the attributes:
 *   systemID, streamID, gain, type, sysType, ttl, sps, sps_denom 
 *  are equal in two GcfSeg, else function returns False (0) */
int cmpSegHead(GcfSeg *s1, GcfSeg *s2) {
   return !strcmp(s1->systemID,s2->systemID) && !strcmp(s1->streamID,s2->streamID) && 
      s1->gain==s2->gain && s1->type==s2->type && s1->sysType==s2->sysType && 
      s1->ttl==s2->ttl && s1->sps==s2->sps && s1->sps_denom==s2->sps_denom;
}



/* function CheckSegAligned() checks if GcfSeg s1 and GcfSeg s2 are aligned in time */
int CheckSegAligned(GcfSeg *s1, GcfSeg *s2, double tol) {
   double diff;
   int ret=-2;
   if (s1->sps == s2->sps && s1->sps_denom == s2->sps_denom && (s1->err == s2->err || (s1->err >= 0 && s2->err >= 0))) {
      diff = gcfSegEnd(s1)-gcfSegStart(s2);
      if (fabs(diff) < tol/gcfSegSps(s1)) ret = 0;
      else if (diff < 0) ret = -1;
      else ret = 1;
   }
   return ret;
}


/* function add_GcfSeg() copies a GcfSeg struct to a GcfFile struct */
void add_GcfSeg(GcfFile *obj, GcfSeg seg, int mode, double tol) {
   GcfSeg *s;
   int32 dd = -1, ddd;
   int i, j=obj->n_seg, chunk;
   chunk = MAX_DATA_BLOCK*10;
   if (mode < 2 && !seg.err && seg.n_data) {
      for (i=0; i<obj->n_seg; i++) {
         s = &obj->seg[i];
         if (!s->err && cmpSegHead(s,&seg)) {
            // look for previous segment with matching header         
            if (!CheckSegAligned(s,&seg,tol)) {
               // seg follows where s ends
               if (seg.n_alloc && s->n_alloc) {
                  if (dd < 0) {
                     // first match get jump in data (we use this to place new segment next to segment with 
                     //  less jump in amplitude if there are multiple matches)
                     dd = (int32)labs(s->RIC - seg.FIC);
                     j = i;
                  } else if ((ddd = (int32)labs(s->RIC - seg.FIC)) < dd) {
                     // found another aligned data vector where jump in data is less previously seen, use this instead
                     dd = ddd;
                     j = i;
                  }
               } else {
                  // we have no data vectors, simply set matching segment to first found
                  if (dd < 0) {
                     dd = 0;
                     j = i;
                  }
               }
            } else if (!CheckSegAligned(&seg,s,tol)) {
               // s follows where s ends
               if (seg.n_alloc && s->n_alloc) {
                  // new segment predates 
                  if (dd < 0) {
                     // first match get jump in data
                     dd = (int32)labs(s->FIC - seg.RIC);
                     j = -i-1;
                  } else if ((ddd = (int32)labs(s->FIC - seg.RIC)) < dd) {
                     dd = ddd;
                     j = -i-1;
                  }
               } else {
                  // we have no data vectors, simply set matching segment to first found
                  if (dd < 0) {
                     dd = 0;
                     j = -i-1;
                  }
               }
            }
         }
      }
      if (j<obj->n_seg) {
         i = (j<0) ? -(j+1): j; 
         if (seg.n_alloc && obj->seg[i].n_alloc-obj->seg[i].n_data < seg.n_data) {
            // re-allocate memory
            realloc_GcfSeg(&obj->seg[i],obj->seg[i].n_data+seg.n_data+chunk);
         }
      }
   }
   
   if (j == obj->n_seg) {
      // no match was found (or we are not to merge aligned segmets) make room for the segment
      if (j == obj->n_alloc) {
         // increase number of segments in file, we do this 5 segments at the time
         realloc_GcfFile(obj,obj->n_alloc+5);
      }
      s = &obj->seg[j];
      // allocate memory for new segment (allocate more than nedded to reduce allocations)
      // then set segment meta data --- 2017-03-10 added seg.n_data to logical expression below, cps  
      if (seg.n_alloc && seg.n_data) realloc_GcfSeg(s,seg.n_data+10*1024);
      strncpy(s->systemID,seg.systemID,sizeof(seg.systemID));
      strncpy(s->streamID,seg.streamID,sizeof(seg.streamID));
      s->start = seg.start;
      s->t_numerator = seg.t_numerator;
      s->t_denominator = seg.t_denominator;
      s->gain = seg.gain;
      s->type = seg.type;
      s->sysType = seg.sysType;
      s->ttl = seg.ttl;
      s->blk = seg.blk;
      s->err = seg.err;
      s->sps = seg.sps;
      s->sps_denom = seg.sps_denom;
      s->t_leap = seg.t_leap;
      s->compr = seg.compr;
      s->FIC = seg.FIC;
      s->RIC = seg.RIC;
      obj->n_seg += 1;
   } else if (j<0) {
      s = &obj->seg[-(j+1)];
      // segment to add is to be pre-pended to existing segment, update FIC, compression, start time and first block
      s->start = seg.start;
      s->t_numerator = seg.t_numerator;
      s->t_denominator = seg.t_denominator;
      s->t_leap = seg.t_leap;
      s->blk = seg.blk;
      s->compr = seg.compr;
      s->FIC = seg.FIC;
   } else {
      // segment to add is to be appended to existing segment, update RIC
      s = &obj->seg[j];
      s->RIC = seg.RIC;
   }
   
   // finally we copy the data vector --- 2017-03-10 added seg.n_data to logical expression below, cps  
   if (seg.n_alloc && seg.n_data) {
      if (j>=0) {
         // append segment
         memcpy(s->data+s->n_data,seg.data,seg.n_data*sizeof(int32));
      } else {
         // move existing data to make room, then pre-pend segment
         memmove(s->data+seg.n_data,s->data,s->n_data*sizeof(int32));
         memcpy(s->data,seg.data,seg.n_data*sizeof(int32));         
      }
   }
   s->n_data += seg.n_data;
}


/* function merge_GcfFile() merges the segments in a GcfFile object if aligned and metadata agrees */
void merge_GcfFile(GcfFile *obj, int mode, double tol) {
   if (obj->n_seg > 1) {
      int i, j, k, ii, jj, i0, i1, n, aligned, *Ord, *NewOrd, chunk;
      int32 dd;
      double *ord, is,js,ie,je,sps;
      GcfSeg tmp;
   
      // Allocate
      Ord = (int *)malloc(obj->n_seg*sizeof(int));
      NewOrd = (int *)malloc(obj->n_seg*sizeof(int)); 
      ord = (double *)malloc(obj->n_seg*sizeof(double));
      
      
      // chunks to use when reallocating 
      chunk = MAX_DATA_BLOCK*10;
      
      // sort out the order of the segments in the file
      ord[0] = gcfSegStart(&obj->seg[0]);
      Ord[0] = 0;
      for (i=1;i<obj->n_seg;i++) {
         // find first segment among the already sorted ones that is younger (greater time stamp) 
         for (j=0;j<i;j++) {
            if (ord[j] > gcfSegStart(&obj->seg[i])) break;
         }
         if (j != i) {
            // shift elements then insert
            for (k=i; k>j; k--) {
               ord[k] = ord[k-1];
               Ord[k] = Ord[k-1];
            }
         }
         // insert segment 
         ord[j] = gcfSegStart(&obj->seg[i]);
         Ord[j] = i;
      }
      
      // merge segments
      i=0;
      while (i<obj->n_seg-1) {
         if (obj->seg[Ord[i]].n_data && !obj->seg[Ord[i]].err) {
            for (j=i+1; j<obj->n_seg; j++) {
               if (cmpSegHead(&obj->seg[Ord[i]], &obj->seg[Ord[j]]) && obj->seg[Ord[j]].n_data && !obj->seg[Ord[j]].err && !obj->seg[Ord[j]].t_leap) {
                  if ((aligned = CheckSegAligned(&obj->seg[Ord[i]],&obj->seg[Ord[j]],tol)) == -1) {
                     // remaining blocks are all older, no need to search further
                     break;
                  } else {
                     if (!obj->seg[Ord[i]].n_alloc || !obj->seg[Ord[j]].n_alloc) {
                        // at least one of the segments have no data
                        obj->seg[Ord[i]].n_data += obj->seg[Ord[j]].n_data;
                        if (obj->seg[Ord[j]].start - obj->seg[Ord[i]].start < 0) obj->seg[Ord[i]].blk = obj->seg[Ord[j]].blk;   // update first block used in segment
                        free_GcfSeg(&obj->seg[Ord[j]]);
                     } else if (!aligned) {
                        // sanity check 1: check that there are no other segments with same start as obj->seg[Ord[j]], if found 
                        // select the one with smallest jump in data
                        k = 1;
                        dd = (int32)labs(obj->seg[Ord[i]].RIC - obj->seg[Ord[j]].FIC);
                        while (j+k < obj->n_seg) {
                           if (obj->seg[Ord[j+k]].n_data && !obj->seg[Ord[j+k]].err) {
                              if (!CheckSegAligned(&obj->seg[Ord[i]],&obj->seg[Ord[j+k]],tol)) {
                                 // next segment in line is also aligned check header and jump in data
                                 if (cmpSegHead(&obj->seg[Ord[j]], &obj->seg[Ord[j+k]]) && (int32)labs(obj->seg[Ord[i]].RIC - obj->seg[Ord[j+k]].FIC) < dd) {
                                    // next segment have smaller jump in data, break (will look further ahead next iteration)
                                    break;
                                 }
                              } else {
                                 // next segment is not aligned, break
                                 k = 0; // set k = 0 to indicate to use obj->seg[Ord[j]]
                                 break;
                              }
                           }
                           k++;
                        }
                        if (j+k==obj->n_seg) k = 0;
                        if (!k) {
                           // first sanity check passed (if it failed outermost for loop over j will continue to next segment in line)
                           // sanity check 2:
                           //  check that none of the segments prior to obj->seg[Ord[i]] is aligned with obj->seg[Ord[j]] and have 
                           //  a smaller jump in data, if so update i and break for loop over j
                           ii = -1;
                           while (k<j) {
                              if (k != i && obj->seg[Ord[k]].n_data && !obj->seg[Ord[k]].err && !CheckSegAligned(&obj->seg[Ord[k]],&obj->seg[Ord[j]],tol) &&
                                 cmpSegHead(&obj->seg[Ord[k]], &obj->seg[Ord[j]]) && (int32)labs(obj->seg[Ord[k]].RIC - obj->seg[Ord[j]].FIC) < dd) {
                                 // segment k is aligned and have smaller jump in data
                                 ii = k;
                              }
                              k++;
                           }
                           if (ii == -1) ii = i; 
                           if (ii < i) {
                              // sanity check 2 triggered for segment before i, update i and break loop over j
                              i = ii-1;
                              break;
                           } else {
                              // second sanity check either passed or triggered for segment after i, merge segments
                              // re-allocate memory, we do so in chuncks to reduce number of reallocations
                              if (obj->seg[Ord[ii]].n_alloc < obj->seg[Ord[ii]].n_data+obj->seg[Ord[j]].n_data) {
                                 realloc_GcfSeg(&obj->seg[Ord[ii]],obj->seg[Ord[ii]].n_data+obj->seg[Ord[j]].n_data+chunk);                        
                              }
                              memcpy(obj->seg[Ord[ii]].data+obj->seg[Ord[ii]].n_data,obj->seg[Ord[j]].data,obj->seg[Ord[j]].n_data*sizeof(int32));
                              obj->seg[Ord[ii]].n_data += obj->seg[Ord[j]].n_data;
                              obj->seg[Ord[ii]].RIC = obj->seg[Ord[j]].RIC; // 2020-03-01, CPS update RIC
                              if (obj->seg[Ord[j]].start - obj->seg[Ord[ii]].start < 0) obj->seg[Ord[ii]].blk = obj->seg[Ord[j]].blk;   // update first block used in segment
                              free_GcfSeg(&obj->seg[Ord[j]]);
                           }
                        }
                     } else if (!mode) {
                        // check overlapping data, if identical merge segments and throw away overlap
                        sps = gcfSegSps(&obj->seg[Ord[i]]);
                        is = gcfSegStart(&obj->seg[Ord[i]]);
                        js = gcfSegStart(&obj->seg[Ord[j]]);
                        ie = is+obj->seg[Ord[i]].n_data/sps;
                        je = js+obj->seg[Ord[j]].n_data/sps;
                        i0 = (int)round((js-is)*sps);
                        if (fabs(ie-je) < tol/sps || ie < je) {
                           i1 = obj->seg[Ord[i]].n_data;
                        } else i1 = i0+obj->seg[Ord[j]].n_data;
                        for (ii=i0, jj=0; ii<i1; ii++,jj++) {
                           if (obj->seg[Ord[i]].data[ii] != obj->seg[Ord[j]].data[jj]) break;
                        }
                        if (ii==i1) {
                           // overlapping data is the same remove overlap
                           if (jj != obj->seg[Ord[j]].n_data) {
                              // new segments only partly overlap, add none-overlapping data from new segments to old
                              n = obj->seg[Ord[j]].n_data-jj; 
                              if (obj->seg[Ord[i]].n_alloc < obj->seg[Ord[i]].n_data+n) {
                                 // re-allocate memory
                                 realloc_GcfSeg(&obj->seg[Ord[i]],obj->seg[Ord[i]].n_data+n+chunk);
                              }
                              memcpy(obj->seg[Ord[i]].data+obj->seg[Ord[i]].n_data,obj->seg[Ord[j]].data+jj,n*sizeof(int32));
                              obj->seg[Ord[i]].n_data += n;                  // 2020-03-01, CPS update n_data
                              obj->seg[Ord[i]].RIC = obj->seg[Ord[j]].RIC;  // 2020-03-01, CPS update RIC
                           }
                           if (obj->seg[Ord[j]].start - obj->seg[Ord[i]].start < 0) obj->seg[Ord[i]].blk = obj->seg[Ord[j]].blk; // update first block used in segment
                           free_GcfSeg(&obj->seg[Ord[j]]);
                        }
                     }
                  }
               }
            }
         }
         i++;
      }
      
      // get ordering and number of remaining segments
      n = 0;
      for (i=0;i<obj->n_seg; i++) {
         if (obj->seg[Ord[i]].n_data) {
            NewOrd[n] = Ord[i];
            n += 1;
         }
      }
      
      // move data to get properly ordered segments
      for (i=0; i<n; i++) {
         if (NewOrd[i] != i) {
            j = 0;
            if (obj->seg[i].n_data) {
               // site already occupied, cp data to tmp
               tmp = obj->seg[i];
               j = 1;
            }
            obj->seg[i] = obj->seg[NewOrd[i]];
            if (j) obj->seg[NewOrd[i]] = tmp;
            else init_GcfSeg(&obj->seg[NewOrd[i]],0);
            // update position
            for (j=i; j<n; j++) {
               if (NewOrd[j] == i) {
                  NewOrd[j] = NewOrd[i];
                  break;
               }
            }
         }
      }
      // finally set number of segments proper
      obj->n_seg = n;
      
      // release memory
      free(Ord);
      free(NewOrd);
      free(ord);
   }
}


/* return True (1) if on little endian machine */
int is_LittleEndian_gcf(void) {
   static int i = -192; // Just an initial value to compute only once if little endian
   if (i == -192) {
      int j;
      char *cp;
      cp = (char *) & j;
      j = 57;
      i = (*cp == 57);
   }
   return i;
}


/* shift low and high byte in short */
void swab_short(char *val) {
   char temp;
   temp = *val;
   *val = *(val+1);
   *(val+1) = temp;
}


/* shift all bytes in long = 32 bit integer here, hmmm why is the masking needed here (simply to assure we work with 8 bit) */
void swab_long(char *val) {
   char temp;
   temp = *val & 0xff;
   *val = *(val+3) & 0xff;
   *(val+3) = temp & 0xff;
   temp = *(val+1) & 0xff;
   *(val+1) = *(val+2) & 0xff;
   *(val+2) = temp & 0xff;
}


/* closes a file connection */
void closegcf(int32 *fid) {
   close(*fid);
}


/* open a file for reading */
int opengcf(const char *fname, int32 *fid) {
   if ((*fid = open_r(fname, ORFLAG)) <  0 ) {
      return 1;
   }
   return 0;
}


/* reads in data from file*/
int gcf_read(int fd, void *buf, size_t count) {
   int n;
   n = (int)read(fd,buf,count);
   if (n < 0) {
      return -1;
   }
   return(n);
}


/* Function FillBuffer() reads a new datablock
 * 
 * ARGUMENTS:
 *  size      size in bytes of block to read
 *  buffer    will upon successful return hold read data block
 *  fid       location to read from
 * 
 * RETURN
 *  upon successful return function returns number of bytes read, else 0
 */
int FillBuffer(int size, unsigned char buffer[], int32 *fid) {
   int n;
   n = gcf_read(*fid,buffer,size);
   if (n <= 0) return(0);
   return(n);
}


/* Function IDToStr() decodes the either the streamID or the systemID field in a block header
 * of the gcf format
 * 
 * ARGUMENTS:
 *  ID     container of ID to decode
 *  gain   will upon successful return hold gain setting if format is (double) extended, else -1
 *  type   will upon successful return hold Instrument type if format is (double) extended, else 0
 *  Str    will upon successful return hold ID (>6 characters)
 * 
 * RETURN:
 *  if all went well function returns 0 if ID format is not extended (ID is 6 char), 1 if format is 
 *  extended (ID is 5 char) and 2 if format is double extended (ID is 4 char). Note that function will 
 *  not strictly conform to expected number of chars but rather decode for actual number of chars (max 6)
 */
int IDToStr(uint32 ID, int *gain, int *type, char *Str) {
   int ret=0, i, j=5;
   int32 imed;
   uint32 _ID;
   Str[6] = 0;
   _ID = ID;
   *gain = -1;
   *type = 0;
   if (_ID & 0x80000000) {
      // Bit 31 is set -> (double?) extended format, decode gain and type
      if (_ID & 0x04000000) *type = 1;
      *gain = (_ID & 0x38000000) >> 27;
      if (*gain > 1) *gain = 1 << (*gain-1);
      
      if (_ID & 0x40000000) {
         // bit 31 and 30 are set -> double extended format, unmask first 11 bits
         _ID &= 0x001fffff;
         ret = 2;
      } else {
         // bit 31 is set -> double extended format, unmask first 6 bits
         _ID &= 0x03ffffff;
         ret = 1;
      }
   }
   for (i=5; i>=0; i--) {
      imed = _ID % 36;
      if (imed > 9) imed += 7;
      Str[i] = imed + '0';
      _ID /= 36;
      if (!_ID) break;
   }
   if (i) {
      // if less than 6 characters (ok if (double) extended format) shift characters
      for (j=0;j<=5-i;j++) Str[j] = Str[j+i];
      for (j=j;j<=5;j++) Str[j] = 0;
   }
   return(ret);
}


/* Function GcfTime2Unix() converts GCF time (days since
 * 1989-11-17 + seconds since 00:00:00Z) to unix time (seconds
 * since 1970-01-01 00:00:00)
 * 
 * ARGUMENT
 *  time     GCF time with seconds in bits 0-16 and days in bits 17-31 
 *  leap     will upon return be 1 if time is a leapsecond, else 0
 * 
 * RETURN
 *  function returns unix time
 */
gtime GcfTime2Unix(uint32 time, int *leap) {
   gtime ut;
   ut = 627264000 + ((time>>17) * 86400) + (time & 0x0001ffff);
   *leap = (time & 0x0001ffff)==86400 ? 1 : 0;
   return ut;
}


/* function UnixTime2Gcf() converts unix time (seconds since 1970-01-01 00:00:00)
 * to GCF time (days since 1989-11-17 + seconds since 00:00:00Z, leapsecond 
 * indicated by 86400s)  
 * 
 * ARGUMENTS
 *  time       Unix time
 *  leap       1 if time is a leap second, else 0, NOTE: for leap second to be set
 *              input unix time must be compatible (i.e. second part should be 0 or
 *              equivalently 86400)
 *  
 * RETURN
 *  GCF time with seconds in bits 0-16 and days in bits 17-31. NOTE: GCF time starts
 *  at 0, any ealier times will be 0
 */
uint32 UnixTime2Gcf(gtime time, int leap) {
   uint32 gcf_t=0, days;
   time -= 627264000;
   if (time > 0) {
      gcf_t = time%86400;
      days = time/86400;
      if (leap && !gcf_t) {
         days -= 1;
         gcf_t = 86400;
      }
      gcf_t += days<<17;
   }
   return gcf_t;
}


/* function ParseGcfBlockHeader() parses the content in a gcf block header
 * 
 *  NOTE: function does not check that ttl header value is in range
 * 
 * ARGUMETS
 *  bh      structure holding unparsed header
 *  seg     stucture to store parsed header
 *  endian  indicator if on a litle endian machine (1 else 0)
 * 
 * RETURN
 *  if successful function returns 0 else function returns error code set in seg->err
 */
int ParseGcfBlockHeader(BH *bh, GcfSeg *seg, int endian) {
   int gain, type;
   if (endian) {
      swab_long((char*)&bh->systemID);
      swab_long((char*)&bh->streamID);
      swab_long((char*)&bh->time);
   }
   seg->sysType = IDToStr(bh->systemID, &seg->gain, &seg->type, seg->systemID);
   seg->err = IDToStr(bh->streamID, &gain, &type, seg->streamID);
   seg->start = GcfTime2Unix(bh->time,&seg->t_leap);   
   seg->sps = bh->sps;
   switch (seg->sps) {
   case 157:
      seg->sps = 1;
      seg->sps_denom = 10;
      break;
   case 161:
      seg->sps = 1;
      seg->sps_denom = 8;
      break;
   case 162:
      seg->sps = 1;
      seg->sps_denom = 5;
      break;
   case 164:
      seg->sps = 1;
      seg->sps_denom = 4;
      break;
   case 167:
      seg->sps = 1;
      seg->sps_denom = 2;
      break;
   case 171:
      seg->sps = 400;
      seg->t_denominator = 8;
      break;
   case 174:
      seg->sps = 500;
      seg->t_denominator = 2;
      break;
   case 175:
      seg->sps = 800;
      seg->t_denominator = 16;
      break;
   case 176:
      seg->sps = 1000;
      seg->t_denominator = 4;
      break;
   case 179:
      seg->sps = 2000;
      seg->t_denominator = 8;
      break;
   case 181:
      seg->sps = 4000;
      seg->t_denominator = 16;
      break;
   case 182:
      seg->sps = 625;
      seg->t_denominator = 5;
      break;
   case 191:
      seg->sps = 1250;
      seg->t_denominator = 5;
      break;
   case 193:
      seg->sps = 2500;
      seg->t_denominator = 10;
      break;
   case 194:
      seg->sps = 5000;
      seg->t_denominator = 20;
      break;
   }
   if (seg->sps > 250) { 
      seg->t_numerator = ((bh->comp & 0xf0) >> 4) + ((bh->comp & 0x08) << 1); // 2020-03-01 cps, updated to include fourth bit as MSB
   } else if (seg->sps == 0) seg->err = -1;
   bh->comp = bh->comp & 0x07;  // 2020-03-01 cps, updated to only use first 3 bits (were first 4 -> & 0x0f), also make sure to 
                                // only use first 3 bits as it can not be guaranted that bits 3-7 are 00000 for sps <= 250
                                // according to Fish at Guralp (email conversation 2020-03-01)
   seg->compr = bh->comp;
   seg->ttl = bh->ttl;
   seg->n_data = bh->nrec*bh->comp;
   
   return seg->err;
}


/* funtion decode() decodes data in a gcf block
 *
 * ARGUMENTS
 *  buff      pointer to memory where FIC (forward integration constant) exists
 *  compr     compression level (1 - 32 bit, 2 - 16 bit, 4 - 8 bit; signed diff)
 *  n         expected number of data samples to decode
 *  y         data vector to put converted samples in
 *  endian    endianess of current machine
 *  FIC       Will upon retun hold the Forward Integration Constant
 *  err       Will upon return hold an error code if an error occured else be untouched,
 *             error codes are:
 *               3  Unknown compression code
 *              10  Failure to verify decoded data (last data != RIC)
 *              11  First difference not 0
 *              21  error 10 + 21
 * 
 * RETURN
 *  if sucessful function returns RIC (reverse integration constant) that should
 *  be equal that last sample in the data vector if properly decoded
 */ 
int32 decode(char *buff, uint8 compr, int32 n, int32 *y, int endian, int32 *FIC, int *err) {
   int32 i,RIC,ival,size,csize,off;
   int16 sval;
   size = 4; 
   if (endian) swab_long(&buff[0]);
   memcpy(FIC,buff,size);
   y[0] = *FIC;
   i = 0;
   n--;
   switch (compr) {
   case 4:
      csize = 1;          // size of difference
      y[i] += buff[size]; // add first difference (should be 0)
      if (y[i] != *FIC) *err = 11; 
      off = size+csize;
      while(n--) {
         y[i+1] = y[i] + buff[i+off];
         i++;
      }
      memcpy(&RIC,&buff[i+off],size);
      if (endian) swab_long((char*)&RIC);
      // check that last data point verfies against RIC
      if (y[i] != RIC) *err = *err == 11 ? 21 : 10;
      return(RIC);
   case 2:
      csize = 2;        // size of difference
      // add first difference (should be 0)
      memcpy(&sval,&buff[size],csize);
      if (endian) swab_short((char*)&sval);
      y[i] += sval;
      if (y[i] != *FIC) *err = 11; 
      off = size+csize;
      while(n--) {
         memcpy(&sval,&buff[i*csize+off],csize);
         if (endian) swab_short((char*)&sval);
         y[i+1] = y[i] + sval;
         i++;
      }
      memcpy(&RIC,&buff[i*csize+off],size);
      if (endian) swab_long((char*)&RIC);
      // check that last data point verfies against RIC
      if (y[i] != RIC) *err = *err == 11 ? 21 : 10;
      return(RIC);
   case 1:
      csize = 4;        // size of difference
      // add first difference (should be 0)
      memcpy(&ival,&buff[size],csize);
      if (endian) swab_short((char*)&ival);
      ival &= 0xffffffff;   // this really shouldn't be neccesary
      y[i] += ival;
      if (y[i] != *FIC) *err = 11; 
      off = size+csize;
      while(n--) {
         memcpy(&ival,&buff[i*csize+off],csize);
         if (endian) swab_long((char*)&ival);
         ival &= 0xffffffff;   // this really shouldn't be neccesary
         y[i+1] = y[i] + ival;
         i++;
      }
      memcpy(&RIC,&buff[i*csize+off],size);
      if (endian) swab_long((char*)&RIC);
      // check that last data point verfies against RIC
      if (y[i] != RIC) *err = *err == 11 ? 21 : 10;
      return(RIC);
   default:
      *err = 3;
      return(UNKNOWN);
   }
}


/* function ProcData() processes the data part of a gcf block
 * 
 * ARGUMENTS
 *  y         data vector to store decoded data in
 *  buffer    gcf block to process
 *  compress  compression level used
 *  no_of_s   expected number of samples in block
 *  endian    endianess of machine
 *  FIC       Will upon return hold the Forward Integration Constant
 *  RIC       Will upon retun hold the Reverse Integration Constant
 *  err       Will upon return hold an error code if an error occured else be untouched,
 *             error codes are:
 *               3  Unknown compression code
 *              10  Failure to verify decoded data (last data != RIC)
 *              11  First difference not 0
 * 
 * RETURN
 *  if successful function returns number of data samples parsed, else -1
 */
int32 ProcData(int32 *y, unsigned char buffer[], uint8 compress, int32 no_of_s, int endian, int32 *FIC, int32 *RIC, int *err) {
   int32 i;
   *RIC = decode((char *)&buffer[16],compress,no_of_s,y,endian,FIC,err);
   i = no_of_s;
   if ((y[no_of_s-1]-*RIC) != 0.0) {
      return(-1);
   }
   return(i);
}


/* function StrToID() encodes a streamID or systemID into gcf block header format
 * 
 * ARGUMENTS
 *   Str      streamID/systemID to encode
 *   ID       will upon return hold encoded ID
 */
void StrToID(char *Str, uint32 *ID) {
   int i=0,c;
   *ID = 0;
   while (Str[i]) {
      c = Str[i]-'0';
      if (c > 9) c-=7;
      *ID += c;
      if (Str[i+1]) *ID *= 36;
      i++;
   }
}


/* function verify_GcfFile() verifies that a GcfFile
 * struct contains all information neccesary to be written 
 * to file. It also makes sure that streamID and systemID
 * are upper case.
 * 
 * NOTE Function will only report on first encountered problem
 *  if multiple exists
 *
 * ARGUMENTS
 *  obj       GcfFile struct to check
 *  
 * RETURN
 *  0 if ok to write to file
 *  1 if no data or inconsistent segment info
 *  2 if unsupported samplingrate
 *  3 erronous fractional start time
 *  4 unsupported gain
 *  5 erronous instrument type
 *  6 to many characters in systemID
 */ 
int verify_GcfFile(GcfFile *obj) {
   int i,j,n,ret=0;
   
   if (!obj->n_alloc || !obj->n_seg || obj->n_seg > obj->n_alloc) { 
      // no data or inconsistent segment info
      ret = 1;
   } else {
      // check the segments
      i = 0;
      for (i=0; i<obj->n_seg; i++) {
         if (obj->seg[i].n_alloc && obj->seg[i].n_data && obj->seg[i].n_data <= obj->seg[i].n_alloc) break;
         for (j=0; j<(int)strlen(obj->seg[i].streamID); j++) obj->seg[i].streamID[j] = toupper(obj->seg[i].streamID[j]);
         for (j=0; j<(int)strlen(obj->seg[i].systemID); j++) obj->seg[i].systemID[j] = toupper(obj->seg[i].systemID[j]);
      }
      if (i < obj->n_seg) {
         for (i=0; i<obj->n_seg; i++) {
            if (obj->seg[i].sps_denom != 1) {
               if (obj->seg[i].sps != 1) ret = 2;
               else if (obj->seg[i].sps_denom != 2 && obj->seg[i].sps_denom != 4 && obj->seg[i].sps_denom != 5 &&
                  obj->seg[i].sps_denom != 8 && obj->seg[i].sps_denom != 10) ret = 2;
            } else if (obj->seg[i].sps < 1) ret = 2;
            else if (obj->seg[i].sps < 251 && obj->seg[i].t_numerator != 0 && obj->seg[i].t_numerator != 1) ret = 3;
            else if (obj->seg[i].sps > 250 && obj->seg[i].sps != 400 && obj->seg[i].sps != 500 && obj->seg[i].sps != 800 && 
               obj->seg[i].sps != 1000 &&  obj->seg[i].sps != 2000 &&  obj->seg[i].sps != 4000 && obj->seg[i].sps != 625 && 
               obj->seg[i].sps != 1250  && obj->seg[i].sps != 2500 &&  obj->seg[i].sps != 5000) ret = 2; 
//             else if (obj->seg[i].sps > 250 && obj->seg[i].t_denominator < 1) ret = 3; // this is actually not used so no need to check
            else if (((n=(int)strlen(obj->seg[i].systemID)) > 6 || (obj->seg[i].sysType == 1 && n > 5) || (obj->seg[i].sysType == 2 && n > 4))) ret = 6;
            if (!ret) {
               if (obj->seg[i].gain > -1) {
                  if (obj->seg[i].gain && obj->seg[i].gain != 1 && obj->seg[i].gain != 2 && obj->seg[i].gain != 4 && obj->seg[i].gain != 8 && 
                     obj->seg[i].gain != 16 && obj->seg[i].gain != 32 && obj->seg[i].gain != 64) ret = 4;
                  else if (obj->seg[i].type < 0 || obj->seg[i].type > 1 || obj->seg[i].sysType < 0 || obj->seg[i].sysType > 2) ret = 5;
               }
            } 
            
            if (ret) break;
         }
      } else ret = 1;
   }
   return ret;
}


/* function parse_gcf_block() parses a 1024 byte gcf data block */
int parse_gcf_block(unsigned char buffer[1024], GcfSeg *seg, int mode, int endian) {
   BH *bh=NULL;
   int ret = 0;
   
   // reset seg and set bh 
   init_GcfSeg(seg,1);
   bh = (BH*)&buffer[0];
   // parse block header
   if (ParseGcfBlockHeader(bh, seg, endian) >= 0) {
      if (bh->comp != 1 && bh->comp != 2 && bh->comp != 4) {
         // unknown compression 
         seg->err = 3;
      } else if (seg->n_data > MAX_DATA_BLOCK || seg->n_data <= 0) {
         // number of data samples indicate in header is greater than what can be stored in block
         seg->err = 4;
      } else {
         // all info needed to decode data is available
         if (seg->start < 0) seg->err = 5;
         if (mode >= 0) {
            // allocate memory and decode data
            if ((ProcData(seg->data,buffer,bh->comp,seg->n_data,endian,&seg->FIC,&seg->RIC,&seg->err)) < 0) {
               // there was a problem decoding the datablock
               ret = seg->err;
            }
         }
      }
   }  
   if (!ret) ret = seg->err;
   return ret;
}


/* function read_gcf() parses a gcf data file */
int read_gcf(const char *f, GcfFile *obj, int mode) {
   int ret=0, endian, d=0, err=0, b1 = 0; 
   int32 fid=0, n_alloc=0;
   GcfSeg seg;
   double tol = 1.E-3;
   unsigned char buffer[1024]; 
   
   // initiate and allocate segment
   init_GcfSeg(&seg,0);
   if (mode >= 0) realloc_GcfSeg(&seg, MAX_DATA_BLOCK);
   
   // adjust mode if nedded
   if (mode > 2) {
      mode = 2;
      b1 = 1;
   }
   
   // get endianess of current machine
   endian = is_LittleEndian_gcf();
   if (opengcf(f,&fid)) ret=-1;
   else {
      while(FillBuffer(1024,buffer,&fid)) {
         obj->n_blk += 1;
         if ((err=parse_gcf_block(buffer,&seg,mode,endian)) < 0) {
            // not a data block
            d++;
         } else if (err >= 10) {
            // there were some issues with the data block
            obj->n_errData++;
         }
         if (seg.err > 0 && seg.err < 10) {
            obj->n_errHead++;
         }
         
         // all done add segment
         seg.blk = obj->n_blk-1;
         if (mode >= 0 && (seg.err == 3 || seg.err == 4)) {
            // no data were actually decoded, temporarly set n_alloc to 0 to avoid adding non-existing data to GcfFile
            n_alloc = seg.n_alloc;
            seg.n_alloc = 0;
         }
         add_GcfSeg(obj,seg,abs(mode),tol);
         if (mode >= 0 && (seg.err == 3 || seg.err == 4)) seg.n_alloc = n_alloc;
         if (b1) break;
      }
      closegcf(&fid);
   }
   // free segment
   free_GcfSeg(&seg);

   // merge segments if asked for
   if (abs(mode) < 2) merge_GcfFile(obj,mode,tol);
   if (!ret && obj->n_blk==d) ret = 1 + endian; // no data blocks in file   
   return ret;
}


/* function write_gcf() writes a gcf data file */
int write_gcf(const char *f, GcfFile *obj) {
   int FID,i,j,k,n,m,t_numerator,d0,d1,c,cprev,ret=1,dsps=1, dd=1, endian,csize=1,leap;
   gtime dt;
   uint32 x;
   int32 FIC, RIC, d32;
   int16 d16;
   signed char d8;
   unsigned char buffer[1024]; 
   BH *bh=NULL;
   
   // get endianess of current machine
   endian = is_LittleEndian_gcf();
   
   // check that obj contains some data
   if (!(ret=verify_GcfFile(obj))) {
      // open file with read/write for owner and read for group and others
      if((FID = open_w(f, OWFLAG, FPERM)) >= 0) {
         bh = (BH*)&buffer[0];
         ret = 0;
         
         // loop over the segments
         for (i=0; i<obj->n_seg;i++) {               
            if (obj->seg[i].n_data) {
               // initialize header
               memset(buffer,0,16);
               if (obj->seg[i].sps <= 250) t_numerator = 0;
               else t_numerator = obj->seg[i].t_numerator;
               
               // encode the systemID
               x = 0;
               StrToID(obj->seg[i].systemID, &x);
               if (obj->seg[i].sysType > 0) {
                  // (double) extended sysid, add info on type then on gain
                  if (obj->seg[i].type == 1) {
                     // set bit 26
                     x |= 0x04000000;
                  }
                  if (obj->seg[i].gain > 0) {
                     // set bits 27-29
                     switch(obj->seg[i].gain) {
                     case 1:
                        // 001
                        x |= 0x08000000;
                        break;
                     case 2:
                        // 010
                        x |= 0x10000000;
                        break;
                     case 4:
                        // 011
                        x |= 0x18000000;
                        break;
                     case 8:
                        // 100
                        x |= 0x20000000;
                        break;
                     case 16:
                        // 101
                        x |= 0x28000000;
                        break;
                     case 32:
                        // 110
                        x |= 0x30000000;
                        break;
                     case 64:
                        // 111
                        x |= 0x38000000;
                        break;
                     }                        
                  }
                  // finally set bit 31 to indicate extended format
                  x |= 0x80000000;
                  if (obj->seg[i].sysType == 2) x |= 0x40000000; // Double extended format set bit 30 
               }    
               bh->systemID = x;
               if (endian) swab_long((char*)&bh->systemID);
               
               // encode the streamID
               x = 0;
               StrToID(obj->seg[i].streamID, &x);
               bh->streamID = x;
               if (endian) swab_long((char*)&bh->streamID);
               
               // convert integer time stamp to gcf time, ttl and sps
               bh->ttl = obj->seg[i].ttl & 0xff;
               if (obj->seg[i].sps_denom > 1) {
                  switch(obj->seg[i].sps_denom) {
                  case 2:
                     // 0.5 Hz
                     bh->sps = 167 & 0xff;
                     break;
                  case 4:
                     // 0.25 Hz
                     bh->sps = 164 & 0xff;
                     break;
                  case 5:
                     // 0.2 Hz
                     bh->sps = 162 & 0xff;
                     break;
                  case 8:
                     // 0.125 Hz
                     bh->sps = 161 & 0xff;
                     break;
                  case 10:
                     // 0.1 Hz
                     bh->sps = 157 & 0xff;
                     break;
                  }
               } else if (obj->seg[i].sps > 250) {
                  switch(obj->seg[i].sps) {
                  case 400:
                     bh->sps = 171 & 0xff;
                     dd = 8;
                     break;
                  case 500:
                     bh->sps = 174 & 0xff;
                     dd = 2;
                     break;
                  case 800:
                     bh->sps = 175 & 0xff;
                     dd = 16;
                  case 1000:
                     bh->sps = 176 & 0xff;
                     dd = 4;
                     break;
                  case 2000:
                     bh->sps = 179 & 0xff;
                     dd = 8;
                     break;
                  case 4000:
                     bh->sps = 181 & 0xff;
                     dd = 16;
                     break;
                  case 625:
                     bh->sps = 182 & 0xff;
                     dd = 5;
                     break;
                  case 1250:
                     bh->sps = 191 & 0xff;
                     dd = 5;
                     break;
                  case 2500:
                     bh->sps = 193 & 0xff;
                     dd = 10;
                     break;
                  case 5000:
                     bh->sps = 194 & 0xff;
                     dd = 20;
                     break;
                  }
                  dsps = obj->seg[i].sps/dd; // number of samples per gcf block must be an integer number of dsps
               } else bh->sps = obj->seg[i].sps & 0xff;
               
               // split and compress the data
               d1 = 0;
               dt = 0;
               leap = obj->seg[i].t_leap;
               while (d1 < obj->seg[i].n_data) {
                  // initialize data vector
                  memset(buffer+16,0,1008);
                  
                  // set time stamp
                  bh->time = UnixTime2Gcf(obj->seg[i].start+dt,leap);
                  if (endian) swab_long((char*)&bh->time);
                  leap = 0;
                  
                  d0 = d1;  // first data point
                  d1++;     // last data point
                  // find the compression level
                  d32 = obj->seg[i].data[d1]-obj->seg[i].data[d0];
                  if (d32 < -32768 || d32 > 32767) {
                     // need 4 byte for each difference, maximum number of data points are 250
                     c = 4;
                     d1 = d0+249;  // 249 here to prevent compression to get swaped to 2 later
                  } else {                        
                     if (d32 < 128 && d32 >= -128) c = 1; // initial difference can do with 1 byte
                     else c = 2;  // initial difference needs 2 byte
                     
                     // scan until block is full ( < 1000 bytes) or all data is covered
                     while (d1+1 < obj->seg[i].n_data && (d1-d0+1)*c < 1000) {
                        cprev = c;
                        d32 = obj->seg[i].data[d1+1]-obj->seg[i].data[d1];
                        d1++;
                        if (d32 > 127 || d32 < -128) {
                           if (d32 < 32768 && d32 >= -32768) {
                              c = 2;
                           } else {
                              c = 4;
                              d1 = obj->seg[i].n_data > d0+249 ? d0+249:obj->seg[i].n_data-1;
                              break;
                           }
                        }
                        if ((d1-d0+1)*c > 1000) {
                           d1--;
                           if(cprev < c) {
                              c = cprev; 
                              break;
                           }
                        }
                     }
                  }
                  
                  // if it fits use a lower compression, actually this would optimally be done after adjusting last 
                  // sample to use, but that would complicate the algorithm, in any it is not really needed
                  if ((d1-d0+1) <= 250) c = 4;
                  else if ((d1-d0+1) <= 500) c = 2;   
                  
                  // adjust compression code, for readability perhaps use another variable, well well...
                  if (c==1) c=4; 
                  else if (c==4) c=1;
                  
                  // set compresion code in header
                  bh->comp = c;
                  
                  // adjust the last sample if nedded, skip if these are the last few data points or sps < 1 Hz
                  if (d1 < obj->seg[i].n_data-1 && obj->seg[i].sps_denom == 1) {
                     // sampling rate is greater than 1 Hz
                     m = 0;
                     if (dsps != 1) {
                        // sampling rate is greater than 250 Hz, block may start and end at fractional time
                        if (t_numerator) {
                           // add fractional start time to bits 4-7 with MSB in bit 3 in compression code entry in header
                           bh->comp += (t_numerator & 0xf)<<4; // First 4 bits (0-3) moved into bit 4-7
                           bh->comp += (t_numerator & 0x10)>>1; // MSB (bit 4) moved into bit 3
                        }
                        m = (d1-d0+1)/dsps;
                        n = m*dsps;
                        d1 = d0+n-1;
                        dt += (m+t_numerator)/dd;
                        t_numerator = (m+t_numerator)%dd;
                     } else {
                        // block should end at integer second
                        m = (d1-d0+1)/obj->seg[i].sps; 
                        n = m*obj->seg[i].sps;
                        d1 = d0+n-1;
                        dt += m; 
                     }
                  } else {
                     if (d1 >= obj->seg[i].n_data-1) {
                        if (dsps != 1 && t_numerator) {
                           // last data block for > 250 Hz data, add fractional start time of block
                           // to bits 4-7 with MSB in bit 3 in compression code entry in header
                           bh->comp += (t_numerator & 0xf)<<4; // First 4 bits (0-3) moved into bit 4-7
                           bh->comp += (t_numerator & 0x10)>>1; // MSB (bit 4) moved into bit 3
                        }
                        d1 = obj->seg[i].n_data-1;
                     }
                     n = d1-d0+1;
                     dt += n*obj->seg[i].sps_denom;
                  }
                  
                  // set number of 4 byte records in header
                  bh->nrec = (n/c) & 0xff;
                  
                  // encode data, first we write FIC
                  FIC = obj->seg[i].data[d0];
                  if (endian) swab_long((char *)&FIC);
                  memcpy(&buffer[16],&FIC,4);         
                  j = 0;
                  switch (c) {       
                  case 1:
                     csize = 4;
                     for (k=d0; k<d1; k++) { 
                        j++; // NOTE: we increment before as the first difference of the first data is always 0
                        d32 = obj->seg[i].data[k+1]-obj->seg[i].data[k];
                        if (endian) swab_long((char*)&d32);
                        memcpy(&buffer[20+j*csize],&d32,csize);
                     }  
                     break;                      
                  case 2:
                     csize = 2;
                     for (k=d0; k<d1; k++) { 
                        j++; // NOTE: we increment before as the first difference of the first data is always 0
                        d16 = obj->seg[i].data[k+1]-obj->seg[i].data[k];
                        if (endian) swab_short((char*)&d16);
                        memcpy(&buffer[20+j*csize],&d16,csize);
                     }      
                     break;         
                  case 4:
                     csize = 1;
                     for (k=d0; k<d1; k++) { 
                        j++; // NOTE: we increment before as the first difference of the first data is always 0
                        d8 = obj->seg[i].data[k+1]-obj->seg[i].data[k];
                        buffer[20+j*csize] = d8;
                     }    
                     break;     
                  }
                  // and lastly write RIC
                  RIC = obj->seg[i].data[d1];
                  if (endian) swab_long((char*)&RIC);
                  memcpy(&buffer[20+(j+1)*csize],&RIC,4);
                  
                  // write block
                  if (write(FID,&buffer,1024) != 1024) {
                     // something went wrong writing, break and return
                     ret = -2; 
                     break;
                  }
                  if (ret) break;
                  d1++;   // do not write d1 again.
               }
            }
         }
         close(FID);
      } else ret = -1;
   }
   return ret;
}
