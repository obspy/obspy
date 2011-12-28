/***************************************************************************
 * tracelist.c:
 *
 * Routines to handle TraceList and related structures.
 *
 * Written by Chad Trabant, IRIS Data Management Center
 *
 * modified: 2011.304
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "libmseed.h"


MSTraceSeg *mstl_msr2seg (MSRecord *msr, hptime_t endtime);
MSTraceSeg *mstl_addmsrtoseg (MSTraceSeg *seg, MSRecord *msr, hptime_t endtime, flag whence);
MSTraceSeg *mstl_addsegtoseg (MSTraceSeg *seg1, MSTraceSeg *seg2);


/***************************************************************************
 * mstl_init:
 *
 * Initialize and return a MSTraceList struct, allocating memory if
 * needed.  If the supplied MSTraceList is not NULL any associated
 * memory it will be freed including data at prvtptr pointers.
 *
 * Returns a pointer to a MSTraceList struct on success or NULL on error.
 ***************************************************************************/
MSTraceList *
mstl_init ( MSTraceList *mstl )
{
  if ( mstl )
    {
      mstl_free (&mstl, 1);
    }
  
  mstl = (MSTraceList *) malloc (sizeof(MSTraceList));
  
  if ( mstl == NULL )
    {
      ms_log (2, "mstl_init(): Cannot allocate memory\n");
      return NULL;
    }
  
  memset (mstl, 0, sizeof (MSTraceList));
  
  return mstl;
} /* End of mstl_init() */


/***************************************************************************
 * mstl_free:
 *
 * Free all memory associated with a MSTraceList struct and set the
 * pointer to 0.
 *
 * If the freeprvtptr flag is true any private pointer data will also
 * be freed when present.
 ***************************************************************************/
void
mstl_free ( MSTraceList **ppmstl, flag freeprvtptr )
{
  MSTraceID *id = 0;
  MSTraceID *nextid = 0;
  MSTraceSeg *seg = 0;
  MSTraceSeg *nextseg = 0;  
  
  if ( ! ppmstl )
    return;
  
  if ( *ppmstl )
    {
      /* Free any associated traces */
      id = (*ppmstl)->traces;
      while ( id )
	{
	  nextid = id->next;
	  
	  /* Free any associated trace segments */
	  seg = id->first;
	  while ( seg )
	    {
	      nextseg = seg->next;
	      
	      /* Free private pointer data if present and requested*/
	      if ( freeprvtptr && seg->prvtptr )
		free (seg->prvtptr);
	      
	      /* Free data array if allocated */
	      if ( seg->datasamples )
		free (seg->datasamples);
	      
	      free (seg);
	      seg = nextseg;
	    }
	  
	  /* Free private pointer data if present and requested*/
	  if ( freeprvtptr && id->prvtptr )
	    free (id->prvtptr);
	  
	  free (id);
	  id = nextid;
	}
      
      free (*ppmstl);
      
      *ppmstl = NULL;
    }
  
  return;
} /* End of mstl_free() */


/***************************************************************************
 * mstl_addmsr:
 *
 * Add data coverage from an MSRecord to a MSTraceList by searching the
 * list for the appropriate MSTraceID and MSTraceSeg and either adding
 * data to it or creating a new MStraceID and/or MSTraceSeg if needed.
 *
 * If the dataquality flag is true the data quality bytes must also
 * match otherwise they are ignored.
 *
 * If the autoheal flag is true extra processing is invoked to conjoin
 * trace segments that fit together after the MSRecord coverage is
 * added.  For segments that are removed, any memory at the prvtptr
 * will be freed.
 *
 * An MSTraceList is always maintained with the MSTraceIDs in
 * descending alphanumeric order.  MSTraceIDs are always maintained
 * with MSTraceSegs in data time time order.
 *
 * Return a pointer to the MSTraceSeg updated or 0 on error.
 ***************************************************************************/
MSTraceSeg *
mstl_addmsr ( MSTraceList *mstl, MSRecord *msr, flag dataquality,
	      flag autoheal, double timetol, double sampratetol )
{
  MSTraceID *id = 0;
  MSTraceID *searchid = 0;
  MSTraceID *ltid = 0;
  
  MSTraceSeg *seg = 0;
  MSTraceSeg *searchseg = 0;
  MSTraceSeg *segbefore = 0;
  MSTraceSeg *segafter = 0;
  MSTraceSeg *followseg = 0;
  
  hptime_t endtime;
  hptime_t pregap;
  hptime_t postgap;
  hptime_t lastgap;
  hptime_t firstgap;
  hptime_t hpdelta;
  hptime_t hptimetol = 0;
  hptime_t nhptimetol = 0;
  
  char srcname[45];
  char *s1, *s2;
  flag whence;
  flag lastratecheck;
  flag firstratecheck;
  int mag;
  int cmp;
  int ltmag;
  int ltcmp;
  
  if ( ! mstl || ! msr )
    return 0;
  
  /* Calculate end time for MSRecord */
  if ( (endtime = msr_endtime (msr)) == HPTERROR )
    {
      ms_log (2, "mstl_addmsr(): Error calculating record end time\n");
      return 0;
    }
  
  /* Generate source name string */
  if ( ! msr_srcname (msr, srcname, dataquality) )
    {
      ms_log (2, "mstl_addmsr(): Error generating srcname for MSRecord\n");
      return 0;
    }
  
  /* Search for matching trace ID starting with last accessed ID and
     then looping through the trace ID list. */
  if ( mstl->last )
    {
      s1 = mstl->last->srcname;
      s2 = srcname;
      while ( *s1 == *s2++ )
	{
	  if ( *s1++ == '\0' )
	    break;
	}
      cmp = (*s1 - *--s2);
      
      if ( ! cmp )
	{
	  id = mstl->last;
	}
      else
	{
	  /* Loop through trace ID list searching for a match, simultaneously
	   * track the source name which is closest but less than the MSRecord
	   * to allow for later insertion with sort order. */
	  searchid = mstl->traces;
	  ltcmp = 0;
	  ltmag = 0;
	  while ( searchid )
	    {
	      /* Compare source names */
	      s1 = searchid->srcname;
	      s2 = srcname;
	      mag = 0;
	      while ( *s1 == *s2++ )
		{
		  mag++;
		  if ( *s1++ == '\0' )
		    break;
		}
	      cmp = (*s1 - *--s2);
	      
	      /* If source names did not match track closest "less than" value 
	       * and continue searching. */
	      if ( cmp != 0 )
		{
		  if ( cmp < 0 )
		    {
		      if ( (ltcmp == 0 || cmp >= ltcmp) && mag >= ltmag )
			{
			  ltcmp = cmp;
			  ltmag = mag;
			  ltid = searchid;
			}
		      else if ( mag > ltmag )
			{
			  ltcmp = cmp;
			  ltmag = mag;
			  ltid = searchid;
			}
		    }
		  
		  searchid = searchid->next;
		  continue;
		}
	      
	      /* If we made it this far we found a match */
	      id = searchid;
	      break;
	    }
	}
    } /* Done searching for match in trace ID list */
  
  /* If no matching ID was found create new MSTraceID and MSTraceSeg entries */
  if ( ! id )
    {
      if ( ! (id = (MSTraceID *) calloc (1, sizeof(MSTraceID))) )
	{
	  ms_log (2, "mstl_addmsr(): Error allocating memory\n");
	  return 0;
	}
      
      /* Populate MSTraceID */
      strcpy (id->network, msr->network);
      strcpy (id->station, msr->station);
      strcpy (id->location, msr->location);
      strcpy (id->channel, msr->channel);
      id->dataquality = msr->dataquality;
      strcpy (id->srcname, srcname);
      
      id->earliest = msr->starttime;
      id->latest = endtime;
      id->numsegments = 1;
      
      if ( ! (seg = mstl_msr2seg (msr, endtime)) )
	{
	  return 0;
	}
      id->first = id->last = seg;
      
      /* Add new MSTraceID to MSTraceList */
      if ( ! mstl->traces || ! ltid )
	{
	  id->next = mstl->traces;
	  mstl->traces = id;
	}
      else
	{
	  id->next = ltid->next;
	  ltid->next = id;
	}
      
      mstl->numtraces++;
    }
  /* Add data coverage to the matching MSTraceID */
  else
    {
      /* Calculate high-precision sample period */
      hpdelta = (hptime_t) (( msr->samprate ) ? (HPTMODULUS / msr->samprate) : 0.0);
      
      /* Calculate high-precision time tolerance */
      if ( timetol == -1.0 )
	hptimetol = (hptime_t) (0.5 * hpdelta);   /* Default time tolerance is 1/2 sample period */
      else if ( timetol >= 0.0 )
	hptimetol = (hptime_t) (timetol * HPTMODULUS);
      
      nhptimetol = ( hptimetol ) ? -hptimetol : 0;
      
      /* last/firstgap are negative when the record overlaps the trace
       * segment and positive when there is a time gap. */
      
      /* Gap relative to the last segment */
      lastgap = msr->starttime - id->last->endtime - hpdelta;
      
      /* Gap relative to the first segment */
      firstgap = id->first->starttime - endtime - hpdelta;
      
      /* Sample rate tolerance checks for first and last segments */
      if ( sampratetol == -1.0 )
	{
	  lastratecheck = MS_ISRATETOLERABLE (msr->samprate, id->last->samprate);
	  firstratecheck = MS_ISRATETOLERABLE (msr->samprate, id->first->samprate);
	}
      else
	{
	  lastratecheck = (ms_dabs (msr->samprate - id->last->samprate) > sampratetol) ? 0 : 1;
	  firstratecheck = (ms_dabs (msr->samprate - id->first->samprate) > sampratetol) ? 0 : 1;
	}
      
      /* Search first for the simple scenarios in order of likelihood:
       * - Record fits at end of last segment
       * - Record fits after all coverage
       * - Record fits before all coverage
       * - Record fits at beginning of first segment
       *
       * If none of those scenarios are true search the complete segment list.
       */
      
      /* Record coverage fits at end of last segment */
      if ( lastgap <= hptimetol && lastgap >= nhptimetol && lastratecheck )
	{
	  if ( ! mstl_addmsrtoseg (id->last, msr, endtime, 1) )
	    return 0;
	  
	  seg = id->last;
	  
	  if ( endtime > id->latest )
	    id->latest = endtime;
	}
      /* Record coverage is after all other coverage */
      else if ( (msr->starttime - hpdelta - hptimetol) > id->latest )
	{
	  if ( ! (seg = mstl_msr2seg (msr, endtime)) )
	    return 0;
	  
	  /* Add to end of list */
	  id->last->next = seg;
	  seg->prev = id->last;
	  id->last = seg;
	  id->numsegments++;
	  
	  if ( endtime > id->latest )
	    id->latest = endtime;
	}
      /* Record coverage is before all other coverage */
      else if ( (endtime + hpdelta + hptimetol) < id->earliest )
	{
	  if ( ! (seg = mstl_msr2seg (msr, endtime)) )
	    return 0;
	  
	  /* Add to beginning of list */
	  id->first->prev = seg;
	  seg->next = id->first;
	  id->first = seg;
	  id->numsegments++;
	  
	  if ( msr->starttime < id->earliest )
	    id->earliest = msr->starttime;
	}
      /* Record coverage fits at beginning of first segment */
      else if ( firstgap <= hptimetol && firstgap >= nhptimetol && firstratecheck )
	{
	  if ( ! mstl_addmsrtoseg (id->first, msr, endtime, 2) )
	    return 0;
	  
	  seg = id->first;
	  
	  if ( msr->starttime < id->earliest )
	    id->earliest = msr->starttime;
	}
      /* Search complete segment list for matches */
      else
	{
	  searchseg = id->first;
	  segbefore = 0;     /* Find segment that record fits before */
	  segafter = 0;      /* Find segment that record fits after */
	  followseg = 0;     /* Track segment that record follows in time order */
	  while ( searchseg )
	    {
	      if ( msr->starttime > searchseg->starttime )
		followseg = searchseg;
	      
	      whence = 0;
	      
	      postgap = msr->starttime - searchseg->endtime - hpdelta;
	      if ( ! segbefore && postgap <= hptimetol && postgap >= nhptimetol )
		whence = 1;
	      
	      pregap = searchseg->starttime - endtime - hpdelta;
	      if ( ! segafter && pregap <= hptimetol && pregap >= nhptimetol )
		whence = 2;
	      
	      if ( ! whence )
		{
		  searchseg = searchseg->next;
		  continue;
		}
	      
	      if ( sampratetol == -1.0 )
		{
		  if ( ! MS_ISRATETOLERABLE (msr->samprate, searchseg->samprate) )
		    {
		      searchseg = searchseg->next;
		      continue;
		    }
		}
	      else
		{
		  if ( ms_dabs (msr->samprate - searchseg->samprate) > sampratetol )
		    {
		      searchseg = searchseg->next;
		      continue;
		    }
		}
	      
	      if ( whence == 1 )
		segbefore = searchseg;
	      else
		segafter = searchseg;
	      
	      /* Done searching if not autohealing */
	      if ( ! autoheal )
		break;
	      
	      /* Done searching if both before and after segments are found */
	      if ( segbefore && segafter )
		break;
	      
	      searchseg = searchseg->next;
	    } /* Done looping through segments */
	  
	  /* Add MSRecord coverage to end of segment before */
	  if ( segbefore )
	    {
	      if ( ! mstl_addmsrtoseg (segbefore, msr, endtime, 1) )
		{
		  return 0;
		}
	      
	      /* Merge two segments that now fit if autohealing */
	      if ( autoheal && segafter && segbefore != segafter )
		{
		  /* Add segafter coverage to segbefore */
		  if ( ! mstl_addsegtoseg (segbefore, segafter) )
		    {
		      return 0;
		    }
		  
		  /* Shift last segment pointer if it's going to be removed */
		  if ( segafter == id->last )
		    id->last = id->last->prev;
		  
		  /* Remove segafter from list */
		  if ( segafter->prev )
		    segafter->prev->next = segafter->next;
		  if ( segafter->next )
		    segafter->next->prev = segafter->prev;
		  
		  /* Free data samples, private data and segment structure */
		  if (segafter->datasamples)
		    free (segafter->datasamples);
		  
		  if (segafter->prvtptr)
		    free (segafter->prvtptr);
		  
		  free (segafter);
		}
	      
	      seg = segbefore;
	    }
	  /* Add MSRecord coverage to beginning of segment after */
	  else if ( segafter )
	    {
	      if ( ! mstl_addmsrtoseg (segafter, msr, endtime, 2) )
		{
		  return 0;
		}
	      
	      seg = segafter;
	    }
	  /* Add MSRecord coverage to new segment */
	  else
	    {
	      /* Create new segment */
	      if ( ! (seg = mstl_msr2seg (msr, endtime)) )
		{
		  return 0;
		}
	      
	      /* Add new segment as first in list */
	      if ( ! followseg )
		{
		  seg->next = id->first;
		  if ( id->first )
		    id->first->prev = seg;
		  
		  id->first = seg;
		}
	      /* Add new segment after the followseg segment */
	      else
		{
		  seg->next = followseg->next;
		  seg->prev = followseg;
		  if ( followseg->next )
		    followseg->next->prev = seg;
		  followseg->next = seg;
		  
		  if ( followseg == id->last )
		    id->last = seg;
		}
	      
	      id->numsegments++;
	    }
	  
	  /* Track earliest and latest times */
	  if ( msr->starttime < id->earliest )
	    id->earliest = msr->starttime;
	  
	  if ( endtime > id->latest )
	    id->latest = endtime;
	} /* End of searching segment list */
    } /* End of adding coverage to matching ID */
  
  /* Sort modified segment into place, logic above should limit these to few shifts if any */
  while ( seg->next && ( seg->starttime > seg->next->starttime ||
			 (seg->starttime == seg->next->starttime && seg->endtime < seg->next->endtime) ) )
    {
      /* Move segment down list, swap seg and seg->next */
      segafter = seg->next;
      
      if ( seg->prev )
	seg->prev->next = segafter;
      
      if ( segafter->next )
	segafter->next->prev = seg;
      
      segafter->prev = seg->prev;
      seg->prev = segafter;
      seg->next = segafter->next;
      segafter->next = seg;
      
      /* Reset first and last segment pointers if replaced */
      if ( id->first == seg )
	id->first = segafter;
      
      if ( id->last == segafter )
	id->last = seg;
    }
  while ( seg->prev && ( seg->starttime < seg->prev->starttime ||
			 (seg->starttime == seg->prev->starttime && seg->endtime > seg->prev->endtime) ) )
    {
      /* Move segment up list, swap seg and seg->prev */
      segbefore = seg->prev;
      
      if ( seg->next )
	seg->next->prev = segbefore;
      
      if ( segbefore->prev )
	segbefore->prev->next = seg;
      
      segbefore->next = seg->next;
      seg->next = segbefore;
      seg->prev = segbefore->prev;
      segbefore->prev = seg;
      
      /* Reset first and last segment pointers if replaced */
      if ( id->first == segbefore )
	id->first = seg;
      
      if ( id->last == seg )
	id->last = segbefore;
    }
  
  /* Set MSTraceID as last accessed */
  mstl->last = id;
  
  return seg;
}  /* End of mstl_addmsr() */


/***************************************************************************
 * mstl_msr2seg:
 *
 * Create an MSTraceSeg structure from an MSRecord structure.
 *
 * Return a pointer to a MSTraceSeg otherwise 0 on error.
 ***************************************************************************/
MSTraceSeg *
mstl_msr2seg (MSRecord *msr, hptime_t endtime)
{
  MSTraceSeg *seg = 0;
  int samplesize;
  
  if ( ! (seg = (MSTraceSeg *) calloc (1, sizeof(MSTraceSeg))) )
    {
      ms_log (2, "mstl_addmsr(): Error allocating memory\n");
      return 0;
    }
  
  /* Populate MSTraceSeg */
  seg->starttime = msr->starttime;
  seg->endtime = endtime;
  seg->samprate = msr->samprate;
  seg->samplecnt = msr->samplecnt;
  seg->sampletype = msr->sampletype;
  seg->numsamples = msr->numsamples;
  
  /* Allocate space for and copy datasamples */
  if ( msr->datasamples && msr->numsamples )
    {
      samplesize = ms_samplesize (msr->sampletype);
      
      if ( ! (seg->datasamples = malloc (samplesize * msr->numsamples)) )
	{
	  ms_log (2, "mstl_msr2seg(): Error allocating memory\n");
	  return 0;
	}
      
      /* Copy data samples from MSRecord to MSTraceSeg */
      memcpy (seg->datasamples, msr->datasamples, samplesize * msr->numsamples);
    }
  
  return seg;
} /* End of mstl_msr2seg() */


/***************************************************************************
 * mstl_addmsrtoseg:
 *
 * Add data coverage from a MSRecord structure to a MSTraceSeg structure.
 *
 * Data coverage is added to the beginning or end of MSTraceSeg
 * according to the whence flag:
 * 1 : add coverage to the end
 * 2 : add coverage to the beginninig
 *
 * Return a pointer to a MSTraceSeg otherwise 0 on error.
 ***************************************************************************/
MSTraceSeg *
mstl_addmsrtoseg (MSTraceSeg *seg, MSRecord *msr, hptime_t endtime, flag whence)
{
  int samplesize = 0;
  void *newdatasamples;
  
  if ( ! seg || ! msr )
    return 0;

  /* Allocate more memory for data samples if included */
  if ( msr->datasamples && msr->numsamples > 0 )
    {
      if ( msr->sampletype != seg->sampletype )
	{
	  ms_log (2, "mstl_addmsrtoseg(): MSRecord sample type (%c) does not match segment sample type (%c)\n",
		  msr->sampletype, seg->sampletype);
	  return 0;
	}
      
      if ( ! (samplesize = ms_samplesize (msr->sampletype)) )
	{
	  ms_log (2, "mstl_addmsrtoseg(): Unknown sample size for sample type: %c\n", msr->sampletype);
	  return 0;
	}
      
      if ( ! (newdatasamples = realloc (seg->datasamples, (seg->numsamples + msr->numsamples) * samplesize)) )
	{
	  ms_log (2, "mstl_addmsrtoseg(): Error allocating memory\n");
	  return 0;
	}
      
      seg->datasamples = newdatasamples;
    }
  
  /* Add coverage to end of segment */
  if ( whence == 1 )
    {
      seg->endtime = endtime;
      seg->samplecnt += msr->samplecnt;
      
      if ( msr->datasamples && msr->numsamples > 0 )
	{
	  memcpy ((char *)seg->datasamples + (seg->numsamples * samplesize),
                  msr->datasamples,
                  msr->numsamples * samplesize);
	  
	  seg->numsamples += msr->numsamples;
	}
    }
  /* Add coverage to beginning of segment */
  else if ( whence == 2 )
    {
      seg->starttime = msr->starttime;
      seg->samplecnt += msr->samplecnt;
      
      if ( msr->datasamples && msr->numsamples > 0 )
	{
	  memmove ((char *)seg->datasamples + (msr->numsamples * samplesize),
		   seg->datasamples,
		   seg->numsamples * samplesize);
	  
	  memcpy (seg->datasamples,
                  msr->datasamples,
                  msr->numsamples * samplesize);
	  
	  seg->numsamples += msr->numsamples;
	}
    }
  else
    {
      ms_log (2, "mstl_addmsrtoseg(): unrecognized whence value: %d\n", whence);
      return 0;
    }
  
  return seg;
} /* End of mstl_addmsrtoseg() */


/***************************************************************************
 * mstl_addsegtoseg:
 *
 * Add data coverage from seg2 to seg1.
 *
 * Return a pointer to a seg1 otherwise 0 on error.
 ***************************************************************************/
MSTraceSeg *
mstl_addsegtoseg (MSTraceSeg *seg1, MSTraceSeg *seg2)
{
  int samplesize = 0;
  void *newdatasamples;
  
  if ( ! seg1 || ! seg2 )
    return 0;
  
  /* Allocate more memory for data samples if included */
  if ( seg2->datasamples && seg2->numsamples > 0 )
    {
      if ( seg2->sampletype != seg1->sampletype )
	{
	  ms_log (2, "mstl_addsegtoseg(): MSTraceSeg sample types do not match (%c and %c)\n",
		  seg1->sampletype, seg2->sampletype);
	  return 0;
	}
      
      if ( ! (samplesize = ms_samplesize (seg1->sampletype)) )
	{
	  ms_log (2, "mstl_addsegtoseg(): Unknown sample size for sample type: %c\n", seg1->sampletype);
	  return 0;
	}
      
      if ( ! (newdatasamples = realloc (seg1->datasamples, (seg1->numsamples + seg2->numsamples) * samplesize)) )
	{
	  ms_log (2, "mstl_addsegtoseg(): Error allocating memory\n");
	  return 0;
	}
      
      seg1->datasamples = newdatasamples;
    }
  
  /* Add seg2 coverage to end of seg1 */
  seg1->endtime = seg2->endtime;
  seg1->samplecnt += seg2->samplecnt;
  
  if ( seg2->datasamples && seg2->numsamples > 0 )
    {
      memcpy ((char *)seg1->datasamples + (seg1->numsamples * samplesize),
	      seg2->datasamples,
	      seg2->numsamples * samplesize);
      
      seg1->numsamples += seg2->numsamples;
    }
  
  return seg1;
} /* End of mstl_addsegtoseg() */


/***************************************************************************
 * mstl_printtracelist:
 *
 * Print trace list summary information for the specified MSTraceList.
 *
 * By default only print the srcname, starttime and endtime for each
 * trace.  If details is greater than 0 include the sample rate,
 * number of samples and a total trace count.  If gaps is greater than
 * 0 and the previous trace matches (srcname & samprate) include the
 * gap between the endtime of the last trace and the starttime of the
 * current trace.
 *
 * The timeformat flag can either be:
 * 0 : SEED time format (year, day-of-year, hour, min, sec)
 * 1 : ISO time format (year, month, day, hour, min, sec)
 * 2 : Epoch time, seconds since the epoch
 ***************************************************************************/
void
mstl_printtracelist ( MSTraceList *mstl, flag timeformat,
		      flag details, flag gaps )
{
  MSTraceID *id = 0;
  MSTraceSeg *seg = 0;
  char stime[30];
  char etime[30];
  char gapstr[20];
  flag nogap;
  double gap;
  double delta;
  int tracecnt = 0;
  int segcnt = 0;
  
  if ( ! mstl )
    {
      return;
    }
  
  /* Print out the appropriate header */
  if ( details > 0 && gaps > 0 )
    ms_log (0, "   Source                Start sample             End sample        Gap  Hz  Samples\n");
  else if ( details <= 0 && gaps > 0 )
    ms_log (0, "   Source                Start sample             End sample        Gap\n");
  else if ( details > 0 && gaps <= 0 )
    ms_log (0, "   Source                Start sample             End sample        Hz  Samples\n");
  else
    ms_log (0, "   Source                Start sample             End sample\n");
  
  /* Loop through trace list */
  id = mstl->traces;  
  while ( id )
    {
      /* Loop through segment list */
      seg = id->first;
      while ( seg )
	{
	  /* Create formatted time strings */
	  if ( timeformat == 2 )
	    {
	      snprintf (stime, sizeof(stime), "%.6f", (double) MS_HPTIME2EPOCH(seg->starttime) );
	      snprintf (etime, sizeof(etime), "%.6f", (double) MS_HPTIME2EPOCH(seg->endtime) );
	    }
	  else if ( timeformat == 1 )
	    {
	      if ( ms_hptime2isotimestr (seg->starttime, stime, 1) == NULL )
		ms_log (2, "Cannot convert trace start time for %s\n", id->srcname);
	      
	      if ( ms_hptime2isotimestr (seg->endtime, etime, 1) == NULL )
		ms_log (2, "Cannot convert trace end time for %s\n", id->srcname);
	    }
	  else
	    {
	      if ( ms_hptime2seedtimestr (seg->starttime, stime, 1) == NULL )
		ms_log (2, "Cannot convert trace start time for %s\n", id->srcname);
	      
	      if ( ms_hptime2seedtimestr (seg->endtime, etime, 1) == NULL )
		ms_log (2, "Cannot convert trace end time for %s\n", id->srcname);
	    }
	  
	  /* Print segment info at varying levels */
	  if ( gaps > 0 )
	    {
	      gap = 0.0;
	      nogap = 0;
	      
	      if ( seg->prev )
		gap = (double) (seg->starttime - seg->prev->endtime) / HPTMODULUS;
	      else
		nogap = 1;
	      
	      /* Check that any overlap is not larger than the trace coverage */
	      if ( gap < 0.0 )
		{
		  delta = ( seg->samprate ) ? (1.0 / seg->samprate) : 0.0;
		  
		  if ( (gap * -1.0) > (((double)(seg->endtime - seg->starttime)/HPTMODULUS) + delta) )
		    gap = -(((double)(seg->endtime - seg->starttime)/HPTMODULUS) + delta);
		}
	      
	      /* Fix up gap display */
	      if ( nogap )
		snprintf (gapstr, sizeof(gapstr), " == ");
	      else if ( gap >= 86400.0 || gap <= -86400.0 )
		snprintf (gapstr, sizeof(gapstr), "%-3.1fd", (gap / 86400));
	      else if ( gap >= 3600.0 || gap <= -3600.0 )
		snprintf (gapstr, sizeof(gapstr), "%-3.1fh", (gap / 3600));
	      else if ( gap == 0.0 )
		snprintf (gapstr, sizeof(gapstr), "-0  ");
	      else
		snprintf (gapstr, sizeof(gapstr), "%-4.4g", gap);
	      
	      if ( details <= 0 )
		ms_log (0, "%-17s %-24s %-24s %-4s\n",
			id->srcname, stime, etime, gapstr);
	      else
		ms_log (0, "%-17s %-24s %-24s %-s %-3.3g %-lld\n",
			id->srcname, stime, etime, gapstr, seg->samprate, (long long int)seg->samplecnt);
	    }
	  else if ( details > 0 && gaps <= 0 )
	    ms_log (0, "%-17s %-24s %-24s %-3.3g %-lld\n",
		    id->srcname, stime, etime, seg->samprate, (long long int)seg->samplecnt);
	  else
	    ms_log (0, "%-17s %-24s %-24s\n", id->srcname, stime, etime);
	  
	  segcnt++;
	  seg = seg->next;
	}
      
      tracecnt++;
      id = id->next;
    }
  
  if ( tracecnt != mstl->numtraces )
    ms_log (2, "mstl_printtracelist(): number of traces in trace list is inconsistent\n");
  
  if ( details > 0 )
    ms_log (0, "Total: %d trace(s) with %d segment(s)\n", tracecnt, segcnt);
  
  return;
}  /* End of mstl_printtracelist() */


/***************************************************************************
 * mstl_printsynclist:
 *
 * Print SYNC trace list summary information for the specified MSTraceList.
 *
 * The SYNC header line will be created using the supplied dccid, if
 * the pointer is NULL the string "DCC" will be used instead.
 *
 * If the subsecond flag is true the segment start and end times will
 * include subsecond precision, otherwise they will be truncated to
 * integer seconds.
 *
 ***************************************************************************/
void
mstl_printsynclist ( MSTraceList *mstl, char *dccid, flag subsecond )
{
  MSTraceID *id = 0;
  MSTraceSeg *seg = 0;
  char starttime[30];
  char endtime[30];
  char yearday[10];
  time_t now;
  struct tm *nt;
  
  if ( ! mstl )
    {
      return;
    }
  
  /* Generate current time stamp */
  now = time (NULL);
  nt = localtime ( &now ); nt->tm_year += 1900; nt->tm_yday += 1;
  snprintf ( yearday, sizeof(yearday), "%04d,%03d", nt->tm_year, nt->tm_yday);
  
  /* Print SYNC header line */
  ms_log (0, "%s|%s\n", (dccid)?dccid:"DCC", yearday);
  
  /* Loop through trace list */
  id = mstl->traces;  
  while ( id )
    {
      /* Loop through segment list */
      seg = id->first;
      while ( seg )
	{
	  ms_hptime2seedtimestr (seg->starttime, starttime, subsecond);
	  ms_hptime2seedtimestr (seg->endtime, endtime, subsecond);
	  
	  /* Print SYNC line */
	  ms_log (0, "%s|%s|%s|%s|%s|%s||%.10g|%lld|||||||%s\n",
		  id->network, id->station, id->location, id->channel,
		  starttime, endtime, seg->samprate, (long long int)seg->samplecnt,
		  yearday);
	  
	  seg = seg->next;
	}
      
      id = id->next;
    }
  
  return;
}  /* End of mstl_printsynclist() */


/***************************************************************************
 * mstl_printgaplist:
 *
 * Print gap/overlap list summary information for the specified
 * MSTraceList.  Overlaps are printed as negative gaps.
 *
 * If mingap and maxgap are not NULL their values will be enforced and
 * only gaps/overlaps matching their implied criteria will be printed.
 *
 * The timeformat flag can either be:
 * 0 : SEED time format (year, day-of-year, hour, min, sec)
 * 1 : ISO time format (year, month, day, hour, min, sec)
 * 2 : Epoch time, seconds since the epoch
 ***************************************************************************/
void
mstl_printgaplist (MSTraceList *mstl, flag timeformat,
		   double *mingap, double *maxgap)
{
  MSTraceID *id = 0;
  MSTraceSeg *seg = 0;

  char time1[30], time2[30];
  char gapstr[30];
  double gap;
  double delta;
  double nsamples;
  flag printflag;
  int gapcnt = 0;
  
  if ( ! mstl )
    return;
  
  if ( ! mstl->traces )
    return;
  
  ms_log (0, "   Source                Last Sample              Next Sample       Gap  Samples\n");
  
  id = mstl->traces;
  while ( id )
    {
      seg = id->first;
      while ( seg->next )
	{
	  /* Skip segments with 0 sample rate, usually from SOH records */
	  if ( seg->samprate == 0.0 )
	    {
	      seg = seg->next;
	      continue;
	    }
	  
	  gap = (double) (seg->next->starttime - seg->endtime) / HPTMODULUS;
	  
	  /* Check that any overlap is not larger than the trace coverage */
	  if ( gap < 0.0 )
	    {
	      delta = ( seg->next->samprate ) ? (1.0 / seg->next->samprate) : 0.0;
	      
	      if ( (gap * -1.0) > (((double)(seg->next->endtime - seg->next->starttime)/HPTMODULUS) + delta) )
		gap = -(((double)(seg->next->endtime - seg->next->starttime)/HPTMODULUS) + delta);
	    }
	  
	  printflag = 1;
	      
	  /* Check gap/overlap criteria */
	  if ( mingap )
	    if ( gap < *mingap )
	      printflag = 0;
	  
	  if ( maxgap )
	    if ( gap > *maxgap )
	      printflag = 0;
	  
	  if ( printflag )
	    {
	      nsamples = ms_dabs(gap) * seg->samprate;
		  
	      if ( gap > 0.0 )
		nsamples -= 1.0;
	      else
		nsamples += 1.0;
	      
	      /* Fix up gap display */
	      if ( gap >= 86400.0 || gap <= -86400.0 )
		snprintf (gapstr, sizeof(gapstr), "%-3.1fd", (gap / 86400));
	      else if ( gap >= 3600.0 || gap <= -3600.0 )
		snprintf (gapstr, sizeof(gapstr), "%-3.1fh", (gap / 3600));
	      else if ( gap == 0.0 )
		snprintf (gapstr, sizeof(gapstr), "-0  ");
	      else
		snprintf (gapstr, sizeof(gapstr), "%-4.4g", gap);
	      
	      /* Create formatted time strings */
	      if ( timeformat == 2 )
		{
		  snprintf (time1, sizeof(time1), "%.6f", (double) MS_HPTIME2EPOCH(seg->endtime) );
		  snprintf (time2, sizeof(time2), "%.6f", (double) MS_HPTIME2EPOCH(seg->next->starttime) );
		}
	      else if ( timeformat == 1 )
		{
		  if ( ms_hptime2isotimestr (seg->endtime, time1, 1) == NULL )
		    ms_log (2, "Cannot convert trace end time for %s\n", id->srcname);
		  
		  if ( ms_hptime2isotimestr (seg->next->starttime, time2, 1) == NULL )
		    ms_log (2, "Cannot convert next trace start time for %s\n", id->srcname);
		}
	      else
		{
		  if ( ms_hptime2seedtimestr (seg->endtime, time1, 1) == NULL )
		    ms_log (2, "Cannot convert trace end time for %s\n", id->srcname);
		  
		  if ( ms_hptime2seedtimestr (seg->next->starttime, time2, 1) == NULL )
		    ms_log (2, "Cannot convert next trace start time for %s\n", id->srcname);
		}
	      
	      ms_log (0, "%-17s %-24s %-24s %-4s %-.8g\n",
		      id->srcname, time1, time2, gapstr, nsamples);
	      
	      gapcnt++;
	    }
	      
	  seg = seg->next;
	}
      
      id = id->next;
    }
  
  ms_log (0, "Total: %d gap(s)\n", gapcnt);
  
  return;
}  /* End of mstl_printgaplist() */
