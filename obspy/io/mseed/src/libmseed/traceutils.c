/***************************************************************************
 * traceutils.c:
 *
 * Generic routines to handle Traces.
 *
 * Written by Chad Trabant, IRIS Data Management Center
 *
 * modified: 2015.108
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "libmseed.h"

static int mst_groupsort_cmp (MSTrace *mst1, MSTrace *mst2, flag quality);

/***************************************************************************
 * mst_init:
 *
 * Initialize and return a MSTrace struct, allocating memory if needed.
 * If the specified MSTrace includes data samples they will be freed.
 *
 * Returns a pointer to a MSTrace struct on success or NULL on error.
 ***************************************************************************/
MSTrace *
mst_init (MSTrace *mst)
{
  /* Free datasamples, prvtptr and stream state if present */
  if (mst)
  {
    if (mst->datasamples)
      free (mst->datasamples);

    if (mst->prvtptr)
      free (mst->prvtptr);

    if (mst->ststate)
      free (mst->ststate);
  }
  else
  {
    mst = (MSTrace *)malloc (sizeof (MSTrace));
  }

  if (mst == NULL)
  {
    ms_log (2, "mst_init(): Cannot allocate memory\n");
    return NULL;
  }

  memset (mst, 0, sizeof (MSTrace));

  return mst;
} /* End of mst_init() */

/***************************************************************************
 * mst_free:
 *
 * Free all memory associated with a MSTrace struct and set the pointer
 * to 0.
 ***************************************************************************/
void
mst_free (MSTrace **ppmst)
{
  if (ppmst && *ppmst)
  {
    /* Free datasamples if present */
    if ((*ppmst)->datasamples)
      free ((*ppmst)->datasamples);

    /* Free private memory if present */
    if ((*ppmst)->prvtptr)
      free ((*ppmst)->prvtptr);

    /* Free stream processing state if present */
    if ((*ppmst)->ststate)
      free ((*ppmst)->ststate);

    free (*ppmst);

    *ppmst = 0;
  }
} /* End of mst_free() */

/***************************************************************************
 * mst_initgroup:
 *
 * Initialize and return a MSTraceGroup struct, allocating memory if
 * needed.  If the supplied MSTraceGroup is not NULL any associated
 * memory it will be freed.
 *
 * Returns a pointer to a MSTraceGroup struct on success or NULL on error.
 ***************************************************************************/
MSTraceGroup *
mst_initgroup (MSTraceGroup *mstg)
{
  MSTrace *mst  = 0;
  MSTrace *next = 0;

  if (mstg)
  {
    mst = mstg->traces;

    while (mst)
    {
      next = mst->next;
      mst_free (&mst);
      mst = next;
    }
  }
  else
  {
    mstg = (MSTraceGroup *)malloc (sizeof (MSTraceGroup));
  }

  if (mstg == NULL)
  {
    ms_log (2, "mst_initgroup(): Cannot allocate memory\n");
    return NULL;
  }

  memset (mstg, 0, sizeof (MSTraceGroup));

  return mstg;
} /* End of mst_initgroup() */

/***************************************************************************
 * mst_freegroup:
 *
 * Free all memory associated with a MSTraceGroup struct and set the
 * pointer to 0.
 ***************************************************************************/
void
mst_freegroup (MSTraceGroup **ppmstg)
{
  MSTrace *mst  = 0;
  MSTrace *next = 0;

  if (*ppmstg)
  {
    mst = (*ppmstg)->traces;

    while (mst)
    {
      next = mst->next;
      mst_free (&mst);
      mst = next;
    }

    free (*ppmstg);

    *ppmstg = 0;
  }
} /* End of mst_freegroup() */

/***************************************************************************
 * mst_findmatch:
 *
 * Traverse the MSTrace chain starting at 'startmst' until a MSTrace
 * is found that matches the given name identifiers.  If the dataquality
 * byte is not 0 it must also match.
 *
 * Return a pointer a matching MSTrace otherwise 0 if no match found.
 ***************************************************************************/
MSTrace *
mst_findmatch (MSTrace *startmst, char dataquality,
               char *network, char *station, char *location, char *channel)
{
  int idx;

  if (!startmst)
    return 0;

  while (startmst)
  {
    if (dataquality && dataquality != startmst->dataquality)
    {
      startmst = startmst->next;
      continue;
    }

    /* Compare network */
    idx = 0;
    while (network[idx] == startmst->network[idx])
    {
      if (network[idx] == '\0')
        break;
      idx++;
    }
    if (network[idx] != '\0' || startmst->network[idx] != '\0')
    {
      startmst = startmst->next;
      continue;
    }
    /* Compare station */
    idx = 0;
    while (station[idx] == startmst->station[idx])
    {
      if (station[idx] == '\0')
        break;
      idx++;
    }
    if (station[idx] != '\0' || startmst->station[idx] != '\0')
    {
      startmst = startmst->next;
      continue;
    }
    /* Compare location */
    idx = 0;
    while (location[idx] == startmst->location[idx])
    {
      if (location[idx] == '\0')
        break;
      idx++;
    }
    if (location[idx] != '\0' || startmst->location[idx] != '\0')
    {
      startmst = startmst->next;
      continue;
    }
    /* Compare channel */
    idx = 0;
    while (channel[idx] == startmst->channel[idx])
    {
      if (channel[idx] == '\0')
        break;
      idx++;
    }
    if (channel[idx] != '\0' || startmst->channel[idx] != '\0')
    {
      startmst = startmst->next;
      continue;
    }

    /* A match was found if we made it this far */
    break;
  }

  return startmst;
} /* End of mst_findmatch() */

/***************************************************************************
 * mst_findadjacent:
 *
 * Find a MSTrace in a MSTraceGroup matching the given name
 * identifiers, samplerate and is adjacent with a time span.  If the
 * dataquality byte is not 0 it must also match.
 *
 * The time tolerance and sample rate tolerance are used to determine
 * if traces abut.  If timetol is -1.0 the default tolerance of 1/2
 * the sample period will be used.  If samprratetol is -1.0 the
 * default tolerance check of abs(1-sr1/sr2) < 0.0001 is used (defined
 * in libmseed.h).  If timetol or sampratetol is -2.0 the respective
 * tolerance check will not be performed.
 *
 * The 'whence' flag will be set, when a matching MSTrace is found, to
 * indicate where the indicated time span is adjacent to the MSTrace
 * using the following values:
 * 1: time span fits at the end of the MSTrace
 * 2: time span fits at the beginning of the MSTrace
 *
 * Return a pointer a matching MSTrace and set the 'whence' flag
 * otherwise 0 if no match found.
 ***************************************************************************/
MSTrace *
mst_findadjacent (MSTraceGroup *mstg, flag *whence, char dataquality,
                  char *network, char *station, char *location, char *channel,
                  double samprate, double sampratetol,
                  hptime_t starttime, hptime_t endtime, double timetol)
{
  MSTrace *mst = 0;
  hptime_t pregap;
  hptime_t postgap;
  hptime_t hpdelta;
  hptime_t hptimetol  = 0;
  hptime_t nhptimetol = 0;
  int idx;

  if (!mstg)
    return 0;

  *whence = 0;

  /* Calculate high-precision sample period */
  hpdelta = (hptime_t) ((samprate) ? (HPTMODULUS / samprate) : 0.0);

  /* Calculate high-precision time tolerance */
  if (timetol == -1.0)
    hptimetol = (hptime_t) (0.5 * hpdelta); /* Default time tolerance is 1/2 sample period */
  else if (timetol >= 0.0)
    hptimetol = (hptime_t) (timetol * HPTMODULUS);

  nhptimetol = (hptimetol) ? -hptimetol : 0;

  mst = mstg->traces;

  while (mst)
  {
    /* post/pregap are negative when the record overlaps the trace
       * segment and positive when there is a time gap. */
    postgap = starttime - mst->endtime - hpdelta;

    pregap = mst->starttime - endtime - hpdelta;

    /* If not checking the time tolerance decide if beginning or end is a better fit */
    if (timetol == -2.0)
    {
      if (ms_dabs ((double)postgap) < ms_dabs ((double)pregap))
        *whence = 1;
      else
        *whence = 2;
    }
    else
    {
      if (postgap <= hptimetol && postgap >= nhptimetol)
      {
        /* Span fits right at the end of the trace */
        *whence = 1;
      }
      else if (pregap <= hptimetol && pregap >= nhptimetol)
      {
        /* Span fits right at the beginning of the trace */
        *whence = 2;
      }
      else
      {
        /* Span does not fit with this Trace */
        mst = mst->next;
        continue;
      }
    }

    /* Perform samprate tolerance check if requested */
    if (sampratetol != -2.0)
    {
      /* Perform default samprate tolerance check if requested */
      if (sampratetol == -1.0)
      {
        if (!MS_ISRATETOLERABLE (samprate, mst->samprate))
        {
          mst = mst->next;
          continue;
        }
      }
      /* Otherwise check against the specified sample rate tolerance */
      else if (ms_dabs (samprate - mst->samprate) > sampratetol)
      {
        mst = mst->next;
        continue;
      }
    }

    /* Compare data qualities */
    if (dataquality && dataquality != mst->dataquality)
    {
      mst = mst->next;
      continue;
    }

    /* Compare network */
    idx = 0;
    while (network[idx] == mst->network[idx])
    {
      if (network[idx] == '\0')
        break;
      idx++;
    }
    if (network[idx] != '\0' || mst->network[idx] != '\0')
    {
      mst = mst->next;
      continue;
    }
    /* Compare station */
    idx = 0;
    while (station[idx] == mst->station[idx])
    {
      if (station[idx] == '\0')
        break;
      idx++;
    }
    if (station[idx] != '\0' || mst->station[idx] != '\0')
    {
      mst = mst->next;
      continue;
    }
    /* Compare location */
    idx = 0;
    while (location[idx] == mst->location[idx])
    {
      if (location[idx] == '\0')
        break;
      idx++;
    }
    if (location[idx] != '\0' || mst->location[idx] != '\0')
    {
      mst = mst->next;
      continue;
    }
    /* Compare channel */
    idx = 0;
    while (channel[idx] == mst->channel[idx])
    {
      if (channel[idx] == '\0')
        break;
      idx++;
    }
    if (channel[idx] != '\0' || mst->channel[idx] != '\0')
    {
      mst = mst->next;
      continue;
    }

    /* A match was found if we made it this far */
    break;
  }

  return mst;
} /* End of mst_findadjacent() */

/***************************************************************************
 * mst_addmsr:
 *
 * Add MSRecord time coverage to a MSTrace.  The start or end time will
 * be updated and samples will be copied if they exist.  No checking
 * is done to verify that the record matches the trace in any way.
 *
 * If whence is 1 the coverage will be added at the end of the trace,
 * whereas if whence is 2 the coverage will be added at the beginning
 * of the trace.
 *
 * Return 0 on success and -1 on error.
 ***************************************************************************/
int
mst_addmsr (MSTrace *mst, MSRecord *msr, flag whence)
{
  int samplesize = 0;

  if (!mst || !msr)
    return -1;

  /* Reallocate data sample buffer if samples are present */
  if (msr->datasamples && msr->numsamples >= 0)
  {
    /* Check that the entire record was decompressed */
    if (msr->samplecnt != msr->numsamples)
    {
      ms_log (2, "mst_addmsr(): Sample counts do not match, record not fully decompressed?\n");
      ms_log (2, "  The sample buffer will likely contain a discontinuity.\n");
    }

    if ((samplesize = ms_samplesize (msr->sampletype)) == 0)
    {
      ms_log (2, "mst_addmsr(): Unrecognized sample type: '%c'\n",
              msr->sampletype);
      return -1;
    }

    if (msr->sampletype != mst->sampletype)
    {
      ms_log (2, "mst_addmsr(): Mismatched sample type, '%c' and '%c'\n",
              msr->sampletype, mst->sampletype);
      return -1;
    }

    mst->datasamples = realloc (mst->datasamples,
                                (size_t) (mst->numsamples * samplesize + msr->numsamples * samplesize));

    if (mst->datasamples == NULL)
    {
      ms_log (2, "mst_addmsr(): Cannot allocate memory\n");
      return -1;
    }
  }

  /* Add samples at end of trace */
  if (whence == 1)
  {
    if (msr->datasamples && msr->numsamples >= 0)
    {
      memcpy ((char *)mst->datasamples + (mst->numsamples * samplesize),
              msr->datasamples,
              (size_t) (msr->numsamples * samplesize));

      mst->numsamples += msr->numsamples;
    }

    mst->endtime = msr_endtime (msr);

    if (mst->endtime == HPTERROR)
    {
      ms_log (2, "mst_addmsr(): Error calculating record end time\n");
      return -1;
    }
  }

  /* Add samples at the beginning of trace */
  else if (whence == 2)
  {
    if (msr->datasamples && msr->numsamples >= 0)
    {
      /* Move any samples to end of buffer */
      if (mst->numsamples > 0)
      {
        memmove ((char *)mst->datasamples + (msr->numsamples * samplesize),
                 mst->datasamples,
                 (size_t) (mst->numsamples * samplesize));
      }

      memcpy (mst->datasamples,
              msr->datasamples,
              (size_t) (msr->numsamples * samplesize));

      mst->numsamples += msr->numsamples;
    }

    mst->starttime = msr->starttime;
  }

  /* If two different data qualities reset the MSTrace.dataquality to 0 */
  if (mst->dataquality && msr->dataquality && mst->dataquality != msr->dataquality)
    mst->dataquality = 0;

  /* Update MSTrace sample count */
  mst->samplecnt += msr->samplecnt;

  return 0;
} /* End of mst_addmsr() */

/***************************************************************************
 * mst_addspan:
 *
 * Add a time span to a MSTrace.  The start or end time will be updated
 * and samples will be copied if they are provided.  No checking is done to
 * verify that the record matches the trace in any way.
 *
 * If whence is 1 the coverage will be added at the end of the trace,
 * whereas if whence is 2 the coverage will be added at the beginning
 * of the trace.
 *
 * Return 0 on success and -1 on error.
 ***************************************************************************/
int
mst_addspan (MSTrace *mst, hptime_t starttime, hptime_t endtime,
             void *datasamples, int64_t numsamples, char sampletype,
             flag whence)
{
  int samplesize = 0;

  if (!mst)
    return -1;

  if (datasamples && numsamples > 0)
  {
    if ((samplesize = ms_samplesize (sampletype)) == 0)
    {
      ms_log (2, "mst_addspan(): Unrecognized sample type: '%c'\n",
              sampletype);
      return -1;
    }

    if (sampletype != mst->sampletype)
    {
      ms_log (2, "mst_addspan(): Mismatched sample type, '%c' and '%c'\n",
              sampletype, mst->sampletype);
      return -1;
    }

    mst->datasamples = realloc (mst->datasamples,
                                (size_t) (mst->numsamples * samplesize + numsamples * samplesize));

    if (mst->datasamples == NULL)
    {
      ms_log (2, "mst_addspan(): Cannot allocate memory\n");
      return -1;
    }
  }

  /* Add samples at end of trace */
  if (whence == 1)
  {
    if (datasamples && numsamples > 0)
    {
      memcpy ((char *)mst->datasamples + (mst->numsamples * samplesize),
              datasamples,
              (size_t) (numsamples * samplesize));

      mst->numsamples += numsamples;
    }

    mst->endtime = endtime;
  }

  /* Add samples at the beginning of trace */
  else if (whence == 2)
  {
    if (datasamples && numsamples > 0)
    {
      /* Move any samples to end of buffer */
      if (mst->numsamples > 0)
      {
        memmove ((char *)mst->datasamples + (numsamples * samplesize),
                 mst->datasamples,
                 (size_t) (mst->numsamples * samplesize));
      }

      memcpy (mst->datasamples,
              datasamples,
              (size_t) (numsamples * samplesize));

      mst->numsamples += numsamples;
    }

    mst->starttime = starttime;
  }

  /* Update MSTrace sample count */
  if (numsamples > 0)
    mst->samplecnt += numsamples;

  return 0;
} /* End of mst_addspan() */

/***************************************************************************
 * mst_addmsrtogroup:
 *
 * Add data samples from a MSRecord to a MSTrace in a MSTraceGroup by
 * searching the group for the approriate MSTrace and either adding data
 * to it or creating a new MSTrace if no match found.
 *
 * Matching traces are found using the mst_findadjacent() routine.  If
 * the dataquality flag is true the data quality bytes must also match
 * otherwise they are ignored.
 *
 * Return a pointer to the MSTrace updated or 0 on error.
 ***************************************************************************/
MSTrace *
mst_addmsrtogroup (MSTraceGroup *mstg, MSRecord *msr, flag dataquality,
                   double timetol, double sampratetol)
{
  MSTrace *mst = 0;
  hptime_t endtime;
  flag whence;
  char dq;

  if (!mstg || !msr)
    return 0;

  dq = (dataquality) ? msr->dataquality : 0;

  endtime = msr_endtime (msr);

  if (endtime == HPTERROR)
  {
    ms_log (2, "mst_addmsrtogroup(): Error calculating record end time\n");
    return 0;
  }

  /* Find matching, time adjacent MSTrace */
  mst = mst_findadjacent (mstg, &whence, dq,
                          msr->network, msr->station, msr->location, msr->channel,
                          msr->samprate, sampratetol,
                          msr->starttime, endtime, timetol);

  /* If a match was found update it otherwise create a new MSTrace and
     add to end of MSTrace chain */
  if (mst)
  {
    /* Records with no time coverage do not contribute to a trace */
    if (msr->samplecnt <= 0 || msr->samprate <= 0.0)
      return mst;

    if (mst_addmsr (mst, msr, whence))
    {
      return 0;
    }
  }
  else
  {
    mst = mst_init (NULL);

    mst->dataquality = dq;

    strncpy (mst->network, msr->network, sizeof (mst->network));
    strncpy (mst->station, msr->station, sizeof (mst->station));
    strncpy (mst->location, msr->location, sizeof (mst->location));
    strncpy (mst->channel, msr->channel, sizeof (mst->channel));

    mst->starttime  = msr->starttime;
    mst->samprate   = msr->samprate;
    mst->sampletype = msr->sampletype;

    if (mst_addmsr (mst, msr, 1))
    {
      mst_free (&mst);
      return 0;
    }

    /* Link new MSTrace into the end of the chain */
    if (!mstg->traces)
    {
      mstg->traces = mst;
    }
    else
    {
      MSTrace *lasttrace = mstg->traces;

      while (lasttrace->next)
        lasttrace = lasttrace->next;

      lasttrace->next = mst;
    }

    mstg->numtraces++;
  }

  return mst;
} /* End of mst_addmsrtogroup() */

/***************************************************************************
 * mst_addtracetogroup:
 *
 * Add a MSTrace to a MSTraceGroup at the end of the MSTrace chain.
 *
 * Return a pointer to the MSTrace added or 0 on error.
 ***************************************************************************/
MSTrace *
mst_addtracetogroup (MSTraceGroup *mstg, MSTrace *mst)
{
  MSTrace *lasttrace;

  if (!mstg || !mst)
    return 0;

  if (!mstg->traces)
  {
    mstg->traces = mst;
  }
  else
  {
    lasttrace = mstg->traces;

    while (lasttrace->next)
      lasttrace = lasttrace->next;

    lasttrace->next = mst;
  }

  mst->next = 0;

  mstg->numtraces++;

  return mst;
} /* End of mst_addtracetogroup() */

/***************************************************************************
 * mst_groupheal:
 *
 * Check if traces in MSTraceGroup can be healed, if contiguous segments
 * belong together they will be merged.  This routine is only useful
 * if the trace group was assembled from segments out of time order
 * (e.g. a file of Mini-SEED records not in time order) but forming
 * contiguous time coverage.  The MSTraceGroup will be sorted using
 * mst_groupsort() before healing.
 *
 * The time tolerance and sample rate tolerance are used to determine
 * if the traces are indeed the same.  If timetol is -1.0 the default
 * tolerance of 1/2 the sample period will be used.  If samprratetol
 * is -1.0 the default tolerance check of abs(1-sr1/sr2) < 0.0001 is
 * used (defined in libmseed.h).
 *
 * Return number of trace mergings on success otherwise -1 on error.
 ***************************************************************************/
int
mst_groupheal (MSTraceGroup *mstg, double timetol, double sampratetol)
{
  int mergings         = 0;
  MSTrace *curtrace    = 0;
  MSTrace *nexttrace   = 0;
  MSTrace *searchtrace = 0;
  MSTrace *prevtrace   = 0;
  int8_t merged        = 0;
  double postgap, pregap, delta;

  if (!mstg)
    return -1;

  /* Sort MSTraceGroup before any healing */
  if (mst_groupsort (mstg, 1))
    return -1;

  curtrace = mstg->traces;

  while (curtrace)
  {
    nexttrace = mstg->traces;
    prevtrace = mstg->traces;

    while (nexttrace)
    {
      searchtrace = nexttrace;
      nexttrace   = searchtrace->next;

      /* Do not process the same MSTrace we are trying to match */
      if (searchtrace == curtrace)
      {
        prevtrace = searchtrace;
        continue;
      }

      /* Check if this trace matches the curtrace */
      if (strcmp (searchtrace->network, curtrace->network) ||
          strcmp (searchtrace->station, curtrace->station) ||
          strcmp (searchtrace->location, curtrace->location) ||
          strcmp (searchtrace->channel, curtrace->channel))
      {
        prevtrace = searchtrace;
        continue;
      }

      /* Perform default samprate tolerance check if requested */
      if (sampratetol == -1.0)
      {
        if (!MS_ISRATETOLERABLE (searchtrace->samprate, curtrace->samprate))
        {
          prevtrace = searchtrace;
          continue;
        }
      }
      /* Otherwise check against the specified sample rates tolerance */
      else if (ms_dabs (searchtrace->samprate - curtrace->samprate) > sampratetol)
      {
        prevtrace = searchtrace;
        continue;
      }

      merged = 0;

      /* post/pregap are negative when searchtrace overlaps curtrace
         segment and positive when there is a time gap. */
      delta = (curtrace->samprate) ? (1.0 / curtrace->samprate) : 0.0;

      postgap = ((double)(searchtrace->starttime - curtrace->endtime) / HPTMODULUS) - delta;

      pregap = ((double)(curtrace->starttime - searchtrace->endtime) / HPTMODULUS) - delta;

      /* Calculate default time tolerance (1/2 sample period) if needed */
      if (timetol == -1.0)
        timetol = 0.5 * delta;

      /* Fits right at the end of curtrace */
      if (ms_dabs (postgap) <= timetol)
      {
        /* Merge searchtrace with curtrace */
        mst_addspan (curtrace, searchtrace->starttime, searchtrace->endtime,
                     searchtrace->datasamples, searchtrace->numsamples,
                     searchtrace->sampletype, 1);

        /* If no data is present, make sure sample count is updated */
        if (searchtrace->numsamples <= 0)
          curtrace->samplecnt += searchtrace->samplecnt;

        /* If qualities do not match reset the indicator */
        if (curtrace->dataquality != searchtrace->dataquality)
          curtrace->dataquality = 0;

        merged = 1;
      }

      /* Fits right at the beginning of curtrace */
      else if (ms_dabs (pregap) <= timetol)
      {
        /* Merge searchtrace with curtrace */
        mst_addspan (curtrace, searchtrace->starttime, searchtrace->endtime,
                     searchtrace->datasamples, searchtrace->numsamples,
                     searchtrace->sampletype, 2);

        /* If no data is present, make sure sample count is updated */
        if (searchtrace->numsamples <= 0)
          curtrace->samplecnt += searchtrace->samplecnt;

        /* If qualities do not match reset the indicator */
        if (curtrace->dataquality != searchtrace->dataquality)
          curtrace->dataquality = 0;

        merged = 1;
      }

      /* If searchtrace was merged with curtrace remove it from the chain */
      if (merged)
      {
        /* Re-link trace chain and free searchtrace */
        if (searchtrace == mstg->traces)
          mstg->traces = nexttrace;
        else
          prevtrace->next = nexttrace;

        mst_free (&searchtrace);

        mstg->numtraces--;
        mergings++;
      }
      else
      {
        prevtrace = searchtrace;
      }
    }

    curtrace = curtrace->next;
  }

  return mergings;
} /* End of mst_groupheal() */

/***************************************************************************
 * mst_groupsort:
 *
 * Sort a MSTraceGroup using a mergesort algorithm.  MSTrace entries
 * are compared using the mst_groupsort_cmp() function.
 *
 * The mergesort implementation was inspired by the listsort function
 * published and copyright 2001 by Simon Tatham.
 *
 * Return 0 on success and -1 on error.
 ***************************************************************************/
int
mst_groupsort (MSTraceGroup *mstg, flag quality)
{
  MSTrace *p, *q, *e, *top, *tail;
  int nmerges;
  int insize, psize, qsize, i;

  if (!mstg)
    return -1;

  if (!mstg->traces)
    return 0;

  top    = mstg->traces;
  insize = 1;

  for (;;)
  {
    p    = top;
    top  = NULL;
    tail = NULL;

    nmerges = 0; /* count number of merges we do in this pass */

    while (p)
    {
      nmerges++; /* there exists a merge to be done */

      /* step `insize' places along from p */
      q     = p;
      psize = 0;
      for (i = 0; i < insize; i++)
      {
        psize++;
        q = q->next;
        if (!q)
          break;
      }

      /* if q hasn't fallen off end, we have two lists to merge */
      qsize = insize;

      /* now we have two lists; merge them */
      while (psize > 0 || (qsize > 0 && q))
      {
        /* decide whether next element of merge comes from p or q */
        if (psize == 0)
        { /* p is empty; e must come from q. */
          e = q;
          q = q->next;
          qsize--;
        }
        else if (qsize == 0 || !q)
        { /* q is empty; e must come from p. */
          e = p;
          p = p->next;
          psize--;
        }
        else if (mst_groupsort_cmp (p, q, quality) <= 0)
        { /* First element of p is lower (or same), e must come from p. */
          e = p;
          p = p->next;
          psize--;
        }
        else
        { /* First element of q is lower; e must come from q. */
          e = q;
          q = q->next;
          qsize--;
        }

        /* add the next element to the merged list */
        if (tail)
          tail->next = e;
        else
          top = e;

        tail = e;
      }

      /* now p has stepped `insize' places along, and q has too */
      p = q;
    }

    tail->next = NULL;

    /* If we have done only one merge, we're finished. */
    if (nmerges <= 1) /* allow for nmerges==0, the empty list case */
    {
      mstg->traces = top;

      return 0;
    }

    /* Otherwise repeat, merging lists twice the size */
    insize *= 2;
  }
} /* End of mst_groupsort() */

/***************************************************************************
 * mst_groupsort_cmp:
 *
 * Compare two MSTrace entities for the purposes of sorting a
 * MSTraceGroup.  Criteria for MSTrace comparison are (in order of
 * testing): source name, start time, descending endtime (longest
 * trace first) and sample rate.
 *
 * Return 1 if mst1 is "greater" than mst2, otherwise return 0.
 ***************************************************************************/
static int
mst_groupsort_cmp (MSTrace *mst1, MSTrace *mst2, flag quality)
{
  char src1[50], src2[50];
  int strcmpval;

  if (!mst1 || !mst2)
    return -1;

  mst_srcname (mst1, src1, quality);
  mst_srcname (mst2, src2, quality);

  strcmpval = strcmp (src1, src2);

  /* If the source names do not match make sure the "greater" string is 2nd,
   * otherwise, if source names do match, make sure the later start time is 2nd
   * otherwise, if start times match, make sure the earlier end time is 2nd
   * otherwise, if end times match, make sure the highest sample rate is 2nd
   */
  if (strcmpval > 0)
  {
    return 1;
  }
  else if (strcmpval == 0)
  {
    if (mst1->starttime > mst2->starttime)
    {
      return 1;
    }
    else if (mst1->starttime == mst2->starttime)
    {
      if (mst1->endtime < mst2->endtime)
      {
        return 1;
      }
      else if (mst1->endtime == mst2->endtime)
      {
        if (!MS_ISRATETOLERABLE (mst1->samprate, mst2->samprate) &&
            mst1->samprate > mst2->samprate)
        {
          return 1;
        }
      }
    }
  }

  return 0;
} /* End of mst_groupsort_cmp() */

/***************************************************************************
 * mst_convertsamples:
 *
 * Convert the data samples associated with an MSTrace to another data
 * type.  ASCII data samples cannot be converted, if supplied or
 * requested an error will be returned.
 *
 * When converting float & double sample types to integer type a
 * simple rounding is applied by adding 0.5 to the sample value before
 * converting (truncating) to integer.
 *
 * If the truncate flag is true data samples will be truncated to
 * integers even if loss of sample precision is detected.  If the
 * truncate flag is false (0) and loss of precision is detected an
 * error is returned.
 *
 * Returns 0 on success, and -1 on failure.
 ***************************************************************************/
int
mst_convertsamples (MSTrace *mst, char type, flag truncate)
{
  int32_t *idata;
  float *fdata;
  double *ddata;
  int64_t idx;

  if (!mst)
    return -1;

  /* No conversion necessary, report success */
  if (mst->sampletype == type)
    return 0;

  if (mst->sampletype == 'a' || type == 'a')
  {
    ms_log (2, "mst_convertsamples: cannot convert ASCII samples to/from numeric type\n");
    return -1;
  }

  idata = (int32_t *)mst->datasamples;
  fdata = (float *)mst->datasamples;
  ddata = (double *)mst->datasamples;

  /* Convert to 32-bit integers */
  if (type == 'i')
  {
    if (mst->sampletype == 'f') /* Convert floats to integers with simple rounding */
    {
      for (idx = 0; idx < mst->numsamples; idx++)
      {
        /* Check for loss of sub-integer */
        if (!truncate && (fdata[idx] - (int32_t)fdata[idx]) > 0.000001)
        {
          ms_log (1, "mst_convertsamples: Warning, loss of precision when converting floats to integers, loss: %g\n",
                  (fdata[idx] - (int32_t)fdata[idx]));
          return -1;
        }

        idata[idx] = (int32_t) (fdata[idx] + 0.5);
      }
    }
    else if (mst->sampletype == 'd') /* Convert doubles to integers with simple rounding */
    {
      for (idx = 0; idx < mst->numsamples; idx++)
      {
        /* Check for loss of sub-integer */
        if (!truncate && (ddata[idx] - (int32_t)ddata[idx]) > 0.000001)
        {
          ms_log (1, "mst_convertsamples: Warning, loss of precision when converting doubles to integers, loss: %g\n",
                  (ddata[idx] - (int32_t)ddata[idx]));
          return -1;
        }

        idata[idx] = (int32_t) (ddata[idx] + 0.5);
      }

      /* Reallocate buffer for reduced size needed */
      if (!(mst->datasamples = realloc (mst->datasamples, (size_t) (mst->numsamples * sizeof (int32_t)))))
      {
        ms_log (2, "mst_convertsamples: cannot re-allocate buffer for sample conversion\n");
        return -1;
      }
    }

    mst->sampletype = 'i';
  } /* Done converting to 32-bit integers */

  /* Convert to 32-bit floats */
  else if (type == 'f')
  {
    if (mst->sampletype == 'i') /* Convert integers to floats */
    {
      for (idx     = 0; idx < mst->numsamples; idx++)
        fdata[idx] = (float)idata[idx];
    }
    else if (mst->sampletype == 'd') /* Convert doubles to floats */
    {
      for (idx     = 0; idx < mst->numsamples; idx++)
        fdata[idx] = (float)ddata[idx];

      /* Reallocate buffer for reduced size needed */
      if (!(mst->datasamples = realloc (mst->datasamples, (size_t) (mst->numsamples * sizeof (float)))))
      {
        ms_log (2, "mst_convertsamples: cannot re-allocate buffer after sample conversion\n");
        return -1;
      }
    }

    mst->sampletype = 'f';
  } /* Done converting to 32-bit floats */

  /* Convert to 64-bit doubles */
  else if (type == 'd')
  {
    if (!(ddata = (double *)malloc ((size_t) (mst->numsamples * sizeof (double)))))
    {
      ms_log (2, "mst_convertsamples: cannot allocate buffer for sample conversion to doubles\n");
      return -1;
    }

    if (mst->sampletype == 'i') /* Convert integers to doubles */
    {
      for (idx     = 0; idx < mst->numsamples; idx++)
        ddata[idx] = (double)idata[idx];

      free (idata);
    }
    else if (mst->sampletype == 'f') /* Convert floats to doubles */
    {
      for (idx     = 0; idx < mst->numsamples; idx++)
        ddata[idx] = (double)fdata[idx];

      free (fdata);
    }

    mst->datasamples = ddata;
    mst->sampletype  = 'd';
  } /* Done converting to 64-bit doubles */

  return 0;
} /* End of mst_convertsamples() */

/***************************************************************************
 * mst_srcname:
 *
 * Generate a source name string for a specified MSTrace in the
 * format: 'NET_STA_LOC_CHAN[_QUAL]'.  The quality is added to the
 * srcname if the quality flag argument is 1 and mst->dataquality is
 * not zero.  The passed srcname must have enough room for the
 * resulting string.
 *
 * Returns a pointer to the resulting string or NULL on error.
 ***************************************************************************/
char *
mst_srcname (MSTrace *mst, char *srcname, flag quality)
{
  char *src = srcname;
  char *cp  = srcname;

  if (!mst || !srcname)
    return NULL;

  /* Build the source name string */
  cp = mst->network;
  while (*cp)
  {
    *src++ = *cp++;
  }
  *src++ = '_';
  cp     = mst->station;
  while (*cp)
  {
    *src++ = *cp++;
  }
  *src++ = '_';
  cp     = mst->location;
  while (*cp)
  {
    *src++ = *cp++;
  }
  *src++ = '_';
  cp     = mst->channel;
  while (*cp)
  {
    *src++ = *cp++;
  }

  if (quality && mst->dataquality)
  {
    *src++ = '_';
    *src++ = mst->dataquality;
  }

  *src = '\0';

  return srcname;
} /* End of mst_srcname() */

/***************************************************************************
 * mst_printtracelist:
 *
 * Print trace list summary information for the specified MSTraceGroup.
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
mst_printtracelist (MSTraceGroup *mstg, flag timeformat,
                    flag details, flag gaps)
{
  MSTrace *mst = 0;
  char srcname[50];
  char prevsrcname[50];
  char stime[30];
  char etime[30];
  char gapstr[20];
  flag nogap;
  double gap;
  double delta;
  double prevsamprate;
  hptime_t prevendtime;
  int tracecnt = 0;

  if (!mstg)
    return;

  mst = mstg->traces;

  /* Print out the appropriate header */
  if (details > 0 && gaps > 0)
    ms_log (0, "   Source                Start sample             End sample        Gap  Hz  Samples\n");
  else if (details <= 0 && gaps > 0)
    ms_log (0, "   Source                Start sample             End sample        Gap\n");
  else if (details > 0 && gaps <= 0)
    ms_log (0, "   Source                Start sample             End sample        Hz  Samples\n");
  else
    ms_log (0, "   Source                Start sample             End sample\n");

  prevsrcname[0] = '\0';
  prevsamprate   = -1.0;
  prevendtime    = 0;

  while (mst)
  {
    mst_srcname (mst, srcname, 1);

    /* Create formatted time strings */
    if (timeformat == 2)
    {
      snprintf (stime, sizeof (stime), "%.6f", (double)MS_HPTIME2EPOCH (mst->starttime));
      snprintf (etime, sizeof (etime), "%.6f", (double)MS_HPTIME2EPOCH (mst->endtime));
    }
    else if (timeformat == 1)
    {
      if (ms_hptime2isotimestr (mst->starttime, stime, 1) == NULL)
        ms_log (2, "Cannot convert trace start time for %s\n", srcname);

      if (ms_hptime2isotimestr (mst->endtime, etime, 1) == NULL)
        ms_log (2, "Cannot convert trace end time for %s\n", srcname);
    }
    else
    {
      if (ms_hptime2seedtimestr (mst->starttime, stime, 1) == NULL)
        ms_log (2, "Cannot convert trace start time for %s\n", srcname);

      if (ms_hptime2seedtimestr (mst->endtime, etime, 1) == NULL)
        ms_log (2, "Cannot convert trace end time for %s\n", srcname);
    }

    /* Print trace info at varying levels */
    if (gaps > 0)
    {
      gap   = 0.0;
      nogap = 0;

      if (!strcmp (prevsrcname, srcname) && prevsamprate != -1.0 &&
          MS_ISRATETOLERABLE (prevsamprate, mst->samprate))
        gap = (double)(mst->starttime - prevendtime) / HPTMODULUS;
      else
        nogap = 1;

      /* Check that any overlap is not larger than the trace coverage */
      if (gap < 0.0)
      {
        delta = (mst->samprate) ? (1.0 / mst->samprate) : 0.0;

        if ((gap * -1.0) > (((double)(mst->endtime - mst->starttime) / HPTMODULUS) + delta))
          gap = -(((double)(mst->endtime - mst->starttime) / HPTMODULUS) + delta);
      }

      /* Fix up gap display */
      if (nogap)
        snprintf (gapstr, sizeof (gapstr), " == ");
      else if (gap >= 86400.0 || gap <= -86400.0)
        snprintf (gapstr, sizeof (gapstr), "%-3.1fd", (gap / 86400));
      else if (gap >= 3600.0 || gap <= -3600.0)
        snprintf (gapstr, sizeof (gapstr), "%-3.1fh", (gap / 3600));
      else if (gap == 0.0)
        snprintf (gapstr, sizeof (gapstr), "-0  ");
      else
        snprintf (gapstr, sizeof (gapstr), "%-4.4g", gap);

      if (details <= 0)
        ms_log (0, "%-17s %-24s %-24s %-4s\n",
                srcname, stime, etime, gapstr);
      else
        ms_log (0, "%-17s %-24s %-24s %-s %-3.3g %-" PRId64 "\n",
                srcname, stime, etime, gapstr, mst->samprate, mst->samplecnt);
    }
    else if (details > 0 && gaps <= 0)
      ms_log (0, "%-17s %-24s %-24s %-3.3g %-" PRId64 "\n",
              srcname, stime, etime, mst->samprate, mst->samplecnt);
    else
      ms_log (0, "%-17s %-24s %-24s\n", srcname, stime, etime);

    if (gaps > 0)
    {
      strcpy (prevsrcname, srcname);
      prevsamprate = mst->samprate;
      prevendtime  = mst->endtime;
    }

    tracecnt++;
    mst = mst->next;
  }

  if (tracecnt != mstg->numtraces)
    ms_log (2, "mst_printtracelist(): number of traces in trace group is inconsistent\n");

  if (details > 0)
    ms_log (0, "Total: %d trace segment(s)\n", tracecnt);

} /* End of mst_printtracelist() */

/***************************************************************************
 * mst_printsynclist:
 *
 * Print SYNC trace list summary information for the specified MSTraceGroup.
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
mst_printsynclist (MSTraceGroup *mstg, char *dccid, flag subsecond)
{
  MSTrace *mst = 0;
  char stime[30];
  char etime[30];
  char yearday[10];
  time_t now;
  struct tm *nt;

  if (!mstg)
  {
    return;
  }

  /* Generate current time stamp */
  now = time (NULL);
  nt  = localtime (&now);
  nt->tm_year += 1900;
  nt->tm_yday += 1;
  snprintf (yearday, sizeof (yearday), "%04d,%03d", nt->tm_year, nt->tm_yday);

  /* Print SYNC header line */
  ms_log (0, "%s|%s\n", (dccid) ? dccid : "DCC", yearday);

  /* Loope through trace list */
  mst = mstg->traces;
  while (mst)
  {
    ms_hptime2seedtimestr (mst->starttime, stime, subsecond);
    ms_hptime2seedtimestr (mst->endtime, etime, subsecond);

    /* Print SYNC line */
    ms_log (0, "%s|%s|%s|%s|%s|%s||%.10g|%" PRId64 "|||||||%s\n",
            mst->network, mst->station, mst->location, mst->channel,
            stime, etime, mst->samprate, mst->samplecnt,
            yearday);

    mst = mst->next;
  }
} /* End of mst_printsynclist() */

/***************************************************************************
 * mst_printgaplist:
 *
 * Print gap/overlap list summary information for the specified
 * MSTraceGroup.  Overlaps are printed as negative gaps.  The trace
 * summary information in the MSTraceGroup is logically inverted so gaps
 * for like channels are identified.
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
mst_printgaplist (MSTraceGroup *mstg, flag timeformat,
                  double *mingap, double *maxgap)
{
  MSTrace *mst;
  char src1[50], src2[50];
  char time1[30], time2[30];
  char gapstr[30];
  double gap;
  double delta;
  double nsamples;
  flag printflag;
  int gapcnt = 0;

  if (!mstg)
    return;

  if (!mstg->traces)
    return;

  mst = mstg->traces;

  ms_log (0, "   Source                Last Sample              Next Sample       Gap  Samples\n");

  while (mst->next)
  {
    mst_srcname (mst, src1, 1);
    mst_srcname (mst->next, src2, 1);

    if (!strcmp (src1, src2))
    {
      /* Skip MSTraces with 0 sample rate, usually from SOH records */
      if (mst->samprate == 0.0)
      {
        mst = mst->next;
        continue;
      }

      /* Check that sample rates match using default tolerance */
      if (!MS_ISRATETOLERABLE (mst->samprate, mst->next->samprate))
      {
        ms_log (2, "%s Sample rate changed! %.10g -> %.10g\n",
                src1, mst->samprate, mst->next->samprate);
      }

      gap = (double)(mst->next->starttime - mst->endtime) / HPTMODULUS;

      /* Check that any overlap is not larger than the trace coverage */
      if (gap < 0.0)
      {
        delta = (mst->next->samprate) ? (1.0 / mst->next->samprate) : 0.0;

        if ((gap * -1.0) > (((double)(mst->next->endtime - mst->next->starttime) / HPTMODULUS) + delta))
          gap = -(((double)(mst->next->endtime - mst->next->starttime) / HPTMODULUS) + delta);
      }

      printflag = 1;

      /* Check gap/overlap criteria */
      if (mingap)
        if (gap < *mingap)
          printflag = 0;

      if (maxgap)
        if (gap > *maxgap)
          printflag = 0;

      if (printflag)
      {
        nsamples = ms_dabs (gap) * mst->samprate;

        if (gap > 0.0)
          nsamples -= 1.0;
        else
          nsamples += 1.0;

        /* Fix up gap display */
        if (gap >= 86400.0 || gap <= -86400.0)
          snprintf (gapstr, sizeof (gapstr), "%-3.1fd", (gap / 86400));
        else if (gap >= 3600.0 || gap <= -3600.0)
          snprintf (gapstr, sizeof (gapstr), "%-3.1fh", (gap / 3600));
        else if (gap == 0.0)
          snprintf (gapstr, sizeof (gapstr), "-0  ");
        else
          snprintf (gapstr, sizeof (gapstr), "%-4.4g", gap);

        /* Create formatted time strings */
        if (timeformat == 2)
        {
          snprintf (time1, sizeof (time1), "%.6f", (double)MS_HPTIME2EPOCH (mst->endtime));
          snprintf (time2, sizeof (time2), "%.6f", (double)MS_HPTIME2EPOCH (mst->next->starttime));
        }
        else if (timeformat == 1)
        {
          if (ms_hptime2isotimestr (mst->endtime, time1, 1) == NULL)
            ms_log (2, "Cannot convert trace end time for %s\n", src1);

          if (ms_hptime2isotimestr (mst->next->starttime, time2, 1) == NULL)
            ms_log (2, "Cannot convert next trace start time for %s\n", src1);
        }
        else
        {
          if (ms_hptime2seedtimestr (mst->endtime, time1, 1) == NULL)
            ms_log (2, "Cannot convert trace end time for %s\n", src1);

          if (ms_hptime2seedtimestr (mst->next->starttime, time2, 1) == NULL)
            ms_log (2, "Cannot convert next trace start time for %s\n", src1);
        }

        ms_log (0, "%-17s %-24s %-24s %-4s %-.8g\n",
                src1, time1, time2, gapstr, nsamples);

        gapcnt++;
      }
    }

    mst = mst->next;
  }

  ms_log (0, "Total: %d gap(s)\n", gapcnt);

} /* End of mst_printgaplist() */

/***************************************************************************
 * mst_pack:
 *
 * Pack MSTrace data into Mini-SEED records using the specified record
 * length, encoding format and byte order.  The datasamples array and
 * numsamples field will be adjusted (reduced) based on how many
 * samples were packed.
 *
 * As each record is filled and finished they are passed to
 * record_handler which expects 1) a char * to the record, 2) the
 * length of the record and 3) a pointer supplied by the original
 * caller containing optional private data (handlerdata).  It is the
 * responsibility of record_handler to process the record, the memory
 * will be re-used or freed when record_handler returns.
 *
 * If the flush flag is > 0 all of the data will be packed into data
 * records even though the last one will probably not be filled.
 *
 * If the mstemplate argument is not NULL it will be used as the
 * template for the packed Mini-SEED records.  Otherwise a new
 * MSRecord will be initialized and populated from values in the
 * MSTrace.  The reclen, encoding and byteorder arguments take
 * precedence over those in the template.  The start time, sample
 * rate, datasamples, numsamples and sampletype values from the
 * template will be preserved.
 *
 * Returns the number of records created on success and -1 on error.
 ***************************************************************************/
int
mst_pack (MSTrace *mst, void (*record_handler) (char *, int, void *),
          void *handlerdata, int reclen, flag encoding, flag byteorder,
          int64_t *packedsamples, flag flush, flag verbose,
          MSRecord *mstemplate)
{
  MSRecord *msr;
  char srcname[50];
  int trpackedrecords     = 0;
  int64_t trpackedsamples = 0;
  int samplesize;
  int64_t bufsize;

  hptime_t preservestarttime   = 0;
  double preservesamprate      = 0.0;
  void *preservedatasamples    = 0;
  int64_t preservenumsamples   = 0;
  char preservesampletype      = 0;
  StreamState *preserveststate = 0;

  if (packedsamples)
    *packedsamples = 0;

  /* Allocate stream processing state space if needed */
  if (!mst->ststate)
  {
    mst->ststate = (StreamState *)malloc (sizeof (StreamState));
    if (!mst->ststate)
    {
      ms_log (2, "mst_pack(): Could not allocate memory for StreamState\n");
      return -1;
    }
    memset (mst->ststate, 0, sizeof (StreamState));
  }

  if (mstemplate)
  {
    msr = mstemplate;

    preservestarttime   = msr->starttime;
    preservesamprate    = msr->samprate;
    preservedatasamples = msr->datasamples;
    preservenumsamples  = msr->numsamples;
    preservesampletype  = msr->sampletype;
    preserveststate     = msr->ststate;
  }
  else
  {
    msr = msr_init (NULL);

    if (msr == NULL)
    {
      ms_log (2, "mst_pack(): Error initializing msr\n");
      return -1;
    }

    msr->dataquality = 'D';
    strcpy (msr->network, mst->network);
    strcpy (msr->station, mst->station);
    strcpy (msr->location, mst->location);
    strcpy (msr->channel, mst->channel);
  }

  /* Setup MSRecord template for packing */
  msr->reclen    = reclen;
  msr->encoding  = encoding;
  msr->byteorder = byteorder;

  msr->starttime   = mst->starttime;
  msr->samprate    = mst->samprate;
  msr->datasamples = mst->datasamples;
  msr->numsamples  = mst->numsamples;
  msr->sampletype  = mst->sampletype;
  msr->ststate     = mst->ststate;

  /* Sample count sanity check */
  if (mst->samplecnt != mst->numsamples)
  {
    ms_log (2, "mst_pack(): Sample counts do not match, abort\n");
    return -1;
  }

  /* Pack data */
  trpackedrecords = msr_pack (msr, record_handler, handlerdata, &trpackedsamples, flush, verbose);

  if (verbose > 1)
  {
    ms_log (1, "Packed %d records for %s trace\n", trpackedrecords, mst_srcname (mst, srcname, 1));
  }

  /* Adjust MSTrace start time, data array and sample count */
  if (trpackedsamples > 0)
  {
    /* The new start time was calculated my msr_pack */
    mst->starttime = msr->starttime;

    samplesize = ms_samplesize (mst->sampletype);
    bufsize    = (mst->numsamples - trpackedsamples) * samplesize;

    if (bufsize)
    {
      memmove (mst->datasamples,
               (char *)mst->datasamples + (trpackedsamples * samplesize),
               (size_t)bufsize);

      mst->datasamples = realloc (mst->datasamples, (size_t)bufsize);

      if (mst->datasamples == NULL)
      {
        ms_log (2, "mst_pack(): Cannot (re)allocate datasamples buffer\n");
        return -1;
      }
    }
    else
    {
      if (mst->datasamples)
        free (mst->datasamples);
      mst->datasamples = 0;
    }

    mst->samplecnt -= trpackedsamples;
    mst->numsamples -= trpackedsamples;
  }

  /* Reinstate preserved values if a template was used */
  if (mstemplate)
  {
    msr->starttime   = preservestarttime;
    msr->samprate    = preservesamprate;
    msr->datasamples = preservedatasamples;
    msr->numsamples  = preservenumsamples;
    msr->sampletype  = preservesampletype;
    msr->ststate     = preserveststate;
  }
  else
  {
    msr->datasamples = 0;
    msr->ststate     = 0;
    msr_free (&msr);
  }

  if (packedsamples)
    *packedsamples = trpackedsamples;

  return trpackedrecords;
} /* End of mst_pack() */

/***************************************************************************
 * mst_packgroup:
 *
 * Pack MSTraceGroup data into Mini-SEED records by calling mst_pack()
 * for each MSTrace in the group.
 *
 * Returns the number of records created on success and -1 on error.
 ***************************************************************************/
int
mst_packgroup (MSTraceGroup *mstg, void (*record_handler) (char *, int, void *),
               void *handlerdata, int reclen, flag encoding, flag byteorder,
               int64_t *packedsamples, flag flush, flag verbose,
               MSRecord *mstemplate)
{
  MSTrace *mst;
  int trpackedrecords     = 0;
  int64_t trpackedsamples = 0;
  char srcname[50];

  if (!mstg)
  {
    return -1;
  }

  if (packedsamples)
    *packedsamples = 0;

  mst = mstg->traces;

  while (mst)
  {
    if (mst->numsamples <= 0)
    {
      if (verbose > 1)
      {
        mst_srcname (mst, srcname, 1);
        ms_log (1, "No data samples for %s, skipping\n", srcname);
      }
    }
    else
    {
      trpackedrecords += mst_pack (mst, record_handler, handlerdata, reclen,
                                   encoding, byteorder, &trpackedsamples, flush,
                                   verbose, mstemplate);

      if (trpackedrecords == -1)
        break;

      if (packedsamples)
        *packedsamples += trpackedsamples;
    }

    mst = mst->next;
  }

  return trpackedrecords;
} /* End of mst_packgroup() */
