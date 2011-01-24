/***************************************************************************
 * genutils.c
 *
 * Generic utility routines
 *
 * Written by Chad Trabant
 * ORFEUS/EC-Project MEREDIAN
 * IRIS Data Management Center
 *
 * modified: 2009.353
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "libmseed.h"

static hptime_t ms_time2hptime_int (int year, int day, int hour,
				    int min, int sec, int usec);


/***************************************************************************
 * ms_recsrcname:
 *
 * Generate a source name string for a specified raw data record in
 * the format: 'NET_STA_LOC_CHAN' or, if the quality flag is true:
 * 'NET_STA_LOC_CHAN_QUAL'.  The passed srcname must have enough room
 * for the resulting string.
 *
 * Returns a pointer to the resulting string or NULL on error.
 ***************************************************************************/
char *
ms_recsrcname (char *record, char *srcname, flag quality)
{
  struct fsdh_s *fsdh;
  char network[6];
  char station[6];
  char location[6];
  char channel[6];
  
  if ( ! record )
    return NULL;
  
  fsdh = (struct fsdh_s *) record;
  
  ms_strncpclean (network, fsdh->network, 2);
  ms_strncpclean (station, fsdh->station, 5);
  ms_strncpclean (location, fsdh->location, 2);
  ms_strncpclean (channel, fsdh->channel, 3);
  
  /* Build the source name string including the quality indicator*/
  if ( quality )
    sprintf (srcname, "%s_%s_%s_%s_%c",
             network, station, location, channel, fsdh->dataquality);
  
  /* Build the source name string without the quality indicator*/
  else
    sprintf (srcname, "%s_%s_%s_%s", network, station, location, channel);
  
  return srcname;
} /* End of ms_recsrcname() */


/***************************************************************************
 * ms_splitsrcname:
 *
 * Split srcname into separate components: "NET_STA_LOC_CHAN[_QUAL]".
 * Memory for each component must already be allocated.  If a specific
 * component is not desired set the appropriate argument to NULL.
 *
 * Returns 0 on success and -1 on error.
 ***************************************************************************/
int
ms_splitsrcname (char *srcname, char *net, char *sta, char *loc, char *chan,
		 char *qual)
{
  char *id;
  char *ptr, *top, *next;
  int sepcnt = 0;
  
  if ( ! srcname )
    return -1;
  
  /* Verify number of separating underscore characters */
  id = srcname;
  while ( (id = strchr (id, '_')) )
    {
      id++;
      sepcnt++;
    }
  
  /* Either 3 or 4 separating underscores are required */
  if ( sepcnt != 3 && sepcnt != 4 )
    {
      return -1;
    }
  
  /* Duplicate srcname */
  if ( ! (id = strdup(srcname)) )
    {
      fprintf (stderr, "ms_splitsrcname(): Error duplicating srcname string");
      return -1;
    }
  
  /* Network */
  top = id;
  if ( (ptr = strchr (top, '_')) )
    {
      next = ptr + 1;
      *ptr = '\0';
      
      if ( net )
	strcpy (net, top);
      
      top = next;
    }
  /* Station */
  if ( (ptr = strchr (top, '_')) )
    {
      next = ptr + 1;
      *ptr = '\0';
      
      if ( sta )
	strcpy (sta, top);
      
      top = next;
    }
  /* Location */
  if ( (ptr = strchr (top, '_')) )
    {
      next = ptr + 1;
      *ptr = '\0';
      
      if ( loc )
	strcpy (loc, top);
      
      top = next;
    }
  /* Channel & optional Quality */
  if ( (ptr = strchr (top, '_')) )
    {
      next = ptr + 1;
      *ptr = '\0';
      
      if ( chan )
	strcpy (chan, top);
      
      top = next;
      
      /* Quality */
      if ( *top && qual )
	{
	  /* Quality is a single character */
	  *qual = *top;
	}
    }
  /* Otherwise only Channel */
  else if ( *top && chan )
    {
      strcpy (chan, top);
    }
  
  /* Free duplicated stream ID */
  if ( id )
    free (id);
  
  return 0;
}  /* End of ms_splitsrcname() */


/***************************************************************************
 * ms_strncpclean:
 *
 * Copy up to 'length' characters from 'source' to 'dest' while
 * removing all spaces.  The result is left justified and always null
 * terminated.  The destination string must have enough room needed
 * for the non-space characters within 'length' and the null
 * terminator, a maximum of 'length + 1'.
 * 
 * Returns the number of characters (not including the null terminator) in
 * the destination string.
 ***************************************************************************/
int
ms_strncpclean (char *dest, const char *source, int length)
{
  int sidx, didx;
  
  if ( ! dest )
    return 0;
  
  if ( ! source )
    {
      *dest = '\0';
      return 0;
    }

  for ( sidx=0, didx=0; sidx < length ; sidx++ )
    {
      if ( *(source+sidx) == '\0' )
	{
	  break;
	}

      if ( *(source+sidx) != ' ' )
	{
	  *(dest+didx) = *(source+sidx);
	  didx++;
	}
    }

  *(dest+didx) = '\0';
  
  return didx;
}  /* End of ms_strncpclean() */


/***************************************************************************
 * ms_strncpopen:
 *
 * Copy 'length' characters from 'source' to 'dest', padding the right
 * side with spaces and leave open-ended.  The result is left
 * justified and *never* null terminated (the open-ended part).  The
 * destination string must have enough room for 'length' characters.
 * 
 * Returns the number of characters copied from the source string.
 ***************************************************************************/
int
ms_strncpopen (char *dest, const char *source, int length)
{
  int didx;
  int dcnt = 0;
  int term = 0;
  
  if ( ! dest )
    return 0;
  
  if ( ! source )
    {
      for ( didx=0; didx < length ; didx++ )
	{
	  *(dest+didx) = ' ';
	}
      
      return 0;
    }
  
  for ( didx=0; didx < length ; didx++ )
    {
      if ( !term )
	if ( *(source+didx) == '\0' )
	  term = 1;
      
      if ( !term )
	{
	  *(dest+didx) = *(source+didx);
	  dcnt++;
	}
      else
	{
	  *(dest+didx) = ' ';
	}
    }
  
  return dcnt;
}  /* End of ms_strncpopen() */


/***************************************************************************
 * ms_doy2md:
 *
 * Compute the month and day-of-month from a year and day-of-year.
 *
 * Year is expected to be in the range 1900-2100, jday is expected to
 * be in the range 1-366, month will be in the range 1-12 and mday
 * will be in the range 1-31.
 *
 * Returns 0 on success and -1 on error.
 ***************************************************************************/
int
ms_doy2md(int year, int jday, int *month, int *mday)
{
  int idx;
  int leap;
  int days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  
  /* Sanity check for the supplied year */
  if ( year < 1900 || year > 2100 )
    {
      ms_log (2, "ms_doy2md(): year (%d) is out of range\n", year);
      return -1;
    }
  
  /* Test for leap year */
  leap = ( ((year%4 == 0) && (year%100 != 0)) || (year%400 == 0) ) ? 1 : 0;

  /* Add a day to February if leap year */
  if ( leap )
    days[1]++;

  if (jday > 365+leap || jday <= 0)
    {
      ms_log (2, "ms_doy2md(): day-of-year (%d) is out of range\n", jday);
      return -1;
    }
    
  for ( idx=0; idx < 12; idx++ )
    {
      jday -= days[idx];

      if ( jday <= 0 )
	{
	  *month = idx + 1;
	  *mday = days[idx] + jday;
	  break;
	}
    }

  return 0;
}  /* End of ms_doy2md() */


/***************************************************************************
 * ms_md2doy:
 *
 * Compute the day-of-year from a year, month and day-of-month.
 *
 * Year is expected to be in the range 1900-2100, month is expected to
 * be in the range 1-12, mday is expected to be in the range 1-31 and
 * jday will be in the range 1-366.
 *
 * Returns 0 on success and -1 on error.
 ***************************************************************************/
int
ms_md2doy(int year, int month, int mday, int *jday)
{
  int idx;
  int leap;
  int days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  
  /* Sanity check for the supplied parameters */
  if ( year < 1900 || year > 2100 )
    {
      ms_log (2, "ms_md2doy(): year (%d) is out of range\n", year);
      return -1;
    }
  if ( month < 1 || month > 12 )
    {
      ms_log (2, "ms_md2doy(): month (%d) is out of range\n", month);
      return -1;
    }
  if ( mday < 1 || mday > 31 )
    {
      ms_log (2, "ms_md2doy(): day-of-month (%d) is out of range\n", mday);
      return -1;
    }
  
  /* Test for leap year */
  leap = ( ((year%4 == 0) && (year%100 != 0)) || (year%400 == 0) ) ? 1 : 0;
  
  /* Add a day to February if leap year */
  if ( leap )
    days[1]++;
  
  /* Check that the day-of-month jives with specified month */
  if ( mday > days[month-1] )
    {
      ms_log (2, "ms_md2doy(): day-of-month (%d) is out of range for month %d\n",
	       mday, month);
      return -1;
    }

  *jday = 0;
  month--;
  
  for ( idx=0; idx < 12; idx++ )
    {
      if ( idx == month )
	{
	  *jday += mday;
	  break;
	}
      
      *jday += days[idx];
    }
  
  return 0;
}  /* End of ms_md2doy() */


/***************************************************************************
 * ms_btime2hptime:
 *
 * Convert a binary SEED time structure to a high precision epoch time
 * (1/HPTMODULUS second ticks from the epoch).  The algorithm used is
 * a specific version of a generalized function in GNU glibc.
 *
 * Returns a high precision epoch time on success and HPTERROR on
 * error.
 ***************************************************************************/
hptime_t
ms_btime2hptime (BTime *btime)
{
  hptime_t hptime;
  int shortyear;
  int a4, a100, a400;
  int intervening_leap_days;
  int days;
  
  if ( ! btime )
    return HPTERROR;
  
  shortyear = btime->year - 1900;

  a4 = (shortyear >> 2) + 475 - ! (shortyear & 3);
  a100 = a4 / 25 - (a4 % 25 < 0);
  a400 = a100 >> 2;
  intervening_leap_days = (a4 - 492) - (a100 - 19) + (a400 - 4);
  
  days = (365 * (shortyear - 70) + intervening_leap_days + (btime->day - 1));
  
  hptime = (hptime_t ) (60 * (60 * (24 * days + btime->hour) + btime->min) + btime->sec) * HPTMODULUS
    + (btime->fract * (HPTMODULUS / 10000));
    
  return hptime;
}  /* End of ms_btime2hptime() */


/***************************************************************************
 * ms_btime2isotimestr:
 *
 * Build a time string in ISO recommended format from a BTime struct.
 *
 * The provided isostimestr must have enough room for the resulting time
 * string of 25 characters, i.e. '2001-07-29T12:38:00.0000' + NULL.
 *
 * Returns a pointer to the resulting string or NULL on error.
 ***************************************************************************/
char *
ms_btime2isotimestr (BTime *btime, char *isotimestr)
{  
  int month = 0;
  int mday = 0;
  int ret;

  if ( ! isotimestr )
    return NULL;

  if ( ms_doy2md (btime->year, btime->day, &month, &mday) )
    {
      ms_log (2, "ms_btime2isotimestr(): Error converting year %d day %d\n",
	      btime->year, btime->day);
      return NULL;
    }
  
  ret = snprintf (isotimestr, 25, "%4d-%02d-%02dT%02d:%02d:%02d.%04d",
		  btime->year, month, mday,
		  btime->hour, btime->min, btime->sec, btime->fract);
  
  if ( ret != 24 )
    return NULL;
  else
    return isotimestr;
}  /* End of ms_btime2isotimestr() */


/***************************************************************************
 * ms_btime2mdtimestr:
 *
 * Build a time string in month-day format from a BTime struct.
 * 
 * The provided isostimestr must have enough room for the resulting time
 * string of 25 characters, i.e. '2001-07-29 12:38:00.0000' + NULL.
 *
 * Returns a pointer to the resulting string or NULL on error.
 ***************************************************************************/
char *
ms_btime2mdtimestr (BTime *btime, char *mdtimestr)
{ 
  int month = 0;
  int mday = 0;
  int ret;
  
  if ( ! mdtimestr )
    return NULL;
  
  if ( ms_doy2md (btime->year, btime->day, &month, &mday) )
    {
      ms_log (2, "ms_btime2mdtimestr(): Error converting year %d day %d\n",
              btime->year, btime->day);
      return NULL;
    }
  
  ret = snprintf (mdtimestr, 25, "%4d-%02d-%02d %02d:%02d:%02d.%04d",
                  btime->year, month, mday,
                  btime->hour, btime->min, btime->sec, btime->fract);

  if ( ret != 24 )
    return NULL;
  else
    return mdtimestr;
}  /* End of ms_btime2mdtimestr() */


/***************************************************************************
 * ms_btime2seedtimestr:
 *
 * Build a SEED time string from a BTime struct.
 *
 * The provided seedtimestr must have enough room for the resulting time
 * string of 23 characters, i.e. '2001,195,12:38:00.0000' + NULL.
 *
 * Returns a pointer to the resulting string or NULL on error.
 ***************************************************************************/
char *
ms_btime2seedtimestr (BTime *btime, char *seedtimestr)
{
  int ret;
  
  if ( ! seedtimestr )
    return NULL;
  
  ret = snprintf (seedtimestr, 23, "%4d,%03d,%02d:%02d:%02d.%04d",
		  btime->year, btime->day,
		  btime->hour, btime->min, btime->sec, btime->fract);
  
  if ( ret != 22 )
    return NULL;
  else
    return seedtimestr;
}  /* End of ms_btime2seedtimestr() */


/***************************************************************************
 * ms_hptime2btime:
 *
 * Convert a high precision epoch time to a SEED binary time
 * structure.  The microseconds beyond the 1/10000 second range are
 * truncated and *not* rounded, this is intentional and necessary.
 *
 * Returns 0 on success and -1 on error.
 ***************************************************************************/
int
ms_hptime2btime (hptime_t hptime, BTime *btime)
{
  struct tm *tm;
  int isec;
  int ifract;
  int bfract;
  time_t tsec;
  
  if ( btime == NULL )
    return -1;
  
  /* Reduce to Unix/POSIX epoch time and fractional seconds */
  isec = MS_HPTIME2EPOCH(hptime);
  ifract = hptime - ((hptime_t)isec * HPTMODULUS);
  
  /* BTime only has 1/10000 second precision */
  bfract = ifract / (HPTMODULUS / 10000);
  
  /* Adjust for negative epoch times, round back when needed */
  if ( hptime < 0 && ifract != 0 )
    {
      /* Isolate microseconds between 1e-4 and 1e-6 precision and adjust bfract if not zero */
      if ( ifract - bfract * (HPTMODULUS / 10000) )
	bfract -= 1;
      
      isec -= 1;
      bfract = 10000 - (-bfract);
    }

  tsec = (time_t) isec;
  if ( ! (tm = gmtime ( &tsec )) )
    return -1;
  
  btime->year   = tm->tm_year + 1900;
  btime->day    = tm->tm_yday + 1;
  btime->hour   = tm->tm_hour;
  btime->min    = tm->tm_min;
  btime->sec    = tm->tm_sec;
  btime->unused = 0;
  btime->fract  = (uint16_t) bfract;
  
  return 0;
}  /* End of ms_hptime2btime() */


/***************************************************************************
 * ms_hptime2isotimestr:
 *
 * Build a time string in ISO recommended format from a high precision
 * epoch time.
 *
 * The provided isostimestr must have enough room for the resulting time
 * string of 27 characters, i.e. '2001-07-29T12:38:00.000000' + NULL.
 *
 * The 'subseconds' flag controls whenther the sub second portion of the
 * time is included or not.
 *
 * Returns a pointer to the resulting string or NULL on error.
 ***************************************************************************/
char *
ms_hptime2isotimestr (hptime_t hptime, char *isotimestr, flag subseconds)
{
  struct tm *tm;
  int isec;
  int ifract;
  int ret;
  time_t tsec;

  if ( isotimestr == NULL )
    return NULL;

  /* Reduce to Unix/POSIX epoch time and fractional seconds */
  isec = MS_HPTIME2EPOCH(hptime);
  ifract = (hptime_t) hptime - (isec * HPTMODULUS);
  
  /* Adjust for negative epoch times */
  if ( hptime < 0 && ifract != 0 )
    {
      isec -= 1;
      ifract = HPTMODULUS - (-ifract);
    }

  tsec = (time_t) isec;
  if ( ! (tm = gmtime ( &tsec )) )
    return NULL;
  
  if ( subseconds )
    /* Assuming ifract has at least microsecond precision */
    ret = snprintf (isotimestr, 27, "%4d-%02d-%02dT%02d:%02d:%02d.%06d",
                    tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                    tm->tm_hour, tm->tm_min, tm->tm_sec, ifract);
  else
    ret = snprintf (isotimestr, 20, "%4d-%02d-%02dT%02d:%02d:%02d",
                    tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                    tm->tm_hour, tm->tm_min, tm->tm_sec);

  if ( ret != 26 && ret != 19 )
    return NULL;
  else
    return isotimestr;
}  /* End of ms_hptime2isotimestr() */


/***************************************************************************
 * ms_hptime2mdtimestr:
 *
 * Build a time string in month-day format from a high precision
 * epoch time.
 *
 * The provided mdtimestr must have enough room for the resulting time
 * string of 27 characters, i.e. '2001-07-29 12:38:00.000000' + NULL.
 *
 * The 'subseconds' flag controls whenther the sub second portion of the
 * time is included or not.
 *
 * Returns a pointer to the resulting string or NULL on error.
 ***************************************************************************/
char *
ms_hptime2mdtimestr (hptime_t hptime, char *mdtimestr, flag subseconds)
{
  struct tm *tm;
  int isec;
  int ifract;
  int ret;
  time_t tsec;

  if ( mdtimestr == NULL )
    return NULL;

  /* Reduce to Unix/POSIX epoch time and fractional seconds */
  isec = MS_HPTIME2EPOCH(hptime);
  ifract = (hptime_t) hptime - (isec * HPTMODULUS);

  /* Adjust for negative epoch times */
  if ( hptime < 0 && ifract != 0 )
    {
      isec -= 1;
      ifract = HPTMODULUS - (-ifract);
    }

  tsec = (time_t) isec;
  if ( ! (tm = gmtime ( &tsec )) )
    return NULL;

  if ( subseconds )
    /* Assuming ifract has at least microsecond precision */
    ret = snprintf (mdtimestr, 27, "%4d-%02d-%02d %02d:%02d:%02d.%06d",
                    tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                    tm->tm_hour, tm->tm_min, tm->tm_sec, ifract);
  else
    ret = snprintf (mdtimestr, 20, "%4d-%02d-%02d %02d:%02d:%02d",
                    tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                    tm->tm_hour, tm->tm_min, tm->tm_sec);

  if ( ret != 26 && ret != 19 )
    return NULL;
  else
    return mdtimestr;
}  /* End of ms_hptime2mdtimestr() */


/***************************************************************************
 * ms_hptime2seedtimestr:
 *
 * Build a SEED time string from a high precision epoch time.
 *
 * The provided seedtimestr must have enough room for the resulting time
 * string of 25 characters, i.e. '2001,195,12:38:00.000000\n'.
 *
 * The 'subseconds' flag controls whenther the sub second portion of the
 * time is included or not.
 *
 * Returns a pointer to the resulting string or NULL on error.
 ***************************************************************************/
char *
ms_hptime2seedtimestr (hptime_t hptime, char *seedtimestr, flag subseconds)
{
  struct tm *tm;
  int isec;
  int ifract;
  int ret;
  time_t tsec;
  
  if ( seedtimestr == NULL )
    return NULL;
  
  /* Reduce to Unix/POSIX epoch time and fractional seconds */
  isec = MS_HPTIME2EPOCH(hptime);
  ifract = (hptime_t) hptime - (isec * HPTMODULUS);
  
  /* Adjust for negative epoch times */
  if ( hptime < 0 && ifract != 0 )
    {
      isec -= 1;
      ifract = HPTMODULUS - (-ifract);
    }

  tsec = (time_t) isec;
  if ( ! (tm = gmtime ( &tsec )) )
    return NULL;
  
  if ( subseconds )
    /* Assuming ifract has at least microsecond precision */
    ret = snprintf (seedtimestr, 25, "%4d,%03d,%02d:%02d:%02d.%06d",
		    tm->tm_year + 1900, tm->tm_yday + 1,
		    tm->tm_hour, tm->tm_min, tm->tm_sec, ifract);
  else
    /* Assuming ifract has at least microsecond precision */
    ret = snprintf (seedtimestr, 18, "%4d,%03d,%02d:%02d:%02d",
                    tm->tm_year + 1900, tm->tm_yday + 1,
                    tm->tm_hour, tm->tm_min, tm->tm_sec);
  
  if ( ret != 24 && ret != 17 )
    return NULL;
  else
    return seedtimestr;
}  /* End of ms_hptime2seedtimestr() */


/***************************************************************************
 * ms_time2hptime_int:
 *
 * Convert specified time values to a high precision epoch time.  This
 * is an internal version which does no range checking, it is assumed
 * that checking the range for each value has already been done.
 *
 * Returns epoch time on success and HPTERROR on error.
 ***************************************************************************/
static hptime_t
ms_time2hptime_int (int year, int day, int hour, int min, int sec, int usec)
{
  BTime btime;
  hptime_t hptime;
  
  memset (&btime, 0, sizeof(BTime));
  btime.day = 1;
  
  /* Convert integer seconds using ms_btime2hptime */
  btime.year = (int16_t) year;
  btime.day = (int16_t) day;
  btime.hour = (uint8_t) hour;
  btime.min = (uint8_t) min;
  btime.sec = (uint8_t) sec;
  btime.fract = 0;

  hptime = ms_btime2hptime (&btime);
  
  if ( hptime == HPTERROR )
    {
      ms_log (2, "ms_time2hptime(): Error converting with ms_btime2hptime()\n");
      return HPTERROR;
    }
  
  /* Add the microseconds */
  hptime += (hptime_t) usec * (1000000 / HPTMODULUS);
  
  return hptime;
}  /* End of ms_time2hptime_int() */


/***************************************************************************
 * ms_time2hptime:
 *
 * Convert specified time values to a high precision epoch time.  This
 * is essentially a frontend for ms_time2hptime that does range
 * checking for each input value.
 *
 * Expected ranges:
 * year : 1900 - 2100
 * day  : 1 - 366
 * hour : 0 - 23
 * min  : 0 - 59
 * sec  : 0 - 60
 * usec : 0 - 999999
 *
 * Returns epoch time on success and HPTERROR on error.
 ***************************************************************************/
hptime_t
ms_time2hptime (int year, int day, int hour, int min, int sec, int usec)
{
  if ( year < 1900 || year > 2100 )
    {
      ms_log (2, "ms_time2hptime(): Error with year value: %d\n", year);
      return HPTERROR;
    }
  
  if ( day < 1 || day > 366 )
    {
      ms_log (2, "ms_time2hptime(): Error with day value: %d\n", day);
      return HPTERROR;
    }
  
  if ( hour < 0 || hour > 23 )
    {
      ms_log (2, "ms_time2hptime(): Error with hour value: %d\n", hour);
      return HPTERROR;
    }
  
  if ( min < 0 || min > 59 )
    {
      ms_log (2, "ms_time2hptime(): Error with minute value: %d\n", min);
      return HPTERROR;
    }
  
  if ( sec < 0 || sec > 60 )
    {
      ms_log (2, "ms_time2hptime(): Error with second value: %d\n", sec);
      return HPTERROR;
    }
  
  if ( usec < 0 || usec > 999999 )
    {
      ms_log (2, "ms_time2hptime(): Error with microsecond value: %d\n", usec);
      return HPTERROR;
    }
  
  return ms_time2hptime_int (year, day, hour, min, sec, usec);
}  /* End of ms_time2hptime() */


/***************************************************************************
 * ms_seedtimestr2hptime:
 * 
 * Convert a SEED time string to a high precision epoch time.  SEED
 * time format is "YYYY[,DDD,HH,MM,SS.FFFFFF]", the delimiter can be a
 * comma [,], colon [:] or period [.] except for the fractional
 * seconds which must start with a period [.].
 *
 * The time string can be "short" in which case the omitted values are
 * assumed to be zero (with the exception of DDD which is assumed to
 * be 1): "YYYY,DDD,HH" assumes MM, SS and FFFF are 0.  The year is
 * required, otherwise there wouldn't be much for a date.
 *
 * Ranges are checked for each value.
 *
 * Returns epoch time on success and HPTERROR on error.
 ***************************************************************************/
hptime_t
ms_seedtimestr2hptime (char *seedtimestr)
{
  int fields;
  int year = 0;
  int day  = 1;
  int hour = 0;
  int min  = 0;
  int sec  = 0;
  float fusec = 0.0;
  int usec = 0;
  
  fields = sscanf (seedtimestr, "%d%*[,:.]%d%*[,:.]%d%*[,:.]%d%*[,:.]%d%f",
		   &year, &day, &hour, &min, &sec, &fusec);
  
  /* Convert fractional seconds to microseconds */
  if ( fusec != 0.0 )
    {
      usec = (int) (fusec * 1000000.0 + 0.5);
    }
  
  if ( fields < 1 )
    {
      ms_log (2, "ms_seedtimestr2hptime(): Error converting time string: %s\n", seedtimestr);
      return HPTERROR;
    }
  
  if ( year < 1900 || year > 3000 )
    {
      ms_log (2, "ms_seedtimestr2hptime(): Error with year value: %d\n", year);
      return HPTERROR;
    }

  if ( day < 1 || day > 366 )
    {
      ms_log (2, "ms_seedtimestr2hptime(): Error with day value: %d\n", day);
      return HPTERROR;
    }
  
  if ( hour < 0 || hour > 23 )
    {
      ms_log (2, "ms_seedtimestr2hptime(): Error with hour value: %d\n", hour);
      return HPTERROR;
    }
  
  if ( min < 0 || min > 59 )
    {
      ms_log (2, "ms_seedtimestr2hptime(): Error with minute value: %d\n", min);
      return HPTERROR;
    }
  
  if ( sec < 0 || sec > 60 )
    {
      ms_log (2, "ms_seedtimestr2hptime(): Error with second value: %d\n", sec);
      return HPTERROR;
    }
  
  if ( usec < 0 || usec > 999999 )
    {
      ms_log (2, "ms_seedtimestr2hptime(): Error with fractional second value: %d\n", usec);
      return HPTERROR;
    }
  
  return ms_time2hptime_int (year, day, hour, min, sec, usec);
}  /* End of ms_seedtimestr2hptime() */


/***************************************************************************
 * ms_timestr2hptime:
 * 
 * Convert a generic time string to a high precision epoch time.
 * SEED time format is "YYYY[/MM/DD HH:MM:SS.FFFF]", the delimiter can
 * be a dash [-], slash [/], colon [:], or period [.] and between the
 * date and time a 'T' or a space may be used.  The fracttional
 * seconds must begin with a period [.].
 *
 * The time string can be "short" in which case the omitted values are
 * assumed to be zero (with the exception of month and day which are
 * assumed to be 1): "YYYY/MM/DD" assumes HH, MM, SS and FFFF are 0.
 * The year is required, otherwise there wouldn't be much for a date.
 *
 * Ranges are checked for each value.
 *
 * Returns epoch time on success and HPTERROR on error.
 ***************************************************************************/
hptime_t
ms_timestr2hptime (char *timestr)
{
  int fields;
  int year = 0;
  int mon  = 1;
  int mday = 1;
  int day  = 1;
  int hour = 0;
  int min  = 0;
  int sec  = 0;
  float fusec = 0.0;
  int usec = 0;
    
  fields = sscanf (timestr, "%d%*[-/:.]%d%*[-/:.]%d%*[-/:.T ]%d%*[-/:.]%d%*[- /:.]%d%f",
		   &year, &mon, &mday, &hour, &min, &sec, &fusec);
  
  /* Convert fractional seconds to microseconds */
  if ( fusec != 0.0 )
    {
      usec = (int) (fusec * 1000000.0 + 0.5);
    }

  if ( fields < 1 )
    {
      ms_log (2, "ms_timestr2hptime(): Error converting time string: %s\n", timestr);
      return HPTERROR;
    }
  
  if ( year < 1900 || year > 3000 )
    {
      ms_log (2, "ms_timestr2hptime(): Error with year value: %d\n", year);
      return HPTERROR;
    }
  
  if ( mon < 1 || mon > 12 )
    {
      ms_log (2, "ms_timestr2hptime(): Error with month value: %d\n", mon);
      return HPTERROR;
    }

  if ( mday < 1 || mday > 31 )
    {
      ms_log (2, "ms_timestr2hptime(): Error with day value: %d\n", mday);
      return HPTERROR;
    }

  /* Convert month and day-of-month to day-of-year */
  if ( ms_md2doy (year, mon, mday, &day) )
    {
      return HPTERROR;
    }
  
  if ( hour < 0 || hour > 23 )
    {
      ms_log (2, "ms_timestr2hptime(): Error with hour value: %d\n", hour);
      return HPTERROR;
    }
  
  if ( min < 0 || min > 59 )
    {
      ms_log (2, "ms_timestr2hptime(): Error with minute value: %d\n", min);
      return HPTERROR;
    }
  
  if ( sec < 0 || sec > 60 )
    {
      ms_log (2, "ms_timestr2hptime(): Error with second value: %d\n", sec);
      return HPTERROR;
    }
  
  if ( usec < 0 || usec > 999999 )
    {
      ms_log (2, "ms_timestr2hptime(): Error with fractional second value: %d\n", usec);
      return HPTERROR;
    }
  
  return ms_time2hptime_int (year, day, hour, min, sec, usec);
}  /* End of ms_timestr2hptime() */


/***************************************************************************
 * ms_nomsamprate:
 *
 * Calculate a sample rate from SEED sample rate factor and multiplier
 * as stored in the fixed section header of data records.
 * 
 * Returns the positive sample rate.
 ***************************************************************************/
double
ms_nomsamprate (int factor, int multiplier)
{
  double samprate = 0.0;
  
  if ( factor > 0 )
    samprate = (double) factor;
  else if ( factor < 0 )
    samprate = -1.0 / (double) factor;
  if ( multiplier > 0 )
    samprate = samprate * (double) multiplier;
  else if ( multiplier < 0 )
    samprate = -1.0 * (samprate / (double) multiplier);
  
  return samprate;
}  /* End of ms_nomsamprate() */


/***************************************************************************
 * ms_genfactmult:
 *
 * Generate an approriate SEED sample rate factor and multiplier from
 * a double precision sample rate.
 * 
 * Returns 0 on success and -1 on error.
 ***************************************************************************/
int
ms_genfactmult (double samprate, int16_t *factor, int16_t *multiplier)
{
  int num, den;
  
  /* This routine does not support very high or negative sample rates,
     even though high rates are possible in Mini-SEED */
  if ( samprate > 32727.0 || samprate < 0.0 )
    {
      ms_log (2, "ms_genfactmult(): samprate out of range: %g\n", samprate);
      return -1;
    }
  
  /* If the sample rate is integer set the factor and multipler in the
     obvious way, otherwise derive a (potentially approximate)
     numerator and denominator for the given samprate */
  if ( (samprate - (int16_t) samprate) < 0.000001 )
    {
      *factor = (int16_t) samprate;
      if ( *factor )
	*multiplier = 1;
    }
  else
    {
      ms_ratapprox (samprate, &num, &den, 32727, 1e-12);
      
      /* Negate the multiplier to denote a division factor */
      *factor = (int16_t ) num;
      *multiplier = (int16_t) -den;
    }
  
  return 0;
}  /* End of ms_genfactmult() */


/***************************************************************************
 * ms_ratapprox:
 *
 * Find an approximate rational number for a real through continued
 * fraction expansion.  Given a double precsion 'real' find a
 * numerator (num) and denominator (den) whose absolute values are not
 * larger than 'maxval' while trying to reach a specified 'precision'.
 * 
 * Returns the number of iterations performed.
 ***************************************************************************/
int
ms_ratapprox (double real, int *num, int *den, int maxval, double precision)
{
  double realj, preal;
  char pos;  
  int pnum, pden;
  int iterations = 1;
  int Aj1, Aj2, Bj1, Bj2;
  int bj = 0;
  int Aj = 0;
  int Bj = 1;
  
  if ( real >= 0.0 ) { pos = 1; realj = real; }
  else               { pos = 0; realj = -real; }
  
  preal = realj;
  
  bj = (int) (realj + precision);
  realj = 1 / (realj - bj);
  Aj = bj; Aj1 = 1;
  Bj = 1;  Bj1 = 0;
  *num = pnum = Aj;
  *den = pden = Bj;
  if ( !pos ) *num = -*num;
  
  while ( ms_dabs(preal - (double)Aj/(double)Bj) > precision &&
	  Aj < maxval && Bj < maxval )
    {
      Aj2 = Aj1; Aj1 = Aj;
      Bj2 = Bj1; Bj1 = Bj;
      bj = (int) (realj + precision);
      realj = 1 / (realj - bj);
      Aj = bj * Aj1 + Aj2;
      Bj = bj * Bj1 + Bj2;
      *num = pnum;
      *den = pden;
      if ( !pos ) *num = -*num;
      pnum = Aj;
      pden = Bj;
      
      iterations++;
    }
  
  if ( pnum < maxval && pden < maxval )
    {
      *num = pnum;
      *den = pden;
      if ( !pos ) *num = -*num;
    }
  
  return iterations;
}


/***************************************************************************
 * ms_bigendianhost:
 *
 * Determine the byte order of the host machine.  Due to the lack of
 * portable defines to determine host byte order this run-time test is
 * provided.  The code below actually tests for little-endianess, the
 * only other alternative is assumed to be big endian.
 * 
 * Returns 0 if the host is little endian, otherwise 1.
 ***************************************************************************/
int
ms_bigendianhost ()
{
  int16_t host = 1;
  return !(*((int8_t *)(&host)));
}  /* End of ms_bigendianhost() */


/***************************************************************************
 * ms_dabs:
 *
 * Determine the absolute value of an input double, actually just test
 * if the input double is positive multiplying by -1.0 if not and
 * return it.
 * 
 * Returns the positive value of input double.
 ***************************************************************************/
double
ms_dabs (double val)
{
  if ( val < 0.0 )
    val *= -1.0;
  return val;
}  /* End of ms_dabs() */


/***************************************************************************
 * ms_parse_raw:
 *
 * Parse and verify a SEED data record header (fixed section and
 * blockettes) at the lowest level, printing error messages for
 * invalid header values and optionally print raw header values.  The
 * memory at 'record' is assumed to be a Mini-SEED record.  Not every
 * possible test is performed, common errors and those causing
 * libmseed parsing to fail should be detected.
 *
 * The 'details' argument is interpreted as follows:
 *
 * details:
 *  0 = only print error messages for invalid header fields
 *  1 = print basic fields in addition to invalid field errors
 *  2 = print all fields in addition to invalid field errors
 *
 * The 'swapflag' argument is interpreted as follows:
 *
 * swapflag:
 *  1 = swap multibyte quantities
 *  0 = do no swapping
 * -1 = autodetect byte order using year test, swap if needed
 *
 * Any byte swapping performed by this routine is applied directly to
 * the memory reference by the record pointer.
 *
 * This routine is primarily intended to diagnose invalid Mini-SEED headers.
 *
 * Returns 0 when no errors were detected or a positive count of
 * errors detected.
 ***************************************************************************/
int
ms_parse_raw (char *record, int maxreclen, flag details, flag swapflag)
{
  struct fsdh_s *fsdh;
  double nomsamprate;
  char srcname[50];
  char *X;
  char b;
  int retval = 0;
  int b1000encoding = -1;
  int b1000reclen = -1;
  int endofblockettes = -1;
  int idx;
  
  if ( ! record )
    return 1;
  
  /* Generate a source name string */
  srcname[0] = '\0';
  ms_recsrcname (record, srcname, 1);
  
  fsdh = (struct fsdh_s *) record;
  
  /* Check to see if byte swapping is needed by testing the year */
  if ( swapflag == -1 &&
       ((fsdh->start_time.year < 1920) ||
	(fsdh->start_time.year > 2050)) )
    swapflag = 1;
  else
    swapflag = 0;
  
  if ( details > 1 )
    {
      if ( swapflag == 1 )
	ms_log (0, "Swapping multi-byte quantities in header\n");
      else
	ms_log (0, "Not swapping multi-byte quantities in header\n");
    }
  
  /* Swap byte order */
  if ( swapflag )
    {
      MS_SWAPBTIME (&fsdh->start_time);
      ms_gswap2a (&fsdh->numsamples);
      ms_gswap2a (&fsdh->samprate_fact);
      ms_gswap2a (&fsdh->samprate_mult);
      ms_gswap4a (&fsdh->time_correct);
      ms_gswap2a (&fsdh->data_offset);
      ms_gswap2a (&fsdh->blockette_offset);
    }
  
  /* Validate fixed section header fields */
  X = record;  /* Pointer of convenience */
  
  /* Check record sequence number, 6 ASCII digits */
  if ( ! isdigit((unsigned char) *(X)) || ! isdigit ((unsigned char) *(X+1)) ||
       ! isdigit((unsigned char) *(X+2)) || ! isdigit ((unsigned char) *(X+3)) ||
       ! isdigit((unsigned char) *(X+4)) || ! isdigit ((unsigned char) *(X+5)) )
    {
      ms_log (2, "%s: Invalid sequence number: '%c%c%c%c%c%c'\n", srcname, X, X+1, X+2, X+3, X+4, X+5);
      retval++;
    }
  
  /* Check header/quality indicator */
  if ( ! MS_ISDATAINDICATOR(*(X+6)) )
    {
      ms_log (2, "%s: Invalid header indicator (DRQM): '%c'\n", srcname, X+6);
      retval++;
    }
  
  /* Check reserved byte, space or NULL */
  if ( ! (*(X+7) == ' ' || *(X+7) == '\0') )
    {
      ms_log (2, "%s: Invalid fixed section reserved byte (Space): '%c'\n", srcname, X+7);
      retval++;
    }
  
  /* Check station code, 5 alphanumerics or spaces */
  if ( ! (isalnum((unsigned char) *(X+8)) || *(X+8) == ' ') ||
       ! (isalnum((unsigned char) *(X+9)) || *(X+9) == ' ') ||
       ! (isalnum((unsigned char) *(X+10)) || *(X+10) == ' ') ||
       ! (isalnum((unsigned char) *(X+11)) || *(X+11) == ' ') ||
       ! (isalnum((unsigned char) *(X+12)) || *(X+12) == ' ') )
    {
      ms_log (2, "%s: Invalid station code: '%c%c%c%c%c'\n", srcname, X+8, X+9, X+10, X+11, X+12);
      retval++;
    }
  
  /* Check location ID, 2 alphanumerics or spaces */
  if ( ! (isalnum((unsigned char) *(X+13)) || *(X+13) == ' ') ||
       ! (isalnum((unsigned char) *(X+14)) || *(X+14) == ' ') )
    {
      ms_log (2, "%s: Invalid location ID: '%c%c'\n", srcname, X+13, X+14);
      retval++;
    }
  
  /* Check channel codes, 3 alphanumerics or spaces */
  if ( ! (isalnum((unsigned char) *(X+15)) || *(X+15) == ' ') ||
       ! (isalnum((unsigned char) *(X+16)) || *(X+16) == ' ') ||
       ! (isalnum((unsigned char) *(X+17)) || *(X+17) == ' ') )
    {
      ms_log (2, "%s: Invalid channel codes: '%c%c%c'\n", srcname, X+15, X+16, X+17);
      retval++;
    }
  
  /* Check network code, 2 alphanumerics or spaces */
  if ( ! (isalnum((unsigned char) *(X+18)) || *(X+18) == ' ') ||
       ! (isalnum((unsigned char) *(X+19)) || *(X+19) == ' ') )
    {
      ms_log (2, "%s: Invalid network code: '%c%c'\n", srcname, X+18, X+19);
      retval++;
    }
  
  /* Check start time fields */
  if ( fsdh->start_time.year < 1920 || fsdh->start_time.year > 2050 )
    {
      ms_log (2, "%s: Unlikely start year (1920-2050): '%d'\n", srcname, fsdh->start_time.year);
      retval++;
    }
  if ( fsdh->start_time.day < 1 || fsdh->start_time.day > 366 )
    {
      ms_log (2, "%s: Invalid start day (1-366): '%d'\n", srcname, fsdh->start_time.day);
      retval++;
    }
  if ( fsdh->start_time.hour > 23 )
    {
      ms_log (2, "%s: Invalid start hour (0-23): '%d'\n", srcname, fsdh->start_time.hour);
      retval++;
    }
  if ( fsdh->start_time.min > 59 )
    {
      ms_log (2, "%s: Invalid start minute (0-59): '%d'\n", srcname, fsdh->start_time.min);
      retval++;
    }
  if ( fsdh->start_time.sec > 60 )
    {
      ms_log (2, "%s: Invalid start second (0-60): '%d'\n", srcname, fsdh->start_time.sec);
      retval++;
    }
  if ( fsdh->start_time.fract > 9999 )
    {
      ms_log (2, "%s: Invalid start fractional seconds (0-9999): '%d'\n", srcname, fsdh->start_time.fract);
      retval++;
    }
  
  /* Check number of samples, max samples in 4096-byte Steim-2 encoded record: 6601 */
  if ( fsdh->numsamples > 20000 )
    {
      ms_log (2, "%s: Unlikely number of samples (>20000): '%d'\n", srcname, fsdh->numsamples);
      retval++;
    }
  
  /* Sanity check that there is space for blockettes when both data and blockettes are present */
  if ( fsdh->numsamples > 0 && fsdh->numblockettes > 0 && fsdh->data_offset <= fsdh->blockette_offset )
    {
      ms_log (2, "%s: No space for %d blockettes, data offset: %d, blockette offset: %d\n", srcname,
	      fsdh->numblockettes, fsdh->data_offset, fsdh->blockette_offset);
      retval++;
    }
  
  
  /* Print raw header details */
  if ( details >= 1 )
    {
      /* Determine nominal sample rate */
      nomsamprate = ms_nomsamprate (fsdh->samprate_fact, fsdh->samprate_mult);
  
      /* Print header values */
      ms_log (0, "RECORD -- %s\n", srcname);
      ms_log (0, "        sequence number: '%c%c%c%c%c%c'\n", fsdh->sequence_number[0], fsdh->sequence_number[1], fsdh->sequence_number[2],
	      fsdh->sequence_number[3], fsdh->sequence_number[4], fsdh->sequence_number[5]);
      ms_log (0, " data quality indicator: '%c'\n", fsdh->dataquality);
      if ( details > 0 )
        ms_log (0, "               reserved: '%c'\n", fsdh->reserved);
      ms_log (0, "           station code: '%c%c%c%c%c'\n", fsdh->station[0], fsdh->station[1], fsdh->station[2], fsdh->station[3], fsdh->station[4]);
      ms_log (0, "            location ID: '%c%c'\n", fsdh->location[0], fsdh->location[1]);
      ms_log (0, "          channel codes: '%c%c%c'\n", fsdh->channel[0], fsdh->channel[1], fsdh->channel[2]);
      ms_log (0, "           network code: '%c%c'\n", fsdh->network[0], fsdh->network[1]);
      ms_log (0, "             start time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", fsdh->start_time.year, fsdh->start_time.day,
	      fsdh->start_time.hour, fsdh->start_time.min, fsdh->start_time.sec, fsdh->start_time.fract, fsdh->start_time.unused);
      ms_log (0, "      number of samples: %d\n", fsdh->numsamples);
      ms_log (0, "     sample rate factor: %d  (%.10g samples per second)\n",
              fsdh->samprate_fact, nomsamprate);
      ms_log (0, " sample rate multiplier: %d\n", fsdh->samprate_mult);
      
      /* Print flag details if requested */
      if ( details > 1 )
        {
          /* Activity flags */
	  b = fsdh->act_flags;
	  ms_log (0, "         activity flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
		  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
		  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
	  if ( b & 0x01 ) ms_log (0, "                         [Bit 0] Calibration signals present\n");
	  if ( b & 0x02 ) ms_log (0, "                         [Bit 1] Time correction applied\n");
	  if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Beginning of an event, station trigger\n");
	  if ( b & 0x08 ) ms_log (0, "                         [Bit 3] End of an event, station detrigger\n");
	  if ( b & 0x10 ) ms_log (0, "                         [Bit 4] A positive leap second happened in this record\n");
	  if ( b & 0x20 ) ms_log (0, "                         [Bit 5] A negative leap second happened in this record\n");
	  if ( b & 0x40 ) ms_log (0, "                         [Bit 6] Event in progress\n");
	  if ( b & 0x80 ) ms_log (0, "                         [Bit 7] Undefined bit set\n");
	  
	  /* I/O and clock flags */
	  b = fsdh->io_flags;
	  ms_log (0, "    I/O and clock flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
		  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
		  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
	  if ( b & 0x01 ) ms_log (0, "                         [Bit 0] Station volume parity error possibly present\n");
	  if ( b & 0x02 ) ms_log (0, "                         [Bit 1] Long record read (possibly no problem)\n");
	  if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Short record read (record padded)\n");
	  if ( b & 0x08 ) ms_log (0, "                         [Bit 3] Start of time series\n");
	  if ( b & 0x10 ) ms_log (0, "                         [Bit 4] End of time series\n");
	  if ( b & 0x20 ) ms_log (0, "                         [Bit 5] Clock locked\n");
	  if ( b & 0x40 ) ms_log (0, "                         [Bit 6] Undefined bit set\n");
	  if ( b & 0x80 ) ms_log (0, "                         [Bit 7] Undefined bit set\n");
	  
	  /* Data quality flags */
	  b = fsdh->dq_flags;
	  ms_log (0, "     data quality flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
		  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
		  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
	  if ( b & 0x01 ) ms_log (0, "                         [Bit 0] Amplifier saturation detected\n");
	  if ( b & 0x02 ) ms_log (0, "                         [Bit 1] Digitizer clipping detected\n");
	  if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Spikes detected\n");
	  if ( b & 0x08 ) ms_log (0, "                         [Bit 3] Glitches detected\n");
	  if ( b & 0x10 ) ms_log (0, "                         [Bit 4] Missing/padded data present\n");
	  if ( b & 0x20 ) ms_log (0, "                         [Bit 5] Telemetry synchronization error\n");
	  if ( b & 0x40 ) ms_log (0, "                         [Bit 6] A digital filter may be charging\n");
	  if ( b & 0x80 ) ms_log (0, "                         [Bit 7] Time tag is questionable\n");
	}
      
      ms_log (0, "   number of blockettes: %d\n", fsdh->numblockettes);
      ms_log (0, "        time correction: %ld\n", (long int) fsdh->time_correct);
      ms_log (0, "            data offset: %d\n", fsdh->data_offset);
      ms_log (0, " first blockette offset: %d\n", fsdh->blockette_offset);
    } /* Done printing raw header details */
  
  
  /* Validate and report information in the blockette chain */
  if ( fsdh->blockette_offset > 46 && fsdh->blockette_offset < maxreclen )
    {
      int blkt_offset = fsdh->blockette_offset;
      int blkt_count = 0;
      int blkt_length;
      uint16_t blkt_type;
      uint16_t next_blkt;
      char *blkt_desc;
      
      /* Traverse blockette chain */
      while ( blkt_offset != 0 && blkt_offset < maxreclen )
	{
	  /* Every blockette has a similar 4 byte header: type and next */
	  memcpy (&blkt_type, record + blkt_offset, 2);
	  memcpy (&next_blkt, record + blkt_offset+2, 2);
	  
	  if ( swapflag )
	    {
	      ms_gswap2 (&blkt_type);
	      ms_gswap2 (&next_blkt);
	    }
	  
	  /* Print common header fields */
	  if ( details >= 1 )
	    {
	      blkt_desc =  ms_blktdesc(blkt_type);
	      ms_log (0, "          BLOCKETTE %u: (%s)\n", blkt_type, (blkt_desc) ? blkt_desc : "Unknown");
	      ms_log (0, "              next blockette: %u\n", next_blkt);
	    }
	  
	  blkt_length = ms_blktlen (blkt_type, record + blkt_offset, swapflag);
	  if ( blkt_length == 0 )
	    {
	      ms_log (2, "%s: Unknown blockette length for type %d\n", srcname, blkt_type);
	      retval++;
	    }
	  
	  /* Track end of blockette chain */
	  endofblockettes = blkt_offset + blkt_length - 1;
	  
	  /* Sanity check that the blockette is contained in the record */
	  if ( endofblockettes > maxreclen )
	    {
	      ms_log (2, "%s: Blockette type %d at offset %d with length %d does not fix in record (%d)\n",
		      srcname, blkt_type, blkt_offset, blkt_length, maxreclen);
	      retval++;
	      break;
	    }
	  
	  if ( blkt_type == 100 )
	    {
	      struct blkt_100_s *blkt_100 = (struct blkt_100_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		ms_gswap4 (&blkt_100->samprate);
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "          actual sample rate: %.10g\n", blkt_100->samprate);
		  
		  if ( details > 1 )
		    {
		      b = blkt_100->flags;
		      ms_log (0, "             undefined flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			      bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			      bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		      
		      ms_log (0, "          reserved bytes (3): %u,%u,%u\n",
			      blkt_100->reserved[0], blkt_100->reserved[1], blkt_100->reserved[2]);
		    }
		}
	    }
	  
	  else if ( blkt_type == 200 )
	    {
	      struct blkt_200_s *blkt_200 = (struct blkt_200_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  ms_gswap4 (&blkt_200->amplitude);
		  ms_gswap4 (&blkt_200->period);
		  ms_gswap4 (&blkt_200->background_estimate);
		  MS_SWAPBTIME (&blkt_200->time);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "            signal amplitude: %g\n", blkt_200->amplitude);
		  ms_log (0, "               signal period: %g\n", blkt_200->period);
		  ms_log (0, "         background estimate: %g\n", blkt_200->background_estimate);
		  
		  if ( details > 1 )
		    {
		      b = blkt_200->flags;
		      ms_log (0, "       event detection flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			      bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			      bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		      if ( b & 0x01 ) ms_log (0, "                         [Bit 0] 1: Dilatation wave\n");
		      else            ms_log (0, "                         [Bit 0] 0: Compression wave\n");
		      if ( b & 0x02 ) ms_log (0, "                         [Bit 1] 1: Units after deconvolution\n");
		      else            ms_log (0, "                         [Bit 1] 0: Units are digital counts\n");
		      if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Bit 0 is undetermined\n");
		      ms_log (0, "               reserved byte: %u\n", blkt_200->reserved);
		    }
		  
		  ms_log (0, "           signal onset time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_200->time.year, blkt_200->time.day,
			  blkt_200->time.hour, blkt_200->time.min, blkt_200->time.sec, blkt_200->time.fract, blkt_200->time.unused);
		  ms_log (0, "               detector name: %.24s\n", blkt_200->detector);
		}
	    }
	  
	  else if ( blkt_type == 201 )
	    {
	      struct blkt_201_s *blkt_201 = (struct blkt_201_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  ms_gswap4 (&blkt_201->amplitude);
		  ms_gswap4 (&blkt_201->period);
		  ms_gswap4 (&blkt_201->background_estimate);
		  MS_SWAPBTIME (&blkt_201->time);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "            signal amplitude: %g\n", blkt_201->amplitude);
		  ms_log (0, "               signal period: %g\n", blkt_201->period);
		  ms_log (0, "         background estimate: %g\n", blkt_201->background_estimate);
		  
		  b = blkt_201->flags;
		  ms_log (0, "       event detection flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		  if ( b & 0x01 ) ms_log (0, "                         [Bit 0] 1: Dilation wave\n");
		  else            ms_log (0, "                         [Bit 0] 0: Compression wave\n");
		  
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_201->reserved);
		  ms_log (0, "           signal onset time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_201->time.year, blkt_201->time.day,
			  blkt_201->time.hour, blkt_201->time.min, blkt_201->time.sec, blkt_201->time.fract, blkt_201->time.unused);
		  ms_log (0, "                  SNR values: ");
		  for (idx=0; idx < 6; idx++) ms_log (0, "%u  ", blkt_201->snr_values[idx]);
		  ms_log (0, "\n");
		  ms_log (0, "              loopback value: %u\n", blkt_201->loopback);
		  ms_log (0, "              pick algorithm: %u\n", blkt_201->pick_algorithm);
		  ms_log (0, "               detector name: %.24s\n", blkt_201->detector);
		}
	    }
	  
	  else if ( blkt_type == 300 )
	    {
	      struct blkt_300_s *blkt_300 = (struct blkt_300_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  MS_SWAPBTIME (&blkt_300->time);
		  ms_gswap4 (&blkt_300->step_duration);
		  ms_gswap4 (&blkt_300->interval_duration);
		  ms_gswap4 (&blkt_300->amplitude);
		  ms_gswap4 (&blkt_300->reference_amplitude);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "      calibration start time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_300->time.year, blkt_300->time.day,
			  blkt_300->time.hour, blkt_300->time.min, blkt_300->time.sec, blkt_300->time.fract, blkt_300->time.unused);
		  ms_log (0, "      number of calibrations: %u\n", blkt_300->numcalibrations);
		  
		  b = blkt_300->flags;
		  ms_log (0, "           calibration flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		  if ( b & 0x01 ) ms_log (0, "                         [Bit 0] First pulse is positive\n");
		  if ( b & 0x02 ) ms_log (0, "                         [Bit 1] Calibration's alternate sign\n");
		  if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Calibration was automatic\n");
		  if ( b & 0x08 ) ms_log (0, "                         [Bit 3] Calibration continued from previous record(s)\n");
		  
		  ms_log (0, "               step duration: %u\n", blkt_300->step_duration);
		  ms_log (0, "           interval duration: %u\n", blkt_300->interval_duration);
		  ms_log (0, "            signal amplitude: %g\n", blkt_300->amplitude);
		  ms_log (0, "        input signal channel: %.3s", blkt_300->input_channel);
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_300->reserved);
		  ms_log (0, "         reference amplitude: %u\n", blkt_300->reference_amplitude);
		  ms_log (0, "                    coupling: %.12s\n", blkt_300->coupling);
		  ms_log (0, "                     rolloff: %.12s\n", blkt_300->rolloff);
		}
	    }
	  
	  else if ( blkt_type == 310 )
	    {
	      struct blkt_310_s *blkt_310 = (struct blkt_310_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  MS_SWAPBTIME (&blkt_310->time);
		  ms_gswap4 (&blkt_310->duration);
		  ms_gswap4 (&blkt_310->period);
		  ms_gswap4 (&blkt_310->amplitude);
		  ms_gswap4 (&blkt_310->reference_amplitude);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "      calibration start time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_310->time.year, blkt_310->time.day,
			  blkt_310->time.hour, blkt_310->time.min, blkt_310->time.sec, blkt_310->time.fract, blkt_310->time.unused);
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_310->reserved1);
		  
		  b = blkt_310->flags;
		  ms_log (0, "           calibration flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		  if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Calibration was automatic\n");
		  if ( b & 0x08 ) ms_log (0, "                         [Bit 3] Calibration continued from previous record(s)\n");
		  if ( b & 0x10 ) ms_log (0, "                         [Bit 4] Peak-to-peak amplitude\n");
		  if ( b & 0x20 ) ms_log (0, "                         [Bit 5] Zero-to-peak amplitude\n");
		  if ( b & 0x40 ) ms_log (0, "                         [Bit 6] RMS amplitude\n");
		  
		  ms_log (0, "        calibration duration: %u\n", blkt_310->duration);
		  ms_log (0, "               signal period: %g\n", blkt_310->period);
		  ms_log (0, "            signal amplitude: %g\n", blkt_310->amplitude);
		  ms_log (0, "        input signal channel: %.3s", blkt_310->input_channel);
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_310->reserved2);	      
		  ms_log (0, "         reference amplitude: %u\n", blkt_310->reference_amplitude);
		  ms_log (0, "                    coupling: %.12s\n", blkt_310->coupling);
		  ms_log (0, "                     rolloff: %.12s\n", blkt_310->rolloff);
		}
	    }
	  
	  else if ( blkt_type == 320 )
	    {
	      struct blkt_320_s *blkt_320 = (struct blkt_320_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  MS_SWAPBTIME (&blkt_320->time);
		  ms_gswap4 (&blkt_320->duration);
		  ms_gswap4 (&blkt_320->ptp_amplitude);
		  ms_gswap4 (&blkt_320->reference_amplitude);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "      calibration start time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_320->time.year, blkt_320->time.day,
			  blkt_320->time.hour, blkt_320->time.min, blkt_320->time.sec, blkt_320->time.fract, blkt_320->time.unused);
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_320->reserved1);
		  
		  b = blkt_320->flags;
		  ms_log (0, "           calibration flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		  if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Calibration was automatic\n");
		  if ( b & 0x08 ) ms_log (0, "                         [Bit 3] Calibration continued from previous record(s)\n");
		  if ( b & 0x10 ) ms_log (0, "                         [Bit 4] Random amplitudes\n");
		  
		  ms_log (0, "        calibration duration: %u\n", blkt_320->duration);
		  ms_log (0, "      peak-to-peak amplitude: %g\n", blkt_320->ptp_amplitude);
		  ms_log (0, "        input signal channel: %.3s", blkt_320->input_channel);
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_320->reserved2);
		  ms_log (0, "         reference amplitude: %u\n", blkt_320->reference_amplitude);
		  ms_log (0, "                    coupling: %.12s\n", blkt_320->coupling);
		  ms_log (0, "                     rolloff: %.12s\n", blkt_320->rolloff);
		  ms_log (0, "                  noise type: %.8s\n", blkt_320->noise_type);
		}
	    }
	  
	  else if ( blkt_type == 390 )
	    {
	      struct blkt_390_s *blkt_390 = (struct blkt_390_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  MS_SWAPBTIME (&blkt_390->time);
		  ms_gswap4 (&blkt_390->duration);
		  ms_gswap4 (&blkt_390->amplitude);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "      calibration start time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_390->time.year, blkt_390->time.day,
			  blkt_390->time.hour, blkt_390->time.min, blkt_390->time.sec, blkt_390->time.fract, blkt_390->time.unused);
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_390->reserved1);
		  
		  b = blkt_390->flags;
		  ms_log (0, "           calibration flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		  if ( b & 0x04 ) ms_log (0, "                         [Bit 2] Calibration was automatic\n");
		  if ( b & 0x08 ) ms_log (0, "                         [Bit 3] Calibration continued from previous record(s)\n");
		  
		  ms_log (0, "        calibration duration: %u\n", blkt_390->duration);
		  ms_log (0, "            signal amplitude: %g\n", blkt_390->amplitude);
		  ms_log (0, "        input signal channel: %.3s", blkt_390->input_channel);
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_390->reserved2);
		}
	    }

	  else if ( blkt_type == 395 )
	    {
	      struct blkt_395_s *blkt_395 = (struct blkt_395_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		MS_SWAPBTIME (&blkt_395->time);
	      
	      if ( details >= 1 )
		{ 
		  ms_log (0, "        calibration end time: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_395->time.year, blkt_395->time.day,
			  blkt_395->time.hour, blkt_395->time.min, blkt_395->time.sec, blkt_395->time.fract, blkt_395->time.unused);
		  if ( details > 1 )
		    ms_log (0, "          reserved bytes (2): %u,%u\n",
			    blkt_395->reserved[0], blkt_395->reserved[1]);
		}
	    }
	  
	  else if ( blkt_type == 400 )
	    {
	      struct blkt_400_s *blkt_400 = (struct blkt_400_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  ms_gswap4 (&blkt_400->azimuth);
		  ms_gswap4 (&blkt_400->slowness);
		  ms_gswap4 (&blkt_400->configuration);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "      beam azimuth (degrees): %g\n", blkt_400->azimuth);
		  ms_log (0, "  beam slowness (sec/degree): %g\n", blkt_400->slowness);
		  ms_log (0, "               configuration: %u\n", blkt_400->configuration);
		  if ( details > 1 )
		    ms_log (0, "          reserved bytes (2): %u,%u\n",
			    blkt_400->reserved[0], blkt_400->reserved[1]);
		}
	    }

	  else if ( blkt_type == 405 )
	    {
	      struct blkt_405_s *blkt_405 = (struct blkt_405_s *) (record + blkt_offset + 4);
	      uint16_t firstvalue = blkt_405->delay_values[0];  /* Work on a private copy */
	      
	      if ( swapflag )
		ms_gswap2 (&firstvalue);
	      
	      if ( details >= 1 )
		ms_log (0, "           first delay value: %u\n", firstvalue);
	    }
	  
	  else if ( blkt_type == 500 )
	    {
	      struct blkt_500_s *blkt_500 = (struct blkt_500_s *) (record + blkt_offset + 4);
	      
	      if ( swapflag )
		{
		  ms_gswap4 (&blkt_500->vco_correction);
		  MS_SWAPBTIME (&blkt_500->time);
		  ms_gswap4 (&blkt_500->exception_count);
		}
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "              VCO correction: %g%%\n", blkt_500->vco_correction);
		  ms_log (0, "           time of exception: %d,%d,%d:%d:%d.%04d (unused: %d)\n", blkt_500->time.year, blkt_500->time.day,
			  blkt_500->time.hour, blkt_500->time.min, blkt_500->time.sec, blkt_500->time.fract, blkt_500->time.unused);
		  ms_log (0, "                        usec: %d\n", blkt_500->usec);
		  ms_log (0, "           reception quality: %u%%\n", blkt_500->reception_qual);
		  ms_log (0, "             exception count: %u\n", blkt_500->exception_count);
		  ms_log (0, "              exception type: %.16s\n", blkt_500->exception_type);
		  ms_log (0, "                 clock model: %.32s\n", blkt_500->clock_model);
		  ms_log (0, "                clock status: %.128s\n", blkt_500->clock_status);
		}
	    }
	  
	  else if ( blkt_type == 1000 )
	    {
	      struct blkt_1000_s *blkt_1000 = (struct blkt_1000_s *) (record + blkt_offset + 4);
	      char order[40];
	      
	      /* Calculate record size in bytes as 2^(blkt_1000->rec_len) */
	      b1000reclen = (unsigned int) 1 << blkt_1000->reclen;
	      
	      /* Big or little endian? */
	      if (blkt_1000->byteorder == 0)
		strncpy (order, "Little endian", sizeof(order)-1);
	      else if (blkt_1000->byteorder == 1)
		strncpy (order, "Big endian", sizeof(order)-1);
	      else
		strncpy (order, "Unknown value", sizeof(order)-1);
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "                    encoding: %s (val:%u)\n",
			  (char *) ms_encodingstr (blkt_1000->encoding), blkt_1000->encoding);
		  ms_log (0, "                  byte order: %s (val:%u)\n",
			  order, blkt_1000->byteorder);
		  ms_log (0, "               record length: %d (val:%u)\n",
			  b1000reclen, blkt_1000->reclen);
		  
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_1000->reserved);
		}
	      
	      /* Save encoding format */
	      b1000encoding = blkt_1000->encoding;
	      
	      /* Sanity check encoding format */
	      if ( ! (b1000encoding >= 0 && b1000encoding <= 5) &&
		   ! (b1000encoding >= 10 && b1000encoding <= 19) &&
		   ! (b1000encoding >= 30 && b1000encoding <= 33) )
		{
		  ms_log (2, "%s: Blockette 1000 encoding format invalid (0-5,10-19,30-33): %d\n", srcname, b1000encoding);
		  retval++;
		}
	      
	      /* Sanity check byte order flag */
	      if ( blkt_1000->byteorder != 0 && blkt_1000->byteorder != 1 )
		{
		  ms_log (2, "%s: Blockette 1000 byte order flag invalid (0 or 1): %d\n", srcname, blkt_1000->byteorder);
		  retval++;
		}
	    }
	  
	  else if ( blkt_type == 1001 )
	    {
	      struct blkt_1001_s *blkt_1001 = (struct blkt_1001_s *) (record + blkt_offset + 4);
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "              timing quality: %u%%\n", blkt_1001->timing_qual);
		  ms_log (0, "                micro second: %d\n", blkt_1001->usec);
		  
		  if ( details > 1 )
		    ms_log (0, "               reserved byte: %u\n", blkt_1001->reserved);
		  
		  ms_log (0, "                 frame count: %u\n", blkt_1001->framecnt);
		}
	    }
	  
	  else if ( blkt_type == 2000 )
	    {
	      struct blkt_2000_s *blkt_2000 = (struct blkt_2000_s *) (record + blkt_offset + 4);
	      char order[40];
	      
	      if ( swapflag )
		{
		  ms_gswap2 (&blkt_2000->length);
		  ms_gswap2 (&blkt_2000->data_offset);
		  ms_gswap4 (&blkt_2000->recnum);
		}
	      
	      /* Big or little endian? */
	      if (blkt_2000->byteorder == 0)
		strncpy (order, "Little endian", sizeof(order)-1);
	      else if (blkt_2000->byteorder == 1)
		strncpy (order, "Big endian", sizeof(order)-1);
	      else
		strncpy (order, "Unknown value", sizeof(order)-1);
	      
	      if ( details >= 1 )
		{
		  ms_log (0, "            blockette length: %u\n", blkt_2000->length);
		  ms_log (0, "                 data offset: %u\n", blkt_2000->data_offset);
		  ms_log (0, "               record number: %u\n", blkt_2000->recnum);
		  ms_log (0, "                  byte order: %s (val:%u)\n",
			  order, blkt_2000->byteorder);
		  b = blkt_2000->flags;
		  ms_log (0, "                  data flags: [%u%u%u%u%u%u%u%u] 8 bits\n",
			  bit(b,0x01), bit(b,0x02), bit(b,0x04), bit(b,0x08),
			  bit(b,0x10), bit(b,0x20), bit(b,0x40), bit(b,0x80));
		  
		  if ( details > 1 )
		    {
		      if ( b & 0x01 ) ms_log (0, "                         [Bit 0] 1: Stream oriented\n");
		      else            ms_log (0, "                         [Bit 0] 0: Record oriented\n");
		      if ( b & 0x02 ) ms_log (0, "                         [Bit 1] 1: Blockette 2000s may NOT be packaged\n");
		      else            ms_log (0, "                         [Bit 1] 0: Blockette 2000s may be packaged\n");
		      if ( ! (b & 0x04) && ! (b & 0x08) )
			ms_log (0, "                      [Bits 2-3] 00: Complete blockette\n");
		      else if ( ! (b & 0x04) && (b & 0x08) )
			ms_log (0, "                      [Bits 2-3] 01: First blockette in span\n");
		      else if ( (b & 0x04) && (b & 0x08) )
			ms_log (0, "                      [Bits 2-3] 11: Continuation blockette in span\n");
		      else if ( (b & 0x04) && ! (b & 0x08) )
			ms_log (0, "                      [Bits 2-3] 10: Final blockette in span\n");
		      if ( ! (b & 0x10) && ! (b & 0x20) )
			ms_log (0, "                      [Bits 4-5] 00: Not file oriented\n");
		      else if ( ! (b & 0x10) && (b & 0x20) )
			ms_log (0, "                      [Bits 4-5] 01: First blockette of file\n");
		      else if ( (b & 0x10) && ! (b & 0x20) )
			ms_log (0, "                      [Bits 4-5] 10: Continuation of file\n");
		      else if ( (b & 0x10) && (b & 0x20) )
			ms_log (0, "                      [Bits 4-5] 11: Last blockette of file\n");
		    }
		  
		  ms_log (0, "           number of headers: %u\n", blkt_2000->numheaders);
		  
		  /* Crude display of the opaque data headers */
		  if ( details > 1 )
		    ms_log (0, "                     headers: %.*s\n",
			    (blkt_2000->data_offset - 15), blkt_2000->payload);
		}
	    }
	  
	  else
	    {
	      ms_log (2, "%s: Unrecognized blockette type: %d\n", srcname, blkt_type);
	      retval++;
	    }
	  
	  /* Sanity check the next blockette offset */
	  if ( next_blkt && next_blkt <= endofblockettes )
	    {
	      ms_log (2, "%s: Next blockette offset (%d) is within current blockette ending at byte %d\n",
		      srcname, next_blkt, endofblockettes);
	      blkt_offset = 0;
	    }
	  else
	    {
	      blkt_offset = next_blkt;
	    }
	  
	  blkt_count++;
	} /* End of looping through blockettes */
      
      /* Check that the blockette offset is within the maximum record size */
      if ( blkt_offset > maxreclen )
	{
	  ms_log (2, "%s: Blockette offset (%d) beyond maximum record length (%d)\n", srcname, blkt_offset, maxreclen);
	  retval++;
	}
      
      /* Check that the data and blockette offsets are within the record */
      if ( b1000reclen && fsdh->data_offset > b1000reclen )
	{
	  ms_log (2, "%s: Data offset (%d) beyond record length (%d)\n", srcname, fsdh->data_offset, b1000reclen);
	  retval++;
	}
      if ( b1000reclen && fsdh->blockette_offset > b1000reclen )
	{
	  ms_log (2, "%s: Blockette offset (%d) beyond record length (%d)\n", srcname, fsdh->blockette_offset, b1000reclen);
	  retval++;
	}
      
      /* Check that the data offset is beyond the end of the blockettes */
      if ( fsdh->numsamples && fsdh->data_offset <= endofblockettes )
	{
	  ms_log (2, "%s: Data offset (%d) is within blockette chain (end of blockettes: %d)\n", srcname, fsdh->data_offset, endofblockettes);
	  retval++;
	}
      
      /* Check that the correct number of blockettes were parsed */
      if ( fsdh->numblockettes != blkt_count )
	{
	  ms_log (2, "%s: Specified number of blockettes (%d) not equal to those parsed (%d)\n", srcname, fsdh->numblockettes, blkt_count);
	  retval++;
	}
    }
  
  return retval;
} /* End of ms_parse_raw() */
