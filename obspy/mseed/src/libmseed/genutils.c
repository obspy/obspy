/***************************************************************************
 * genutils.c
 *
 * Generic utility routines
 *
 * Written by Chad Trabant
 * ORFEUS/EC-Project MEREDIAN
 * IRIS Data Management Center
 *
 * modified: 2013.053
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "lmplatform.h"
#include "libmseed.h"

static hptime_t ms_time2hptime_int (int year, int day, int hour,
				    int min, int sec, int usec);

static struct tm *ms_gmtime_r (int64_t *timep, struct tm *result);


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
      ms_log (2, "ms_splitsrcname(): Error duplicating srcname string");
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
 * ms_strncpcleantail:
 *
 * Copy up to 'length' characters from 'source' to 'dest' without any
 * trailing spaces.  The result is left justified and always null
 * terminated.  The destination string must have enough room needed
 * for the characters within 'length' and the null terminator, a
 * maximum of 'length + 1'.
 *
 * Returns the number of characters (not including the null terminator) in
 * the destination string.
 ***************************************************************************/
int
ms_strncpcleantail (char *dest, const char *source, int length)
{
  int idx, pretail;
  
  if ( ! dest )
    return 0;
  
  if ( ! source )
    {
      *dest = '\0';
      return 0;
    }
  
  *(dest+length) = '\0';
  
  pretail = 0;
  for ( idx=length-1; idx >= 0 ; idx-- )
    {
      if ( ! pretail && *(source+idx) == ' ' )
	{
	  *(dest+idx) = '\0';
	}
      else
	{
	  pretail++;
	  *(dest+idx) = *(source+idx);
	}
    }
  
  return pretail;
}  /* End of ms_strncpcleantail() */


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
 * Year is expected to be in the range 1800-5000, jday is expected to
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
  if ( year < 1800 || year > 5000 )
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
 * Year is expected to be in the range 1800-5000, month is expected to
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
  if ( year < 1800 || year > 5000 )
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
  
  hptime = (hptime_t ) (60 * (60 * ((hptime_t) 24 * days + btime->hour) + btime->min) + btime->sec) * HPTMODULUS
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
  struct tm tms;
  int64_t isec;
  int ifract;
  int bfract;
  
  if ( btime == NULL )
    return -1;
  
  /* Reduce to Unix/POSIX epoch time and fractional seconds */
  isec = MS_HPTIME2EPOCH(hptime);
  ifract = (int)(hptime - (isec * HPTMODULUS));
  
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

  if ( ! (ms_gmtime_r (&isec, &tms)) )
    return -1;
  
  btime->year   = tms.tm_year + 1900;
  btime->day    = tms.tm_yday + 1;
  btime->hour   = tms.tm_hour;
  btime->min    = tms.tm_min;
  btime->sec    = tms.tm_sec;
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
  struct tm tms;
  int64_t isec;
  int ifract;
  int ret;

  if ( isotimestr == NULL )
    return NULL;

  /* Reduce to Unix/POSIX epoch time and fractional seconds */
  isec = MS_HPTIME2EPOCH(hptime);
  ifract = (int)(hptime - (isec * HPTMODULUS));
  
  /* Adjust for negative epoch times */
  if ( hptime < 0 && ifract != 0 )
    {
      isec -= 1;
      ifract = HPTMODULUS - (-ifract);
    }
  
  if ( ! (ms_gmtime_r (&isec, &tms)) )
    return NULL;
  
  if ( subseconds )
    /* Assuming ifract has at least microsecond precision */
    ret = snprintf (isotimestr, 27, "%4d-%02d-%02dT%02d:%02d:%02d.%06d",
                    tms.tm_year + 1900, tms.tm_mon + 1, tms.tm_mday,
                    tms.tm_hour, tms.tm_min, tms.tm_sec, ifract);
  else
    ret = snprintf (isotimestr, 20, "%4d-%02d-%02dT%02d:%02d:%02d",
                    tms.tm_year + 1900, tms.tm_mon + 1, tms.tm_mday,
                    tms.tm_hour, tms.tm_min, tms.tm_sec);

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
  struct tm tms;
  int64_t isec;
  int ifract;
  int ret;
  
  if ( mdtimestr == NULL )
    return NULL;

  /* Reduce to Unix/POSIX epoch time and fractional seconds */
  isec = MS_HPTIME2EPOCH(hptime);
  ifract = (int)(hptime - (isec * HPTMODULUS));

  /* Adjust for negative epoch times */
  if ( hptime < 0 && ifract != 0 )
    {
      isec -= 1;
      ifract = HPTMODULUS - (-ifract);
    }
  
  if ( ! (ms_gmtime_r (&isec, &tms)) )
    return NULL;

  if ( subseconds )
    /* Assuming ifract has at least microsecond precision */
    ret = snprintf (mdtimestr, 27, "%4d-%02d-%02d %02d:%02d:%02d.%06d",
                    tms.tm_year + 1900, tms.tm_mon + 1, tms.tm_mday,
                    tms.tm_hour, tms.tm_min, tms.tm_sec, ifract);
  else
    ret = snprintf (mdtimestr, 20, "%4d-%02d-%02d %02d:%02d:%02d",
                    tms.tm_year + 1900, tms.tm_mon + 1, tms.tm_mday,
                    tms.tm_hour, tms.tm_min, tms.tm_sec);

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
  struct tm tms;
  int64_t isec;
  int ifract;
  int ret;
  
  if ( seedtimestr == NULL )
    return NULL;
  
  /* Reduce to Unix/POSIX epoch time and fractional seconds */
  isec = MS_HPTIME2EPOCH(hptime);
  ifract = (int)(hptime - (isec * HPTMODULUS));
  
  /* Adjust for negative epoch times */
  if ( hptime < 0 && ifract != 0 )
    {
      isec -= 1;
      ifract = HPTMODULUS - (-ifract);
    }
  
  if ( ! (ms_gmtime_r (&isec, &tms)) )
    return NULL;
  
  if ( subseconds )
    /* Assuming ifract has at least microsecond precision */
    ret = snprintf (seedtimestr, 25, "%4d,%03d,%02d:%02d:%02d.%06d",
		    tms.tm_year + 1900, tms.tm_yday + 1,
		    tms.tm_hour, tms.tm_min, tms.tm_sec, ifract);
  else
    ret = snprintf (seedtimestr, 18, "%4d,%03d,%02d:%02d:%02d",
                    tms.tm_year + 1900, tms.tm_yday + 1,
                    tms.tm_hour, tms.tm_min, tms.tm_sec);
  
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
 * year : 1800 - 5000
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
  if ( year < 1800 || year > 5000 )
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
 * Convert a SEED time string (day-of-year style) to a high precision
 * epoch time.  The time format expected is
 * "YYYY[,DDD,HH,MM,SS.FFFFFF]", the delimiter can be a dash [-],
 * comma [,], colon [:] or period [.].  Additionally a [T] or space
 * may be used to seprate the day and hour fields.  The fractional
 * seconds ("FFFFFF") must begin with a period [.] if present.
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
  
  fields = sscanf (seedtimestr, "%d%*[-,:.]%d%*[-,:.Tt ]%d%*[-,:.]%d%*[-,:.]%d%f",
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
  
  if ( year < 1800 || year > 5000 )
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
 * Convert a generic time string to a high precision epoch time.  The
 * time format expected is "YYYY[/MM/DD HH:MM:SS.FFFF]", the delimiter
 * can be a dash [-], comma[,], slash [/], colon [:], or period [.].
 * Additionally a 'T' or space may be used between the date and time
 * fields.  The fractional seconds ("FFFFFF") must begin with a period
 * [.] if present.
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
  
  fields = sscanf (timestr, "%d%*[-,/:.]%d%*[-,/:.]%d%*[-,/:.Tt ]%d%*[-,/:.]%d%*[-,/:.]%d%f",
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
  
  if ( year < 1800 || year > 5000 )
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
  if ( samprate > 32767.0 || samprate < 0.0 )
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
      ms_ratapprox (samprate, &num, &den, 32767, 1e-12);
      
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
 * ms_gmtime_r:
 *
 * An internal version of gmtime_r() that is 64-bit compliant and
 * works with years beyond 2038.
 *
 * The original was called pivotal_gmtime_r() by Paul Sheer, all
 * required copyright and other hoohas are below.  Modifications were
 * made to integrate the original to this code base, avoid name
 * collisions and formatting so I could read it.
 * 
 * Returns a pointer to the populated tm struct on success and NULL on error.
 ***************************************************************************/

/* pivotal_gmtime_r - a replacement for gmtime/localtime/mktime
                      that works around the 2038 bug on 32-bit
                      systems. (Version 4)

   Copyright (C) 2009  Paul Sheer

   Redistribution and use in source form, with or without modification,
   is permitted provided that the above copyright notice, this list of
   conditions, the following disclaimer, and the following char array
   are retained.

   Redistribution and use in binary form must reproduce an
   acknowledgment: 'With software provided by http://2038bug.com/' in
   the documentation and/or other materials provided with the
   distribution, and wherever such acknowledgments are usually
   accessible in Your program.

   This software is provided "AS IS" and WITHOUT WARRANTY, either
   express or implied, including, without limitation, the warranties of
   NON-INFRINGEMENT, MERCHANTABILITY or FITNESS FOR A PARTICULAR
   PURPOSE. THE ENTIRE RISK AS TO THE QUALITY OF THIS SOFTWARE IS WITH
   YOU. Under no circumstances and under no legal theory, whether in
   tort (including negligence), contract, or otherwise, shall the
   copyright owners be liable for any direct, indirect, special,
   incidental, or consequential damages of any character arising as a
   result of the use of this software including, without limitation,
   damages for loss of goodwill, work stoppage, computer failure or
   malfunction, or any and all other commercial damages or losses. This
   limitation of liability shall not apply to liability for death or
   personal injury resulting from copyright owners' negligence to the
   extent applicable law prohibits such limitation. Some jurisdictions
   do not allow the exclusion or limitation of incidental or
   consequential damages, so this exclusion and limitation may not apply
   to You.

*/

const char pivotal_gmtime_r_stamp_lm[] =
  "pivotal_gmtime_r. Copyright (C) 2009  Paul Sheer. Terms and "
  "conditions apply. Visit http://2038bug.com/ for more info.";

static const int tm_days[4][13] = {
  {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
  {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
  {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365},
  {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366},
};

#define TM_LEAP_CHECK(n) ((!(((n) + 1900) % 400) || (!(((n) + 1900) % 4) && (((n) + 1900) % 100))) != 0)
#define TM_WRAP(a,b,m)   ((a) = ((a) <  0  ) ? ((b)--, (a) + (m)) : (a))

static struct tm *
ms_gmtime_r (int64_t *timep, struct tm *result)
{
  int v_tm_sec, v_tm_min, v_tm_hour, v_tm_mon, v_tm_wday, v_tm_tday;
  int leap;
  long m;
  int64_t tv;
  
  if ( ! timep || ! result )
    return NULL;
  
  tv = *timep;
  
  v_tm_sec = ((int64_t) tv % (int64_t) 60);
  tv /= 60;
  v_tm_min = ((int64_t) tv % (int64_t) 60);
  tv /= 60;
  v_tm_hour = ((int64_t) tv % (int64_t) 24);
  tv /= 24;
  v_tm_tday = (int)tv;
  
  TM_WRAP (v_tm_sec, v_tm_min, 60);
  TM_WRAP (v_tm_min, v_tm_hour, 60);
  TM_WRAP (v_tm_hour, v_tm_tday, 24);
  
  if ( (v_tm_wday = (v_tm_tday + 4) % 7) < 0 )
    v_tm_wday += 7;
  
  m = (long) v_tm_tday;
  
  if ( m >= 0 )
    {
      result->tm_year = 70;
      leap = TM_LEAP_CHECK (result->tm_year);
      
      while ( m >= (long) tm_days[leap + 2][12] )
	{
	  m -= (long) tm_days[leap + 2][12];
	  result->tm_year++;
	  leap = TM_LEAP_CHECK (result->tm_year);
	}
      
      v_tm_mon = 0;
      
      while ( m >= (long) tm_days[leap][v_tm_mon] )
	{
	  m -= (long) tm_days[leap][v_tm_mon];
	  v_tm_mon++;
	}
    }
  else
    {
      result->tm_year = 69;
      leap = TM_LEAP_CHECK (result->tm_year);
      
      while ( m < (long) -tm_days[leap + 2][12] )
	{
	  m += (long) tm_days[leap + 2][12];
	  result->tm_year--;
	  leap = TM_LEAP_CHECK (result->tm_year);
	}
      
      v_tm_mon = 11;
      
      while ( m < (long) -tm_days[leap][v_tm_mon] )
	{
	  m += (long) tm_days[leap][v_tm_mon];
	  v_tm_mon--;
	}
      
      m += (long) tm_days[leap][v_tm_mon];
    }
  
  result->tm_mday = (int) m + 1;
  result->tm_yday = tm_days[leap + 2][v_tm_mon] + m;
  result->tm_sec = v_tm_sec;
  result->tm_min = v_tm_min;
  result->tm_hour = v_tm_hour;
  result->tm_mon = v_tm_mon;
  result->tm_wday = v_tm_wday;
  
  return result;
}  /* End of ms_gmtime_r() */
