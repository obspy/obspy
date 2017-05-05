/***************************************************************************
 * lmplatform.c:
 *
 * Platform portability routines.
 *
 * modified: 2017.118
 ***************************************************************************/

/* Define _LARGEFILE_SOURCE to get ftello/fseeko on some systems (Linux) */
#define _LARGEFILE_SOURCE 1

#include "libmseed.h"

/* Size of off_t data type as determined at build time */
int LM_SIZEOF_OFF_T = sizeof(off_t);

/***************************************************************************
 * lmp_ftello:
 *
 * Return the current file position for the specified descriptor using
 * the system's closest match to the POSIX ftello.
 ***************************************************************************/
off_t
lmp_ftello (FILE *stream)
{
#if defined(LMP_WIN)
  return (off_t)ftell (stream);

#else
  return (off_t)ftello (stream);

#endif
} /* End of lmp_ftello() */

/***************************************************************************
 * lmp_fseeko:
 *
 * Seek to a specific file position for the specified descriptor using
 * the system's closest match to the POSIX fseeko.
 ***************************************************************************/
int
lmp_fseeko (FILE *stream, off_t offset, int whence)
{
#if defined(LMP_WIN)
  return (int)fseek (stream, (long int)offset, whence);

#else
  return (int)fseeko (stream, offset, whence);

#endif
} /* End of lmp_fseeko() */
