/***************************************************************************
 * lmplatform.h:
 * 
 * Platform specific headers.  This file provides a basic level of platform
 * portability.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public License
 * as published by the Free Software Foundation; either version 2 of
 * the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License (GNU-LGPL) for more details.  The
 * GNU-LGPL and further information can be found here:
 * http://www.gnu.org/
 *
 * Written by Chad Trabant, IRIS Data Management Center
 *
 * modified: 2010.304
 ***************************************************************************/

#ifndef LMPLATFORM_H
#define LMPLATFORM_H 1

#ifdef __cplusplus
extern "C" {
#endif

  /* On some platforms (e.g. ARM) structures are aligned on word boundaries
     by adding padding between the elements.  This library uses structs that
     map to SEED header/blockette structures that are required to have a
     layout exactly as specified, i.e. no padding.

     If "ATTRIBUTE_PACKED" is defined at compile time (e.g. -DATTRIBUTE_PACKED)
     the preprocessor will use the define below to add the "packed" attribute 
     to effected structs.  This attribute is supported by GCC and increasingly
     more compilers.
  */
#if defined(ATTRIBUTE_PACKED)
  #define LMP_PACKED __attribute__((packed))
#else
  #define LMP_PACKED
#endif

  /* Make some guesses about the system libraries based
   * on the architecture.  Currently the assumptions are:
   * Linux => glibc2 libraries (LMP_GLIBC2)
   * Sun => Solaris libraties (LMP_SOLARIS)
   * BSD => BSD libraries, including Apple Mac OS X (LMP_BSD)
   * WIN32 => WIN32 and Windows Sockets 2 (LMP_WIN32)
   */

#if defined(__linux__) || defined(__linux)
  #define LMP_GLIBC2 1

  #include <stdlib.h>
  #include <stdio.h>
  #include <unistd.h>
  #include <stdarg.h>
  #include <inttypes.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netdb.h>
  #include <sys/time.h>
  #include <string.h>
  #include <ctype.h>
  #include <features.h>
  
#elif defined(__sun__) || defined(__sun)
  #define LMP_SOLARIS 1

  #include <stdlib.h>
  #include <stdio.h>
  #include <unistd.h>
  #include <stdarg.h>
  #include <inttypes.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netdb.h>
  #include <sys/time.h>
  #include <string.h>
  #include <ctype.h>
  
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
  #define LMP_BSD 1

  #include <stdlib.h>
  #include <stdio.h>
  #include <unistd.h>
  #include <stdarg.h>
  #include <inttypes.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netdb.h>
  #include <sys/time.h>
  #include <string.h>
  #include <ctype.h>

#elif defined(WIN32) || defined(WIN64)
  #define LMP_WIN32 1

  #include <windows.h>
  #include <stdarg.h>
  #include <winsock.h>
  #include <stdio.h>
  #include <sys/types.h>
  #include <ctype.h>

  #if defined(_MSC_VER)
    #define snprintf _snprintf
    #define vsnprintf _vsnprintf
    #define strcasecmp _stricmp
    #define strncasecmp _strnicmp
    #define strtoull _strtoui64
    #define strdup _strdup
    #define fileno _fileno
  #endif

  typedef signed char int8_t;
  typedef unsigned char uint8_t;
  typedef signed short int int16_t;
  typedef unsigned short int uint16_t;
  typedef signed int int32_t;
  typedef unsigned int uint32_t;
  typedef signed __int64 int64_t;
  typedef unsigned __int64 uint64_t;

#else
  #include <stdlib.h>
  #include <stdio.h>
  #include <unistd.h>
  #include <stdarg.h>
  #include <inttypes.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netdb.h>
  #include <sys/time.h>
  #include <string.h>
  #include <ctype.h>

  typedef signed char int8_t;
  typedef unsigned char uint8_t;
  typedef signed short int int16_t;
  typedef unsigned short int uint16_t;
  typedef signed int int32_t;
  typedef unsigned int uint32_t;
  typedef signed long long int64_t;
  typedef unsigned long long uint64_t;

#endif

extern off_t lmp_ftello (FILE *stream);
extern int lmp_fseeko (FILE *stream, off_t offset, int whence);

#ifdef __cplusplus
}
#endif
 
#endif /* LMPLATFORM_H */
