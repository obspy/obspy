/***************************************************************************
 * lmplatform.h:
 *
 * Platform specific headers.  This file provides a basic level of platform
 * portability.
 *
 * modified: 2016.275
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

/* C99 standard headers */
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>
#include <ctype.h>

/* Set architecture specific defines and features */
#if defined(__linux__) || defined(__linux) || defined(__CYGWIN__)
  #define LMP_LINUX 1
  #define LMP_GLIBC2 1 /* Deprecated */

  #include <unistd.h>
  #include <inttypes.h>

#elif defined(__sun__) || defined(__sun)
  #define LMP_SOLARIS 1

  #include <unistd.h>
  #include <inttypes.h>

#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
  #define LMP_BSD 1

  #include <unistd.h>
  #include <inttypes.h>

#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  #define LMP_WIN 1
  #define LMP_WIN32 1 /* Deprecated */

  #include <windows.h>
  #include <sys/types.h>

  /* For MSVC 2012 and earlier define standard int types, otherwise use inttypes.h */
  #if defined(_MSC_VER) && _MSC_VER <= 1700
    typedef signed char int8_t;
    typedef unsigned char uint8_t;
    typedef signed short int int16_t;
    typedef unsigned short int uint16_t;
    typedef signed int int32_t;
    typedef unsigned int uint32_t;
    typedef signed __int64 int64_t;
    typedef unsigned __int64 uint64_t;
  #else
    #include <inttypes.h>
  #endif

  #if defined(_MSC_VER)
    #if !defined(PRId64)
      #define PRId64 "I64d"
    #endif
    #if !defined(SCNd64)
      #define SCNd64 "I64d"
    #endif

    #define snprintf _snprintf
    #define vsnprintf _vsnprintf
    #define strcasecmp _stricmp
    #define strncasecmp _strnicmp
    #define strtoull _strtoui64
    #define strdup _strdup
    #define fileno _fileno
  #endif

  #if defined(__MINGW32__) || defined(__MINGW64__)
    #include <fcntl.h>

    #define fstat _fstat
    #define stat _stat
  #endif

#endif

extern off_t lmp_ftello (FILE *stream);
extern int lmp_fseeko (FILE *stream, off_t offset, int whence);

#ifdef __cplusplus
}
#endif

#endif /* LMPLATFORM_H */
