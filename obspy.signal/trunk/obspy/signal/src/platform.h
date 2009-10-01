#ifndef PLATFORM_H
#define PLATFORM_H 1

#ifdef __cplusplus
extern "C" {
#endif
  
#if defined(__linux__) || defined(__linux)
  #include <values.h>
  #include <malloc.h>
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
  #include <limits.h>
  #include <sys/malloc.h>
#elif defined(WIN32)
  #include <limits.h>
  #include <malloc.h>
#else
  #include <limits.h>
  #include <malloc.h>
#endif


#ifdef __cplusplus
}
#endif
 
#endif
