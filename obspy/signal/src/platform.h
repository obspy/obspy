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
#elif defined(_WIN32) || defined(_WIN64)
  // Windows 32 & 64
  #include <limits.h>
  #include <malloc.h>
  #define _USE_MATH_DEFINES
  #include <math.h>
  #if defined(_MSC_VER)
    // MSC
    #define fmax max
    #define fmin min
  #endif
#else
  #include <limits.h>
  #include <malloc.h>
#endif


#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif


#ifdef __cplusplus
}
#endif
 
#endif
