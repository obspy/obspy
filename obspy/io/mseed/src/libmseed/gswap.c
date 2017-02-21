/***************************************************************************
 * gswap.c:
 *
 * Functions for generalized, in-place byte swapping between LSBF and
 * MSBF byte orders.
 *
 * Some standard integer types are needed, namely uint8_t and
 * uint32_t, (these are normally declared by including inttypes.h or
 * stdint.h).  Each function expects it's input to be a void pointer
 * to a quantity of the appropriate size.
 *
 * There are two versions of most routines, one that works on
 * quantities regardless of alignment (gswapX) and one that works on
 * memory aligned quantities (gswapXa).  The memory aligned versions
 * (gswapXa) are much faster than the other versions (gswapX), but the
 * memory *must* be aligned.
 *
 * Written by Chad Trabant,
 *   IRIS Data Management Center
 *
 * Version: 2010.006
 ***************************************************************************/

#include "lmplatform.h"

/* Swap routines that work on any (aligned or not) quantities */

void
ms_gswap2 (void *data2)
{
  uint8_t temp;

  union {
    uint8_t c[2];
  } dat;

  memcpy (&dat, data2, 2);
  temp     = dat.c[0];
  dat.c[0] = dat.c[1];
  dat.c[1] = temp;
  memcpy (data2, &dat, 2);
}

void
ms_gswap3 (void *data3)
{
  uint8_t temp;

  union {
    uint8_t c[3];
  } dat;

  memcpy (&dat, data3, 3);
  temp     = dat.c[0];
  dat.c[0] = dat.c[2];
  dat.c[2] = temp;
  memcpy (data3, &dat, 3);
}

void
ms_gswap4 (void *data4)
{
  uint8_t temp;

  union {
    uint8_t c[4];
  } dat;

  memcpy (&dat, data4, 4);
  temp     = dat.c[0];
  dat.c[0] = dat.c[3];
  dat.c[3] = temp;
  temp     = dat.c[1];
  dat.c[1] = dat.c[2];
  dat.c[2] = temp;
  memcpy (data4, &dat, 4);
}

void
ms_gswap8 (void *data8)
{
  uint8_t temp;

  union {
    uint8_t c[8];
  } dat;

  memcpy (&dat, data8, 8);
  temp     = dat.c[0];
  dat.c[0] = dat.c[7];
  dat.c[7] = temp;

  temp     = dat.c[1];
  dat.c[1] = dat.c[6];
  dat.c[6] = temp;

  temp     = dat.c[2];
  dat.c[2] = dat.c[5];
  dat.c[5] = temp;

  temp     = dat.c[3];
  dat.c[3] = dat.c[4];
  dat.c[4] = temp;
  memcpy (data8, &dat, 8);
}

/* Swap routines that work on memory aligned quantities */

void
ms_gswap2a (void *data2)
{
  uint16_t *data = data2;

  *data = (((*data >> 8) & 0xff) | ((*data & 0xff) << 8));
}

void
ms_gswap4a (void *data4)
{
  uint32_t *data = data4;

  *data = (((*data >> 24) & 0xff) | ((*data & 0xff) << 24) |
           ((*data >> 8) & 0xff00) | ((*data & 0xff00) << 8));
}

void
ms_gswap8a (void *data8)
{
  uint32_t *data4 = data8;
  uint32_t h0, h1;

  h0 = data4[0];
  h0 = (((h0 >> 24) & 0xff) | ((h0 & 0xff) << 24) |
        ((h0 >> 8) & 0xff00) | ((h0 & 0xff00) << 8));

  h1 = data4[1];
  h1 = (((h1 >> 24) & 0xff) | ((h1 & 0xff) << 24) |
        ((h1 >> 8) & 0xff00) | ((h1 & 0xff00) << 8));

  data4[0] = h1;
  data4[1] = h0;
}
