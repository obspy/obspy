/***************************************************************************
 * lookup.c:
 *
 * Generic lookup routines for Mini-SEED information.
 *
 * Written by Chad Trabant, ORFEUS/EC-Project MEREDIAN
 *
 * modified: 2006.346
 ***************************************************************************/

#include <string.h>

#include "libmseed.h"

/***************************************************************************
 * ms_samplesize():
 *
 * Returns the sample size based on type code or 0 for unknown.
 ***************************************************************************/
uint8_t
ms_samplesize (const char sampletype)
{
  switch (sampletype)
  {
  case 'a':
    return 1;
  case 'i':
  case 'f':
    return 4;
  case 'd':
    return 8;
  default:
    return 0;
  } /* end switch */

} /* End of ms_samplesize() */

/***************************************************************************
 * ms_encodingstr():
 *
 * Returns a string describing a data encoding format.
 ***************************************************************************/
char *
ms_encodingstr (const char encoding)
{
  switch (encoding)
  {
  case 0:
    return "ASCII text";
  case 1:
    return "16 bit integers";
  case 2:
    return "24 bit integers";
  case 3:
    return "32 bit integers";
  case 4:
    return "IEEE floating point";
  case 5:
    return "IEEE double precision float";
  case 10:
    return "STEIM 1 Compression";
  case 11:
    return "STEIM 2 Compression";
  case 12:
    return "GEOSCOPE Muxed 24 bit int";
  case 13:
    return "GEOSCOPE Muxed 16/3 bit gain/exp";
  case 14:
    return "GEOSCOPE Muxed 16/4 bit gain/exp";
  case 15:
    return "US National Network compression";
  case 16:
    return "CDSN 16 bit gain ranged";
  case 17:
    return "Graefenberg 16 bit gain ranged";
  case 18:
    return "IPG - Strasbourg 16 bit gain";
  case 19:
    return "STEIM 3 Compression";
  case 30:
    return "SRO Gain Ranged Format";
  case 31:
    return "HGLP Format";
  case 32:
    return "DWWSSN Format";
  case 33:
    return "RSTN 16 bit gain ranged";
  default:
    return "Unknown format code";
  } /* end switch */

} /* End of ms_encodingstr() */

/***************************************************************************
 * ms_blktdesc():
 *
 * Return a string describing a given blockette type or NULL if the
 * type is unknown.
 ***************************************************************************/
char *
ms_blktdesc (uint16_t blkttype)
{
  switch (blkttype)
  {
  case 100:
    return "Sample Rate";
  case 200:
    return "Generic Event Detection";
  case 201:
    return "Murdock Event Detection";
  case 300:
    return "Step Calibration";
  case 310:
    return "Sine Calibration";
  case 320:
    return "Pseudo-random Calibration";
  case 390:
    return "Generic Calibration";
  case 395:
    return "Calibration Abort";
  case 400:
    return "Beam";
  case 500:
    return "Timing";
  case 1000:
    return "Data Only SEED";
  case 1001:
    return "Data Extension";
  case 2000:
    return "Opaque Data";
  } /* end switch */

  return NULL;

} /* End of ms_blktdesc() */

/***************************************************************************
 * ms_blktlen():
 *
 * Returns the total length of a given blockette type in bytes or 0 if
 * type unknown.
 ***************************************************************************/
uint16_t
ms_blktlen (uint16_t blkttype, const char *blkt, flag swapflag)
{
  uint16_t blktlen = 0;

  switch (blkttype)
  {
  case 100: /* Sample Rate */
    blktlen = 12;
    break;
  case 200: /* Generic Event Detection */
    blktlen = 28;
    break;
  case 201: /* Murdock Event Detection */
    blktlen = 36;
    break;
  case 300: /* Step Calibration */
    blktlen = 32;
    break;
  case 310: /* Sine Calibration */
    blktlen = 32;
    break;
  case 320: /* Pseudo-random Calibration */
    blktlen = 28;
    break;
  case 390: /* Generic Calibration */
    blktlen = 28;
    break;
  case 395: /* Calibration Abort */
    blktlen = 16;
    break;
  case 400: /* Beam */
    blktlen = 16;
    break;
  case 500: /* Timing */
    blktlen = 8;
    break;
  case 1000: /* Data Only SEED */
    blktlen = 8;
    break;
  case 1001: /* Data Extension */
    blktlen = 8;
    break;
  case 2000: /* Opaque Data */
    /* First 2-byte field after the blockette header is the length */
    if (blkt)
    {
      memcpy ((void *)&blktlen, blkt + 4, sizeof (int16_t));
      if (swapflag)
        ms_gswap2 (&blktlen);
    }
    break;
  } /* end switch */

  return blktlen;

} /* End of ms_blktlen() */

/***************************************************************************
 * ms_errorstr():
 *
 * Return a string describing a given libmseed error code or NULL if the
 * code is unknown.
 ***************************************************************************/
char *
ms_errorstr (int errorcode)
{
  switch (errorcode)
  {
  case MS_ENDOFFILE:
    return "End of file reached";
  case MS_NOERROR:
    return "No error";
  case MS_GENERROR:
    return "Generic error";
  case MS_NOTSEED:
    return "No SEED data detected";
  case MS_WRONGLENGTH:
    return "Length of data read does not match record length";
  case MS_OUTOFRANGE:
    return "SEED record length out of range";
  case MS_UNKNOWNFORMAT:
    return "Unknown data encoding format";
  case MS_STBADCOMPFLAG:
    return "Bad Steim compression flag(s) detected";
  } /* end switch */

  return NULL;

} /* End of ms_blktdesc() */
