/*--------------------------------------------------------------------
# Filename: coordtrans.c
#  Purpose: Transform Coordinates
#   Author: Matthias Ohrnberger, Matthias Hardt
# Copyright (C) M. Ohrnberger, M. Hardt
#---------------------------------------------------------------------*/
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define	E_RAD	6378.163
#define E_FLAT  298.26
#define DRAD	1.7453292e-2
#define DRLT	9.9330647e-1


void utl_geo_km(double orig_lon, double orig_lat, double rota, double *lon, double *lat) {
   double	lat_fac;	/* conversion factor for latitude in km */
   double	lon_fac;	/* conversion factor for longitude in km */
   double	snr;		/* sin of rotation angle */
   double	csr;		/* cos of rotation angle */
   double	dlt1;
   double	dlt2;
   double	del;
   double	radius;
   double	tmp;
   double	tmp_x, tmp_y;

   /* convert everything to minutes */
   orig_lat = 60.0f * orig_lat;
   orig_lon = 60.0f * orig_lon;

   /* latitude */
   dlt1 = atan(DRLT * tan(orig_lat * DRAD/60.0));
   dlt2 = atan(DRLT * tan((orig_lat +1.0) * DRAD/60.0));
   del  = dlt2 - dlt1;
   radius = E_RAD * (1.0 - (sin(dlt1)*sin(dlt1) / E_FLAT));
   lat_fac = radius * del;

   /* longitude */
   del = acos(1.0 - (1.0 - cos(DRAD/60.0)) * cos(dlt1) * cos(dlt1));
   dlt2 = radius * del;
   lon_fac = dlt2 / cos(dlt1);

   /* rotation */
   snr = sin(rota * DRAD);
   csr = cos(rota * DRAD);


   *lat *= 60.0f; 
   *lon *= 60.0f;

   tmp_x = (*lon) - orig_lon;
   tmp_y = (*lat) - orig_lat;

   tmp   = atan(DRLT * tan(DRAD * ((*lat)+orig_lat)/120.0));
   tmp_x = tmp_x * lon_fac * cos(tmp);
   tmp_y = tmp_y * lat_fac;

   *lon = csr*tmp_x - snr*tmp_y;
   *lat = csr*tmp_y + snr*tmp_x;
}

void utl_lonlat(double orig_lon,double orig_lat,double x,double y,double *lon,double *lat) {
   double        olon;
   double        olat;
   double       lat_fac;        /* conversion factor for latitude in km */
   double       lon_fac;        /* conversion factor for longitude in km */
   double       snr;            /* sin of rotation angle */
   double       csr;            /* cos of rotation angle */
   double       dlt1;
   double       dlt2;
   double       del;
   double       radius;
   double        tmp;
   double        tmp_x, tmp_y;
   double 	rota=0.0;

   /* convert everything to minutes */
   orig_lat = 60.0f * orig_lat;
   orig_lon = 60.0f * orig_lon;
   olon = orig_lon;
   olat = orig_lat;

   /* latitude */
   dlt1 = atan(DRLT * tan(olat * DRAD/60.0));
   dlt2 = atan(DRLT * tan((olat +1.0) * DRAD/60.0));
   del  = dlt2 - dlt1;
   radius = E_RAD * (1.0 - (sin(dlt1)*sin(dlt1) / E_FLAT));
   lat_fac = radius * del;

   /* longitude */
   del = acos(1.0 - (1.0 - cos(DRAD/60.0)) * cos(dlt1) * cos(dlt1));
   dlt2 = radius * del;
   lon_fac = dlt2 / cos(dlt1);

   /* rotation */
   snr = sin(rota * DRAD);
   csr = cos(rota * DRAD);


    tmp_x = snr*y + csr*x;
    tmp_y = csr*y - snr*x;

    tmp_y = tmp_y/lat_fac;
    tmp_y += olat;

    tmp = atan(DRLT * tan(DRAD * (tmp_y+orig_lat)/120.0));
    tmp_x = tmp_x / (lon_fac * cos(tmp));
    tmp_x += olon;

    *lon = tmp_x/60.0f;
    *lat = tmp_y/60.0f;
}

