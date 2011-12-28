#/usr/bin/env sh
#
# Signs the file ObsPy.dmg with the privat key stored at the location indicated by the enviroment
# variable $OBSPY_PRIVAT_KEY_PATH
#
# Use 'export OBSPY_PRIVAT_KEY_PATH="/path/to/privat_key.pem"' to set it before running this script
# 

echo ======================================================================
echo Signature of the image:
openssl dgst -sha1 -binary < ObsPy.dmg | openssl dgst -dss1 -sign "$OBSPY_PRIVAT_KEY_PATH" | openssl enc -base64
echo Size:
stat -f %z ObsPy.dmg
echo ======================================================================
