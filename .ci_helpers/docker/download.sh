#!/bin/sh

# ==== ONLY EDIT WITHIN THIS BLOCK =====
DATA_PATH=/usr/local/apache2/htdocs/data

ek60_source='https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK60/Summer2017-D20170615-T190214.raw'
echo "Downloading ${ek60_source}"
ek60_name=$(basename $ek60_source)
wget -O ${DATA_PATH}/${ek60_name} $ek60_source

ek80_source='https://ncei-wcsd-archive.s3-us-west-2.amazonaws.com/data/raw/Bell_M._Shimada/SH1707/EK80/D20170826-T205615.raw'
echo "Downloading ${ek80_source}"
ek80_name=$(basename $ek80_source)
wget -O ${DATA_PATH}/${ek80_name} $ek80_source

azfp_source='https://rawdata.oceanobservatories.org/files/CE01ISSM/R00007/instrmts/dcl37/ZPLSC_sn55075/ce01issm_zplsc_55075_recovered_2017-10-27/DATA/201703/17032923.01A'
echo "Downloading ${azfp_source}"
azfp_name=$(basename $azfp_source)
wget -O ${DATA_PATH}/${azfp_name} $azfp_source

azfp_xml_source='https://rawdata.oceanobservatories.org/files/CE01ISSM/R00007/instrmts/dcl37/ZPLSC_sn55075/ce01issm_zplsc_55075_recovered_2017-10-27/DATA/201703/17032922.XML'
echo "Downloading ${azfp_xml_source}"
azfp_xml_name=$(basename $azfp_xml_source)
wget -O ${DATA_PATH}/${azfp_xml_name} $azfp_xml_source

# ==== ONLY EDIT WITHIN THIS BLOCK =====

exec "$@"