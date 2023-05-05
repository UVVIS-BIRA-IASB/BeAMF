Calculating the AMF
====================

What is needed for the AMF
---------------------------

All parameters that can be set should be given in a json file, an example is provided below. 
As input the following can be provided:

* general settings for AMF
* LUT
* Terrain height (optional)
* Albedo (optional)
* Profile (optional)

Also the output path have to be specified. And a Background correction (optional) can also be provided. 
  

Example of a use case:
-----------------------

A simple example to calculate TROPOMI HCHO AMF based on QDOAS output in Harp format

.. code-block:: RST
				
   beamf -c harp_hcho.json
   


harp_hcho.json is the configuration file for AMF tool. The general setting includes:

#. molecule(HCHO)
#. wavelength (340nm)
#. amftrop_flag( including total
#. tropospheric and stratospheric AMF calculation),
#. vcd_flag(including a VCD conversion=SCD/AMF)
#. cldcorr_flag  (including cloud correction)
#. sza_max (75, AMF is only calculated for SZA<75 degree).

LUT
"""

box-AMF and Radiance LUT (as a function of wavelength, pressure grid (normalized by surface pressure), surface pressure, solar zenith angle, viewing zenith angle, relative azimuth angle, and
surface albedo)

Input files
"""""""""""

TROPOMI HCHO SCD file with HARP format (file_type=0). The other input variables are not specified, which is defined internally based on “molecule” and “file_type”. All auxiliary data is
pre-calculated (from operational L2 file), and terrain height, albedo, cloud, and profile settings are not required for this case.

Output
"""""""

Append into the same file as input (out_file_mode=out_file_type=0, out_file=””). Output variables include amf, averaging_kernel, cloud_radiance fraction, vcd. Out_var_name only set “cloud_radiance_fraction” = “cloud_radiance_fraction”, the default variable name in HARP is “cloud_fraction”. The other variables are using default name based on HARP format.

It will take ~5 min to process one orbit of TROPOMI data. 



Command line options
---------------------

In addition, variables in configuration file can be modified by command line as well, for example:

.. code-block:: RST

   > beamf -c harp_hcho.json -w 350.0

AMF calculation uses 350nm instead 340nm (settings in configuration file)

.. code-block:: RST
				
   > bamf -c harp_hcho.json -i ./2022/07/01/S5P_RPRO_L1B_RA_BD3_20220701T024730_20220701T042859_24427_03_020100_20230104T141057_w320_h2co_radasref.nc

Command line options have precedence over json config file. 

auxiliary data
---------------

The auxiliary data can be calculated in the AMF tool (interpolated from gridded data into the satellite pixels), this can be done by settings input information in surface pressure, profile etc.

Background correction 
----------------------

* based on configuration file “harp_hcho_bc.json”.
*  bc_flag or sts_flag = True: background correction or stratospheric correction based on the reference sector method (sts_flag: data analysis based on SCD/geometric AMF; bc_flag: data analysis based on SCD).
* inp_file = “./2022/07/01/*.nc”. Background correction is only based on the daily satellite measurement (at least 10 files)
*  bc_test_flag = True: visualized the fitting results. (if you want to do data process, this flag should be switch off)

   
.. list-table::
   :header-rows: 1

   -
   
	  - harp name
	  - unit attribute
	  - description
   
   -
   
	  - HCHO_column_number_density_amf
	  - 1
	  - amf of hcho
