#!/usr/bin/env python3
# coding: utf-8

import argparse
import json
import time
# import importlib
from . import master as amf

# importlib.reload(amf)

# declare list of constants
__ra__ = 6.0221367e23
__deltaT__ = 0.0065
__dryair__ = 28.9644
__rdry__ = 287.0
__g__ = 9.80665
__abc__ = -__g__ / __deltaT__ / __rdry__


# if __name__ == "__main__":
def cml():

    parser = argparse.ArgumentParser(
        description="Air Mass Factor Calculation",
        formatter_class=argparse.RawTextHelpFormatter
    )

    helpstr = "Filename for configuration data"
    parser.add_argument(
        "-c",
        "--configuration_file",
        dest="configuration_file",
        required=True,
        help=helpstr,
        type=str,
        metavar="FILENAME"
    )

    # 1. General Settings
    helpstr = "Verbose mode"
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        help=helpstr
    )

    # molecule in HARP format
    helpstr = (
        "Molecule for AMF calculation:\n"
        "NO2, HCHO, SO2, O3, CHOCHO, O4"
    )
    parser.add_argument(
        "-m",
        "--molecule",
        dest="molecule",
        choices=["NO2", "HCHO", "SO2", "O3", "CHOCHO", "O4"],
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Wavelength(s) for AMF calculation"
    parser.add_argument(
        "-w",
        "--wavelength",
        dest="wavelength",
        required=False,
        help=helpstr,
        type=float,
        nargs="+"
    )

    helpstr = (
        "Wavelength selection flag for multi-wavelength AMF calculation:\n"
        "False: AMF is calculated for all wvs \n"
        "True: AMF is only calculated for selected wv for each pixel"
    )
    parser.add_argument(
        "--wvflg",
        dest="wv_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = "Flag for geometric AMF calculation"
    parser.add_argument(
        "--amfgflg",
        dest="amfgeo_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = (
        "Flag for tropospheric AMF calculation:\n"
        "False: Calculate total AMF\n"
        "True: Calculate total, tropospheric and stratospheric AMF"
    )
    parser.add_argument(
        "--amftropflg",
        dest="amftrop_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = "Flag for background correction calculation"
    parser.add_argument(
        "--bcflg",
        dest="bc_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = (
        "Flag for Stratosphere-Troposphere Separation calculation\n"
        "It will be used when bc_flg=True:\n"
        "False: Background correction calculation is based on SCD data\n"
        "True: Background correction calculation is based on SCD/AMFgeo data"
    )
    parser.add_argument(
        "--stsflg",
        dest="sts_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = "Flag for effective cloud fraction calculation"
    parser.add_argument(
        "--cfflg",
        dest="cf_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = (
        "Flag for vertical column density conversion\n"
        "It will be used only for single wavelength AMF calculation under "
        "two conditions:\n"
        "1. amftrop_flag=False, calculate total vertical column density\n"
        "2. amftrop_flag=True and bc_flag=True, calculate total and "
        "tropospheric vertical column density"
    )
    parser.add_argument(
        "--vcdflg",
        dest="vcd_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = "Flag for cloud correction"
    parser.add_argument(
        "--cldcorrflg",
        dest="cldcorr_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = (
        "CF threshold for cloud correction\n"
        "Cloud correction is only applied when CF>cfthreshold"
    )
    parser.add_argument(
        "--cldcorrcfthld",
        dest="cldcorr_cfthreshold",
        required=False,
        help=helpstr,
        type=float
    )

    helpstr = (
        "Units for CF threshold value used in cloud correction:\n"
        "0: Cloud fraction\n"
        "1: Cloud radiance fraction (intensity-weighted cloud fraction)"
    )
    parser.add_argument(
        "--cldcorrcfu",
        dest="cldcorr_cfunits",
        choices=[0, 1],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Flag for temperature correction in AMF calculation"
    parser.add_argument(
        "--tcorrflg",
        dest="tcorr_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = (
        "Temperature correction mode:\n"
        "0: Polynomial\n"
        "1: 1 / polynomial\n"
        "2: (T_ref - T_coeff) / (T - T_coeff)\n"
        "3: (T - T_coeff) / (T_ref - T_coeff)"
    )
    parser.add_argument(
        "--tcorrm",
        dest="tcorr_mode",
        choices=[0, 1, 2, 3],
        required=False,
        help=helpstr,
        type=int,
        nargs="+"
    )

    helpstr = "Coefficients used in temperature correction formula"
    parser.add_argument(
        "--tcorrtcoeffs",
        dest="tcorr_tcoeffs",
        required=False,
        help=helpstr,
        type=float,
        nargs="+",
        action="append"
    )

    helpstr = "Reference temperature used in temperature correction formula"
    parser.add_argument(
        "--tcorrtref",
        dest="tcorr_tref",
        required=False,
        help=helpstr,
        type=float,
        nargs="+"
    )

    helpstr = (
        "Flag to correct surface pressure due to surface altitude difference "
        "between trace gas profile and satellite pixels (Zhou et al., 2009)"
    )
    parser.add_argument(
        "--spcorrflg",
        dest="spcorr_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = (
        "Units of SZA/VZA for box-AMF LUT interpolation:\n"
        "0: Degree\n"
        "1: Cosine of angle"
    )
    parser.add_argument(
        "--geou",
        dest="geo_units",
        choices=[0, 1],
        required=False,
        help=helpstr,
        type=int
    )

    # 1.1 valid data range
    helpstr = "Minimum SZA (degree) to process AMF calculation"
    parser.add_argument(
        "--szamin",
        dest="sza_min",
        required=False,
        help=helpstr,
        type=float
    )

    helpstr = "Maximum SZA (degree) to process AMF calculation"
    parser.add_argument(
        "--szamax",
        dest="sza_max",
        required=False,
        help=helpstr,
        type=float
    )

    helpstr = "Minimum VZA (degree) to process AMF calculation"
    parser.add_argument(
        "--vzamin",
        dest="vza_min",
        required=False,
        help=helpstr,
        type=float
    )

    helpstr = "Maximum VZA (degree) to process AMF calculation"
    parser.add_argument(
        "--vzamax",
        dest="vza_max",
        required=False,
        help=helpstr,
        type=float
    )

    helpstr = "Minimum latitude (degree) to process AMF calculation"
    parser.add_argument(
        "--latmin",
        dest="lat_min",
        required=False,
        help=helpstr,
        type=float
    )

    helpstr = "Maximum latitude (degree) to process AMF calculation"
    parser.add_argument(
        "--latmax",
        dest="lat_max",
        required=False,
        help=helpstr,
        type=float
    )

    helpstr = "Minimum longitude (degree) to process AMF calculation"
    parser.add_argument(
        "--lonmin",
        dest="lon_min",
        required=False,
        help=helpstr,
        type=float
    )

    helpstr = "Maximum longitude (degree) to process AMF calculation"
    parser.add_argument(
        "--lonmax",
        dest="lon_max",
        required=False,
        help=helpstr,
        type=float
    )

    # 2. Input
    # 2.0 Box-AMF LUT Input
    helpstr = "Filename for box-AMF/radiance LUT"
    parser.add_argument(
        "--lutf",
        dest="lut_file",
        required=False,
        help=helpstr,
        type=str,
        metavar="FILENAME"
    )

    helpstr = "Variable (radiance) name in LUT file"
    parser.add_argument(
        "--lutrad",
        dest="lut_rad_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (box-AMF) name in LUT file"
    parser.add_argument(
        "--lutamf",
        dest="lut_amf_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "List of dimension names in LUT file:\n"
        "Including (wavelengths), normalized pressure profile, surface "
        "pressure, SZA, VZA, RAA"
    )
    parser.add_argument(
        "--lutvar",
        dest="lut_var_name",
        required=False,
        help=helpstr,
        type=str,
        nargs="+"
    )

    helpstr = "Dimension (albedo) name(s) in LUT file"
    parser.add_argument(
        "--lutalb",
        dest="lut_alb_name",
        required=False,
        help=helpstr,
        type=str,
        nargs="+"
    )

    helpstr = "Dimension (other variable) name(s) in LUT file"
    parser.add_argument(
        "--lutother",
        dest="lut_other_name",
        required=False,
        help=helpstr,
        type=str,
        nargs="+"
    )

    # 2.1 General Input
    helpstr = (
        "Filename(s) for general input:\n"
        "Such as QDOAS output and satellite L1/L2 data, "
        "which include geometry, geolocation, time etc."
    )
    parser.add_argument(
        "-i",
        "--inpf",
        dest="inp_file",
        required=False,
        help=helpstr,
        type=str,
        metavar="FILENAME",
        nargs="+"
    )

    helpstr = (
        "Type of general input file:\n"
        "0: HARP format\n"
        "1: TROPOMI L2\n"
        "2: OMI L2\n"
        "3: QA4ECV L2\n"
        "4: GOME-2 operational L2\n"
        "9: Customized netCDF4 file"
    )
    parser.add_argument(
        "-t",
        "--filetype",
        dest="file_type",
        choices=[0, 1, 2, 3, 4, 9],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Variable (wavelength index) name in inp_file"
    parser.add_argument(
        "--wvidx",
        dest="wvidx_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (slant column density) name in inp_file"
    parser.add_argument(
        "--scd",
        dest="scd_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (intensity) name in inp_file"
    parser.add_argument(
        "--intensity",
        dest="intens_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (latitude) name in inp_file"
    parser.add_argument(
        "--lat",
        dest="lat_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (longitude) name in inp_file"
    parser.add_argument(
        "--lon",
        dest="lon_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (latitude corners) name in inp_file"
    parser.add_argument(
        "--latcor",
        dest="latcor_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (longitude corners) name in inp_file"
    parser.add_argument(
        "--loncor",
        dest="loncor_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (SZA) name in inp_file"
    parser.add_argument(
        "--sza",
        dest="sza_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (VZA) name in inp_file"
    parser.add_argument(
        "--vza",
        dest="vza_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "List of variable (RAA) name(s) in inp_file:\n"
        "One string: RAA\n"
        "Two strings: SAA and VAA"
    )
    parser.add_argument(
        "--raa",
        dest="raa_name",
        required=False,
        help=helpstr,
        type=str,
        nargs="+"
    )

    helpstr = (
        "Mode for definition of relative azimuth angle in inp_file"
        "(in order to consistent with the RAA definition in LUT):\n"
        "0: Relative azimuth angle = 180-RAA or 180-(SAA-VAA)\n"
        "1: Relative azimuth angle = RAA or SAA-VAA"
    )
    parser.add_argument(
        "--raam",
        dest="raa_mode",
        choices=[0, 1],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Mode for measurement time in inp_file:\n"
        "0: Time + timedelta\n"
        "1: Time"
    )
    parser.add_argument(
        "--timem",
        dest="time_mode",
        choices=[0, 1],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Variable (time) name in inp_file"
    parser.add_argument(
        "--time",
        dest="time_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Units for time in inp_file:\n"
        "0: Seconds since the reference time (time_mode=0/1)\n"
        "1: Days since the reference time (time_mode=0)\n"
        "2: yyyymmdd (time_mode=0)\n"
        "3: (year,month,day,hour,minute,second,mirosecond) (time_mode=1)"
    )
    parser.add_argument(
        "--timeu",
        dest="time_units",
        choices=[0, 1, 2, 3],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Variable (timedelta) name in inp_file"
    parser.add_argument(
        "--timedelta",
        dest="timedelta_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Units for timedelta in inp_file:\n"
        "0: Milliseconds\n"
        "1: Seconds\n"
        "2: hhmmss\n"
        "3: Fractional time (days)\n"
        "4: Fractional time (hours)"
    )
    parser.add_argument(
        "--timedeltau",
        dest="timedelta_units",
        choices=[0, 1, 2, 3, 4],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Reference for measurement time in inp_file:\n"
        "year, month, day, (hour, minute, second)"
    )
    parser.add_argument(
        "--timeref",
        dest="time_reference",
        required=False,
        help=helpstr,
        type=int,
        nargs="+"
    )

    # 2.2 Terrain Height Input
    helpstr = (
        "Mode for terrain height:\n"
        "0: Values for satellite measurement are directly from th_file\n"
        "1: Gridded data from th_file, and interpolated into satellite pixels"
    )
    parser.add_argument(
        "--thm",
        dest="th_mode",
        choices=[0, 1],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Filename(s) for terrain height\n"
        "If not set and th_mode=0, th_file=inp_file"
    )
    parser.add_argument(
        "--thf",
        dest="th_file",
        required=False,
        help=helpstr,
        type=str,
        metavar="FILENAME",
        nargs="+"
    )

    helpstr = "Variable (terrain height) name in th_file"
    parser.add_argument(
        "--th",
        dest="th_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Units for terrain height in th_file:\n"
        "0: m\n"
        "1: km"
    )
    parser.add_argument(
        "--thu",
        dest="th_units",
        choices=[0, 1],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Dimension (latitude) name in th_file"
    parser.add_argument(
        "--thlat",
        dest="th_lat_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Dimension (longitude) name in th_file"
    parser.add_argument(
        "--thlon",
        dest="th_lon_name",
        required=False,
        help=helpstr,
        type=str
    )

    # 2.3 Surface Albedo Input
    helpstr = (
        "Mode for surface albedo data:\n"
        "0: Values for satellite measurement are directly from alb_file\n"
        "1: Gridded data from alb_file, and interpolated into satellite "
        "pixels\n"
        "2: Gridded data with VZA dependence from alb_file, and "
        "interpolated into satellite pixels\n"
        "3: Gridded MODIS-type BRDF data\n"
        "9: OMI LER climatology"
    )
    parser.add_argument(
        "--albm",
        dest="alb_mode",
        choices=[0, 1, 2, 3, 9],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Filename(s) for surface albedo data\n"
        "If not set and alb_mode=0, alb_file=inp_file"
    )
    parser.add_argument(
        "--albf",
        dest="alb_file",
        required=False,
        help=helpstr,
        type=str,
        metavar="FILENAME",
        nargs="+"
    )

    helpstr = "Variable (albedo) name(s) in alb_file"
    parser.add_argument(
        "--alb",
        dest="alb_name",
        required=False,
        help=helpstr,
        type=str,
        nargs="+"
    )

    helpstr = "Scaling factor(s) for albedo value in alb_file"
    parser.add_argument(
        "--albfactor",
        dest="alb_factor",
        required=False,
        help=helpstr,
        type=float,
        nargs="+"
    )

    helpstr = (
        "Mode for time in alb_file:\n"
        "0: Single (no time dimension)\n"
        "1: Monthy climatology\n"
        "2: Defined by alb_time_name (days of year)"
    )
    parser.add_argument(
        "--albtimem",
        dest="alb_time_mode",
        choices=[0, 1, 2],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Dimension (time) name in alb_file"
    parser.add_argument(
        "--albtime",
        dest="alb_time_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Dimension (latitude) name in alb_file"
    parser.add_argument(
        "--alblat",
        dest="alb_lat_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Dimension (longitude) name in alb_file"
    parser.add_argument(
        "--alblon",
        dest="alb_lon_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Dimension (wavelength) name in alb_file"
    parser.add_argument(
        "--albwv",
        dest="alb_wv_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Wavelength(s) for surface albedo, linear interpolation from "
        "gridded dataset.\n"
        "This can be different with the wavelength for AMF calculation"
    )
    parser.add_argument(
        "--albwvs",
        dest="albwvs",
        required=False,
        help=helpstr,
        type=float,
        nargs="+"
    )

    helpstr = (
        "Whether it is regional albedo map:\n"
        "False: Global map\n"
        "True: Regional map"
    )
    parser.add_argument(
        "--albregflg",
        dest="alb_region_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = (
        "Sign for VZA when alb_mode=2:\n"
        "False: VZA is negative when RAA>90°\n"
        "True: VZA is negative when RAA<90°"
    )
    parser.add_argument(
        "--albvzasign",
        dest="alb_vza_sign",
        action="store_true",
        required=False,
        help=helpstr
    )

    # 2.4 Cloud Input
    helpstr = (
        "Mode for cloud data for AMF calculation:\n"
        "0: Directly use cloud parameters (CF, CP, CA) from cld_file\n"
        "1: Adjustment CA' = 0.8 and CF' = CF · CA / 0.8 for AMF calculation"
    )
    parser.add_argument(
        "--cldm",
        dest="cld_mode",
        choices=[0, 1],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Filename(s) for cloud data\n"
        "If not set, cld_file=inp_file"
    )
    parser.add_argument(
        "--cldf",
        dest="cld_file",
        required=False,
        help=helpstr,
        type=str,
        metavar="FILENAME",
        nargs="+"
    )

    helpstr = "Variable (cloud fraction) in cld_file"
    parser.add_argument(
        "--cf",
        dest="cf_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (cloud pressure) in cld_file"
    parser.add_argument(
        "--cp",
        dest="cp_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Units for cloud pressure in cld_file:\n"
        "0: Pa\n"
        "1: hPa"
    )
    parser.add_argument(
        "--cpu",
        dest="cp_units",
        choices=[0, 1],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Variable (cloud albedo) in cld_file\n"
        "If no set, use the default value 0.8"
    )
    parser.add_argument(
        "--ca",
        dest="ca_name",
        required=False,
        help=helpstr,
        type=str
    )

    # 2.5 Profile Input
    helpstr = (
        "Mode of profile file:\n"
        "0: Values for satellite measurement are directly from pro_file\n"
        "1: Gridded data from pro_file, and interpolated into satellite "
        "pixels\n"
        "2: Same as 1 but monthly climatology at satellite overpass time\n"
        "3: Gridded data without time dimension"
    )
    parser.add_argument(
        "--prom",
        dest="pro_mode",
        choices=[0, 1, 2, 3],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Filename(s) for profile data\n"
        "If not set and pro_mode=0, then pro_file=inp_file"
    )
    parser.add_argument(
        "--prof",
        dest="pro_file",
        required=False,
        help=helpstr,
        type=str,
        metavar="FILENAME",
        nargs="+"
    )

    helpstr = "Variable (trace gas profile) name in pro_file"
    parser.add_argument(
        "--pro",
        dest="pro_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Units for trace gas profile in pro_file:\n"
        "0: Volume mixing ratio\n"
        "1: Column number density"
    )
    parser.add_argument(
        "--prou",
        dest="pro_units",
        choices=[0, 1],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Variable (temperature profile) name in pro_file"
    parser.add_argument(
        "--tpro",
        dest="tpro_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Units for temperature profile in pro_file:\n"
        "0: Kelvin\n"
        "1: Celsius\n"
        "2: Fahrenheit"
    )
    parser.add_argument(
        "--tprou",
        dest="tpro_units",
        choices=[0, 1, 2],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Variable (tropopause layer) name in pro_file"
    parser.add_argument(
        "--protrop",
        dest="tropopause_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Mode for tropopause in pro_file:\n"
        "0: Layer index\n"
        "1: Tropopause pressure (the units is the same as surface pressure)"
    )
    parser.add_argument(
        "--protropm",
        dest="tropopause_mode",
        choices=[0, 1],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Variable (terrain height) name in pro_file"
    parser.add_argument(
        "--proth",
        dest="pro_th_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Units for terrain height in pro_file:\n"
        "0: m\n"
        "1: km"
    )
    parser.add_argument(
        "--prothu",
        dest="pro_th_units",
        choices=[0, 1],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Variable (surface pressure) name in pro_file"
    parser.add_argument(
        "--prosp",
        dest="pro_sp_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Units for pressure in pro_file:\n"
        "0: Pa\n"
        "1: hPa"
    )
    parser.add_argument(
        "--prospu",
        dest="pro_sp_units",
        choices=[0, 1],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Dimension (latitude) name in pro_file"
    parser.add_argument(
        "--prolat",
        dest="pro_lat_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Dimension (longitude) name in pro_file"
    parser.add_argument(
        "--prolon",
        dest="pro_lon_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Mode for time in pro_file:\n"
        "0: Days since pro_time_reference"
    )
    parser.add_argument(
        "--protimem",
        dest="pro_time_mode",
        choices=[0],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Dimension (time) name in pro_file"
    parser.add_argument(
        "--protime",
        dest="pro_time_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Reference time for time in pro_file:\n"
        "year, month, day, (hour, minute, second)"
    )
    parser.add_argument(
        "--protimeref",
        dest="pro_time_reference",
        required=False,
        help=helpstr,
        type=int,
        nargs="+"
    )

    helpstr = (
        "Mode for profile pressure grid:\n"
        "0: Model layer midpoint pressure = a + b · sp\n"
        "1: Pressure grid (midpoint)\n"
        "2: Model layer boundary pressure = a + b · sp\n"
        "3: Pressure grid (boundary)"
    )
    parser.add_argument(
        "--progridm",
        dest="pro_grid_mode",
        choices=[0, 1, 2, 3],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "List of variable (profile pressure grid) name in pro_file"
    parser.add_argument(
        "--progrid",
        dest="pro_grid_name",
        required=False,
        help=helpstr,
        type=str,
        nargs="+"
    )

    helpstr = (
        "Whether it is regional profile data:\n"
        "False: Global map\n"
        "True: Regional map"
    )
    parser.add_argument(
        "--proregflg",
        dest="pro_region_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = (
        "Filename(s) for other variable file\n"
        "If not set, then var_file=inp_file"
    )
    parser.add_argument(
        "--varf",
        dest="var_file",
        required=False,
        help=helpstr,
        type=str,
        metavar="FILENAME",
        nargs="+"
    )

    helpstr = "Variable (other variable) name in var_file/inp_file"
    parser.add_argument(
        "--var",
        dest="var_name",
        required=False,
        help=helpstr,
        type=str,
        nargs="+"
    )

    # 3. Background correction
    helpstr = "Data range (longitude) for background correction"
    parser.add_argument(
        "--bclonlim",
        dest="bc_lon_lim",
        required=False,
        help=helpstr,
        type=float,
        nargs=2
    )

    helpstr = "Data range (latitude) for background correction"
    parser.add_argument(
        "--bclatlim",
        dest="bc_lat_lim",
        required=False,
        help=helpstr,
        type=float,
        nargs=2
    )

    helpstr = "Data range (SZA) for background correction"
    parser.add_argument(
        "--bcszalim",
        dest="bc_sza_lim",
        required=False,
        help=helpstr,
        type=float,
        nargs=2
    )

    helpstr = "Data range (VZA) for background correction"
    parser.add_argument(
        "--bcvzalim",
        dest="bc_vza_lim",
        required=False,
        help=helpstr,
        type=float,
        nargs=2
    )

    helpstr = (
        "Target parameter used for background correction:\n"
        "lat\n"
        "sza\n"
        "cossza"
    )
    parser.add_argument(
        "--bcvar",
        dest="bc_x_name",
        choices=["lat", "sza", "cossza"],
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Interval for target parameter used in background correction"
    parser.add_argument(
        "--bcvarinterval",
        dest="bc_x_interval",
        required=False,
        help=helpstr,
        type=float,
        nargs="+"
    )

    helpstr = (
        "Minimum number of samples for target parameter for statistics in "
        "each interval used for background correction"
    )
    parser.add_argument(
        "--bcvarsample",
        dest="bc_x_sample_limit",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Flag to show background correction result:\n"
        "False: Data processing\n"
        "True: Display fitting result"
    )
    parser.add_argument(
        "--bctestflag",
        dest="bc_test_flag",
        required=False,
        action="store_true",
        help=helpstr
    )

    # 4. Output
    helpstr = (
        "Type of output file:\n"
        "0: HARP format\n"
        "1: TROPOMI L2\n"
        "2: OMI L2\n"
        "3: QA4ECV L2\n"
        "4: GOME-2 operational L2\n"
        "9: Customized netCDF4 file"
    )
    parser.add_argument(
        "--outft",
        dest="out_file_type",
        choices=[0, 1, 2, 3, 4, 9],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Mode for output file:\n"
        "0: Write in an existing file\n"
        "1: Write in a new file"
    )
    parser.add_argument(
        "--outfm",
        dest="out_file_mode",
        choices=[0, 1],
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Filepath or Filename(s) for output file"
    parser.add_argument(
        "-o",
        "--outf",
        dest="out_file",
        required=False,
        help=helpstr,
        type=str,
        metavar="FILENAME",
        nargs="+"
    )
    # output variable flag and name are not listed in command line arguments
    # only in json configuration file

    # -------------------------------main-------------------------------------
    print("BeAMF tool is running")
    a = time.time()
    # Start processing
    amfcal = amf.amf()

    # read command line arguments
    var1 = parser.parse_args()

    # read information from configuration file
    with open(var1.configuration_file, "r") as fid:
        var0 = json.load(fid)

    # process AMF calculation based on the input information
    # from configuration file and command line arguments
    amfcal(var0, var1)
    print("Running time: " + str(time.time() - a))
    print("Program is end!")
