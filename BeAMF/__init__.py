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

       
    parser = argparse.ArgumentParser(description="Air Mass Factor Calculation")

    helpstr = "Configuration file"
    parser.add_argument(
        "-c", "--config",
        dest="configuration_file",
        required=True,
        type=str,
        help=helpstr,
        metavar="FILENAME"
    )

    # 1. Settings
    helpstr = "Verbose mode"
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        help=helpstr
    )

    # molecular in HARP format
    helpstr = "Molecular for AMF calculation (NO2, HCHO, SO2, O3, C2H2O2, O4)"
    parser.add_argument(
        "-m", "--molec",
        dest="molecular",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Wavelengths for AMF calculation"
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
        "Wavelengths selection flag for multi-wavelength AMF calculation: "
        "True: for each pixel, AMF is only calculated for selected wv; "
        "False: AMF is calculate for all wvs"
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
        "Flag for tropospheric AMF calculation"
        "False: only total AMF "
        "True: Total/tropospheric/stratospheric AMF"
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

    helpstr = "Flag for effective cloud fraction calculation"
    parser.add_argument(
        "--cfflg",
        dest="cf_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = "Flag for vertical column density calculation"
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
        "CF threshold for cloud correction "
        "(cloud correction is only applied when cf/crf>cfthreshold)"
    )
    parser.add_argument(
        "--cldcorrcfthld",
        dest="cldcorr_cfthreshold",
        required=False,
        help=helpstr,
        type=float
    )

    helpstr = (
        "Units for CF threshold for cloud correction: "
        "0: cloud fraction "
        "1: cloud radiance fraction"
    )
    parser.add_argument(
        "--cldcorrcfu",
        dest="cldcorr_cfunits",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Flag for temperature correction for cross section in AMF calculation"
    )
    parser.add_argument(
        "--tcorrflg",
        dest="tcorr_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = (
        "Temperature correction mode: "
        "0: polynomial; "
        "1: 1 / polynomial; "
        "2: (tref - tcoeff) / (t - tcoeff); "
        "3: (t - tcoeff) / (tref - tcoeff); "
    )
    parser.add_argument(
        "-tcorrm",
        dest="tcorr_mode",
        required=False,
        help=helpstr,
        type=int,
        nargs="+"
    )

    helpstr = "Coefficients for temperature correction formula"
    parser.add_argument(
        "--tcorrcoeffs",
        dest="tcorr_coeffs",
        required=False,
        help=helpstr,
        type=float,
        nargs="+",
        action="append"
    )

    helpstr = "Temperature reference for temperature correction mode"
    parser.add_argument(
        "--tcorrref",
        dest="tcorr_ref",
        required=False,
        help=helpstr,
        type=float,
        nargs="+"
    )

    helpstr = (
        "Flag for surface pressure correction (Zhou et al., 2009) "
        "due to the difference of surface altitude between profile and "
        "satellite pixels"
    )
    parser.add_argument(
        "--spcorrflg",
        dest="spcorr_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = (
        "Geometry units for box-AMF LUT interpolation: "
        "0: degree; "
        "1: cosine of angle"
    )
    parser.add_argument(
        "--geou",
        dest="geo_units",
        required=False,
        help=helpstr,
        type=int
    )

    # 1.1 valid data range
    helpstr = "Minimum SZA (degree)"
    parser.add_argument(
        "--szamin",
        dest="sza_min",
        required=False,
        help=helpstr,
        type=float,
    )

    helpstr = "Maximum SZA (degree)"
    parser.add_argument(
        "--szamax",
        dest="sza_max",
        required=False,
        help=helpstr,
        type=float
    )

    helpstr = "Minimum VZA (degree)"
    parser.add_argument(
        "--vzamin",
        dest="vza_min",
        required=False,
        help=helpstr,
        type=float
    )

    helpstr = "Maximum VZA (degree)"
    parser.add_argument(
        "--vzamax",
        dest="vza_max",
        required=False,
        help=helpstr,
        type=float
    )

    helpstr = "Minimum latitude (degree)"
    parser.add_argument(
        "--latmin",
        dest="lat_min",
        required=False,
        help=helpstr,
        type=float
    )

    helpstr = "Maximum latitude (degree)"
    parser.add_argument(
        "--latmax",
        dest="lat_max",
        required=False,
        help=helpstr,
        type=float
    )

    helpstr = "Minimum longitude (degree)"
    parser.add_argument(
        "--lonmin",
        dest="lon_min",
        required=False,
        help=helpstr,
        type=float
    )

    helpstr = "Maximum longitude (degree)"
    parser.add_argument(
        "--lonmax",
        dest="lon_max",
        required=False,
        help=helpstr,
        type=float
    )

    # 2. Input
    # 2.0 Input for box-AMF data
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
        "List of variable names in LUT file"
        "including wavelengths / normalized pressure grid / surface pressure"
        " / SZA / VZA / RAA (except albedo and other parameters)"
    )
    parser.add_argument(
        "--lutvar",
        dest="lut_var_name",
        required=False,
        help=helpstr,
        type=str,
        nargs="+"
    )

    helpstr = "Variable (albedo) name in LUT file"
    parser.add_argument(
        "--lutalb",
        dest="lut_alb_name",
        required=False,
        help=helpstr,
        type=str,
        nargs="+"
    )

    helpstr = "variable (the others) name in LUT file"
    parser.add_argument(
        "--lutother",
        dest="lut_other_name",
        required=False,
        help=helpstr,
        type=str,
        nargs="+"
    )

    # 2.1 General input
    helpstr = (
        "Input file (such a QDOAS output, Satellite L2 data, "
        "or general input including geometry, surface albedo, profiles etc)"
    )
    parser.add_argument(
        "-i", "--inpf",
        dest="inp_file",
        required=False,
        help=helpstr,
        type=str,
        metavar="FILENAME",
        nargs="+"
    )

    helpstr = (
        "Type of input file: "
        "0: HARP format; "
        "1: TROPOMI L2; "
        "2: OMI L2; "
        "3: QA4ECV L2; "
        "4: GOME-2 operational L2; "
        "9: customized netCDF4 file"
    )
    parser.add_argument(
        "-t",
        "--filetype",
        dest="file_type",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Variable(wavelengths index) name in inp_file"
    parser.add_argument(
        "--wvidxn",
        dest="wvidx_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable(slant column density) name in inp_file"
    parser.add_argument(
        "--scd",
        dest="scd_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable(intensity) name in inp_file"
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

    helpstr = "Variable (longitude corner) name in inp_file"
    parser.add_argument(
        "--loncor",
        dest="loncor_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Variable (SZA) name in inp_file "
        "(specify the path for SZA in inp_file)"
    )
    parser.add_argument(
        "--sza",
        dest="sza_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Variable (VZA) name in inp_file"
        "(specify the path for VZA in inp_file)"
    )
    parser.add_argument(
        "--vza",
        dest="vza_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "List of variable (RAA) name(s) in inp_file: "
        "(specify the path for RAA or SAA/VAA in inp_file)"
        "one string: RAA; "
        "two strings: SAA and VAA."
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
        "Mode for RAA: "
        "0: Relative azimuth angle = 180-RAA or 180-ABS(SAA-VAA); "
        "1: Relative azimuth angle = RAA or ABS(SAA-VAA)"
    )
    parser.add_argument(
        "--raam",
        dest="raa_mode",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Mode for variable time in inp_file: "
        "0: Time + TimeDelta"
        "1: Time"
    )
    parser.add_argument(
        "--timem",
        dest="time_mode",
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
        "Units for time in inp_file: "
        "0: seconds since the reference time;(time_mode=0/1) "
        "1: days since the reference time;(time_mode=0) "
        "2: yyyymmdd;(time_mode=0) "
        "3: year/month/day/hour/minute/second/mirosecond(time_mode=1)"
    )
    parser.add_argument(
        "--timeu",
        dest="time_units",
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
        "Units for timedelta in inp_file: "
        "0: milliseconds;"
        "1: seconds; "
        "2: hhmmss"
        "3: fractional time (days); "
        "4: fractional time (hours)"
    )
    parser.add_argument(
        "--timedeltau",
        dest="timedelta_units",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Reference time for variable time in inp_file: "
        "(year, month, day)"
    )
    parser.add_argument(
        "--timeref",
        dest="time_reference",
        required=False,
        help=helpstr,
        type=int,
        nargs="+"
    )

    # 2.2 Input for Terrain Height
    helpstr = (
        "Model for terrain height: "
        "0: values are directly from th_file; "
        "1: interpolation from th_file into lat/lon in inp_file"
    )
    parser.add_argument(
        "--thm",
        dest="th_mode",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Filename for terrain height "
        "(if not set and th_mode=0, th_file=inp_file)"
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
        "Units for terrin height in th_file: "
        "0: m; "
        "1: km"
    )
    parser.add_argument(
        "--thu",
        dest="th_units",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Variable (latitude) name in th_file"
    parser.add_argument(
        "--thlat",
        dest="th_lat_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (longitude) name in th_file"
    parser.add_argument(
        "--thlon",
        dest="th_lon_name",
        required=False,
        help=helpstr,
        type=str
    )

    # 2.3 Input for surface albedo
    helpstr = (
        "Mode for surface albedo data: "
        "0: directly get from the alb_file; "
        "1: LER climatology; "
        "2: LER with VZA dependence; "
        "3: MODIS-type BRDF"
    )
    parser.add_argument(
        "--albm",
        dest="alb_mode",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Filename for surface albedo data "
        "(if not set and alb_mode=0, alb_file=inp_file)"
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

    helpstr = "Variable (albedo value) name in alb_file"
    parser.add_argument(
        "--alb",
        dest="alb_name",
        required=False,
        help=helpstr,
        type=str,
        nargs="+"
    )

    helpstr = "Scaling factor for albedo value in alb_file"
    parser.add_argument(
        "--albfactor",
        dest="alb_factor",
        required=False,
        help=helpstr,
        type=float,
        nargs="+"
    )

    helpstr = (
        "Mode for time in alb_file: "
        "0: single (no time dimension); "
        "1: monthy climatology; "
        "2: defined by alb_time_name (day of year)"
    )
    parser.add_argument(
        "--albtimem",
        dest="alb_time_mode",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Variable (time) in alb_file"
    parser.add_argument(
        "--albtime",
        dest="alb_time_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (latitude) in alb_file"
    parser.add_argument(
        "--alblat",
        dest="alb_lat_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (longitude) in alb_file"
    parser.add_argument(
        "--alblon",
        dest="alb_lon_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (wavelengths) in alb_file"
    parser.add_argument(
        "--albwvs",
        dest="alb_wv_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Wavelengths for surface albedo, linear interpolation from "
        "climatology dataset. (it can be different with the wavelength "
        "for AMF calculation)"
    )
    parser.add_argument(
        "--albwv",
        dest="albwv",
        required=False,
        help=helpstr,
        type=float,
        nargs="+"
    )

    helpstr = (
        "Whether it is regional albedo map: "
        "False: global map; "
        "True: regional map"
    )
    parser.add_argument(
        "--albregf",
        dest="alb_region_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = (
        "Sign for VZA when alb_mode=2: "
        "False: VZA is negative when RAA>90°; "
        "True: VZA is negative when RAA<90°"
    )
    parser.add_argument(
        "--albvzasign",
        dest="alb_vza_sign",
        action="store_true",
        required=False,
        help=helpstr
    )

    # 2.4 Input for cloud properties
    helpstr = (
        "Mode for cloud properties for AMF calculation: "
        "0: directly using cloud values from the cld_file "
        "1: ca1=0.8 and cf1=cf*ca/0.8 for AMF calculation"
    )
    parser.add_argument(
        "--cldm",
        dest="cld_mode",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Filenames for cloud properties"
        "(if not set, cld_file=inp_file)"
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

    helpstr = "Variable (cloud fraction) in the cld_file"
    parser.add_argument(
        "--cf",
        dest="cf_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (cloud pressure) in the cld_file"
    parser.add_argument(
        "--cp",
        dest="cp_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Units for cloud pressure in the cld_file:"
        "0: Pa; "
        "1: hPa"
    )
    parser.add_argument(
        "--cpu",
        dest="cp_units",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Variable (cloud albedo) in the cld_file "
        "(if no set, the default value is 0.8)"
    )
    parser.add_argument(
        "--ca",
        dest="ca_name",
        required=False,
        help=helpstr,
        type=str
    )

    # 2.5 Input for profiles
    helpstr = (
        "Mode of profile file: "
        "0: values are directly from pro_file; "
        "1: time series of gridded data interpolation from pro_file "
        "into lon/lat in inp_file;"
        "2: the same as 1 but monthly climatology at satellite overpass time"
        "3: gridded data without time dimension"
    )
    parser.add_argument(
        "--prom",
        dest="pro_mode",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Filenames for profile file"
        "(if not set and pro_mode=0, pro_file=inp_file, "
        "if pro_mode=1, pro_file is list of files.)"
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
        "Units for trace gas profile in pro_file: "
        "0: volume mixing ratio; "
        "1: columne number density"
    )
    parser.add_argument(
        "--prou",
        dest="pro_units",
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
        "Units for temperature profile in pro_file: "
        "0: Kelvin; "
        "1: Celsius; "
        "2: Fahrenheit"
    )
    parser.add_argument(
        "--tprou",
        dest="tpro_units",
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
        "Mode for tropopause in pro_file: "
        "0: layer index; "
        "1: tropopause pressure(the units is the same as surface pressure)"
    )
    parser.add_argument(
        "--protropm",
        dest="tropopause_mode",
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
        "Units for terrain height in pro_file: "
        "0: m; "
        "1: km"
    )
    parser.add_argument(
        "--prothu",
        dest="pro_th_units",
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
        "Units for surface pressure in pro_file: "
        "0: Pa; "
        "1: hPa"
    )
    parser.add_argument(
        "--prospu",
        dest="pro_sp_units",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Variable (latitude) name in pro_file"
    parser.add_argument(
        "--prolat",
        dest="pro_lat_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Variable (longitude) name in pro_file"
    parser.add_argument(
        "--prolon",
        dest="pro_lon_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = (
        "Mode for time in pro_file: "
        "0: Days since pro_time_reference"
    )
    parser.add_argument(
        "--protimem",
        dest="pro_time_mode",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Variable (time) name in pro_file"
    parser.add_argument(
        "--protime",
        dest="pro_time_name",
        required=False,
        help=helpstr,
        type=str
    )

    helpstr = "Reference time for time in pro_file"
    parser.add_argument(
        "--protimeref",
        dest="pro_time_reference",
        required=False,
        help=helpstr,
        type=int,
        nargs="+"
    )

    helpstr = (
        "profile pressure grid mode: "
        "0: Model level midpoint pressure = a + b * sp; "
        "1: Pressure (midpoint) grid "
    )
    parser.add_argument(
        "--progridm",
        dest="pro_grid_mode",
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
        "Whether it is regional profile map: "
        "False: global map; "
        "True: regional map"
    )
    parser.add_argument(
        "--proregf",
        dest="pro_region_flag",
        action="store_true",
        required=False,
        help=helpstr
    )

    helpstr = (
        "Filenames for the other varialbe file"
        "(if not set, var_file=inp_file) "
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

    helpstr = "Variable (other variables) name in var_file/inp_file"
    parser.add_argument(
        "--var",
        dest="var_name",
        required=False,
        help=helpstr,
        type=str,
        nargs="+"
    )

    # 3. Background correction
    helpstr = "Data range (Longitude) for background correction"
    parser.add_argument(
        "--bclonlim",
        dest="bc_lon_lim",
        required=False,
        help=helpstr,
        type=float,
        nargs="+"
    )

    helpstr = "Data range (Latitude) for background correction"
    parser.add_argument(
        "--bclatlim",
        dest="bc_lat_lim",
        required=False,
        help=helpstr,
        type=float,
        nargs="+"
    )

    helpstr = "Data range (SZA) for background correction"
    parser.add_argument(
        "--bcszalim",
        dest="bc_sza_lim",
        required=False,
        help=helpstr,
        type=float,
        nargs="+"
    )

    helpstr = "Data range (VZA) for background correction"
    parser.add_argument(
        "--bcvzalim",
        dest="bc_vza_lim",
        required=False,
        help=helpstr,
        type=float,
        nargs="+"
    )

    helpstr = (
        "Target parameter for background correction "
        "(lat, sza or cossza)"
    )
    parser.add_argument(
        "--bcvar",
        dest="bc_x_name",
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
        "Lowest number of samples for target parameter in each interval"
        "for background correction"
    )
    parser.add_argument(
        "--bcvarsample",
        dest="bc_x_sample_limit",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "Flag to show the background correction result: "
        "True: show the result of the fitting; "
        "False: process the AMF calculation"
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
        "output file type: "
        "0: HARP format; "
        "1: TROPOMI format; "
        "2: OMI format; "
        "3: QA4ECV format; "
        "4: GOME-2 operational format; "
        "9: customized format"
    )
    parser.add_argument(
        "--outft",
        dest="out_file_type",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = (
        "output file mode: "
        "0: write in an existing file; "
        "1: create a new file"
    )
    parser.add_argument(
        "--outfm",
        dest="out_file_mode",
        required=False,
        help=helpstr,
        type=int
    )

    helpstr = "Filenames for output file"
    parser.add_argument(
        "-o", "--outf",
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
