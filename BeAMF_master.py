import numpy as np
import netCDF4 as nc
from glob import glob
import os
import bisect
import warnings
# import importlib

import BeAMF_calculation as amf_cal
import BeAMF_terrainheight as amf_th
import BeAMF_albedo as amf_alb
import BeAMF_profile as amf_pro
import BeAMF_output as amf_out
import BeAMF_bc as amf_bc
import BeAMF_function as amf_func

# importlib.reload(amf_cal)
# importlib.reload(amf_th)
# importlib.reload(amf_alb)
# importlib.reload(amf_pro)
# importlib.reload(amf_func)
# importlib.reload(amf_bc)
# importlib.reload(amf_out)

# ignoring any type of warnings
np.seterr(all="ignore")
# ignoring UserWarning warnings
warnings.simplefilter("ignore", UserWarning)


class amf:
    def __init__(self):
        # initial input information variables
        self.info = {
            "configuration_file": "",
            "verbose": False,
            "molecular": "",
            "wavelength": [],
            "wv_flag": False,
            "amfgeo_flag": False,
            "amftrop_flag": False,
            "bc_flag": False,
            "cf_flag": False,
            "vcd_flag": False,
            "cldcorr_flag": False,
            "cfcorr_cfthreshold": 0.0,
            "cfcorr_cfunits": 0,
            "tcorr_flag": False,
            "tcorr_mode": 0,
            "tcorr_coeffs": [],
            "tcorr_ref": [],
            "spcorr_flag": False,
            "geo_units": 0,
            "sza_min": 0.0,
            "sza_max": 90.0,
            "vza_min": 0.0,
            "vza_max": 90.0,
            "lat_min": -90.0,
            "lat_max": 90.0,
            "lon_min": -180.0,
            "lon_max": 180.0,
            "lut_file": "",
            "lut_rad_name": "",
            "lut_amf_name": "",
            "lut_var_name": [],
            "lut_alb_name": [],
            "lut_other_name": [],
            "inp_file": [],
            "file_type": 0,
            "wvidx_name": "",
            "scd_name": "",
            "intens_name": "",
            "lat_name": "",
            "lon_name": "",
            "latcor_name": "",
            "loncor_name": "",
            "sza_name": "",
            "vza_name": "",
            "raa_name": [],
            "raa_mode": 0,
            "time_mode": 0,
            "time_name": "",
            "time_units": 0,
            "timedelta_name": "",
            "timedelta_units": 0,
            "time_reference": [],
            "th_mode": 0,
            "th_file": [],
            "th_name": "",
            "th_units": 0,
            "th_lat_name": "",
            "th_lon_name": "",
            "alb_mode": 0,
            "alb_file": [],
            "alb_name": [],
            "alb_factor": [],
            "alb_lat_name": "",
            "alb_lon_name": "",
            "alb_time_mode": 0,
            "alb_time_name": "",
            "alb_wv_name": "",
            "albwv": [],
            "alb_region_flag": False,
            "alb_vza_sign": False,
            "cld_mode": 0,
            "cld_file": [],
            "cf_name": "",
            "cp_name": "",
            "cp_units": 0,
            "ca_name": "",
            "pro_mode": 0,
            "pro_file": [],
            "pro_name": "",
            "pro_units": 0,
            "tpro_name": "",
            "tpro_units": 0,
            "tropopause_mode": 0,
            "tropopause_name": "",
            "pro_th_name": "",
            "pro_th_units": 0,
            "pro_sp_name": "",
            "pro_sp_units": 0,
            "pro_lat_name": "",
            "pro_lon_name": "",
            "pro_time_name": "",
            "pro_time_mode": 0,
            "pro_time_reference": [],
            "pro_grid_mode": 0,
            "pro_grid_name": [],
            "pro_region_flag": False,
            "var_file": [],
            "var_name": [],
            "bc_lon_lim": [],
            "bc_lat_lim": [],
            "bc_vza_lim": [],
            "bc_sza_lim": [],
            "bc_x_name": "",
            "bc_x_interval": [],
            "bc_x_sample_limit": 0,
            "bc_test_flag": False,
            "out_file_type": 0,
            "out_file_mode": 0,
            "out_file": [],
        }

        self.out = {
            "latitude": np.array([]),
            "longitude": np.array([]),
            "latitudecorners": np.array([]),
            "longitudecorners": np.array([]),
            "solar_zenith_angle": np.array([]),
            "sensor_zenith_angle": np.array([]),
            "solar_azimuth_angle": np.array([]),
            "sensor_azimuth_angle": np.array([]),
            "relative_azimuth_angle": np.array([]),
            "surface_altitude": np.array([]),
            "surface_albedo": [],
            "cloud_fraction": np.array([]),
            "cloud_pressure": np.array([]),
            "cloud_albedo": np.array([]),
            "profile": np.array([]),
            "temperature": np.array([]),
            "pressure": np.array([]),
            "surface_pressure": np.array([]),
            "tropopause": np.array([]),
            "other_variable": [],
            "amf_geo": np.array([]),
            "amf": [],
            "amf_clr": [],
            "amf_cld": [],
            "cloud_radiance_fraction": np.array([]),
            "averaging_kernel": np.array([]),
            "averaging_kernel_clr": np.array([]),
            "scd": np.array([]),
            "vcd": np.array([]),
            "vcdtrop": np.array([]),
            "wvidx": np.array([]),
            "nwv": 0,
            "nalb": 0,
            "nlayer": 0,
            "size": 0,
        }

        self.out_name = {
            "latitude": "",
            "longitude": "",
            "latitudecorners": "",
            "longitudecorners": "",
            "solar_zenith_angle": "",
            "sensor_zenith_angle": "",
            "solar_azimuth_angle": "",
            "sensor_azimuth_angle": "",
            "relative_azimuth_angle": "",
            "surface_altitude": "",
            "surface_albedo": [],
            "cloud_fraction": "",
            "cloud_pressure": "",
            "cloud_albedo": "",
            "profile": "",
            "temperature": "",
            "pressure": "",
            "surface_pressure": "",
            "tropopause": "",
            "other_variable": [],
            "amf_geo": "",
            "amf": [],
            "amf_clr": [],
            "amf_cld": [],
            "cloud_radiance_fraction": "",
            "averaging_kernel": "",
            "averaging_kernel_clr": "",
            "scd": "",
            "vcd": "",
            "vcdtrop": "",
        }

        self.out_flag = {
            "latitude": False,
            "longitude": False,
            "latitudecorners": False,
            "longitudecorners": False,
            "solar_zenith_angle": False,
            "sensor_zenith_angle": False,
            "solar_azimuth_angle": False,
            "sensor_azimuth_angle": False,
            "relative_azimuth_angle": False,
            "surface_altitude": False,
            "surface_albedo": False,
            "cloud_fraction": False,
            "cloud_pressure": False,
            "cloud_albedo": False,
            "profile": False,
            "temperature": False,
            "pressure": False,
            "surface_pressure": False,
            "tropopause": False,
            "other_variable": False,
            "amf_geo": False,
            "amf": True,
            "amf_clr": False,
            "amf_cld": False,
            "cloud_radiance_fraction": False,
            "averaging_kernel": False,
            "averaging_kernel_clr": False,
            "scd": False,
            "vcd": False,
            "vcdtrop": False,
        }

    def copy_variable(self, var0, var1):
        """
        preparation for AMF calculation based on input information
        copy information from configuration file and comannd line arguments to
        input parameters
        Parameters
        ----------
        var0 : dict
            input parameters from configuration file.
        var1 : argparse.Namespace
            input arguments from command line
        -------
        None.
        """

        # var0 --> info
        # all variables
        # 1. settings
        for key in var0["settings"].keys():
            self.info[key] = var0["settings"][key]
        # 2. input
        # 2.0 LUT
        for key in var0["input"]["lut"].keys():
            self.info[key] = var0["input"]["lut"][key]
        # 2.1 general
        for key in var0["input"]["general"].keys():
            self.info[key] = var0["input"]["general"][key]
        # 2.2 terrain height
        for key in var0["input"]["terrain_height"].keys():
            self.info[key] = var0["input"]["terrain_height"][key]
        # 2.3 albedo
        for key in var0["input"]["albedo"].keys():
            self.info[key] = var0["input"]["albedo"][key]
        # 2.4 cloud
        for key in var0["input"]["cloud"].keys():
            self.info[key] = var0["input"]["cloud"][key]
        # 2.5 profiles
        for key in var0["input"]["profile"].keys():
            self.info[key] = var0["input"]["profile"][key]
        # 2.6 other variables
        for key in var0["input"]["var"].keys():
            self.info[key] = var0["input"]["var"][key]
        # 2.7 background correction settings
        for key in var0["bc"].keys():
            self.info[key] = var0["bc"][key]
        # 2.5 output
        for key in var0["output"].keys():
            self.info[key] = var0["output"][key]
        # var1 --> info
        # only when var1 is not default value
        # 1. settings
        if var1.verbose:
            self.info["verbose"] = var1.verbose
        if var1.molecular:
            self.info["molecular"] = var1.molecular
        if var1.wavelength:
            self.info["wavelength"] = var1.wavelength
        if var1.wv_flag:
            self.info["wv_flag"] = var1.wv_flag
        if var1.amfgeo_flag:
            self.info["amfgeo_flag"] = var1.amfgeo_flag
        if var1.amftrop_flag:
            self.info["amftrop_flag"] = var1.amftrop_flag
        if var1.bc_flag:
            self.info["bc_flag"] = var1.bc_flag
        if var1.cf_flag:
            self.info["cf_flag"] = var1.cf_flag
        if var1.vcd_flag:
            self.info["vcd_flag"] = var1.vcd_flag
        if var1.cldcorr_flag:
            self.info["cldcorr_flag"] = var1.cldcorr_flag
            if var1.cldcorr_cfthreshold:
                self.info["cldcorr_cfthreshold"] = var1.cldcorr_cfthreshold
            if var1.cldcorr_cfunits:
                self.info["cldcorr_cfunits"] = var1.cldcorr_cfunits
        if var1.tcorr_flag:
            self.info["tcorr_flag"] = var1.tcorr_flag
            if var1.tcorr_mode:
                self.info["tcorr_mode"] = var1.tcorr_mode
            if var1.tcorr_coeffs:
                self.info["tcorr_coeffs"] = var1.tcorr_coeffs
            if var1.tcorr_ref:
                self.info["tcorr_ref"] = var1.tcorr_ref
        if var1.spcorr_flag:
            self.info["spcorr_flag"] = var1.spcorr_flag
        if var1.geo_units:
            self.info["geo_units"] = var1.geo_units
        # 1.1 criteria for AMF calculation
        if var1.sza_min:
            self.info["sza_min"] = var1.sza_min
        if var1.sza_max:
            self.info["sza_max"] = var1.sza_max
        if var1.vza_min:
            self.info["vza_min"] = var1.vza_min
        if var1.vza_max:
            self.info["vza_max"] = var1.vza_max
        if var1.lat_min:
            self.info["lat_min"] = var1.lat_min
        if var1.lat_max:
            self.info["lat_max"] = var1.lat_max
        if var1.lon_min:
            self.info["lon_min"] = var1.lon_min
        if var1.lon_max:
            self.info["lon_max"] = var1.lon_max
        # 2. input
        # 2.0 box-AMF LUT data
        if var1.lut_file:
            self.info["lut_file"] = var1.lut_file
        if var1.lut_rad_name:
            self.info["lut_rad_name"] = var1.lut_rad_name
        if var1.lut_amf_name:
            self.info["lut_amf_name"] = var1.lut_amf_name
        if var1.lut_var_name:
            self.info["lut_var_name"] = var1.lut_var_name
        if var1.lut_alb_name:
            self.info["lut_alb_name"] = var1.lut_alb_name
        if var1.lut_other_name:
            self.info["lut_other_name"] = var1.lut_other_name
        # 2.1 general input information
        if var1.inp_file:
            self.info["inp_file"] = var1.inp_file
        if var1.file_type:
            self.info["file_type"] = var1.file_type
        if var1.wvidx_name:
            self.info["wvidx_name"] = var1.wvidx_name
        if var1.scd_name:
            self.info["scd_name"] = var1.scd_name
        if var1.intens_name:
            self.info["intens_name"] = var1.intens_name
        if var1.lat_name:
            self.info["lat_name"] = var1.lat_name
        if var1.lon_name:
            self.info["lon_name"] = var1.lon_name
        if var1.latcor_name:
            self.info["latcor_name"] = var1.latcor_name
        if var1.loncor_name:
            self.info["loncor_name"] = var1.loncor_name
        if var1.sza_name:
            self.info["sza_name"] = var1.sza_name
        if var1.vza_name:
            self.info["vza_name"] = var1.vza_name
        if var1.raa_name:
            self.info["raa_name"] = var1.raa_name
        if var1.raa_mode:
            self.info["raa_mode"] = var1.raa_mode
        if var1.time_mode:
            self.info["time_mode"] = var1.time_mode
        if var1.time_name:
            self.info["time_name"] = var1.time_name
        if var1.time_units:
            self.info["time_units"] = var1.time_units
        if var1.timedelta_name:
            self.info["timedelta_name"] = var1.timedelta_name
        if var1.timedelta_units:
            self.info["timedelta_units"] = var1.timedelta_units
        if var1.time_reference:
            self.info["time_reference"] = var1.time_reference
        # 2.2 terrain height data
        if var1.th_mode:
            self.info["th_mode"] = var1.th_mode
        if var1.th_file:
            self.info["th_file"] = var1.th_file
        if var1.th_name:
            self.info["th_name"] = var1.th_name
        if var1.th_units:
            self.info["th_units"] = var1.th_units
        if var1.th_lat_name:
            self.info["th_lat_name"] = var1.th_lat_name
        if var1.th_lon_name:
            self.info["th_lon_name"] = var1.th_lon_name
        # 2.3 surface albedo data
        if var1.alb_mode:
            self.info["alb_mode"] = var1.alb_mode
        if var1.alb_file:
            self.info["alb_file"] = var1.alb_file
        if var1.alb_name:
            self.info["alb_name"] = var1.alb_name
        if var1.alb_factor:
            self.info["alb_factor"] = var1.alb_factor
        if var1.alb_time_mode:
            self.info["alb_time_mode"] = var1.alb_time_mode
        if var1.alb_time_name:
            self.info["alb_time_name"] = var1.alb_time_name
        if var1.alb_lat_name:
            self.info["alb_lat_name"] = var1.alb_lat_name
        if var1.alb_lon_name:
            self.info["alb_lon_name"] = var1.alb_lon_name
        if var1.alb_wv_name:
            self.info["alb_wv_name"] = var1.alb_wv_name
        if var1.albwv:
            self.info["albwv"] = var1.albwv
        if var1.alb_region_flag:
            self.info["alb_region_flag"] = var1.alb_region_flag
        if var1.alb_vza_sign:
            self.info["alb_vza_sign"] = var1.alb_vza_sign
        # 2.4 cloud properties
        if var1.cld_mode:
            self.info["cld_mode"] = var1.cld_mode
        if var1.cld_file:
            self.info["cld_file"] = var1.cld_file
        if var1.cf_name:
            self.info["cf_name"] = var1.cf_name
        if var1.cp_name:
            self.info["cp_name"] = var1.cp_name
        if var1.cp_units:
            self.info["cp_units"] = var1.cp_units
        if var1.ca_name:
            self.info["ca_name"] = var1.ca_name
        # 2.5 profile information
        if var1.pro_mode:
            self.info["pro_mode"] = var1.pro_mode
        if var1.pro_file:
            self.info["pro_file"] = var1.pro_file
        if var1.pro_name:
            self.info["pro_name"] = var1.pro_name
        if var1.pro_units:
            self.info["pro_units"] = var1.pro_units
        if var1.tpro_name:
            self.info["tpro_name"] = var1.tpro_name
        if var1.tpro_units:
            self.info["tpro_units"] = var1.tpro_units
        if var1.tropopause_name:
            self.info["tropopause_name"] = var1.tropopause_name
        if var1.tropopause_mode:
            self.info["tropopause_mode"] = var1.tropopause_mode
        if var1.pro_th_name:
            self.info["pro_th_name"] = var1.pro_th_name
        if var1.pro_th_units:
            self.info["pro_th_units"] = var1.pro_th_units
        if var1.pro_sp_name:
            self.info["pro_sp_name"] = var1.pro_sp_name
        if var1.pro_sp_units:
            self.info["pro_sp_units"] = var1.pro_sp_units
        if var1.pro_lat_name:
            self.info["pro_lat_name"] = var1.pro_lat_name
        if var1.pro_lon_name:
            self.info["pro_lon_name"] = var1.pro_lon_name
        if var1.pro_time_name:
            self.info["pro_time_name"] = var1.pro_time_name
        if var1.pro_time_mode:
            self.info["pro_time_mode"] = var1.pro_time_mode
        if var1.pro_time_reference:
            self.info["pro_time_reference"] = var1.pro_time_reference
        if var1.pro_grid_mode:
            self.info["pro_grid_mode"] = var1.pro_grid_mode
        if var1.pro_grid_name:
            self.info["pro_grid_name"] = var1.pro_grid_name
        if var1.pro_region_flag:
            self.info["pro_region_flag"] = var1.pro_region_flag
        # 2.6 other variable information
        if var1.var_file:
            self.info["var_file"] = var1.var_file
        if var1.var_name:
            self.info["var_name"] = var1.var_name
        # 3. background correction
        if var1.bc_lon_lim:
            self.info["bc_lon_lim"] = var1.bc_lon_lim
        if var1.bc_lat_lim:
            self.info["bc_lat_lim"] = var1.bc_lat_lim
        if var1.bc_vza_lim:
            self.info["bc_vza_lim"] = var1.bc_vza_lim
        if var1.bc_sza_lim:
            self.info["bc_sza_lim"] = var1.bc_sza_lim
        if var1.bc_x_name:
            self.info["bc_x_name"] = var1.bc_x_name
        if var1.bc_x_interval:
            self.info["bc_x_interval"] = var1.bc_x_interval
        if var1.bc_x_sample_limit:
            self.info["bc_x_sample_limit"] = var1.bc_x_sample_limit
        if var1.bc_test_flag:
            self.info["bc_test_flag"] = var1.bc_test_flag
        # 4. output
        if var1.out_file_type:
            self.info["out_file_type"] = var1.out_file_type
        if var1.out_file_mode:
            self.info["out_file_mode"] = var1.out_file_mode
        if var1.out_file:
            self.info["out_file"] = var1.out_file
        # check whether input variable is list or correct datatype
        # variable is a list of integer/float or an integer/float
        keys = ["wavelength", "tcorr_ref", "alb_factor", "albwv"]
        for key in keys:
            if type(self.info[key]) == list:
                for var in self.info[key]:
                    assert type(var) in [int, float], (
                        "datatype of "
                        + key
                        + " is not (list of) integer or float."
                    )
            else:
                assert type(self.info[key]) in [int, float], (
                    "datatype of "
                    + key
                    + " is not (list of) integer or float."
                )
                # convert int/float into a list
                self.info[key] = [self.info[key]]
        # varible is a list of integer/float
        keys = [
            "time_reference",
            "pro_time_reference",
            "bc_lon_lim",
            "bc_lat_lim",
            "bc_vza_lim",
            "bc_sza_lim",
            "bc_x_interval"
        ]
        for key in keys:
            assert type(self.info[key]) == list, (
                key + " is not list of integer/float"
            )
            for var in self.info[key]:
                assert type(var) in [int, float], (
                    "datatype of " + key + " is not list of integer/float."
                )
        # tcorr_mode
        # variable is a list of integer or an integer
        keys = ["tcorr_mode"]
        for key in keys:
            if type(self.info[key]) == list:
                for var in self.info[key]:
                    assert type(var) == int, (
                        "datatype of " + key + " is not (list of) integer."
                    )
            else:
                assert type(self.info[key]) == int, (
                    "datatype of " + key + " is not (list of) integer."
                )
                # convert integer into a list
                self.info[key] = [self.info[key]]
        # variable is a list of string
        keys = ["lut_var_name"]
        for key in keys:
            assert type(self.info[key]) == list, "lut_var_name is not list"
            for var in self.info[key]:
                assert type(var) == str, (
                    "datatype of " + key + " is not list of string."
                )
        # variable is a list of string or a string
        keys = [
            "lut_alb_name",
            "lut_other_name",
            "inp_file",
            "raa_name",
            "th_file",
            "alb_file",
            "alb_name",
            "cld_file",
            "pro_file",
            "pro_grid_name",
            "var_file",
            "var_name",
            "out_file",
        ]
        for key in keys:
            if type(self.info[key]) == list:
                for var in self.info[key]:
                    assert type(var) == str, (
                        "datatype of " + key + " is not (list of) string."
                    )
            else:
                assert type(self.info[key]) == str, (
                    "datatype of " + key + " is not (list of) string."
                )
                # convert string into a list
                if self.info[key]:
                    self.info[key] = [self.info[key]]
                else:
                    self.info[key] = []
        # tcorr_coeffs is a list of list of int/float
        assert (
            type(self.info["tcorr_coeffs"]) == list
        ), "tcorr_coeffs is not list"
        if len(self.info["tcorr_coeffs"]) > 0:
            if type(self.info["tcorr_coeffs"][0]) == list:
                for tcorr_coeffs in self.info["tcorr_coeffs"]:
                    assert type(tcorr_coeffs) == list, (
                        "datatype of tcorr_coeffs is not consistent "
                        "between the elements in the list"
                    )
                    for tcorr_coeff in tcorr_coeffs:
                        assert type(tcorr_coeff) in [int, float], (
                            "datatype of tcorr_coeffs is not (list of) "
                            "integer or float"
                        )
            elif type(self.info["tcorr_coeffs"][0]) in [int, float]:
                for tcorr_coeffs in self.info["tcorr_coeffs"]:
                    assert type(tcorr_coeffs) in [int, float], (
                        "datatype of tcorr_coeffs is not (list of) "
                        "integer or float"
                    )
                self.info["tcorr_coeffs"] = [self.info["tcorr_coeffs"]]
        # variable is a float/integer
        keys = [
            "cldcorr_cfthreshold",
            "sza_min",
            "sza_max",
            "vza_min",
            "vza_max",
            "lat_min",
            "lat_max",
            "lon_min",
            "lon_max",
        ]
        for key in keys:
            assert type(self.info[key]) in [int, float], (
                "datatype of " + key + " is not integer/float."
            )
        # variable is an integer
        keys = [
            "cldcorr_cfunits",
            "geo_units",
            "file_type",
            "raa_mode",
            "time_mode",
            "time_units",
            "timedelta_units",
            "th_mode",
            "th_units",
            "alb_mode",
            "alb_time_mode",
            "cld_mode",
            "cp_units",
            "pro_mode",
            "pro_units",
            "tpro_units",
            "tropopause_mode",
            "pro_th_units",
            "pro_sp_units",
            "pro_time_mode",
            "pro_grid_mode",
            "out_file_type",
            "out_file_mode",
            "bc_x_sample_limit",
        ]
        for key in keys:
            assert type(self.info[key]) == int, (
                "datatype of " + key + " is not integer."
            )
        # variable is a string
        keys = [
            "molecular",
            "lut_file",
            "lut_rad_name",
            "lut_amf_name",
            "wvidx_name",
            "scd_name",
            "intens_name",
            "lon_name",
            "lat_name",
            "latcor_name",
            "loncor_name",
            "sza_name",
            "vza_name",
            "time_name",
            "timedelta_name",
            "th_name",
            "th_lon_name",
            "th_lat_name",
            "alb_lon_name",
            "alb_lat_name",
            "alb_time_name",
            "alb_wv_name",
            "cf_name",
            "cp_name",
            "ca_name",
            "pro_name",
            "tpro_name",
            "tropopause_name",
            "pro_th_name",
            "pro_sp_name",
            "pro_lon_name",
            "pro_lat_name",
            "pro_time_name",
            "bc_x_name",
        ]
        for key in keys:
            assert type(self.info[key]) == str, (
                "datatype of " + key + " is not string."
            )
        # variable is a boolean
        keys = [
            "verbose",
            "wv_flag",
            "amfgeo_flag",
            "amftrop_flag",
            "bc_flag",
            "cf_flag",
            "vcd_flag",
            "cldcorr_flag",
            "tcorr_flag",
            "spcorr_flag",
            "alb_region_flag",
            "alb_vza_sign",
            "pro_region_flag",
            "bc_test_flag",
        ]
        for key in keys:
            assert type(self.info[key]) == bool, (
                "datatype of " + key + " is not boolean."
            )
        # check variables from [output][out_var_flag/out_var_name]
        out_var_flag = self.info["out_var_flag"]
        out_var_name = self.info["out_var_name"]
        for key in out_var_flag:
            assert type(out_var_flag[key]) == bool, (
                "datatype of out_var_flag[" + key + "] is not boolean."
            )
            if out_var_flag[key]:
                if key in [
                    "amf",
                    "amf_clr",
                    "amf_cld",
                    "surface_albedo",
                    "other_variable",
                ]:
                    if type(out_var_name[key]) == list:
                        for var_name in out_var_name[key]:
                            assert type(var_name) == str, (
                                "datatype of out_var_name["
                                + key
                                + "] is not string or list of string."
                            )
                    else:
                        assert type(out_var_name[key]) == str, (
                            + key
                            + "] is not string or list of string."
                        )
                        if out_var_name[key]:
                            self.info["out_var_name"][key] = [
                                out_var_name[key]
                            ]
                        else:
                            self.info["out_var_name"][key] = []
                else:
                    assert type(out_var_name[key]) == str, (
                        "datatype of out_var_name[" + key + "] is not string."
                    )

    def check_setting(self):
        """
        check the input information settings for the AMF calculation.
        Parameters
        """
        # 1. settings
        self.verbose = self.info["verbose"]
        # check molecular is in the list
        # ["NO2", "HCHO", "SO2", "O3", "C2H2O2", "O4"]
        assert self.info["molecular"].upper() in [
            "NO2",
            "HCHO",
            "SO2",
            "O3",
            "C2H2O2",
            "O4",
        ], "molecular is not NO2, HCHO, SO2, O3, C2H2O2, O4"
        # check wavelength range between 300 and 800nm
        self.wavelength = self.info["wavelength"]
        # if wavelength is not set,
        # then use the default value based on molecular.
        if len(self.wavelength) == 0:
            if self.info["molecular"].upper() == "NO2":
                self.wavelength = [440.0]
            elif self.info["molecular"].upper() == "HCHO":
                self.wavelength = [340.0]
            elif self.info["molecular"].upper() == "SO2":
                self.wavelength = [313.0]
            elif self.info["molecular"].upper() == "O3":
                self.wavelength = [334.0]  # OMI DOAS total ozone retrieval
            elif self.info["molecular"].upper() == "C2H2O2":
                self.wavelength = [450.0]
            elif self.info["molecular"].upper() == "O4":
                self.wavelength = [477.0]
        self.wavelength = np.array(self.wavelength)
        self.nwv = self.wavelength.size
        assert all(
            (self.wavelength >= 300.0) & (self.wavelength <= 800.0)
        ), "wavelength is out of range -- between [300,800]nm"

        # if wv_flag is True, then nwv should be larger than 1
        if self.info["wv_flag"]:
            assert self.nwv > 1, "if wv_flag is True, len(wavelength)>1"
        # if bc_flag is True, then nwv should be 1
        if self.info["bc_flag"]:
            assert self.nwv == 1, "if bc_flag is True, len(wavelength)=1"
        # if cf_flag is True, then nwv should be 1, and not cloud correction
        if self.info["cf_flag"]:
            assert self.nwv == 1, "if cf_flag is True, len(wavelength)=1"
            assert (
                not self.cldcorr_flag
            ), "if cf_flag is True, no cloud correction is applied."
        # if vcd_flag is True, then nwv should be 1
        # amftrop_flag=False or amftrop_flag/bc_flag=True
        if self.info["vcd_flag"]:
            assert self.nwv == 1, "if vcd_flag is True, len(wavelength)=1"
            if self.info["amftrop_flag"]:
                assert self.info[
                    "bc_flag"
                ], "if vcd_flag/amftrop_flag=True, then bc_flag=True."
        # check cldcorr mode setting:
        if self.info["cldcorr_flag"]:
            # check cfthreshold should between [0, 1)
            assert (self.info["cldcorr_cfthreshold"] >= 0) & (
                self.info["cldcorr_cfthreshold"] < 1
            ), "cldcorr_cfthreshold is out of range - between [0,1)"
            # check cfunits should be 0 or 1.
            assert self.info["cldcorr_cfunits"] in [
                0,
                1,
            ], "cldcorr_cfunits is out of range -- (0/1)"
        # check tcorr mode settings.
        if self.info["tcorr_flag"]:
            # check tcorr_mode should between [0, 3].
            tcorr_mode = np.array(self.info["tcorr_mode"])
            assert all(
                (tcorr_mode >= 0) & (tcorr_mode <= 3)
            ), "tcorr_mode is out of range -- (0/1/2/3)"
            # check len(tcorr_mode) = len(tcorr_coeffs) = len(tcorr_ref)
            assert (
                len(self.info["tcorr_mode"])
                == len(self.info["tcorr_coeffs"])
                == len(self.info["tcorr_ref"])
            ), "size of tcorr_mode/tcorr_coeffs/tcorr_ref is not the same"
            n = len(self.info["tcorr_mode"])
            if n == 1:
                self.info["tcorr_mode"] = self.info["tcorr_mode"] * self.nwv
                self.info["tcorr_coeffs"] = (
                    self.info["tcorr_coeffs"] * self.nwv
                )
                self.info["tcorr_ref"] = self.info["tcorr_ref"] * self.nwv
            else:
                assert n == self.nwv, (
                    "size of tcorr variables and wavelenth is not the same"
                )
        # check geo_units should between [0, 1].
        assert self.info["geo_units"] in [
            0,
            1,
        ], "geo_units is out of range -- (0/1)"
        # 1.1 criteria for AMF calculation
        # chekc the data range
        assert (
            self.info["sza_min"] < self.info["sza_max"]
        ), "sza_min should be smaller than sza_max"
        assert (
            self.info["vza_min"] < self.info["vza_max"]
        ), "vza_min should be smaller than vza_max"
        assert (
            self.info["lat_min"] < self.info["lat_max"]
        ), "lat_min should be smaller than lat_max"
        assert (
            self.info["lon_min"] < self.info["lon_max"]
        ), "lon_min should be smaller than lon_max"
        if self.info["sza_min"] < 0:
            self.info["sza_min"] = 0.0
            if self.verbose:
                print("Warning: sza_min is set to 0 internally.")
        if self.info["sza_max"] > 90:
            self.info["sza_max"] = 90.0
            if self.verbose:
                print("Warning: sza_max is set to 90 internally.")
        if self.info["vza_min"] < 0:
            self.info["vza_min"] = 0.0
            if self.verbose:
                print("Warning: vza_min is set to 0 internally.")
        if self.info["vza_max"] > 90:
            self.info["vza_max"] = 90.0
            if self.verbose:
                print("Warning: vza_max is set to 90 internally.")
        if self.info["lat_min"] < -90:
            self.info["lat_min"] = -90.0
            if self.verbose:
                print("Warning: lat_min is set to -90 internally.")
        if self.info["lat_max"] > 90:
            self.info["lat_max"] = 90.0
            if self.verbose:
                print("Warning: lat_max is set to 90 internally.")
        if self.info["lon_min"] < -360:
            self.info["lon_min"] = -360.0
            if self.verbose:
                print("Warning: lon_min is set to -360 internally.")
        if self.info["lon_min"] > 360:
            self.info["lon_min"] = 360.0
            if self.verbose:
                print("Warning: lon_min is set to 360 internally.")
        if self.info["lon_max"] < -360:
            self.info["lon_max"] = -360.0
            if self.verbose:
                print("Warning: lon_max is set to -360 internally.")
        if self.info["lon_max"] > 360:
            self.info["lon_max"] = 360.0
            if self.verbose:
                print("Warning: lon_max is set to 360 internally.")
        # set lon_min/lon_max within range of [-180, 180]
        if self.info["lon_min"] < -180:
            self.info["lon_min"] = self.info["lon_min"] + 360.0
        if self.info["lon_min"] > 180:
            self.info["lon_min"] = self.info["lon_min"] - 360.0
        if self.info["lon_max"] < -180:
            self.info["lon_max"] = self.info["lon_max"] + 360.0
        if self.info["lon_max"] > 180:
            self.info["lon_max"] = self.info["lon_max"] - 360.0

        # 2. input files
        # 2.0 LUT file
        # check lut_file is NetCDF (including HDF5) file
        try:
            fid = nc.Dataset(self.info["lut_file"], "r")
            fid.close()
        except IndexError:
            print("lut_file is not netCDF/HDF file")
            raise
        self.lut_file = self.info["lut_file"]
        # variables in LUT
        self.lut_vars = []
        self.lut_vars.append(self.info["lut_rad_name"])
        self.lut_vars.append(self.info["lut_amf_name"])
        assert (
            len(self.info["lut_var_name"]) in [5, 6]
        ), "len(lut_var_name) is not 5/6 (including (wv)/pres/sp/sza/vza/raa)"
        for var in self.info["lut_var_name"]:
            self.lut_vars.append(var)
        # number of albedo parameters
        self.nalb = len(self.info["lut_alb_name"])
        assert self.nalb in [1, 3], "len(lut_alb_name) is incorrect (1,3)"
        for var in self.info["lut_alb_name"]:
            self.lut_vars.append(var)
        # other variables in LUT
        self.nvar = len(self.info["lut_other_name"])
        if len(self.info["lut_other_name"]):
            for var in self.info["lut_other_name"]:
                self.lut_vars.append(var)

        # 2.1 general input file
        # file_type shoud be [0, 1, 2, 3, 4, 9]
        assert self.info["file_type"] in [
            0,
            1,
            2,
            3,
            4,
            9,
        ], "file_type is out of range -- (0/1/2/3/4/9)"
        # check molecular for different file_type
        # if file_type=0/9, for all moleculars
        if self.info["file_type"] == 1:  # TROPOMI
            assert self.info["molecular"].upper() in ["NO2", "HCHO", "SO2"], (
                self.info["molecular"] + "is not in TROPOMI L2 file"
            )
        elif self.info["file_type"] == 2:  # QA4ECV
            assert self.info["molecular"].upper() in ["NO2", "HCHO"], (
                self.info["molecular"] + "is not in QA4ECV L2 file"
            )
        elif self.info["file_type"] == 3:  # OMI
            assert self.info["molecular"].upper() in ["NO2"], (
                self.info["molecular"] + "is not in OMI L2 file"
            )
        elif self.info["file_type"] == 4:  # operational GOME-2
            assert self.info["molecular"].upper() in ["NO2", "HCHO", "SO2"], (
                self.info["molecular"] + "is not in GOME-2 operational L2 file"
            )
        # get all inp_file
        if self.info["inp_file"] == 0:
            assert False, "there is no inp_file"
        elif len(self.info["inp_file"]) == 1:
            self.inp_file = sorted(glob(self.info["inp_file"][0]))
            assert len(self.inp_file), "there is no inp_file"
        else:
            self.inp_file = self.info["inp_file"]
        # for background correction, more than 10 files are needed.
        if self.info["bc_flag"]:
            assert len(self.inp_file) >= 10, (
                "number of inp_file should be larger than 10 when bc_flag=True"
            )

        # 2.2 terrain height (only needed when spcorr_flag is set)
        # define th_mode is in the range [0,1]
        if self.info["spcorr_flag"]:
            assert self.info["th_mode"] in [
                0,
                1,
            ], "th_mode is out of range -- (0/1)"
            # get all th_files
            if len(self.info["th_file"]) == 1:
                self.th_file = sorted(glob(self.info["th_file"][0]))
                assert len(self.th_file), "there is no th_file"
            else:
                self.th_file = self.info["th_file"]

        # 2.3 surface albedo
        # define alb_mode is in the range [0,3]
        assert self.info["alb_mode"] in [
            0,
            1,
            2,
            3,
            9,
        ], "alb_mode is out of range -- (0/1/2/3/9)"
        # get all alb_files
        if len(self.info["alb_file"]) == 1:
            self.alb_file = sorted(glob(self.info["alb_file"][0]))
            assert len(self.alb_file), "there is no alb_file"
        else:
            self.alb_file = self.info["alb_file"]
        # if alb_mode=0 and alb_file is not set, then alb_file=inp_file
        # otherwise, size of alb_file should equal to size of inp_file
        if self.info["alb_mode"] == 0:
            if len(self.alb_file) == 0:
                self.alb_file = self.inp_file
            else:
                assert len(self.alb_file) == len(
                    self.inp_file
                ), "len(alb_file) is not len(inp_file) when th_mode=0"

        # 2.4 cloud
        # only used when cldcorr_flag=True
        if self.info["cldcorr_flag"]:
            # define cld_mode is in the range [0,1]
            assert self.info["cld_mode"] in [
                0,
                1,
            ], "cld_mode is out of range -- [0/1]"
            # get all cld_files
            if len(self.info["cld_file"]) == 1:
                self.cld_file = sorted(glob(self.info["cld_file"][0]))
                assert len(self.cld_file), "there is no cld_file"
            else:
                self.cld_file = self.info["cld_file"]
            # if cld_file is not set, then cld_file=inp_file
            if len(self.cld_file) == 0:
                self.cld_file = self.inp_file
            assert len(self.cld_file) == len(
                self.inp_file
            ), "len(cld_file) is not len(inp_file)"

        # 2.5 profile
        # define pro_mode is in the range [0,1,2,3]
        assert self.info["pro_mode"] in [
            0,
            1,
            2,
            3,
        ], "pro_mode is out of range -- (0/1/2/3)"
        # get all pro_files
        if len(self.info["pro_file"]) == 1:
            self.pro_file = sorted(glob(self.info["pro_file"][0]))
            assert len(self.pro_file), "there is no pro_file"
        else:
            self.pro_file = self.info["pro_file"]
        # for pro_mode=0, and if pro_file is not set, then pro_file=inp_file
        # otherwise, len(pro_file) = inp_file
        # for pro_mode>0, then len(pro_file) >= 1
        if self.info["pro_mode"] == 0:
            if len(self.pro_file) == 0:
                self.pro_file = self.inp_file
            else:
                assert len(self.pro_file) == len(
                    self.inp_file
                ), "len(pro_file) is not len(inp_file)"
        elif self.info["pro_mode"] == 1:
            assert (
                len(self.pro_file) >= 1
            ), "len(pro_file)>=1 when pro_mode=1"
        elif self.info["pro_mode"] == 2:
            assert (
                len(self.pro_file) >= 1
            ), "len(pro_file)>=1 when pro_mode=2"
        else:
            assert (
                len(self.pro_file) == 1
            ), "len(pro_file)=1 when pro_mode=3"

        # 2.6 other variables
        if self.nvar > 0:  # if self.nvar>0, var_file is needed.
            if len(self.info["var_file"]) == 1:
                self.var_file = sorted(glob(self.info["var_file"][0]))
                assert len(self.var_file), "there is no var_file"
            else:
                self.var_file = self.info["var_file"]
            if len(self.info["var_file"]) == 0:
                self.var_file = self.inp_file
            assert len(self.var_file) == len(
                self.inp_file
            ), "len(var_file) is not len(inp_file)"

        # 3. background correction
        if self.info["bc_flag"]:
            # if bc_lon_lim is not set, then set it as [-180, 180]
            # bc_lon_lim should be within [-360, 360]
            if len(self.info["bc_lon_lim"]) == 0:
                self.info["bc_lon_lim"] = [-180, 180]
            elif len(self.info["bc_lon_lim"]) == 2:
                data = self.info["bc_lon_lim"]
                assert (
                    -360 <= data[0] < data[1] <= 360
                ), "bc_lon_lim values are incorrect"
            else:
                assert False, "size of bc_lon_lim is incorrect (2)"
            # if bc_lat_lim is not set, then set it as [-90, 90]
            # bc_lat_lim should be within [-90, 90]
            if len(self.info["bc_lat_lim"]) == 0:
                self.info["bc_lat_lim"] = [-90, 90]
            elif len(self.info["bc_lat_lim"]) == 2:
                data = self.info["bc_lat_lim"]
                assert (
                    -90 <= data[0] < data[1] <= 90
                ), "bc_lat_lim values are incorrect"
            else:
                assert False, "size of bc_lat_lim is incorrect (2)"
            # if bc_vza_lim is not set, then set it as [0, 90]
            # bc_vza_lim should be within [0, 90]
            if len(self.info["bc_vza_lim"]) == 0:
                self.info["bc_vza_lim"] = [0, 90]
            elif len(self.info["bc_vza_lim"]) == 2:
                data = self.info["bc_vza_lim"]
                assert (
                    0 <= data[0] < data[1] <= 90
                ), "bc_vza_lim values are incorrect"
            else:
                assert False, "size of bc_vza_lim is incorrect (2)"
            # if bc_sza_lim is not set, then set it as [0, 100]
            # bc_sza_lim should be within [0, 90]
            if len(self.info["bc_sza_lim"]) == 0:
                self.info["bc_sza_lim"] = [0, 100]
            elif len(self.info["bc_sza_lim"]) == 2:
                data = self.info["bc_sza_lim"]
                assert (
                    0 <= data[0] < data[1] <= 90
                ), "bc_sza_lim values are incorrect"
            else:
                assert False, "size of bc_sza_lim is incorrect (20)"
            # bc_x_name should be "lat", "sza", "cossza"
            assert (
                self.info["bc_x_name"].lower() in ["lat", "sza", "cossza"]
            ), "bc_x_name should be lat, sza or cossza"
            # bc_x_interval: [start, end, npoint-1] value of x
            # normally, len(bc_x_itnerval)=3
            # if len(bc_x_interval)=2, then npoint=21
            # if len(bc_x_interval)>=6, then it is series of
            # x points
            if self.info["bc_x_name"].lower() == "lat":
                xlimit = [-90, 90]
            elif self.info["bc_x_name"].lower() == "sza":
                xlimit = [0, 100]
            else:
                xlimit = [-0.2, 1]
            data0 = np.array(self.info["bc_x_interval"])
            if data0.size == 2:
                assert data0[0] < data0[1], (
                    "bc_x_interval[0] and [1] is incorrect ([0] < [1])"
                )
                if (data0[0] < xlimit[0]) | (data0[1] > xlimit[1]):
                    if self.verbose:
                        print(
                            "warning: bc_x_interval is out of range (" +
                            ",".join(str(d) for d in xlimit) +
                            ")"
                        )
                # set number as default value 20
                data = np.linspace(data0[0], data0[1], num=21)
                self.info["bc_x_interval"] = data
            elif data0.size == 3:
                assert data0[0] < data0[1], (
                    "bc_x_interval[0] and [1] is incorrect ([0] < [1])"
                )
                assert data0[2] >= 5, (
                    "bc_x_interval[2] is incorrect (>=5)"
                )
                if (data0[0] < xlimit[0]) | (data0[1] > xlimit[1]):
                    if self.verbose:
                        print(
                            "warning: bc_x_interval is out of range (" +
                            ",".join(str(d) for d in xlimit) +
                            ")"
                        )
                data = np.linspace(data0[0], data0[1], num=data0[2]+1)
                self.info["bc_x_interval"] = data
            elif data0.size >= 6:
                assert amf_func.monotonic(data0) == 2, (
                    "bc_x_interval is not monotonicity increasing")
                if (data0[0] < xlimit[0]) | (data0[-1] > xlimit[1]):
                    if self.verbose:
                        print(
                            "warning: bc_x_interval is out of range (" +
                            ",".join(str(d) for d in xlimit) +
                            ")"
                        )
                self.info["bc_x_interval"] = data0
            else:
                assert False, "size of bc_x_interval is incorrect (2, 3 or >5)"
            # bc_x_sample_limit is better larger than 50 (warning)
            # It is not allowed to be less than 10
            assert self.info["bc_x_sample_limit"] >= 10, (
                "bc_x_sample_limit is incorrect (>=10)"
            )
            if self.info["bc_x_sample_limit"] < 50:
                if self.verbose:
                    print("warning: bc_x_sample_limit is less than 50.")

        # 4. output
        # check out_file_type range
        assert self.info["out_file_type"] in [
            0,
            1,
            2,
            3,
            4,
            9,
        ], "out_file_type is out of range -- (0/1/2/3/4/9)"
        if self.info["out_file_type"] == 1:  # TROPOMI operational product
            assert self.info["molecular"].upper() in ["NO2", "HCHO", "SO2"], (
                self.info["molecular"] + "is not in TROPOMI L2 file"
            )
        elif self.info["out_file_type"] == 2:  # QA4ECV
            assert self.info["molecular"].upper() in ["NO2", "HCHO"], (
                self.info["molecular"] + "is not in QA4ECV L2 file"
            )
        elif self.info["out_file_type"] == 3:  # OMI
            assert self.info["molecular"].upper() in ["NO2"], (
                self.info["molecular"] + "is not in OMI DOMINO L2 file"
            )
        elif self.info["out_file_type"] == 4:  # operational GOME-2
            assert self.info["molecular"].upper() in ["NO2", "HCHO", "SO2"], (
                self.info["molecular"] + "is not in GOME-2 operational L2 file"
            )
        # check out_file_mode range
        assert self.info["out_file_mode"] in [
            0,
            1,
        ], "out_file_mode is out of range -- (0/1)"
        # when out_file_mode = 0
        # if out_file is not set, then out_file=inp_file, otherwise, out_file
        # can be list of full path filenames or a pathname, and use glob to
        # find all matched files
        if self.info["out_file_mode"] == 0:
            # check output file
            # if not set, then out_file = inp_file, set out_file_type=file_type
            # if len(out_file)=1, then find all file matching out_file pattern
            # if len(out_file)>=1, should check len(out_file)=len(inp_file)
            if len(self.info["out_file"]) == 0:
                self.out_file = self.inp_file
                self.info["out_file_type"] = self.info["file_type"]
            elif len(self.info["out_file"]) == 1:
                self.out_file = sorted(glob(self.info["out_file"][0]))
                assert len(self.out_file) == len(
                    self.inp_file
                ), "len(out_file) is not len(inp_file)"
            else:
                self.out_file = self.info["out_file"]
                assert len(self.out_file) == len(
                    self.inp_file
                ), "len(out_file) is not len(inp_file)"
                for out_file in self.out_file:
                    assert os.path.exists(out_file), (
                        "output file: " + out_file + " does not exist."
                    )
            # if out_file_type = 1/2/3/4, then file_type=out_file_type or 9
            if self.info["out_file_type"] in [1, 2, 3, 4]:
                assert (
                    self.info["file_type"] in [self.info["out_file_type"], 9]
                ), (
                    "file_type should be equal to out_file_type or 9 for "
                    "the operational data format."
                )
            # For Multi-wavelength AMF calculation (nwv>1 & wv_flag=False)
            # and out_file_mode = 0
            # HARP: out_file_mode = 0 -> 1
            # TROPOMI/OMI/GOME-2/etc: only save first wavelength result
            if (self.nwv > 1) & (not self.info["wv_flag"]):
                if self.info["out_file_type"] == 0:
                    if self.verbose:
                        print(
                            "For out_file_type=0 and multi wavelength AMF "
                            "calculation, out_file_mode is set at 1."
                        )
                    self.info["out_file_mode"] = 1
                if self.info["out_file_type"] in [1, 2, 3, 4]:
                    if self.verbose:
                        print(
                            "Warning: For out_file_type="
                            + str(self.info["out_file_type"])
                            + "and multi wavelength AMF calculation"
                            "only output the result for the first wavelength."
                        )
        # when out_file_mode = 1
        # out_file can be a pathname and filename is the same as inp_file
        # or list of full path filenames
        else:
            # Create a new output file only for out_file_type = 0/9
            assert self.info["out_file_type"] in [
                0,
                9,
            ], "out_file_mode=1 only for out_file_type=0/9"
            if len(self.info["out_file"]) == 1:
                # if out_file is a path, then filename is the same as inp_file
                # but suffix uses .nc instead
                if self.info["out_file"][0][-1] in ["/", "\\"]:
                    out_path = self.info["out_file"][0]
                    self.out_file = []
                    for inp_file in self.inp_file:
                        inp_file = inp_file.replace("\\", "/")
                        inp_file = out_path + inp_file.split("/")[-1]
                        # split into filename and suffix
                        [fname, fexten] = os.path.splitext(inp_file)
                        self.out_file.append(fname + ".nc")
                else:
                    self.out_file = self.info["out_file"]
            else:
                self.out_file = self.info["out_file"]
            assert len(self.out_file) == len(
                self.inp_file
            ), "len(out_file) is not len(inp_file)"

    def set_input_variable_name(self):
        """
        setting all default "variable name" based on file_type and molecular
        from the input information
        including:
            wvidx_name / scd_name / intens_name / lat_name / lon_name /
            latcor_name / loncor_name / sza_name / vza_name / raa_name /
            raa_mode / time_mode / time_name / time_units / timedelta_name /
            time_reference / th_name / th_units / alb_name / alb_factor /
            cf_name / cp_name / cp_units / ca_name
        """
        # set default variable name based on file_type
        if self.info["file_type"] == 0:  # HARP format
            self.wvidx_name = ""
            self.scd_name = (
                self.info["molecular"] + "_slant_column_number_density"
            )
            self.intens_name = ""
            self.lat_name = "latitude"
            self.lon_name = "longitude"
            self.latcor_name = "latitude_bounds"
            self.loncor_name = "longitude_bounds"
            self.sza_name = "solar_zenith_angle"
            self.vza_name = "sensor_zenith_angle"
            self.raa_name = ["solar_azimuth_angle", "sensor_azimuth_angle"]
            self.raa_mode = 0
            self.time_mode = 1
            self.time_name = "datetime"
            self.time_units = 0  # 0: seconds since the reference time
            self.timedelta_name = ""
            self.timedelta_units = 0
            self.time_reference = [2010, 1, 1]
            self.th_name = "surface_altitude"
            self.th_units = 0
            self.alb_name = ["surface_albedo"]
            self.alb_factor = [1.0]
            self.cf_name = "cloud_fraction"
            self.cp_name = "cloud_pressure"
            self.cp_units = 0
            self.ca_name = "cloud_albedo"
        elif self.info["file_type"] == 1:  # TROPOMI format
            if self.info["molecular"].upper() == "SO2":
                self.wvidx_name = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "selected_fitting_window_flag"
                )
                self.scd_name = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "fitted_slant_columns_win1"
                )
            else:
                self.wvidx_name = ""
                self.scd_name = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "fitted_slant_columns"
                )
            self.intens_name = ""
            self.lat_name = "/PRODUCT/latitude"
            self.lon_name = "/PRODUCT/longitude"
            self.latcor_name = (
                "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds"
            )
            self.loncor_name = (
                "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds"
            )
            self.sza_name = (
                "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_zenith_angle"
            )
            self.vza_name = (
                "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_zenith_angle"
            )
            self.raa_name = [
                "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_azimuth_angle",
                "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_azimuth_angle",
            ]
            self.raa_mode = 0
            self.time_mode = 0
            self.time_name = "/PRODUCT/time"
            self.time_units = 0
            self.timedelta_name = "/PRODUCT/delta_time"
            self.timedelta_units = 0
            self.time_reference = [2010, 1, 1]
            self.th_name = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude"
            self.th_units = 0
            if self.info["molecular"].upper() == "NO2":
                self.alb_name = [
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/"
                    "surface_albedo_nitrogendioxide_window"
                ]
            elif self.info["molecular"].upper() == "SO2":
                self.alb_name = [
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_albedo_328nm",
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_albedo_328nm",
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_albedo_376nm"
                ]
            else:
                self.alb_name = [
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_albedo"
                ]
            self.alb_factor = [1.0]
            if self.info["molecular"].upper() == "NO2":
                self.cf_name = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "cloud_fraction_crb_nitrogendioxide_window"
                    )
            else:
                self.cf_name = (
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_fraction_crb"
                    )
            self.cp_name = (
                "/PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_pressure_crb"
            )
            self.cp_units = 0
            self.ca_name = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_albedo_crb"
            if self.info["molecular"].upper() == "SO2":
                self.var_name = [
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/"
                    "ozone_total_vertical_column"
                ]
            else:
                self.var_name = ""
        elif self.info["file_type"] == 2:  # QA4ECV product
            self.wvidx_name = ""
            if self.info["molecular"].upper() == "NO2":
                self.scd_name = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/scd_no2"
                )
            elif self.info["molecular"].upper() == "HCHO":
                self.scd_name = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/scd_hcho"
                )
            self.intens_name = ""
            self.lat_name = "/PRODUCT/latitude"
            self.lon_name = "/PRODUCT/longitude"
            self.latcor_name = (
                "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds"
            )
            self.loncor_name = (
                "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds"
            )
            self.sza_name = (
                "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_zenith_angle"
            )
            self.vza_name = (
                "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_zenith_angle"
            )
            self.raa_name = [
                "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/relative_azimuth_angle"
            ]
            self.raa_mode = 1
            self.time_mode = 0
            self.time_name = "/PRODUCT/time"
            self.time_units = 0
            self.timedelta_name = "/PRODUCT/delta_time"
            self.timedelta_units = 0
            self.time_reference = [1995, 1, 1]
            self.th_name = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude"
            self.th_units = 0
            self.alb_factor = [1.0]
            self.cf_name = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_fraction"
            self.cp_name = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_pressure"
            self.cp_units = 1
            if self.info["molecular"].upper() == "NO2":
                self.alb_name = [
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_albedo_no2"
                ]
                self.ca_name = (
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_albedo_no2"
                )
            elif self.info["molecular"].upper() == "HCHO":
                self.alb_name = [
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_albedo_hcho"
                ]
                self.ca_name = (
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_albedo_hcho"
                )
            self.var_name = ""
        elif self.info["file_type"] == 3:  # OMI DOMINO product
            self.wvidx_name = ""
            if self.info["molecular"].upper() == "NO2":
                self.scd_name = (
                    "/HDFEOS/SWATHS/DominoNO2/Data Fields/SlantColumnAmountNO2"
                )
            self.intens_name = ""
            self.sza_name = (
                "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/"
                "SolarZenithAngle"
            )
            self.vza_name = (
                "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/"
                "ViewingZenithAngle"
            )
            self.raa_name = [
                "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/"
                "SolarAzimuthAngle",
                "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/"
                "ViewingAzimuthAngle",
            ]
            self.raa_mode = 0
            self.lat_name = (
                "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/Latitude"
            )
            self.lon_name = (
                "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/Longitude"
            )
            self.latcor_name = (
                "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/"
                "LatitudeCornerpoints"
            )
            self.loncor_name = (
                "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/"
                "LongitudeCornerpoints"
            )
            self.time_mode = 1
            self.time_name = "/HDFEOS/SWATHS/DominoNO2/Geolocation Fields/Time"
            self.time_units = 0
            self.time_reference = [1993, 1, 1]
            self.th_name = "/HDFEOS/SWATHS/DominoNO2/Data Fields/TerrainHeight"
            self.th_units = 0
            self.alb_name = [
                "/HDFEOS/SWATHS/DominoNO2/Data Fields/SurfaceAlbedo"
            ]
            self.alb_factor = [0.0001]
            self.cf_name = "/HDFEOS/SWATHS/DominoNO2/Data Fields/CloudFraction"
            self.cp_name = "/HDFEOS/SWATHS/DominoNO2/Data Fields/CloudPressure"
            self.cp_units = 1
            self.ca_name = ""
            self.var_name = ""
        elif self.info["file_type"] == 4:  # GOME-2 operational product
            self.widx_name = ""
            self.scd_name = ""
            self.intens_name = ""
            self.sza_name = "/GEOLOCATION/SolarZenithAngleCentre"
            self.vza_name = "/GEOLOCATION/LineOfSightZenithAngleCentre"
            self.raa_name = ["/GEOLOCATION/RelativeAzimuthCentre"]
            self.raa_mode = 1
            self.lat_name = "/GEOLOCATION/LatitudeCentre"
            self.lon_name = "/GEOLOCATION/LongitudeCentre"
            self.time_mode = 0
            self.time_name = "/GEOLOCATION/Time"
            self.time_units = 1
            self.timedelta_name = "/GEOLOCATION/Time"
            self.timedelta_units = 0
            self.th_name = "/GEOLOCATION/SurfaceHeight"
            self.alb_name = ["/GEOLOCATION/SurfaceAlbedo"]
            self.cf_name = "/CLOUD_PROPERTIES/CloudFraction"
            self.cp_name = "/CLOUD_PROPERTIES/CloudTopPressure"
            self.ca_name = "/CLOUD_PROPERTIES/CloudTopAlbedo"
            self.var_name = ""
        else:  # customized format
            self.wvidx_name = self.info["wvidx_name"]
            self.scd_name = self.info["scd_name"]
            self.intens_name = self.info["intens_name"]
            self.sza_name = self.info["sza_name"]
            self.vza_name = self.info["vza_name"]
            self.raa_name = self.info["raa_name"]
            self.raa_mode = self.info["raa_mode"]
            self.lat_name = self.info["lat_name"]
            self.lon_name = self.info["lon_name"]
            self.time_mode = self.info["time_mode"]
            self.time_name = self.info["time_name"]
            self.time_units = self.info["time_units"]
            self.timedelta_name = self.info["timedelta_name"]
            self.timedelta_units = self.info["timedelta_units"]
            self.time_reference = self.info["time_reference"]
            self.th_name = self.info["th_name"]
            self.th_units = self.info["th_units"]
            self.alb_name = self.info["alb_name"]
            self.alb_factor = self.info["alb_factor"]
            self.cf_name = self.info["cf_name"]
            self.cp_name = self.info["cp_name"]
            self.cp_units = self.info["cp_units"]
            self.ca_name = self.info["ca_name"]
            self.var_name = self.info["var_name"]

    def set_output_variable_name(self):
        """
        setting all "out_var_name" based on out_file_type and molecular
        from the input information
        and checking if the variables exist in the corresponding file
        """
        if self.info["out_file_type"] == 0:  # HARP format
            molecular = self.info["molecular"]
            # geolocation information
            self.out_name["latitude"] = "latitude"
            self.out_name["longitude"] = "longitude"
            self.out_name["latitudecorners"] = "latitude_bounds"
            self.out_name["longitudecorners"] = "longitude_bounds"
            self.out_name["solar_zenith_angle"] = "solar_zenith_angle"
            self.out_name["sensor_zenith_angle"] = "sensor_zenith_angle"
            self.out_name["solar_azimuth_angle"] = "solar_azimuth_angle"
            self.out_name["sensor_azimuth_angle"] = "sensor_azimuth_angle"
            self.out_name["relative_azimuth_angle"] = "relative_azimuth_angle"
            self.out_name["surface_altitude"] = "surface_altitude"
            self.out_name["surface_albedo"] = ["surface_albedo"]
            self.out_name["cloud_fraction"] = "cloud_fraction"
            self.out_name["cloud_pressure"] = "cloud_pressure"
            self.out_name["cloud_albedo"] = "cloud_albedo"
            if self.info["pro_units"] == 0:  # volume mixing ratio
                self.out_name["profile"] = (
                    molecular + "_volume_mixing_ratio_apriori"
                )
            elif self.info["pro_units"] == 1:  # columne number density
                self.out_name["profile"] = (
                    molecular + "_column_number_density_apriori"
                )
            self.out_name["temperature"] = "temperature"
            self.out_name["pressure"] = "pressure"
            self.out_name["surface_pressure"] = "surface_pressure"
            self.out_name["tropopause"] = "tropopause_pressure"
            # setting amf variable
            if self.info["amftrop_flag"]:
                self.out_name["amf"] = [
                    molecular + "_column_number_density_amf",
                    "tropospheric_" + molecular + "_column_number_density_amf",
                    "stratospheric_"
                    + molecular
                    + "_column_number_density_amf",
                ]
            else:
                self.out_name["amf"] = [
                    molecular + "_column_number_density_amf"
                ]
            self.out_name["averaging_kernel"] = (
                molecular + "_column_number_density_avk"
            )
            # if applying cloud correction in AMF, then save
            # cloud_radiance_fraction, otherwise, save cloud fraction.
            self.out_name["cloud_radiance_fraction"] = "cloud_fraction"
            self.out_name["scd"] = molecular + "_slant_column_number_density"
            self.out_name["vcd"] = molecular + "_column_number_density"
            self.out_name["vcdtrop"] = (
                "tropospheric_" + molecular + "_column_number_density"
            )
            # not available: averaging_kernel_clr, other_variable,
            #    amf_geo, amf_clr, amf_cld
        elif self.info["out_file_type"] == 1:  # TROPOMI format
            self.out_name["latitude"] = "/PRODUCT/latitude"
            self.out_name["longitude"] = "/PRODUCT/longitude"
            self.out_name[
                "latitudecorners"
            ] = "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds"
            self.out_name[
                "longitudecorners"
            ] = "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds"
            self.out_name[
                "solar_zenith_angle"
            ] = "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_zenith_angle"
            self.out_name[
                "sensor_zenith_angle"
            ] = "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_zenith_angle"
            self.out_name[
                "solar_azimuth_angle"
            ] = "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_azimuth_angle"
            self.out_name[
                "sensor_azimuth_angle"
            ] = "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_azimuth_angle"
            self.out_name[
                "surface_altitude"
            ] = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude"
            self.out_name[
                "cloud_pressure"
            ] = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_pressure_crb"
            self.out_name[
                "cloud_albedo"
            ] = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_albedo_crb"
            self.out_name[
                "surface_pressure"
            ] = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_pressure"
            if self.info["molecular"].upper() == "NO2":
                self.out_name["surface_albedo"] = [
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/"
                    "surface_albedo_nitrogendioxide_window"
                ]
                self.out_name["cloud_fraction"] = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "cloud_fraction_crb_nitrogendioxide_window"
                )
                self.out_name[
                    "tropopause"
                ] = "/PRODUCT/tm5_tropopause_layer_index"
                self.out_name["amf"] = [
                    "/PRODUCT/air_mass_factor_total",
                    "/PRODUCT/air_mass_factor_troposphere",
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "air_mass_factor_stratosphere",
                ]
                self.out_name["amf_clr"] = [
                    "",
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "air_mass_factor_clear",
                    "",
                ]
                self.out_name["amf_cld"] = [
                    "",
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "air_mass_factor_cloudy",
                    "",
                ]
                self.out_name["cloud_radiance_fraction"] = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "cloud_radiance_fraction_nitrogendioxide_window"
                )
                self.out_name["averaging_kernel"] = "/PRODUCT/averaging_kernel"
                self.out_name["scd"] = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "nitrogendioxide_slant_column_density"
                )
                self.out_name["vcd"] = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "nitrogendioxide_total_column"
                )
                self.out_name[
                    "vcdtrop"
                ] = "/PRODUCT/nitrogendioxide_tropospheric_column"
                # not availabe: amf_geo, averaging_kernel_clr, pressure,
                # temperature
            elif self.info["molecular"].upper() == "HCHO":
                self.out_name["surface_albedo"] = [
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_albedo"
                ]
                self.out_name[
                    "cloud_fraction"
                ] = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_fraction_crb"
                self.out_name["profile"] = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "formaldehyde_profile_apriori"
                )
                self.out_name["tropopause"] = (
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/"
                    "tm5_tropopause_layer_index"
                )
                self.out_name["amf"] = [
                    "",
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "formaldehyde_tropospheric_air_mass_factor",
                    "",
                ]
                self.out_name["amf_clr"] = [
                    "",
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "formaldehyde_clear_air_mass_factor",
                    "",
                ]
                self.out_name["amf_cld"] = [
                    "",
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "formaldehyde_cloudy_air_mass_factor",
                    "",
                ]
                self.out_name["cloud_radiance_fraction"] = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "cloud_fraction_intensity_weighted"
                )
                self.out_name[
                    "averaging_kernel"
                ] = "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/averaging_kernel"
                # output for TROPOMI SO2/HCHO SCD is not set
                # since the out SCD is for multiple molecules from DOAS fit
                self.out_name[
                    "vcdtrop"
                ] = "/PRODUCT/formaldehyde_tropospheric_vertical_column"
            elif self.info["molecular"].upper() == "SO2":
                self.out_name["surface_albedo"] = [
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_albedo_328nm",
                    "",
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_albedo_376nm",
                ]
                self.out_name[
                    "cloud_fraction"
                ] = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_fraction_crb"
                self.out_name["profile"] = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "sulfurdioxide_profile_apriori"
                )
                self.out_name["other_variable"] = [
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/"
                    "ozone_total_vertical_column"
                ]
                self.out_name["tropopause"] = (
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/"
                    "tm5_tropopause_layer_index"
                )
                self.out_name["amf"] = [
                    "",
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "sulfurdioxide_total_air_mass_factor_polluted",
                    "",
                ]
                self.out_name["amf_clr"] = [
                    "",
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "sulfurdioxide_clear_air_mass_factor_polluted",
                    "",
                ]
                self.out_name["amf_cld"] = [
                    "",
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "sulfurdioxide_cloudy_air_mass_factor_polluted",
                    "",
                ]
                self.out_name["cloud_radiance_fraction"] = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "cloud_fraction_intensity_weighted"
                )
                self.out_name["averaging_kernel"] = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/averaging_kernel"
                )
                # output for TROPOMI SO2/HCHO SCD is not set
                # since the out SCD is for multiple molecules from DOAS fit
                # for SO2, sulfurdioxide_total_vertical_column is tropospheric
                # VCD
                self.out_name[
                    "vcdtrop"
                ] = "/PRODUCT/sulfurdioxide_total_vertical_column"
        elif self.info["out_file_type"] == 2:  # QA4ECV format
            self.out_name["latitude"] = "/PRODUCT/latitude"
            self.out_name["longitude"] = "/PRODUCT/longitude"
            self.out_name[
                "latitudecorners"
            ] = "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds"
            self.out_name[
                "longitudecorners"
            ] = "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds"
            self.out_name[
                "solar_zenith_angle"
            ] = "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_zenith_angle"
            self.out_name[
                "sensor_zenith_angle"
            ] = "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_zenith_angle"
            self.out_name[
                "relative_azimuth_angle"
            ] = "/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/relative_azimuth_angle"
            self.out_name["averaging_kernel"] = "/PRODUCT/averaging_kernel"
            self.out_name["tropopause"] = "/PRODUCT/tm5_tropopause_layer_index"
            self.out_name[
                "cloud_fraction"
            ] = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_fraction"
            self.out_name[
                "cloud_pressure"
            ] = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_pressure"
            self.out_name[
                "cloud_albedo"
            ] = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_albedo"
            self.out_name[
                "surface_altitude"
            ] = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude"
            self.out_name[
                "surface_pressure"
            ] = "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_pressure"
            if self.info["molecular"].upper() == "NO2":
                self.out_name["amf"] = [
                    "/PRODUCT/amf_total",
                    "/PRODUCT/amf_trop",
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/amf_strat",
                ]
                self.out_name["amf_clr"] = [
                    "",
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/amf_clear",
                    "",
                ]
                self.out_name[
                    "amf_geo"
                ] = "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/amf_geo"
                self.out_name["cloud_radiance_fraction"] = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "cloud_radiance_fraction_no2"
                )
                self.out_name["surface_albedo"] = [
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_albedo_no2"
                ]
                self.out_name["scd"] = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/scd_no2"
                )
                self.out_name["vcd"] = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "total_no2_vertical_column"
                )
                self.out_name["vcdtrop"] = (
                    "/PRODUCT/tropospheric_hcho_vertical_column"
                )
            elif self.info["molecular"].upper() == "HCHO":
                self.out_name["amf"] = ["/PRODUCT/amf_trop", "", ""]
                self.out_name["amf_clr"] = [
                    "",
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/amf_clear",
                    "",
                ]
                self.out_name["cloud_radiance_fraction"] = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/"
                    "cloud_radiance_fraction_hcho"
                )
                self.out_name["surface_albedo"] = [
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_albedo_hcho"
                ]
                self.out_name["profile"] = (
                    "/PRODUCT/SUPPORT_DATA/INPUT_DATA/hcho_profile_apriori"
                )
                self.out_name["scd"] = (
                    "/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/scd_hcho"
                )
                self.out_name["vcdtrop"] = (
                    "/PRODUCT/tropospheric_hcho_vertical_column"
                )
        # elif self.info["out_file_type"] == 3:  # OMI format
        # elif self.info["out_file_type"] == 4:  # GOME-2 operational format\

    def check_variable(self):
        # check input/output variable settings

        # check input information
        # 1.0 copy default variables
        # if variable name is not defined in configuration file or command line
        # argument, then it will be copied from default settings based on
        # molecular and sensors.
        if self.info["file_type"] in [0, 1, 2, 3, 4]:
            if len(self.info["wvidx_name"]) > 0:
                self.wvidx_name = self.info["wvidx_name"]
            else:
                self.info["wvidx_name"] = self.wvidx_name
            if len(self.info["scd_name"]) > 0:
                self.scd_name = self.info["scd_name"]
            else:
                self.info["scd_name"] = self.scd_name
            if len(self.info["intens_name"]) > 0:
                self.intens_name = self.info["intens_name"]
            else:
                self.info["intens_name"] = self.intens_name
            if len(self.info["lat_name"]) > 0:
                self.lat_name = self.info["lat_name"]
            else:
                self.info["lat_name"] = self.lat_name
            if len(self.info["lon_name"]) > 0:
                self.lon_name = self.info["lon_name"]
            else:
                self.info["lon_name"] = self.lon_name
            if len(self.info["latcor_name"]) > 0:
                self.latcor_name = self.info["latcor_name"]
            else:
                self.info["latcor_name"] = self.latcor_name
            if len(self.info["loncor_name"]) > 0:
                self.loncor_name = self.info["loncor_name"]
            else:
                self.info["loncor_name"] = self.loncor_name
            if len(self.info["sza_name"]) > 0:
                self.sza_name = self.info["sza_name"]
            else:
                self.info["sza_name"] = self.sza_name
            if len(self.info["vza_name"]) > 0:
                self.vza_name = self.info["vza_name"]
            else:
                self.info["vza_name"] = self.vza_name
            if len(self.info["raa_name"]) > 0:
                self.raa_name = self.info["raa_name"]
            else:
                self.info["raa_name"] = self.raa_name
            if self.info["raa_mode"] > 0:
                self.raa_mode = self.info["raa_mode"]
            else:
                self.info["raa_mode"] = self.raa_mode
            if self.info["time_mode"] > 0:
                self.time_mode = self.info["time_mode"]
            else:
                self.info["time_mode"] = self.time_mode
            if len(self.info["time_name"]) > 0:
                self.time_name = self.info["time_name"]
            else:
                self.info["time_name"] = self.time_name
            if self.info["time_units"] > 0:
                self.time_units = self.info["time_units"]
            else:
                self.info["time_units"] = self.time_units
            if len(self.info["timedelta_name"]) > 0:
                self.timedelta_name = self.info["timedelta_name"]
            else:
                self.info["timedelta_name"] = self.timedelta_name
            if self.info["timedelta_units"] > 0:
                self.timedelta_units = self.info["timedelta_units"]
            else:
                self.info["timedelta_units"] = self.timedelta_units
            if len(self.info["time_reference"]) > 0:
                self.time_reference = self.info["time_reference"]
            else:
                self.info["time_reference"] = self.time_reference
            if len(self.info["th_name"]) > 0:
                self.th_name = self.info["th_name"]
            else:
                self.info["th_name"] = self.th_name
            if self.info["th_units"] > 0:
                self.th_units = self.info["th_units"]
            else:
                self.info["th_units"] = self.th_units
            if len(self.info["alb_name"]) > 0:
                self.alb_name = self.info["alb_name"]
            else:
                self.info["alb_name"] = self.alb_name
            if len(self.info["alb_factor"]) > 0:
                self.alb_factor = self.info["alb_factor"]
            else:
                self.info["alb_factor"] = self.alb_factor
            if len(self.info["cf_name"]) > 0:
                self.cf_name = self.info["cf_name"]
            else:
                self.info["cf_name"] = self.cf_name
            if len(self.info["cp_name"]) > 0:
                self.cp_name = self.info["cp_name"]
            else:
                self.info["cp_name"] = self.cp_name
            if self.info["cp_units"] > 0:
                self.cp_units = self.info["cp_units"]
            else:
                self.info["cp_units"] = self.cp_units
            if len(self.info["ca_name"]) > 0:
                self.ca_name = self.info["ca_name"]
            else:
                self.info["ca_name"] = self.ca_name
            if len(self.info["var_name"]) > 0:
                self.var_name = self.info["var_name"]
            else:
                self.info["var_name"] = self.var_name

        # 1.1 general
        # check if wvidx_name is set when wv_flag=True.
        if self.info["wv_flag"]:
            assert self.info[
                "wvidx_name"
            ], "wvidx_name is not set when wv_flag=True."
        # check if intens_name is set when cf_flag=True.
        if self.info["cf_flag"]:
            assert self.info[
                "intens_name"
            ], "intens_name is not set when cf_flag=True."
        # check if scd_name is set when bc_flag or vcd_flag=True.
        if self.info["bc_flag"] | self.info["vcd_flag"]:
            assert self.info[
                "scd_name"
            ], "scd_name is not set when bc_flag or vcd_flag=True."
        # define raa_mode is in the range [0,1]
        assert self.info["raa_mode"] in [
            0,
            1,
        ], "raa_mode is out of range -- (0/1)"
        # len(raa_name) in [1,2]
        assert len(self.info["raa_name"]) in [
            1,
            2,
        ], "raa_name has 1 or 2 elements"
        # check time mode
        assert self.info["time_mode"] in [
            0,
            1,
        ], "time_mode is out of range -- (0/1)"
        assert self.info["time_units"] in [
            0,
            1,
            2,
            3,
            4,
        ], "time_units is out of range -- (0/1/2/3/4)"
        if self.info["time_mode"] == 0:
            assert self.info["timedelta_units"] in [
                0,
                1,
                2,
                3,
                4,
            ], "timedelta_units is out of range -- (0/1/2/3/4)"
        if self.info["time_units"] in [0, 1]:
            assert len(self.info["time_reference"]) in [3, 6], (
                "time_reference is not a list with length= "
                "3 (year/mon/day) or 6 (year/mon/day/hr/min/sec)"
            )

        # 2.2 terrain height (only needed when spcorr_flag is set)
        if self.info["spcorr_flag"]:
            # check th_name/th_units
            assert self.info["th_name"], "th_name is not set"
            assert self.info["th_units"] in [
                0,
                1,
            ], "th_units is out of range -- (0/1)"
            # if th_mode>=1, then th_file is global digital elevation dataset
            # (a single file):
            # th_name/th_lon_name/th_lat_name are valid
            # if th_mode=0, if th_file is the surface elevation data for the
            # corresponding data pixels. It th_file is not set, then set as
            # inp_file
            # otherwise, should check len(th_file) == len(inp_file)
            if self.info["th_mode"] >= 1:
                assert (
                    len(self.th_file) == 1
                ), "len(th_file) is not 1 when th_mode>=1"
                assert self.info["th_lon_name"], "th_lon_name is not set"
                assert self.info["th_lat_name"], "th_lat_name is not set"
            else:
                if len(self.th_file) == 0:
                    self.th_file = self.inp_file
                else:
                    assert len(self.th_file) == len(
                        self.inp_file
                    ), "len(th_file) is not len(inp_file) when th_mode=0"

        # 2.3 surface albedo
        # alb_mode=0: albedo data for the corresponding satellite pixels
        # alb_mode = 1/2/3: gridded albedo data (LER/LER with VZA/BRDF)
        # alb_mode = 9: OMI LER dataset
        # if alb_mode = 3, then nwv = 1 (AMF only for single wavelength)
        if self.info["alb_mode"] == 3:
            # current only for MODIS-type BRDF parameters (3 parameters)
            assert (
                len(self.info["alb_factor"]) == self.nalb == 3
            ), "len(alb_name)/len(alb_factor) is not 3"
            assert (
                len(self.info["alb_name"]) in [self.nalb, 1]
            ), "len(alb_name) is incorrect (len(lut_alb_name), or 1)"
            assert self.nwv == 1, (
                "len(wavelength) is not 1 when alb_mode=3"
            )
        elif self.info["alb_mode"] == 2:
            # Directional LER
            assert (
                len(self.info["alb_factor"]) == self.nalb == 1
            ), "len(alb_factor)/len(lut_alb_name) is not 1 when alb_mode=2"
            assert self.nwv == 1, (
                "len(wavelength) is not 1 when alb_mode=2"
            )
            assert len(self.info["alb_name"]) in [1, 2], (
                "len(alb_name) is incorrect when alb_mode=2 (1/2)"
            )
        elif self.info["alb_mode"] in [1, 9]:
            # LER
            assert (
                len(self.info["alb_factor"]) == self.nalb == 1
            ), "len(alb_factor)/len(lut_alb_name) is not 1 when alb_mode=1"
            assert (
                len(self.info["alb_name"]) in [1, self.nwv]
            ), (
                "len(alb_name) is incorrect when alb_mode=1 "
                "(1,len(wavelength)"
            )
        elif self.info["alb_mode"] == 0:
            # either BRDF surface treatment or multi-wavelength AMF calculation
            assert (
                (self.nalb == 1) | (self.nwv == 1)
            ), "both len(alb_name) and len(wavelength) > 1 when alb_mode=0"
            assert (
                len(self.info["alb_factor"]) == self.nalb
            ), "len(alb_factor) is not nalb when alb_mode=0"
            assert (
                len(self.info["alb_name"]) in [1, self.nwv, self.nalb]
            ), (
                "len(alb_name) is incorrect when alb_mode=0 "
                "(1,len(wavelength)"
            )
        else:
            assert False, "alb_mode is out of range (0,1,2,3)"
        # check alb_factor > 0
        for alb_factor in self.info["alb_factor"]:
            assert alb_factor > 0, "alb_factor <= 0"
        # if alb_mode>=1, then th_file will be a single file, and
        # alb_lon/lat/(time/wvs)_name are valid
        # if alb_mode=0, if alb_file is not set, then set as inp_file
        # otherwise, len(alb_file) == len(inp_file)
        if self.info["alb_mode"] >= 1:
            assert (
                len(self.alb_file) == 1
            ), "len(alb_file) is not 1 when alb_mode=0"
            assert self.info["alb_lon_name"], "alb_lon_name is not set"
            assert self.info["alb_lat_name"], "alb_lat_name is not set"
            # define alb_time_mode is in the range [0,2]
            assert self.info["alb_time_mode"] in [
                0,
                1,
                2,
            ], "alb_time_mode is out of range -- (0/1/2)"
            # alb_time_name is needed only when alb_time_mode=2
            if self.info["alb_time_mode"] == 2:
                assert self.info[
                    "alb_time_name"
                ], "alb_time_name is not set when alb_time_mode>=2"
            # check albedo wavelength
            # if not set, then albwv = wv
            if len(self.info["albwv"]) == 0:
                self.info["albwv"] = self.info["wavelength"]
            # if size=1, then assume albedo from the same wavelength for
            # all wavelength AMF calculation
            elif len(self.info["albwv"]) == 1:
                self.info["albwv"] = self.info["albwv"] * self.nwv
            assert (
                len(self.info["albwv"]) == self.nwv
            ), "size of albwv is not 1 or size of wavelength"
            # if alb_wv_name is set, then wavelength diemension is included in
            # albedo file
            # otherwise, albwv is not used.
            if len(self.info["alb_wv_name"]) > 0:
                # check albedo wavelength range between 300 and 800nm
                albwv = np.array(self.info["albwv"])
                assert all(
                    (albwv >= 300.0) & (albwv <= 800.0)
                ), "albwv is out of range -- between [300,800]nm"
            else:
                if self.verbose & (self.nwv > 1):
                    print(
                        "warning: alb_wv_name is not set, AMF calculation "
                        "uses the same albedo for all wavelengths."
                    )
            # for BRDF case, only calculate single wavelength
            if self.info["alb_mode"] == 3:
                assert (
                    len(self.info["alb_wv_name"]) == 0
                ), "when alb_mode=3, no wavelength dimension in albedo data"

        # 2.4 cloud
        # define cld_mode is in the range [0,1]
        if self.info["cldcorr_flag"]:
            assert self.info["cf_name"], "cf_name is not set"
            assert self.info["cp_name"], "cp_name is not set"
            assert self.info["cp_units"] in [
                0,
                1,
            ], "cp_units is out of range -- (0/1)"

        # 2.5 profile
        # check time variable when pro_mode=1
        if self.info["pro_mode"] == 1:
            assert self.info["pro_time_name"], "pro_time_name is not set"
            assert (
                self.info["pro_time_mode"] == 0
            ), "pro_time_mode is out of range -- (0)"
            assert len(self.info["pro_time_reference"]) in [3, 6], (
                "pro_time_reference is not a list with length=3 "
                "(year/mon/day) or 6 (year/mon/day/hr/min/sec)"
            )
        if self.info["pro_mode"] >= 1:
            # check lat/lon in pro_file when pro_mode>=1
            assert self.info["pro_lon_name"], "pro_lon_name is not set"
            assert self.info["pro_lat_name"], "pro_lat_name is not set"
        # check trace gas profile
        assert self.info["pro_name"], "pro_name is not set"
        assert self.info["pro_units"] in [
            0,
            1,
        ], "pro_units is out of range -- (0/1)"
        # check pro_grid_mode is in the range [0,1]
        assert self.info["pro_grid_mode"] in [
            0,
            1,
        ], "pro_grid_mode is out of range -- (0/1)"
        # check pro_grid_name
        if self.info["pro_grid_mode"] == 0:
            assert (
                len(self.info["pro_grid_name"]) == 2
            ), "len(pro_grid_name)=2 when pro_grid_mode=0"
        else:
            assert (
                len(self.info["pro_grid_name"]) == 1
            ), "len(pro_grid_name)=1 when pro_grid_mode=1"
        # check surface pressure
        assert self.info["pro_sp_name"], "pro_sp_name is not set"
        assert self.info["pro_sp_units"] in [
            0,
            1,
        ], "pro_sp_units is out of range -- (0/1)"
        # temperature pressure is used for tcorr_flag/spcorr_flag=True
        if self.info["tcorr_flag"] | self.info["spcorr_flag"]:
            assert self.info["tpro_name"], "tpro_name is not set"
            assert self.info["tpro_units"] in [
                0,
                1,
                2,
            ], "tpro_units is out of range -- (0/1/2)"
        # terrain height is used for spcorr_flag=True
        if self.info["spcorr_flag"]:
            assert self.info["pro_th_name"], "pro_th_name is not set"
            assert self.info["pro_th_units"] in [
                0,
                1,
            ], "pro_th_units is out of range -- (0/1)"
        # tropopause only for when amftrop_flag=True
        if self.info["amftrop_flag"]:
            assert self.info["tropopause_name"], "tropopause_name is not set"
            assert self.info["tropopause_mode"] in [
                0,
                1,
            ], "tropopause_mode is out of range -- (0/1)"

        # 2.6 other variables
        assert (
            len(self.info["var_name"]) == self.nvar
        ), "len(lut_other_name) and len(var_name) is inconsistent."

        # 3. output
        # copy out_var_flag from configuration file to self.out_flag
        # reset out_var_name if it is available in the configuration file
        out_flag = self.info["out_var_flag"]
        out_name = self.info["out_var_name"]
        for key in out_flag.keys():
            if out_flag[key]:
                self.out_flag[key] = out_flag[key]
                if len(out_name[key]) > 0:
                    self.out_name[key] = out_name[key]
        # general check and adjustment for out_flag/out_name
        # if amfgeo_flag=False, then out_flag[amf_geo]=False
        if not self.info["amfgeo_flag"]:
            if self.out_flag["amf_geo"]:
                if self.verbose:
                    print(
                        "warning: can not output amf_geo when amfgeo_flag"
                        "=False, set out_var_flag[amf_geo]=False"
                    )
                self.out_flag["amf_geo"] = False
                self.out_name["amf_geo"] = ""
        # if amftrop_flag=False, then out_flag[vcdtrop/tropopause]=False
        if not self.info["amftrop_flag"]:
            if self.out_flag["vcdtrop"]:
                if self.verbose:
                    print(
                        "warning: can not output vcd when amftrop_flag=False,"
                        " set out_var_flag[vcdtrop]=False"
                    )
                self.out_flag["vcdtrop"] = False
                self.out_name["vcdtrop"] = ""
            if self.out_flag["tropopause"]:
                if self.verbose:
                    print(
                        "warning: can not output tropopause when amftrop_flag"
                        "=False, set out_var_flag[tropopause]=False"
                    )
                self.out_flag["tropopause"] = False
                self.out_name["tropopause"] = ""
        # if vcd_flag == False, then out_flag[vcd/vcdtrop]=False
        if not self.info["vcd_flag"]:
            if self.out_flag["vcd"]:
                if self.verbose:
                    print(
                        "warning: can not output vcd when vcd_flag=False,"
                        " set out_var_flag[vcd]=False"
                    )
                self.out_flag["vcd"] = False
                self.out_name["vcd"] = ""
            if self.out_flag["vcdtrop"]:
                if self.verbose:
                    print(
                        "warning: can not output vcdtrop when vcd_flag=False,"
                        " set out_var_flag[vcdtrop]=False"
                    )
                self.out_flag["vcdtrop"] = False
                self.out_name["vcdtrop"] = ""
        # if cldcorr_flag == False, then
        # out_flag[cloud_pressure/cloud_albedo/cloud_radiance_fraction]=False
        if not self.info["cldcorr_flag"]:
            if self.out_flag["cloud_pressure"]:
                if self.verbose:
                    print(
                        "warning: can not output cp when cldcorr_flag=False,"
                        " set out_var_flag[cloud_pressure]=False"
                    )
                self.out_flag["cloud_pressure"] = False
                self.out_name["cloud_pressure"] = ""
            if self.out_flag["cloud_albedo"]:
                if self.verbose:
                    print(
                        "warning: can not output ca when cldcorr_flag=False,"
                        " set out_var_flag[cloud_albedo]=False"
                    )
                self.out_flag["cloud_albedo"] = False
                self.out_name["cloud_albedo"] = ""
            if self.out_flag["cloud_radiance_fraction"]:
                if self.verbose:
                    print(
                        "warning: can not output crf when cldcorr_flag=False,"
                        " set out_var_flag[cloud_radiance_fraction]=False"
                    )
                self.out_flag["cloud_radiance_fraction"] = False
                self.out_name["cloud_radiance_fraction"] = ""
        # if cldcorr_flag and cf_flag == False, then
        # out_flag[cloud_fraction]=False
        if (not self.info["cldcorr_flag"]) & (not self.info["cf_flag"]):
            if self.out_flag["cloud_fraction"]:
                if self.verbose:
                    print(
                        "warning: can not output cf when cldcorr_flag=False"
                        " and cf_flag=False,"
                        " set out_var_flag[cloud_fraction]=False"
                    )
                self.out_flag["cloud_fraction"] = False
                self.out_name["cloud_fraction"] = ""
        # HARP check and adjustment for out_flag/out_name
        if self.info["file_type"] == 0:
            # not standard output for HARP
            keys = ["amf_geo", "amf_clr", "amf_cld", "averaging_kernel_clr"]
            for key in keys:
                if self.out_flag[key]:
                    if self.verbose:
                        print(
                            "warning: out_var_name[" + key + "] is not "
                            "standard output for HARP format"
                            )
            # if both cloud_fraction/cloud_radiance_fraction are set
            # then only output cloud_radiance_fraction
            if self.info["cldcorr_flag"]:
                if self.out_flag["cloud_fraction"]:
                    if self.out_flag["cloud_radiance_fraction"]:
                        if self.verbose:
                            print(
                                "warning: can not output both cloud_fraction"
                                " and cloud_radiance_fraction for HARP format,"
                                " set out_var_flag[cloud_fraction]=False"
                            )
                        self.out_flag["cloud_fraction"] = False
                        self.out_name["cloud_fraction"] = ""
        # TROPOMI check and adjustment for out_flag/out_name
        elif self.info["out_file_type"] == 1:
            # not standard output for TROPOMI
            keys = [
                "relative_azimuth_angle",
                "temperature",
                "pressure",
                "amf_geo",
                "averaging_kernel_clr",
            ]
            for key in keys:
                if self.out_flag[key]:
                    if self.verbose:
                        print(
                            "warning: out_var_name[" + key + "] is not "
                            "standard output for TROPOMI format"
                        )
                    self.out_flag[key] = False
                    self.out_name[key] = ""
            # special parameter for individual species
            if self.info["molecular"].upper() == "NO2":
                keys = ["profile", "other_variable"]
            elif self.info["molecular"].upper() == "HCHO":
                keys = ["scd", "vcd", "other_variable"]
            elif self.info["molecular"].upper() == "SO2":
                keys = ["scd", "vcd"]
            for key in keys:
                if self.out_flag[key]:
                    if self.verbose:
                        print(
                            "warning: out_var_name[" + key + "] is not "
                            "standard output for TROPOMI format"
                        )
                    self.out_flag[key] = False
                    self.out_name[key] = ""

        # QA4ECV check and ajustment for out_flag/out_name
        elif self.info["out_file_type"] == 2:
            # not standard output for HARP
            keys = [
                "solar_azimuth_angle",
                "sensor_azimuth_angle",
                "temperature",
                "pressure",
                "amf_cld",
                "other_variable",
                "averaging_kernel_clr",
            ]
            for key in keys:
                if self.out_flag[key]:
                    if self.verbose:
                        print(
                            "warning: out_var_name[" + key + "] is not "
                            "standard output for QA4ECV format"
                        )
                    self.out_flag[key] = False
                    self.out_name[key] = ""
            # special parameter for individual species
            if self.info["molecular"].upper() == "NO2":
                keys = ["profile"]
            elif self.info["molecular"].upper() == "HCHO":
                keys = ["amf_geo", "vcd"]
            for key in keys:
                if self.out_flag[key]:
                    if self.verbose:
                        print(
                            "warning: out_var_name[" + key + "] is not "
                            "standard output for QA4ECV format"
                        )
                    self.out_flag[key] = False
                    self.out_name[key] = ""

        # check size of output variable
        # out_flag = self.info["out_var_flag"]
        # out_name = self.info["out_var_name"]
        out_flag = self.out_flag
        out_name = self.out_name
        for key in out_flag.keys():
            assert type(out_flag[key]) == bool, (
                "type of " + "out_var_flag[" + key + "] is not boolean"
            )
            if out_flag[key]:
                if key in ["amf", "amf_clr", "amf_cld"]:
                    assert type(out_name[key]) == list, (
                        "out_var_name[" + key + "] is not list"
                    )
                    for name in out_name[key]:
                        assert type(name) == str, (
                            "type of out_var_name[" + key + "] is not string"
                        )
                elif key == "other_variable":
                    assert type(out_name[key]) == list, (
                        "out_var_name[" + key + "] is not list"
                    )
                    for name in out_name[key]:
                        assert type(name) == str, (
                            "type of out_var_name[" + key + "] is not string"
                        )
                elif key == "surface_albedo":
                    assert type(out_name[key]) == list, (
                        "out_var_name[" + key + "] is not list"
                    )
                    for name in out_name[key]:
                        assert type(name) == str, (
                            "type of out_var_name[" + key + "] is not string"
                        )
                else:
                    assert type(out_name[key]) == str, (
                        "out_var_name[" + key + "] is not string"
                    )
        # copy from self.info["out_var_flag"] to self.out_flag
        for key in out_flag.keys():
            self.out_flag[key] = out_flag[key]
            self.out_name[key] = out_name[key]

    # read information for AMF calculation
    def read_info(
        self,
        inp_file,
        th_file=[],
        alb_file=[],
        cld_file=[],
        pro_file=[],
        var_file=[],
    ):
        # initialized extra variables
        # lat/lon/latcor/loncor
        lat = np.array([])
        lon = np.array([])
        latcor = np.array([])
        loncor = np.array([])
        # sza/vza
        sza = np.array([])
        vza = np.array([])
        # cossza/cosvza: cos(sza)/cos(vza)
        cossza = np.array([])
        cosvza = np.array([])
        # saa/vaa
        saa = np.array([])
        vaa = np.array([])
        # scd/intens
        scd = np.array([])
        intens = np.array([])
        # only set when alb_mode/pro_mode>0
        time = np.array([])
        # wavelength index for AMF calculation
        wvidx = np.array([])
        # th data when spcorr_flag is True
        th = np.array([])
        # albedo data
        alb = []
        nalb = 0  # ndimension for surface albedo (last dimension)
        # cloud data when cldcorr_flag is True
        cf = np.array([])
        cp = np.array([])
        ca = np.array([])
        # profile data when pro_mode>0
        pro = np.array([])
        tpro = np.array([])  # only tcorr_flag is True
        pres = np.array([])
        sp = np.array([])
        pro_pam = np.array([])
        pro_pbm = np.array([])
        pro_pai = np.array([])
        pro_pbi = np.array([])
        nlayer = 0
        nlevel = 0
        tropopause = np.array([])  # when amftrop_flag is True
        th1 = np.array([])  # th for profile data when spcorr_flag is True
        var = []

        # read inp_file
        fid = nc.Dataset(inp_file, "r")
        # dimension of the variable (it should be the same for all)
        dim = fid[self.sza_name][:].shape
        size = fid[self.sza_name][:].size
        if len(dim) >= 2:
            if dim == dim[::-1]:
                if self.verbose:
                    print(
                        "Warning: should be sure that the dimension from "
                        "inp_file/th_file/alb_file/pro_file is consistent, "
                        "this can not be checked internally"
                    )
        # assert len(dim) <= 3
        # list of dimension name for sza
        dimname = fid[self.sza_name].dimensions
        # all the input variable should be the same dimensions as sza
        assert (
            fid[self.vza_name].dimensions == dimname
        ), "VZA dimension is incorrect"
        # SZA/VZA values
        sza = amf_func.masked(fid[self.sza_name][:]).ravel()
        sza[sza >= 90] = np.nan
        vza = fid[self.vza_name][:].ravel()
        vza[vza >= 90] = np.nan
        # RAA values
        if len(self.raa_name) == 1:
            assert (
                fid[self.raa_name[0]].dimensions == dimname
            ), "RAA dimension is incorrect"
            raa = amf_func.masked(fid[self.raa_name[0]][:])
            raa[(raa > 360) | (raa < -360)] = np.nan
        else:
            # if two "raa_name", it will be saa and vaa
            assert (
                fid[self.raa_name[0]].dimensions == dimname
            ), "RAA[0] dimension is incorrect"
            assert (
                fid[self.raa_name[1]].dimensions == dimname
            ), "RAA[1] dimension is incorrect"
            saa = amf_func.masked(fid[self.raa_name[0]][:])
            vaa = amf_func.masked(fid[self.raa_name[1]][:])
            saa[(saa > 360) | (saa < -360)] = np.nan
            vaa[(vaa > 360) | (vaa < -360)] = np.nan
            raa = np.abs(saa - vaa)
            saa = saa.ravel()
            vaa = vaa.ravel()
        # ND -> 1D
        raa = raa.ravel()

        # if raa_mode = 0 then use 180-raa
        if self.raa_mode == 0:
            raa = np.abs(180 - raa) % 360
        else:
            raa = np.abs(raa) % 360
        # raa should be in the range of [0, 180]
        raa = np.where(raa > 180, 360 - raa, raa)

        # Longitude / Latitude
        assert (
            fid[self.lon_name].dimensions == dimname
        ), "longitude dimension is incorrect"
        assert (
            fid[self.lat_name].dimensions == dimname
        ), "latitude dimension is incorrect"
        lon = amf_func.masked(fid[self.lon_name][:]).ravel()
        lat = amf_func.masked(fid[self.lat_name][:]).ravel()
        idx = (lon < -361) | (lon > 361) | (lat < -90) | (lat > 90)
        if np.count_nonzero(idx) > 0:
            lon[idx] = np.nan
            lat[idx] = np.nan
        # lon should be in the range of [-180, 180]
        lon = lon % 360
        lon = np.where(lon > 180, lon - 360, lon)

        # longitudecorners / latitudecorners
        # only if out_var_flag=True, then read it.
        if self.info["out_var_flag"]["latitudecorners"]:
            try:
                latcor = amf_func.masked(fid[self.latcor_name][:])
                latcor_dimname = fid[self.latcor_name].dimensions
                if latcor_dimname[:-1] == dimname:
                    assert latcor.shape[-1] == 4, (
                        "latcor dimension is in correct"
                    )
                    latcor = latcor.reshape(-1, 4)
                elif latcor_dimname[1:] == dimname:
                    assert latcor.shape[0] == 4, (
                        "latcor dimension is incorrect"
                    )
                    latcor = latcor.reshape(4, -1).T
                else:
                    assert False, "latcor dimension is incorrect"
            except IndexError:
                if self.verbose:
                    print(
                        "warning: can not read latcor data (latcor name is "
                        "incorrect or does not exist)"
                    )
        if self.info["out_var_flag"]["longitudecorners"]:
            try:
                loncor = amf_func.masked(fid[self.loncor_name][:])
                loncor_dimname = fid[self.loncor_name].dimensions
                if loncor_dimname[:-1] == dimname:
                    assert loncor.shape[-1] == 4, (
                        "loncor dimension is in correct"
                    )
                    loncor = loncor.reshape(-1, 4)
                elif loncor_dimname[1:] == dimname:
                    assert loncor.shape[0] == 4, (
                        "loncor dimension is incorrect"
                    )
                    loncor = loncor.reshape(4, -1).T
                else:
                    assert False, "loncor dimension is incorrect"
            except IndexError:
                if self.verbose:
                    print(
                        "warning: can not read loncor data (loncor name is "
                        "incorrect or does not exist)"
                    )
        # scd (only if bc_flag/vcd_flag=True)
        if self.info["bc_flag"] | self.info["vcd_flag"]:
            scd = amf_func.masked(fid[self.scd_name][:])
            scd_dimname = fid[self.scd_name].dimensions
            if scd_dimname == dimname:
                scd = scd.ravel()
            elif scd_dimname[:-1] == dimname:
                scd = scd[..., 0].ravel()
            elif scd_dimname[1:] == dimname:
                scd = scd[1:, ...].ravel()
            else:
                assert False, "scd dimension is incorrect"

        # intens (only if cf_flag=True)
        if self.info["cf_flag"]:
            intens = amf_func.masked(fid[self.intens_name][:])
            intens_dimname = fid[self.intens_name].dimensions
            if intens_dimname == dimname:
                intens = intens.ravel()
            else:
                assert False, "intens dimension is incorrect"

        # valid pixels
        if self.info["lon_min"] <= self.info["lon_max"]:
            idx = (
                (sza >= self.info["sza_min"]) &
                (sza <= self.info["sza_max"]) &
                (vza >= self.info["vza_min"]) &
                (vza <= self.info["vza_max"]) &
                (lat >= self.info["lat_min"]) &
                (lat <= self.info["lat_max"]) &
                (lon >= self.info["lon_min"]) &
                (lon <= self.info["lon_max"])
            )
        else:
            idx = (
                (sza >= self.info["sza_min"]) &
                (sza <= self.info["sza_max"]) &
                (vza >= self.info["vza_min"]) &
                (vza <= self.info["vza_max"]) &
                (lat <= self.info["lat_min"]) &
                (lat >= self.info["lat_max"]) &
                (
                    (lon >= self.info["lon_min"]) |
                    (lon <= self.info["lon_max"])
                )
            )

        # cosine(SZA/VZA)
        cossza = np.cos(np.radians(sza))
        cosvza = np.cos(np.radians(vza))
        # if wv_flag = True
        if self.info["wv_flag"]:
            assert (
                fid[self.wvidx_name].dimensions == dimname
            ), "window selection index dimension is incorrect"
            wvidx = amf_func.masked(fid[self.wvidx_name][:]).ravel()
        # Time (only alb_mode/pro_mode>0)
        if (self.info["alb_mode"] > 0) | (self.info["pro_mode"] > 0):
            time = amf_func.masked(fid[self.time_name][:])
            # if it is HARP format, try to read time_reference internally
            if self.info["file_type"] == 0:
                try:
                    ustr = fid[self.time_name].units
                    ustr = ustr.split()[-1].split("-")
                    self.time_reference = [int(u) for u in ustr]
                except IndexError:
                    pass
            # date is from the first measurement
            # timedelta0 is fraction of day since date
            date, timedelta0 = amf_func.convert_time(
                time, self.time_units, time_reference=self.time_reference
            )
            # time_mode=0, then time+delta_time
            # time.size=1 or time.size=delta_time.size
            if self.time_mode == 0:
                timedelta = amf_func.masked(fid[self.timedelta_name][:])
                timedelta = amf_func.convert_timedelta(
                    timedelta, self.timedelta_units
                )
                assert (np.array(timedelta0).size == 1) | (
                    timedelta0.shape == timedelta.shape
                ), "dimensions of time and reference time is inconsistent"
                timedelta = timedelta + timedelta0
                if fid[self.timedelta_name].dimensions == dimname[:-1]:
                    timedelta = np.outer(timedelta, np.ones(dim[-1]))
                elif fid[self.timedelta_name].dimensions == dimname[1:]:
                    timedelta = np.outer(np.ones(dim[0]), timedelta)
                else:
                    assert (
                        fid[self.timedelta_name].dimensions == dimname
                    ), "time_delta/time dimension is incorrect"
            else:
                if self.time_units == 3:
                    if fid[self.time_name].dimensions[:-1] == dimname[:-1]:
                        timedelta = np.outer(timedelta0, np.ones(dim[-1]))
                    elif fid[self.time_name].dimensions[:-1] == dimname[1:]:
                        timedelta = np.outer(np.ones(dim[0]), timedelta0)
                    else:
                        assert (
                            fid[self.time_name].dimensions[:-1] == dimname
                        ), "time dimension is incorrect"
                        timedelta = timedelta0
                else:
                    if fid[self.time_name].dimensions == dimname[:-1]:
                        timedelta = np.outer(timedelta0, np.ones(dim[-1]))
                    elif fid[self.time_name].dimensions == dimname[1:]:
                        timedelta = np.outer(np.ones(dim[0]), timedelta0)
                    else:
                        assert (
                            fid[self.time_name].dimensions == dimname
                        ), "time dimension is incorrect"
                        timedelta = timedelta0
            timedelta = timedelta.ravel()
        # close inp_file file
        fid.close()

        # read th file
        # only when spcorr_flag=True & th_mode=0
        if (self.info["spcorr_flag"]) & (self.info["th_mode"] == 0):
            fid = nc.Dataset(th_file, "r")
            th = amf_func.masked(fid[self.th_name][:])
            assert (
                th.shape == dim
            ), "surface altitude (for satellite) dimension is incorrect"
            th = th.ravel()
            # if it is HARP format and th_file=inp_file,
            # try to read self.th_units internally
            if (self.info["file_type"] == 0) & (th_file == inp_file):
                try:
                    ustr = fid[self.th_name].units
                    if ustr == "km":
                        self.th_units = 1
                    elif ustr == "m":
                        self.th_units = 0
                    else:
                        assert False, "th_name units is incorrect"
                except IndexError:
                    pass
            th = amf_func.height_convert(th, self.th_units)
            # close th_file
            fid.close()

        # read alb file
        if self.info["alb_mode"] == 0:
            fid = nc.Dataset(alb_file, "r")
            # for Lambertian surface
            if self.nalb == 1:
                alb = [np.full((self.nwv, size), np.nan)]
                # len(alb_name)=len(wavlength)
                if len(self.alb_name) == self.nwv:
                    for iwv in range(self.nwv):
                        alb0 = (
                            amf_func.masked(fid[self.alb_name[iwv]][:])
                            * self.alb_factor[0]
                        )
                        assert (
                            alb0.shape == dim
                        ), "surface albedo dimension is incorrect"
                        alb[0][iwv] = alb0.ravel()
                # len(alb_name)=1
                elif len(self.alb_name) == 1:
                    alb0 = (
                        amf_func.masked(fid[self.alb_name[0]][:])
                        * self.alb_factor[0]
                    )
                    if alb0.shape == dim:
                        for iwv in range(self.nwv):
                            alb[0][iwv] = alb0.ravel()
                    elif alb0.shape[:-1] == dim:
                        assert alb0.shape[-1] == self.nwv, (
                            "surface albedo dimension is incorrect"
                        )
                        alb[0][:] = alb0.reshape(-1, self.nwv).T
                        if alb0.shape == alb0.shape[::-1]:
                            if self.verbose:
                                print(
                                    "warning: doulbe check surface albedo "
                                    "dimension (last dim is wavelength?)"
                                )
                    elif alb0.shape[1:] == dim:
                        assert alb0.shape[0] == self.nwv, (
                            "surface albedo dimension is incorrect"
                        )
                        alb[0][:] = alb0.reshape(self.nwv, -1)
                    else:
                        assert False, (
                            "surface albedo dimension is incorrect"
                        )
                else:
                    assert False, (
                        "len(alb_name) is incorrect (1 or len(wavelength))"
                    )
            # for BRDF surface
            elif self.nalb == 3:
                if len(self.alb_name) == self.nalb:
                    for ialb in range(self.nalb):
                        alb0 = amf_func.masked(fid[self.alb_name[ialb]][:])
                        assert (
                            alb0.shape == dim
                        ), "surface albedo dimension is incorrect"
                        alb.append(alb0.ravel() * self.alb_factor[ialb])
                elif len(self.alb_name) == 1:
                    alb0 = amf_func.masked(fid[self.alb_name[0]][:])
                    if alb0.shape[:-1] == dim:
                        assert alb0.shape[-1] == self.nalb, (
                            "surface albedo dimension is incorrect"
                        )
                        for ialb in range(self.nalb):
                            alb.append(
                                alb0[..., ialb].ravel()
                                * self.alb_factor[ialb]
                            )
                        if alb0.shape == alb0.shape[::-1]:
                            if self.verbose:
                                print(
                                    "warning: doulbe check surface albedo "
                                    "dimension (last dim is albedo variale?)"
                                )
                    elif alb0.shape[1:] == dim:
                        assert alb0.shape[0] == self.nalb, (
                            "surface albedo dimension is incorrect"
                        )
                        for ialb in range(self.nalb):
                            alb.append(
                                alb0[ialb].ravel() * self.alb_factor[ialb]
                            )
                    else:
                        assert False, (
                            "len(alb_name) is incorrect (1 or nalb)"
                        )
            else:
                assert False, "len(lut_alb_name) is incorrect (1 or 3)"
            # close alb_file
            fid.close()

        # read cld file
        if self.info["cldcorr_flag"]:
            fid = nc.Dataset(cld_file, "r")
            cf = amf_func.masked(fid[self.cf_name][:])
            assert cf.shape == dim, "cf dimension is incorrect"
            cf = cf.ravel()
            cp = amf_func.masked(fid[self.cp_name][:])
            assert cp.shape == dim, "cp dimension is incorrect"
            cp = cp.ravel()
            # if inp_file is HARP format
            # (double) check the units for cloud pressure
            if (self.info["file_type"] == 0) & (cld_file == inp_file):
                try:
                    ustr = fid[self.cp_name].units
                    if ustr == "hPa":
                        self.cp_units = 1
                    elif ustr == "Pa":
                        self.cp_units = 0
                    else:
                        assert False, "cloud pressure units is incorrect"
                except IndexError:
                    pass
            cp = amf_func.pres_convert(cp, self.cp_units)
            # if ca_name is not set, ca=0.8
            # otherwise, read it from ca_name
            try:
                ca = amf_func.masked(fid[self.ca_name][:])
                assert ca.shape == dim, "ca dimension is incorrect"
            except IndexError:
                ca = np.full(dim, 0.8)
                if self.info["file_type"] > 0:
                    if self.verbose:
                        print(
                            "ca_name is not set or set it correctly,"
                            "and put cloud_albedo=0.8 for all cases"
                        )
            ca = ca.ravel()
            ca[np.isnan(cp)] = np.nan
            # correction for cf and ca when cld_mode=1
            if self.info["cld_mode"] == 1:
                cf = cf * ca / 0.8
                ca[:] = 0.8
                ca[np.isnan(cf)] = np.nan
            # close cld_file
            fid.close()

        # read profile
        if self.info["pro_mode"] == 0:
            fid = nc.Dataset(pro_file, "r")
            # surface pressure
            dimname1 = fid[self.info["pro_sp_name"]].dimensions
            # if inp_file is HARP format
            # Try to read the units for pressure
            if (self.info["file_type"] == 0) & (pro_file == inp_file):
                try:
                    ustr = fid[self.info["pro_sp_name"]].units
                    if ustr == "hPa":
                        self.info["pro_sp_units"] = 1
                    elif ustr == "Pa":
                        self.info["pro_sp_units"] = 0
                    else:
                        assert False, "cloud pressure units is incorrect"
                except IndexError:
                    pass
            sp = amf_func.masked(fid[self.info["pro_sp_name"]][:])
            assert sp.shape == dim, "sp dimension is incorrect"
            sp = sp.ravel()
            sp = amf_func.pres_convert(sp, self.info["pro_sp_units"])
            # profile
            pro = amf_func.masked(fid[self.info["pro_name"]][:])
            pro_dimname = fid[self.info["pro_name"]].dimensions
            # pro dimensions convert into [dim, nlayer]
            if pro_dimname[:-1] == dimname1:
                nlayer = pro.shape[-1]
                nlevel = nlayer + 1
                pro = pro.reshape(-1, nlayer)
            elif pro_dimname[1:] == dimname1:
                nlayer = pro.shape[0]
                nlevel = nlayer + 1
                pro = pro.reshape(nlayer, -1).T
            else:
                assert False, "trace gas profile dimension is incorrect"
            # temperature profile
            if (self.info["tcorr_flag"]) | (self.info["spcorr_flag"]):
                tpro = amf_func.masked(fid[self.info["tpro_name"]][:])
                tpro = amf_func.temp_convert(tpro, self.info["tpro_units"])
                tpro_dimname = fid[self.info["tpro_name"]].dimensions
                if tpro_dimname[:-1] == dimname1:
                    tpro = tpro.reshape(-1, nlayer)
                elif tpro_dimname[1:] == dimname1:
                    tpro = tpro.reshape(nlayer, -1).T
                # only surface temperature
                # when tcorr_flag=False/spcorr_flag=True
                elif tpro_dimname == dimname1:
                    assert not self.info[
                        "tcorr_flag"
                    ], "tpro dimension are incorrect!"
                    tpro = tpro.ravel()[:, np.newaxis()]
                else:
                    assert False, "temperature profile dimension is incorrect"
            # surface altitude for profile
            if self.info["spcorr_flag"]:
                # if inp_file is HARP format
                # try to check the units for altitude internally
                if (self.info["file_type"] == 0) & (pro_file == inp_file):
                    try:
                        ustr = fid[self.info["pro_th_name"]].units
                        if ustr == "km":
                            self.info["pro_th_units"] = 1
                        elif ustr == "m":
                            self.info["pro_th_units"] = 0
                        else:
                            assert False, "cloud pressure units is incorrect"
                    except IndexError:
                        pass
                th1 = amf_func.masked(fid[self.info["pro_th_name"]][:])
                th1 = amf_func.height_convert(th1, self.info["pro_th_units"])
                th1_dimname = fid[self.info["pro_th_name"]].dimensions
                assert (
                    th1_dimname == dimname1
                ), "surface altitude (for profile) dimension is incorrect"
                th1 = th1.ravel()
            # read pressure grid
            # pro_mode=0: layer midpoint of pressure coefficients
            if self.info["pro_grid_mode"] == 0:
                pro_pam = amf_func.masked(
                    fid[self.info["pro_grid_name"][0]][:]
                )
                pro_pbm = amf_func.masked(
                    fid[self.info["pro_grid_name"][1]][:]
                )
                pro_pam = amf_func.pres_convert(
                    pro_pam, self.info["pro_sp_units"]
                )
                assert (
                    pro_pam.size == pro_pbm.size == nlayer
                ), "pa/pb coeffs and profile layer is inconsistent"
                if amf_func.monotonic(pro_pbm) in [2, 4]:
                    pro_pam = pro_pam[::-1]
                    pro_pbm = pro_pbm[::-1]
                    pro = pro[:, ::-1]
                    if self.info["tcorr_flag"]:
                        tpro = tpro[:, ::-1]
                pro_pai = np.zeros(nlevel)
                pro_pbi = np.zeros(nlevel)
                pro_pai[0] = 0
                pro_pbi[0] = 1
                for i in range(pro_pam.size):
                    pro_pai[i + 1] = 2 * pro_pam[i] - pro_pai[i]
                    pro_pbi[i + 1] = 2 * pro_pbm[i] - pro_pbi[i]
                # Avoid truncation error
                pro_pai[np.abs(pro_pai) < 1e-3] = 0.
                pro_pbi[np.abs(pro_pbi) < 1e-6] = 0.
                pres = np.outer(np.ones(dim), pro_pai) + np.outer(sp, pro_pbi)
                assert (
                    amf_func.monotonic(pres) == 1
                ), "pressure profile is not monotonicity decreasing"
            # pro_mode=1: layer midpoint of pressure grid
            elif self.info["pro_grid_mode"] == 1:
                pres1 = amf_func.masked(fid[self.info["pro_grid_name"][0]][:])
                pres1_dimname = fid[self.info["pro_grid_name"][0]].dimensions
                pres1 = amf_func.pres_convert(pres1, self.info["pro_sp_units"])
                if pres1_dimname[:-1] == dimname1:
                    assert (
                        pres1.shape[-1] == nlayer
                    ), "pressure profile dimension is incorrect"
                    pres1 = pres1.reshape(-1, nlayer)
                elif pres1_dimname[1:] == dimname1:
                    assert (
                        pres1.shape[0] == nlayer
                    ), "pressure profile dimension is incorrect"
                    pres1 = pres1.reshape(nlayer, -1).T
                else:
                    assert False, "pressure profile dimension is incorrect"
                if amf_func.monotonic(pres1) == 2:
                    pres1 = pres1[:, ::-1]
                    pro = pro[:, ::-1]
                    if self.info["tcorr_flag"]:
                        tpro = tpro[:, ::-1]
                else:
                    assert (
                        amf_func.monotonic(pres1) == 1
                    ), "pressure profile is not monotonicity decreasing"
                pres = np.full((size, nlevel), np.nan)
                pres[:, 0] = sp
                for i in range(nlayer):
                    pres[:, i + 1] = 2 * pres1[:, i] - pres[:, i]
                # Avoid truncation error
                pres[pres < 0] = 0.0
                assert (
                    amf_func.monotonic(pres) == 1
                ), "pressure profile is not monotonicity decreasing"
            # tropopause
            if self.info["amftrop_flag"]:
                tropopause = amf_func.masked(
                    fid[self.info["tropopause_name"]][:]
                )
                tropopause_dimname = fid[
                    self.info["tropopause_name"]
                ].dimensions
                assert (
                    tropopause_dimname == dimname1
                ), "tropopause dimension is incorrect"
                tropopause = tropopause.ravel()
                # calculate tropopause layer when tropopause_mode=1
                # find closest level.
                if self.info["tropopause_mode"] == 1:
                    tropopause = amf_func.pres_convert(
                        tropopause, self.info["pro_sp_units"]
                    )
                    tropopause[~tropopause.mask] = np.nanargmin(
                        np.abs(
                            pres[~tropopause.mask, :]
                            - np.outer(
                                tropopause[~tropopause.mask], np.ones(nlevel)
                            )
                        ),
                        axis=1,
                    )
                tropopause[tropopause.mask] = -1
                tropopause = np.int8(tropopause)
                assert (
                    np.count_nonzero(tropopause == 0) == 0
                ), "tropopause index should be larger than 0"
            # close pro_file
            fid.close()
        # read other variables
        if var_file:
            fid = nc.Dataset(var_file, "r")
            for var_name in self.info["var_name"]:
                var0 = amf_func.masked(fid[var_name][:])
                assert var0.shape == dim, (
                    "variable[" + var_name + "] dimension is incorrect"
                )
                var.append(var0.ravel())
            # close var_file
            fid.close()
        # output (input information)
        data = {
            "dim": dim,
            "size": size,
            "date": date,
            "time": np.float64(timedelta),
            "wvidx": wvidx,
            "scd": np.float64(scd),
            "intens": np.float64(intens),
            "lat": np.float64(lat),
            "lon": np.float64(lon),
            "latcor": latcor,
            "loncor": loncor,
            "sza": np.float64(sza),
            "vza": np.float64(vza),
            "cossza": np.float64(cossza),
            "cosvza": np.float64(cosvza),
            "saa": saa,
            "vaa": vaa,
            "raa": np.float64(raa),
            "th": np.float64(th),
            "alb": [np.float64(a) for a in alb],
            "nalb": nalb,
            "cf": np.float64(cf),
            "cp": np.float64(cp),
            "ca": np.float64(ca),
            "pro": np.float64(pro),
            "tpro": np.float64(tpro),
            "pres": np.float64(pres),
            "sp": np.float64(sp),
            "tropopause": tropopause,
            "th1": np.float64(th1),
            "var": [np.float64(v) for v in var],
            "nlayer": nlayer,
            "nlevel": nlevel,
            "idx": idx,
        }
        return data

    def read_amflut(self, lut_file, lut_vars, wv):
        # lut_vars: including radiance / box-AMF /wavelengths /
        # normalized pressure grid / surface pressure / SZA / VZA / RAA /
        # albedos / other variables
        # radiance: wv x sp x SZA x VZA x RAA x albs x others
        # box-AMF: wv x pre x sp x SZA x VZA x RAA x albs x others
        # if the order of dimension is not same, and then will transpose it.
        fid = nc.Dataset(lut_file, "r")
        # intensity and box-AMF in LUT
        rad0 = fid[lut_vars[0]][:]
        bamf0 = fid[lut_vars[1]][:]
        # other variables
        nvar = len(lut_vars) - 2
        assert (rad0.ndim == nvar - 1) & (
            bamf0.ndim == nvar
        ), "lut_rad.ndim or lut_amf.ndim is not correct"
        dim_rad = fid[lut_vars[0]].dimensions
        dim_bamf = fid[lut_vars[1]].dimensions
        dimorder_rad = []  # dimension orders for radiance
        dimorder_bamf = []  # dimension orders for box-AMF
        var = []  # list of variables in LUT
        # adjust rad/bamf dimensions order to wv x (pre) x sp x SZA x ...
        i = 0
        for lut_var in lut_vars[2:]:
            data = np.float64(fid[lut_var][:])
            assert data.ndim == 1, (
                lut_var + " dimension number in LUT file is incorrect (not 1)"
            )
            dim_var = fid[lut_var].dimensions[0]
            # radiance do not have pressure grid diemension
            if i != nvar - self.nalb - self.nvar - 5:
                dimorder_rad.append(dim_rad.index(dim_var))
            dimorder_bamf.append(dim_bamf.index(dim_var))
            var.append(data)
            i += 1
        fid.close()
        # transpose rad0/amf0 into wv x (pres) x sp x sza x vza x raa x albs x
        # ohters
        rad0 = rad0.transpose(dimorder_rad)
        bamf0 = bamf0.transpose(dimorder_bamf)

        # interpolation into the selected wv
        nwv = len(wv)
        # LUT includes wavelength dimensions:
        if len(var) == 6 + self.nalb + self.nvar:
            # Check the monotonicity of the wavelength variable
            assert (
                amf_func.monotonic(var[0]) == 2
            ), "wavelength in lut_file is not monotonically increasing"
            rad = np.full((nwv,) + rad0.shape[1:], np.nan)
            bamf = np.full((nwv,) + bamf0.shape[1:], np.nan)
            for i in range(nwv):
                idx = bisect.bisect_right(var[0], wv[i])
                if idx == var[0].size:
                    idx = idx - 1
                dis = (
                    (wv[i] - var[0][idx - 1]) / (var[0][idx] - var[0][idx - 1])
                )
                rad[i, ...] = (
                    rad0[idx - 1, ...] * (1 - dis) + rad0[idx, ...] * dis
                )
                bamf[i, ...] = (
                    bamf0[idx - 1, ...] * (1 - dis) + bamf0[idx, ...] * dis
                )
            var[0] = np.array(wv)  # wavelength is replaced
        # LUT without wavelength dimension
        else:
            if (nwv > 1) & self.verbose:
                print("warning: no wavlength dimension in box-AMF LUT, AMF "
                      "calculation is the same for all wavelengths")
            rad = np.full((nwv,) + rad0.shape, np.nan)
            bamf = np.full((nwv,) + bamf0.shape, np.nan)
            for i in range(nwv):
                rad[i, ...] = rad0
                bamf[i, ...] = bamf0
            var.insert(0, np.array(wv))
        # pressure
        if amf_func.monotonic(var[1]) == 1:
            var[1] = var[1][::-1]
            bamf = bamf[:, ::-1, ...]
        assert (
            amf_func.monotonic(var[1]) == 2
        ), "pressure in lut_file is not monotonically increasing"
        # surface pressure
        if amf_func.monotonic(var[2]) == 1:
            var[2] = var[2][::-1]
            rad = rad[:, ::-1, ...]
            bamf = bamf[:, :, ::-1, ...]
        assert (
            amf_func.monotonic(var[2]) == 2
        ), "surface pressure in lut_file is not monotonically increasing"
        # solar/viewing zenith angle (Units: degree)
        if self.info["geo_units"] == 1:  # convert into cosine(SZA/VZA)
            var[3] = np.cos(np.radians(var[3]))
            var[4] = np.cos(np.radians(var[4]))
        if amf_func.monotonic(var[3]) == 1:
            var[3] = var[3][::-1]
            rad = rad[:, :, ::-1, ...]
            bamf = bamf[:, :, :, ::-1, ...]
        assert (
            amf_func.monotonic(var[3]) == 2
        ), "SZA in lut_file is not monotonically increasing"
        if amf_func.monotonic(var[4]) == 1:
            var[4] = var[4][::-1]
            rad = rad[:, :, :, ::-1, ...]
            bamf = bamf[:, :, :, :, ::-1, ...]
        assert (
            amf_func.monotonic(var[4]) == 2
        ), "VZA in lut_file is not monotonically increasing"
        # for the other variables
        for i in range(5, len(var)):
            assert amf_func.monotonic(var[i]) == 2, (
                "var["
                + str(i)
                + "] in lut_file is not monotonically increasing"
            )

        # output
        data = {"var": var, "rad": rad, "bamf": bamf}
        return data

    def cal_vcd(self, inp, amfres, bcres):
        # calculate VCD
        scd = inp["scd"]
        amf = amfres["amf"][0]
        vcd = np.full_like(amfres["amf"][0], np.nan)
        if self.info["amftrop_flag"]:
            amftrop = amfres["amf"][1]
            vcdtrop = np.full_like(amfres["amf"][1], np.nan)
            # remove background column
            if self.info["bc_x_name"].lower() == "lat":
                xdata = inp["lat"]
            elif self.info["bc_x_name"].lower() == "sza":
                xdata = inp["sza"]
            else:
                xdata = inp["cossza"]
            vcdstrat = np.interp(xdata, bcres["x"], bcres["y"])
            amfgeo = amfres["amf_geo"]
        else:
            vcdtrop = np.array([])

        for i in range(self.nwv):
            vcd[i] = scd / amf[i]
            if self.info["amftrop_flag"]:
                vcdtrop[i] = (scd - vcdstrat * amfgeo) / amftrop[i]
        # output
        result = {"vcd": vcd, "vcdtrop": vcdtrop}
        return result

    def copy_output(self, inp, amfres, vcdres=[]):
        # copy output from input information and calculated result

        var_names = {
            "wvidx": "wvidx",
            "latitude": "lat",
            "longitude": "lon",
            "latitudecorners": "latcor",
            "longitudecorners": "loncor",
            "solar_zenith_angle": "sza",
            "sensor_zenith_angle": "vza",
            "solar_azimuth_angle": "saa",
            "sensor_azimuth_angle": "vaa",
            "relative_azimuth_angle": "raa",
            "surface_altitude": "th",
            "surface_albedo": "alb",
            "cloud_fraction": "cf",
            "cloud_pressure": "cp",
            "cloud_albedo": "ca",
            "profile": "pro",
            "temperature": "tpro",
            "pressure": "pres",
            "surface_pressure": "sp",
            "tropopause": "tropopause",
            "other_variable": "var",
        }
        for key in var_names.keys():
            self.out[key] = inp[var_names[key]]

        # AMF result (amf_geo, amf, amf_clr, amf_cld, cloud_radiance fraction,
        # averaging_kernel, averaging_kernel_clr)
        for key in amfres.keys():
            if self.out_flag[key]:
                self.out[key] = amfres[key]

        # VCD result
        if vcdres:
            self.out["vcd"] = vcdres["vcd"]
            self.out["vcdtrop"] = vcdres["vcdtrop"]

        # dimension number
        # number of wavelength/albedo elements/layer/data dimensions
        if self.info["wv_flag"]:
            self.out["nwv"] = 1
        else:
            self.out["nwv"] = self.nwv
        self.out["nalb"] = self.nalb
        self.out["nlayer"] = inp["nlayer"]
        self.out["dim"] = inp["dim"]

    def __call__(self, var0, var1):
        # copy information based on configuration file and command arguments
        self.copy_variable(var0, var1)
        # check input settings
        self.check_setting()
        # set input variable name based on input information
        self.set_input_variable_name()
        # set output variable name based on input information
        self.set_output_variable_name()
        # check input variables
        self.check_variable()

        if self.verbose:
            print("Reading auxiliary dataset ...")
        # read LUT file
        amflut = self.read_amflut(
            self.lut_file, self.lut_vars, self.info["wavelength"]
        )
        # read gridded terrain height data if needed
        if (self.info["spcorr_flag"]) & (self.info["th_mode"] == 1):
            thlut = amf_th.read_thdat(
                self.th_file[0],
                self.th_name,
                self.info["th_lon_name"],
                self.info["th_lat_name"],
                self.th_units,
            )
        # read albedo climatology if needed
        if self.info["alb_mode"] >= 1:
            alblut = amf_alb.read_alblut(self.info)
        # read profile maps if needed
        if self.info["pro_mode"] >= 1:
            prodat = amf_pro.read_prodat(self.info)
        if self.verbose:
            print("Background correction:")
        # background correction calculation
        if self.info["bc_flag"]:
            bcres = amf_bc.bc(
                self.inp_file,
                self.info,
                self.lat_name,
                self.lon_name,
                self.sza_name,
                self.vza_name,
                self.scd_name
            )
            if self.info["bc_test_flag"]:
                return
        if self.verbose:
            print("AMF calculation start:")
        # read variables from each inp_file
        for i in range(len(self.inp_file)):
            if self.verbose:
                print("File: " + self.inp_file[i])
            # read general input information
            inp_file = self.inp_file[i]
            out_file = self.out_file[i]
            # read th file if th_mode = 0
            if (self.info["spcorr_flag"]) & (self.info["th_mode"] == 0):
                th_file = self.th_file[i]
            else:
                th_file = []
            # read alb file if alb_mode = 0
            if self.info["alb_mode"] == 0:
                alb_file = self.alb_file[i]
            else:
                alb_file = []
            # read cld file if cldcorr_flag = True
            if self.info["cldcorr_flag"]:
                cld_file = self.cld_file[i]
            else:
                cld_file = []
            # read profile file if pro_mode = 0
            if self.info["pro_mode"] == 0:
                pro_file = self.pro_file[i]
            else:
                pro_file = []
            if self.info["var_name"]:
                var_file = self.var_file[i]
            else:
                var_file = []
            # read input information from inp_file/th_file/alb_file/cld_file/
            # pro_file
            inp = self.read_info(
                inp_file,
                th_file=th_file,
                alb_file=alb_file,
                cld_file=cld_file,
                pro_file=pro_file,
                var_file=var_file,
            )
            if np.count_nonzero(inp["idx"]) == 0:
                if self.verbose:
                    print("no valid data for " + self.inp_file[i])
                continue
            # if th_mode == 1, then calculate terrain height
            if (self.info["spcorr_flag"]) & (self.info["th_mode"] == 1):
                inp["th"] = amf_th.cal_th(thlut, inp["lon"], inp["lat"])
            # if alb_mode >= 1, then calculate surface albedo
            if self.info["alb_mode"] >= 1:
                inp["alb"] = amf_alb.cal_alb(self.info, inp, alblut)
                inp["nalb"] = self.nalb
            # if pro_mode >= 1, then calculate profile
            if self.info["pro_mode"] >= 1:
                inp["nlayer"] = prodat.nlayer
                inp["nlevel"] = prodat.nlevel
                self.out["nlayer"] = prodat.nlayer
                (
                    inp["pro"],
                    inp["pres"],
                    inp["tpro"],
                    inp["tropopause"],
                    inp["sp"],
                    inp["th1"],
                ) = amf_pro.cal_pro(self.info, inp, prodat)
            # surface pressure correction
            if self.info["spcorr_flag"]:
                sp1 = amf_func.spcorr(
                    inp["sp"], inp["tpro"][:, 0], inp["th1"], inp["th"]
                )
                inp["pres"] = (
                    inp["pres"] / inp["sp"][..., None] * sp1[..., None]
                )
                inp["sp"] = sp1
            # calculate AMF
            amfres = amf_cal.cal_amf(self.info, inp, amflut)
            # calcualte vcd and vcdtrop
            if self.info["bc_flag"] | self.info["vcd_flag"]:
                vcdres = self.cal_vcd(inp, amfres, bcres)
            # copy output data from input variables and AMF result
                self.copy_output(inp, amfres, vcdres=vcdres)
            else:
                self.copy_output(inp, amfres)
            # output
            amf_out.output(
                out_file,
                self.info,
                self.out_flag,
                self.out_name,
                self.out
            )
