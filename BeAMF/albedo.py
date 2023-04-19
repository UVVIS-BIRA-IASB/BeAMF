from netCDF4 import Dataset
import numpy as np
from collections import namedtuple
import calendar
import bisect
from scipy.interpolate import RegularGridInterpolator
import itertools
from . import function as amf_func


# read_alblut: read gridded albedo maps
# cal_alb: calculate albedo based on albedo maps and satellite geolocations,
#          and/or times
def read_alblut(info):
    if info["alb_mode"] == 1:
        data = read_alblut1(info)
    elif info["alb_mode"] == 2:
        data = read_alblut2(info)
    elif info["alb_mode"] == 3:
        data = read_alblut3(info)
    elif info["alb_mode"] == 9:
        data = read_alblut_omi(info)
    else:
        assert False, "alb_mode is out of range"
    return data


def read_alblut1(info):
    """
    read surface albedo climatology when alb_mode=1
    such as LER, BSA/WSA etc. one variable dataset
    input:
        alb_file/alb_name/alb_factor: list with 1 element
        alb_time_name: only set when alb_time_mode>=2
        alb_wv_name: optional dimension
    output: albwvs x time x lon x lat
    """
    # albedo values for AMF calculations
    albwvs = np.array(info["albwvs"])
    nalbwv = albwvs.size
    # variable name copy from info
    alb_file = info["alb_file"][0]
    alb_name = info["alb_name"][0]
    alb_factor = info["alb_factor"][0]
    alb_time_name = info["alb_time_name"]
    alb_wv_name = info["alb_wv_name"]
    alb_lat_name = info["alb_lat_name"]
    alb_lon_name = info["alb_lon_name"]

    # read albedo file
    with Dataset(alb_file, "r") as fid:
        # alb dimension should be time x wv x lat x lon
        alb0 = (
            amf_func.masked(fid[alb_name][:]) * alb_factor
        )
        dimname_alb = fid[alb_name].dimensions
        lat = amf_func.masked(fid[alb_lat_name][:])
        lon = amf_func.masked(fid[alb_lon_name][:])
        # check if lat/lon are 1-D array
        assert (lat.ndim == 1) & (lon.ndim == 1)

        # check alb dimension and change to (time) x wv x lat x lon
        dimorder_alb = []
        # alb_time only for when alb_time_mode>=1
        # alb_time_mode=0, no time dimensions
        if info["alb_time_mode"] >= 1:
            time = amf_func.masked(fid[alb_time_name][:])
            # check if time is 1-D array
            assert time.ndim == 1
            # find the order of dimension in the albedo grid
            # and check it monotonicity
            dimname_var = fid[alb_time_name].dimensions[0]
            dimorder_alb += (dimname_alb.index(dimname_var),)
            if info["alb_time_mode"] >= 2:
                assert amf_func.monotonic(time) == 2, (
                    "time in alb_file is not monotonically increasing"
                )
        # alb_wvs only for when info['alb_wv_name'] is set.
        if len(alb_wv_name) > 0:
            wvs = amf_func.masked(fid[alb_wv_name][:])
            # check if time is 1-D array
            assert wvs.ndim == 1
            # find the order of dimension in the albedo grid
            # and check it monotonicity
            dimname_var = fid[alb_wv_name].dimensions[0]
            dimorder_alb += (dimname_alb.index(dimname_var),)
            assert amf_func.monotonic(wvs) == 2, (
                "wvs in alb_file is not monotonically increasing"
            )
        # find the order of lat/lon dimension in the albedo grid
        dimname_var = fid[alb_lat_name].dimensions[0]
        dimorder_alb += (dimname_alb.index(dimname_var),)
        dimname_var = fid[alb_lon_name].dimensions[0]
        dimorder_alb += (dimname_alb.index(dimname_var),)
        assert len(dimorder_alb) == alb0.ndim

        # convert alb into time x wv x lat x lon
        alb0 = alb0.transpose(dimorder_alb)
        # time dimension set as 1 when alb_time_mode=0
        dim_alb = alb0.shape
        if info["alb_time_mode"] == 0:
            dim_alb = (1,) + dim_alb
        alb0 = alb0.reshape(dim_alb)

        # check and adjustment lat/lon monotonically increasing
        if amf_func.monotonic(lat) == 1:
            lat = lat[::-1]
            alb0 = np.flip(alb0, axis=-2)
        assert amf_func.monotonic(lat) == 2, (
            "lat in alb_file is not monotonically in/decreasing"
        )
        if amf_func.monotonic(lon) == 1:
            lon = lon[::-1]
            alb0 = np.flip(alb0, axis=-1)
        assert amf_func.monotonic(lon) == 2, (
            "lon in alb_file is not monotonically in/decreasing"
        )

        # calculate albedo for the selceted wavelengths
        if len(alb_wv_name) > 0:
            alb1 = np.zeros((nalbwv, dim_alb[0], dim_alb[2], dim_alb[3]))
            # calculate albedo for selected wavelengths
            for i in range(nalbwv):
                assert (albwvs[i] >= wvs[0]) & (albwvs[i] <= wvs[-1]), (
                    "albedo wavelength is out of range"
                )
                idx = bisect.bisect_right(wvs, albwvs[i])
                if idx == wvs.size:
                    idx = wvs.size - 1
                dis = (albwvs[i] - wvs[idx - 1]) / (wvs[idx] - wvs[idx - 1])
                alb1[i, ...] = (
                    alb0[:, idx - 1, ...] * (1 - dis)
                    + alb0[:, idx, ...] * dis
                )
            dim_alb1 = alb1.shape
        # if alb_wv_name is not set, alb set the same for all wavelengths.
        else:
            dim_alb1 = (nalbwv,) + dim_alb
            alb1 = np.full(dim_alb1, np.nan)
            for i in range(nalbwv):
                alb1[i] = alb0
        # copy alb data to alb and set boundary values
        if info["alb_region_flag"]:  # regional map
            if info["alb_time_mode"] == 1:
                alb = np.zeros(
                    (dim_alb1[0], dim_alb1[1] + 2, dim_alb1[2], dim_alb1[3])
                )
                alb[:, 1:-1, :, :] = alb1
                alb[:, 0, :, :] = alb[:, -2, :, :]
                alb[:, -1, :, :] = alb[:, 1, :, :]
        else:  # global map
            if (info["alb_time_mode"] == 0) | (info["alb_time_mode"] == 2):
                alb = np.zeros(
                    (
                        dim_alb1[0],
                        dim_alb1[1],
                        dim_alb1[2] + 2,
                        dim_alb1[3] + 2,
                    )
                )
                alb[:, :, 1:-1, 1:-1] = alb1
                alb[:, :, 0, :] = alb[:, :, 1, :]
                alb[:, :, -1, :] = alb[:, :, -2, :]
                alb[:, :, :, 0] = alb[:, :, :, -2]
                alb[:, :, :, -1] = alb[:, :, :, 1]
            elif info["alb_time_mode"] == 1:
                alb = np.zeros(
                    (
                        dim_alb1[0],
                        dim_alb1[1] + 2,
                        dim_alb1[2] + 2,
                        dim_alb1[3] + 2,
                    )
                )
                alb[:, 1:-1, 1:-1, 1:-1] = alb1
                alb[:, 0, :, :] = alb[:, -2, :, :]
                alb[:, -1, :, :] = alb[:, 1, :, :]
                alb[:, :, 0, :] = alb[:, :, 1, :]
                alb[:, :, -1, :] = alb[:, :, -2, :]
                alb[:, :, :, 0] = alb[:, :, :, -2]
                alb[:, :, :, -1] = alb[:, :, :, 1]
            # final lat/lon grid
            lat = np.insert(lat, 0, -90.0)
            lat = np.append(lat, 90.0)
            lon = np.insert(lon, 0, lon[-1] - 360.0)
            lon = np.append(lon, lon[1] + 360.0)

    alblut = namedtuple("alblut", "alb time wv lat lon")
    return alblut(alb=[alb], time=time, wv=albwvs, lat=lat, lon=lon)


def read_alblut2(info):
    """
    read VZA dependent surface albedo climatology when alb_mode=2
    input:
        alb_file/alb_factor: list with 1 element
        alb_name: list with 1/2 element(s)
        alb_time_name: only set when alb_time_mode>=2
        alb_wv_name: optional dimension
    output: albwvs x time x lon x lat
        if alb_name is one element
        value = alb(0) + alb(1)*vza + ... alb(nvza-1)*vza**(nvza-1)
        if alb_name is two elements
        value = alb0 + alb1(0) + alb1(1)*vza + ... alb1(nvza-1)*vza**(nvza-1)
    """
    # albedo values for AMF calculations
    albwvs = np.array(info["albwvs"])
    nalbwv = albwvs.size
    # variable name copy from info
    alb_file = info["alb_file"][0]
    alb_name = info["alb_name"][:]
    alb_factor = info["alb_factor"][0]
    alb_time_name = info["alb_time_name"]
    alb_wv_name = info["alb_wv_name"]
    alb_lat_name = info["alb_lat_name"]
    alb_lon_name = info["alb_lon_name"]

    # initialized output variables
    wvs = np.array([])
    time = np.array([])

    # check number of alb_name
    assert len(alb_name) in [1, 2]

    # read albedo file
    with Dataset(alb_file, "r") as fid:
        # alb0 has (time),(wv), lat, lon and nvza dimensions
        # len(info['alb_name']) should be 1/2, checked in beprep.check_variable
        if len(alb_name) == 1:
            alb0 = amf_func.masked(fid[alb_name[0]][:]) * alb_factor
            dimname_alb = fid[alb_name[0]].dimensions
        else:
            # VZA dependent term
            alb0 = amf_func.masked(fid[alb_name[1]][:]) * alb_factor
            dimname_alb = fid[alb_name[1]].dimensions
            # LER term
            alb_tmp = amf_func.masked(fid[alb_name[0]][:]) * alb_factor
            dim_tmp = fid[alb_name[0]].dimensions
            # VZA term is first or last dimension
            if dimname_alb[1:] == dim_tmp:
                alb0[0, ...] = alb0[0, ...] + alb_tmp
            else:
                assert dimname_alb[:-1] == dim_tmp
                alb0[..., 0] = alb0[..., 0] + alb_tmp
        lat = amf_func.masked(fid[alb_lat_name][:])
        lon = amf_func.masked(fid[alb_lon_name][:])
        # check if lat/lon are 1-D array
        assert (lat.ndim == 1) & (lon.ndim == 1)

        # check alb dimension and change to time x wv x lat x lon x nvza
        dimorder_alb = []
        # vza dimension is unknown, identify time/wv/lat/lon dimensions,
        # the left is vza dimension
        dim_index = list(np.arange(alb0.ndim))
        # alb_time only for when alb_time_mode>=1
        if info["alb_time_mode"] >= 1:
            time = amf_func.masked(fid[alb_time_name][:])
            # check if time is 1-D array
            assert time.ndim == 1
            # find the order of dimension in the albedo grid
            # and check it monotonicity
            dimname_var = fid[alb_time_name].dimensions[0]
            dimorder_alb += (dimname_alb.index(dimname_var),)
            dim_index.remove(dimname_alb.index(dimname_var))
            if info["alb_time_mode"] >= 2:
                assert amf_func.monotonic(time) == 2, (
                    "time in alb_file is not monotonically increasing"
                )
        # alb_wv only for when info['alb_wv_name'] is set.
        if len(alb_wv_name) > 0:
            wvs = amf_func.masked(fid[alb_wv_name][:])
            # check if wavelength is 1-D array
            assert len(wvs.shape) == 1
            # find the order of dimension in the albedo grid
            # and check it monotonicity
            dimname_var = fid[alb_wv_name].dimensions[0]
            dimorder_alb += (dimname_alb.index(dimname_var),)
            dim_index.remove(dimname_alb.index(dimname_var))
            assert amf_func.monotonic(wvs) == 2, (
                "wvs in alb_file is not monotonically increasing"
            )
        # find the order of lat/lon dimension in the albedo grid
        dimname_var = fid[alb_lat_name].dimensions[0]
        dimorder_alb += (dimname_alb.index(dimname_var),)
        dim_index.remove(dimname_alb.index(dimname_var))
        dimname_var = fid[alb_lon_name].dimensions[0]
        dimorder_alb += (dimname_alb.index(dimname_var),)
        dim_index.remove(dimname_alb.index(dimname_var))

        dimorder_alb += (dim_index[0],)  # for vza dependency
        assert len(dimorder_alb) == alb0.ndim
        alb0 = alb0.transpose(dimorder_alb)
        nvza = alb0.shape[-1]
        # time dimension set as 1 when alb_time_mode=0
        dim_alb = alb0.shape
        if info["alb_time_mode"] == 0:
            dim_alb = (1,) + dim_alb
            alb0 = alb0.reshape(dim_alb)

        # check and adjustment lat/lon monotonically increasing
        if amf_func.monotonic(lat) == 1:
            lat = lat[::-1]
            alb0 = np.flip(alb0, axis=-3)
        assert amf_func.monotonic(lat) == 2, (
            "lat in alb_file is not monotonically in/decreasing"
        )
        if amf_func.monotonic(lon) == 1:
            lon = lon[::-1]
            alb0 = np.flip(alb0, axis=-2)
        assert amf_func.monotonic(lon) == 2, (
            "lon in alb_file is not monotonically in/decreasing"
        )

        # calculate albedo for the selceted wavelengths
        if len(alb_wv_name) > 0:
            alb1 = np.zeros(
                (nalbwv, dim_alb[0], dim_alb[2], dim_alb[3], dim_alb[4])
            )
            # calculate albedo for selected wavelengths
            for i in range(nalbwv):
                assert (albwvs[i] >= wvs[0]) & (albwvs[i] <= wvs[-1]), (
                    "albedo wavelength is out of range"
                )
                idx = bisect.bisect_right(wvs, albwvs[i])
                if idx == wvs.size:
                    idx = wvs.size - 1
                dis = (albwvs[i] - wvs[idx - 1]) / (wvs[idx] - wvs[idx - 1])
                alb1[i, ...] = (
                    alb0[:, idx - 1, ...] * (1 - dis)
                    + alb0[:, idx, ...] * dis
                )
            dim_alb1 = alb1.shape
        # if alb_wv_name is not set, alb set the same for all wavelengths.
        else:
            dim_alb1 = (nalbwv,) + dim_alb
            alb1 = np.full(dim_alb1, np.nan)
            for i in range(nalbwv):
                alb1[i] = alb0
        # copy alb data to alb and set boundary values
        if info["alb_region_flag"]:  # regional map
            if info["alb_time_mode"] == 1:
                alb = np.zeros(
                    (
                        dim_alb1[0],
                        dim_alb1[1] + 2,
                        dim_alb1[2],
                        dim_alb1[3],
                        dim_alb1[4],
                    )
                )
                alb[:, 1:-1, :, :, :] = alb1
                alb[:, 0, :, :, :] = alb[:, -2, :, :, :]
                alb[:, -1, :, :, :] = alb[:, 1, :, :, :]
        else:  # global map
            if (info["alb_time_mode"] == 0) | (info["alb_time_mode"] == 2):
                alb = np.zeros(
                    (
                        dim_alb1[0],
                        dim_alb1[1],
                        dim_alb1[2] + 2,
                        dim_alb1[3] + 2,
                        dim_alb1[4],
                    )
                )
                alb[:, :, 1:-1, 1:-1, :] = alb1
                alb[:, :, 0, :, :] = alb[:, :, 1, :, :]
                alb[:, :, -1, :, :] = alb[:, :, -2, :, :]
                alb[:, :, :, 0, :] = alb[:, :, :, -2, :]
                alb[:, :, :, -1, :] = alb[:, :, :, 1, :]
            elif info["alb_time_mode"] == 1:
                alb = np.zeros(
                    (
                        dim_alb1[0],
                        dim_alb1[1] + 2,
                        dim_alb1[2] + 2,
                        dim_alb1[3] + 2,
                        dim_alb1[4],
                    )
                )
                alb[:, 1:-1, 1:-1, 1:-1, :] = alb1
                alb[:, 0, :, :, :] = alb[:, -2, :, :, :]
                alb[:, -1, :, :, :] = alb[:, 1, :, :, :]
                alb[:, :, 0, :, :] = alb[:, :, 1, :, :]
                alb[:, :, -1, :, :] = alb[:, :, -2, :, :]
                alb[:, :, :, 0, :] = alb[:, :, :, -2, :]
                alb[:, :, :, -1, :] = alb[:, :, :, 1, :]
            # final lat/lon grid
            lat = np.insert(lat, 0, -90.0)
            lat = np.append(lat, 90.0)
            lon = np.insert(lon, 0, lon[-1] - 360.0)
            lon = np.append(lon, lon[1] + 360.0)

    alblut = namedtuple("alblut", "alb time wv lat lon nvza")
    return alblut(alb=[alb], time=time, wv=albwvs, lat=lat, lon=lon, nvza=nvza)


def read_alblut3(info):
    """
    read BRDF surface albedo climatology when alb_mode=3
    list of alb_name: (time) x lon x lat
    calculation only for single wavelength (len(albwvs) = 1)
    """
    # only for single wavelength calculation when alb_mode=3
    assert len(info["albwvs"]) == 1
    # variable name copy from info (no alb_wv_name for this case)
    alb_file = info["alb_file"][0]
    alb_name = info["alb_name"][:]
    alb_factor = info["alb_factor"][:]
    alb_time_name = info["alb_time_name"]
    alb_lat_name = info["alb_lat_name"]
    alb_lon_name = info["alb_lon_name"]
    # number of elements for surface albedo
    nalb = len(alb_name)

    # initialized output variables
    alb = []

    # read albedo file
    with Dataset(alb_file[0], "r") as fid:
        # alb dimension should be time x lat x lon
        for i in range(nalb):
            alb.append(amf_func.masked(fid[alb_name[i]][:]) * alb_factor[i])
            if i == 0:
                dimname_alb = fid[alb_name[i]].dimensions
            else:
                assert fid[alb_name[i]].dimensions == dimname_alb
        # latitude/longitude
        lat = amf_func.masked(fid[alb_lat_name][:])
        lon = amf_func.masked(fid[alb_lon_name][:])
        # check if lat/lon are 1-D array
        assert (lat.ndim == 1) & (lon.ndim == 1)

        # check alb dimension and change to time x wv x lat x lon
        dimorder_alb = []
        # alb_time only for when alb_time_mode >= 0
        if info["alb_time_mode"] >= 1:
            time = amf_func.masked(fid[alb_time_name][:])
            # check if time is 1-D array
            assert time.ndim == 1
            # find the order of dimension in the albedo grid
            # and check it monotonicity
            dimname_var = fid[alb_time_name].dimensions[0]
            dimorder_alb += (dimname_alb.index(dimname_var),)
            if info["alb_time_mode"] >= 2:
                assert amf_func.monotonic(time) == 2, (
                    "time in alb_file is not monotonically increasing"
                )
        # find the order of lat/lon dimension in the albedo grid
        dimname_var = fid[alb_lat_name].dimensions[0]
        dimorder_alb += (dimname_alb.index(dimname_var),)
        dimname_var = fid[alb_lon_name].dimensions[0]
        dimorder_alb += (dimname_alb.index(dimname_var),)
        assert len(dimorder_alb) == alb[0].ndim
        for i in range(nalb):
            alb[i] = alb[i].transpose(dimorder_alb)
        # check and adjustment lat/lon monotonically increasing
        if amf_func.monotonic(lat) == 1:
            lat = lat[::-1]
            for i in range(nalb):
                alb[i] = np.flip(alb[i], axis=-2)
        assert amf_func.monotonic(lat) == 2, (
            "lat in alb_file is not monotonically in/decreasing"
        )
        if amf_func.monotonic(lon) == 1:
            lon = lon[::-1]
            for i in range(nalb):
                alb[i] = np.flip(alb[i], axis=-1)
        assert amf_func.monotonic(lon) == 2, (
            "lon in alb_file is not monotonically in/decreasing"
        )

        # time dimension set as 1 when alb_time_mode=0
        dim_alb = alb[0].shape
        if info["alb_time_mode"] == 0:
            dim_alb = (1,) + dim_alb
            for i in range(nalb):
                alb[i] = alb[i].reshape(dim_alb)
        # add a dimension for wavelength
        dim_alb = (1,) + dim_alb
        for i in range(nalb):
            alb[i] = alb[i].reshape(dim_alb)
        # copy alb data to alb and set boundary values
        if info["alb_region_flag"]:  # regional map
            for i in range(nalb):
                if info["alb_time_mode"] == 1:
                    alb1 = np.zeros(
                        (
                            dim_alb[0],
                            dim_alb[1] + 2,
                            dim_alb[2],
                            dim_alb[3]
                        )
                    )
                    alb1[:, 1:-1, :, :] = alb[i]
                    alb1[:, 0, :, :] = alb1[:, -2, :, :]
                    alb1[:, -1, :, :] = alb1[:, 1, :, :]
                alb[i] = alb1
        else:  # global map
            for i in range(nalb):
                if (info["alb_time_mode"] == 0) | (info["alb_time_mode"] == 2):
                    alb1 = np.zeros(
                        (
                            dim_alb[0],
                            dim_alb[1],
                            dim_alb[2] + 2,
                            dim_alb[3] + 2
                        )
                    )
                    alb1[:, :, 1:-1, 1:-1] = alb[i]
                    alb1[:, :, 0, :] = alb1[:, :, 1, :]
                    alb1[:, :, -1, :] = alb1[:, :, -2, :]
                    alb1[:, :, :, 0] = alb1[:, :, :, -2]
                    alb1[:, :, :, -1] = alb1[:, :, :, 1]
                elif info["alb_time_mode"] == 1:
                    alb1 = np.zeros(
                        (
                            dim_alb[0],
                            dim_alb[1] + 2,
                            dim_alb[2] + 2,
                            dim_alb[3] + 2
                        )
                    )
                    alb1[:, 1:-1, 1:-1, 1:-1] = alb[i]
                    alb1[:, 0, :, :] = alb1[:, -2, :, :]
                    alb1[:, -1, :, :] = alb1[:, 1, :, :]
                    alb1[:, :, 0, :] = alb1[:, :, 1, :]
                    alb1[:, :, -1, :] = alb1[:, :, -2, :]
                    alb1[:, :, :, 0] = alb1[:, :, :, -2]
                    alb1[:, :, :, -1] = alb1[:, :, :, 1]
                alb[i] = alb1
            # final lat/lon grid
            lat = np.insert(lat, 0, -90.0)
            lat = np.append(lat, 90.0)
            lon = np.insert(lon, 0, lon[-1] - 360.0)
            lon = np.append(lon, lon[1] + 360.0)

    alblut = namedtuple("alblut", "alb time lat lon")
    return alblut(alb=alb, time=time, lat=lat, lon=lon)


def read_alblut_omi(info):
    """
    read OMI surface albedo climatology when alb_mode=9
    alb_name: (time) x albwvs x lat x lon (yearly or monthly)
    latitude/longitude need a correction (shift by 0.25 degree)
    """
    # albedo values for AMF calculations
    albwvs = np.array(info["albwvs"])
    nalbwv = albwvs.size
    # variable name copy from info (alb_time_name is not used)
    alb_file = info["alb_file"][0]
    alb_name = info["alb_name"][0]
    alb_factor = info["alb_factor"][0]
    alb_wv_name = info["alb_wv_name"]
    alb_lat_name = info["alb_lat_name"]
    alb_lon_name = info["alb_lon_name"]

    # initialized output variables
    wvs = np.array([])
    time = np.array([])

    if np.abs(alb_factor - 0.001) >= np.finfo(float).eps:
        if info["verbose"]:
            print(
                "warnning: for OMI LER, alb_factor("
                + str(alb_factor)
                + ") is not 0.001"
            )
    assert info["alb_time_mode"] <= 1, (
        "For OMI LER data, alb_time_mode("
        + str(info["alb_time_mode"])
        + ") is not correct (>1)"
    )
    # alb_region_flag = True for OMI albedo
    if info["alb_region_flag"] & info["verbose"]:
        print("warnning: for OMI LER data, alb_region_flag should be False")
    # read albedo file
    with Dataset(alb_file, "r") as fid:
        # alb dimension should be (time) x wv x lat x lon
        alb0 = amf_func.masked(fid[alb_name][:]) * alb_factor
        lat = amf_func.masked(fid[alb_lat_name][:])
        lon = amf_func.masked(fid[alb_lon_name][:])
        wvs = amf_func.masked(fid[alb_wv_name][:])

        # time dimension set as 1 when alb_time_mode=0
        # otherwise, time.size=12
        dim_alb = alb0.shape
        if info["alb_time_mode"] == 0:
            assert alb0.ndim == 3
            dim_alb = (1,) + dim_alb
            alb0 = alb0.reshape(dim_alb)
        else:
            assert info["alb_time_mode"] == 1
            assert (alb0.ndim == 4) & (dim_alb[0] == 12)
        # calculate albedo for the selceted wavelengths
        tempalb = np.zeros((nalbwv, dim_alb[0], dim_alb[2], dim_alb[3]))
        # calculate albedo for selected wavelengths
        for i in range(nalbwv):
            assert (albwvs[i] >= wvs[0]) & (albwvs[i] <= wvs[-1]), (
                "albedo wavelength is out of range"
            )
            idx = bisect.bisect_right(wvs, albwvs[i])
            if idx == wvs.size:
                idx = wvs.size - 1
            dis = (albwvs[i] - wvs[idx - 1]) / (wvs[idx] - wvs[idx - 1])
            tempalb[i, ...] = (
                alb0[:, idx - 1, :, :] * (1 - dis) + alb0[:, idx, :, :] * dis
            )
        # copy alb data to alb and set boundary values
        if info["alb_time_mode"] == 0:
            alb = np.zeros(
                (nalbwv, dim_alb[0], dim_alb[2] + 2, dim_alb[3] + 2)
            )
            alb[:, :, 1:-1, 1:-1] = tempalb
            alb[:, :, 0, :] = alb[:, :, 1, :]
            alb[:, :, -1, :] = alb[:, :, -2, :]
            alb[:, :, :, 0] = alb[:, :, :, -2]
            alb[:, :, :, -1] = alb[:, :, :, 1]
        else:
            alb = np.zeros(
                (nalbwv, dim_alb[0] + 2, dim_alb[2] + 2, dim_alb[3] + 2)
            )
            alb[:, 1:-1, 1:-1, 1:-1] = tempalb
            alb[:, 0, :, :] = alb[:, -2, :, :]
            alb[:, -1, :, :] = alb[:, 1, :, :]
            alb[:, :, 0, :] = alb[:, :, 1, :]
            alb[:, :, -1, :] = alb[:, :, -2, :]
            alb[:, :, :, 0] = alb[:, :, :, -2]
            alb[:, :, :, -1] = alb[:, :, :, 1]
        # final lat/lon grid
        nlat, nlon = lat.size, lon.size
        lat = (np.arange(nlat) + 0.5) / nlat * 180 - 90.0
        lon = (np.arange(nlon) + 0.5) / nlon * 360 - 180.0
        lat = np.insert(lat, 0, -90.0)
        lat = np.append(lat, 90.0)
        lon = np.insert(lon, 0, lon[-1] - 360.0)
        lon = np.append(lon, lon[1] + 360.0)

        alblut = namedtuple("alblut", "alb time wv lat lon")
        return alblut(alb=[alb], time=time, wv=albwvs, lat=lat, lon=lon)


def cal_alb(info, inp, alblut):
    # day of the year for the satellite observation file
    dayofyear = inp["date"].timetuple().tm_yday

    # set time values when alb_time_mode=1/2
    if info["alb_time_mode"] == 1:  # monthly data
        # which depends on if it is leap year
        if calendar.isleap(inp["date"].year):
            days = np.array(
                [
                    -15,
                    16,
                    46,
                    76,
                    106.5,
                    137,
                    167.5,
                    198,
                    229,
                    259.5,
                    290,
                    320.5,
                    351,
                    382,
                ]
            )
        else:
            days = np.array(
                [
                    -15,
                    16,
                    45.5,
                    75,
                    105.5,
                    136,
                    166.5,
                    197,
                    228,
                    258.5,
                    289,
                    319.5,
                    350,
                    381,
                ]
            )
    elif info["alb_time_mode"] == 2:
        days = alblut.time
    # calculate albedo maps for corresponding date
    if info["alb_time_mode"] in [1, 2]:
        idx1 = bisect.bisect_right(days, dayofyear)
        assert (idx1 > 0) & (idx1 < days.size)
        idx0 = idx1 - 1
        dis = (dayofyear - days[idx0]) / (days[idx1] - days[idx0])
        alb_daily = []
        for alb in alblut.alb:
            alb_daily.append(
                alb[:, idx0, ...] * (1 - dis) + alb[:, idx1, ...] * dis
            )
    elif info["alb_time_mode"] == 0:  # no time dimension
        alb_daily = [alb[0][:, 0, ...]]
    if info["alb_mode"] in [1, 9]:
        data = cal_alb1(alb_daily, alblut.lat, alblut.lon, info, inp)
    elif info["alb_mode"] == 2:
        data = cal_alb2(
            alb_daily, alblut.lat, alblut.lon, alblut.nvza, info, inp
        )
    elif info["alb_mode"] == 3:
        data = cal_alb3(alb_daily, alblut.lat, alblut.lon, inp)
    else:
        assert False, "alb_mode is out of range"
    return data


def cal_alb1(albedo, lat, lon, info, inp):
    """
    calculate surface albedo for corresponding satellite pixels based on
    surface albedo database when alb_mode=1/9
    """
    nwv = len(info["wavelength"])
    alb = np.full((nwv, inp["size"]), np.nan)
    if albedo[0].shape[0] == 1:  # no wv dimension in albedo data
        interp_alb = RegularGridInterpolator(
            (lat, lon),
            albedo[0][0, ...],
            bounds_error=False,
            fill_value=np.nan
        )
        alb[0, :] = interp_alb((inp["lat"], inp["lon"]))
        for i in range(1, nwv):
            alb[i, :] = alb[0, :]
    else:  # albedo.shape[0]>1, including wv dimension in albedo data
        for i in range(nwv):
            interp_alb = RegularGridInterpolator(
                (lat, lon),
                albedo[0][i, ...],
                bounds_error=False,
                fill_value=np.nan,
            )
            alb[i, :] = interp_alb((inp["lat"], inp["lon"]))
    return [alb]


def cal_alb2(albedo, lat, lon, nvza, info, inp):
    """
    calculate surface albedo for corresponding satellite pixels based on
    surface albedo database when alb_mode=2 (VZA dependent LER)
    """
    nwv = len(info["wavelength"])
    alb = np.full((nwv, inp["size"]), np.nan)
    vza = inp["vza"]
    # calculate vza in radians
    # vza = np.radians(inp["vza"])
    # check vza sign
    if info["alb_vza_sign"]:
        vza[inp["raa"] > 90] = -vza[inp["raa"] > 90]
    else:
        vza[inp["raa"] <= 90] = -vza[inp["raa"] <= 90]
    if albedo[0].shape[0] == 1:  # no wv dimension in albedo data
        for i in range(nvza):
            interp_alb = RegularGridInterpolator(
                (lat, lon),
                albedo[0][0, ..., i],
                bounds_error=False,
                fill_value=np.nan,
            )
            alb[0, :] = (
                alb[0, :] + interp_alb((inp["lat"], inp["lon"])) * (vza**i)
            )
        for i in range(1, nwv):
            alb[i, :] = alb[0, :]
    else:  # albedo.shape[0]>1, including wv dimension in albedo data
        for i, j in itertools.product(range(nwv), range(nvza)):
            interp_alb = RegularGridInterpolator(
                (lat, lon),
                albedo[0][j, ..., i],
                bounds_error=False,
                fill_value=np.nan,
            )
            alb[i, :] = (
                alb[i, :] + interp_alb((inp["lat"], inp["lon"])) * (vza**j)
            )
    return [alb]


def cal_alb3(albedo, lat, lon, inp):
    """
    calculate surface albedo for corresponding satellite pixels based on
    surface albedo database when alb_mode=3 (BRDF parameters)
    """
    alb = []
    for alb0 in albedo:
        interp_alb = RegularGridInterpolator(
            (lat, lon), alb0, bounds_error=False, fill_value=np.nan
        )
        alb.append(interp_alb((inp["lat"], inp["lon"])))
    return alb
