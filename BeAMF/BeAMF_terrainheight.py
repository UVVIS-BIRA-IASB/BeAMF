from netCDF4 import Dataset
import numpy as np
from collections import namedtuple
from scipy.interpolate import RegularGridInterpolator

from . import BeAMF_function as amf_func


def read_thdat(th_file, th_name, th_lon_name, th_lat_name, th_units):
    """
    read terrain height dataset
    output:
    lon: longitude
    lat: latitude
    th: terrain height with dimensions of lon x lat
    """
    # read th_file
    with Dataset(th_file, "r") as fid:
        th0 = amf_func.masked(fid[th_name][:]).astype("float64")
        dimname_th = fid[th_name].dimensions
        lat = amf_func.masked(fid[th_lat_name][:])
        lon = amf_func.masked(fid[th_lon_name][:])
        # check if lat/lon is 1D array
        assert lat.ndim == lon.ndim == 1

        # check th dimension = lon x lat
        assert th0.ndim == 2, (
            "ndim of DEM data is not 2 (lat x lon)"
        )
        dimorder_th = ()
        dimname_var = fid[th_lon_name].dimensions[0]
        dimorder_th += (dimname_th.index(dimname_var),)
        dimname_var = fid[th_lat_name].dimensions[0]
        dimorder_th += (dimname_th.index(dimname_var),)
        th0 = th0.transpose(dimorder_th)

        # cover the full range of fields
        th = np.full(np.array(th0.shape) + (2, 2), np.nan)
        th[1:-1, 1:-1] = th0
        th[0, :] = th[-2, :]  # longitude
        th[-1, :] = th[1, :]
        th[:, 0] = th[:, 1]  # latitude
        th[:, -1] = th[:, -2]

        # check and adjustment lon/lat monotonically increasing
        if amf_func.monotonic(lon) == 1:
            lon = lon[::-1]
            th = np.flip(th, axis=0)
        assert amf_func.monotonic(lon) == 2, (
            "lon in th_file is not monotonically in/decreasing"
        )
        if amf_func.monotonic(lat) == 1:
            lat = lat[::-1]
            th = np.flip(th, axis=1)
        assert amf_func.monotonic(lat) == 2, (
            "lat in th_file is not monotonically in/decreasing"
        )

        # final lat/lon grid
        lon = np.insert(lon, 0, lon[-1] - 360.0)
        lon = np.append(lon, lon[1] + 360.0)
        lat = np.insert(lat, 0, -90.0)
        lat = np.append(lat, 90.0)

        # change the height units to km
        th = amf_func.height_convert(th, th_units)

        data = namedtuple("data", "th lon lat")
        return data(th=th, lon=lon, lat=lat)


def cal_th(data, lon, lat):
    """
    calculate terrain height for corresponding satellite pixels based on
    digital elevation model (gridded data).
    Input:
    data: gridded terrrain height dataset (including .th, .lat and .lon)
    lat: latitude for satellite pixels
    lon: longitude for satellite pixels
    idx: valid pixel index
    Return:
    th: terrain height values for satellite pixels
    """
    interp_th = RegularGridInterpolator(
        (data.lon, data.lat), data.th, fill_value=np.nan
    )
    th = interp_th((lon, lat))
    return th
