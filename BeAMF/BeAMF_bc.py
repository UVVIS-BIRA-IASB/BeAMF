import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

from . import BeAMF_function as amf_func


__R__ = 6371  # radius of Earth
__Hatm__ = 8.5  # effective scale height of the atmosphere
__r__ = __R__ / __Hatm__


def bc(files, info, lat_name, lon_name, sza_name, vza_name, scd_name):
    """
    A simple background correction based on reference sector approach
    """
    lat = np.array([])
    lon = np.array([])
    sza = np.array([])
    vza = np.array([])
    scd = np.array([])

    lonlim = info["bc_lon_lim"]
    latlim = info["bc_lat_lim"]
    vzalim = info["bc_vza_lim"]
    szalim = info["bc_sza_lim"]
    xpoint = info["bc_x_interval"]
    numlim = info["bc_x_sample_limit"]
    ratio = 2  # sample ratio
    n = xpoint.size
    ns = (n - 1) * ratio + 1
    xpoints = np.interp(np.arange(ns) / ratio, np.arange(n), xpoint)
    xpoint1s = (xpoints[:-ratio] + xpoints[ratio:]) / 2.
    if lonlim[0] < -180:
        lonlim[0] = lonlim[0] + 360.0
    if lonlim[0] > 180:
        lonlim[0] = lonlim[0] - 360.0
    if lonlim[1] < -180:
        lonlim[1] = lonlim[1] + 360.0
    if lonlim[1] > 180:
        lonlim[1] = lonlim[1] - 360.0

    for file in files:
        with nc.Dataset(file, "r") as fid:
            lat0 = amf_func.masked(fid[lat_name][:])
            lon0 = amf_func.masked(fid[lon_name][:])
            vza0 = amf_func.masked(fid[vza_name][:])
            sza0 = amf_func.masked(fid[sza_name][:])
            scd0 = amf_func.masked(fid[scd_name][:])
            dim = lat0.shape
            if scd0.shape[:-1] == dim:
                scd0 = scd0[..., 0]
            elif scd0.shape[1:] == dim:
                scd0 = scd0[0]
            elif scd0.shape == dim:
                scd0 = scd0
            else:
                assert False, 'scd dimensions is incorrect'

            # valid pixels
            if lonlim[0] < lonlim[1]:
                idx = (
                    (lon0 >= lonlim[0])
                    & (lon0 <= lonlim[1])
                    & (lat0 >= latlim[0])
                    & (lat0 <= latlim[1])
                    & (vza0 >= vzalim[0])
                    & (vza0 <= vzalim[1])
                    & (sza0 >= szalim[0])
                    & (sza0 <= szalim[1])
                    & (np.isfinite(scd0))
                )
            else:
                idx = (
                    ((lon0 <= lonlim[1]) | (lon0 >= lonlim[0]))
                    & (lat0 >= latlim[0])
                    & (lat0 <= latlim[1])
                    & (vza0 >= vzalim[0])
                    & (vza0 <= vzalim[1])
                    & (sza0 >= szalim[0])
                    & (sza0 <= szalim[1])
                    & (np.isfinite(scd0))
                )

            if np.count_nonzero(idx) == 0:
                continue
            lat = np.append(lat, lat0[idx])
            vza = np.append(vza, vza0[idx])
            sza = np.append(sza, sza0[idx])
            lon = np.append(lon, lon0[idx])
            scd = np.append(scd, scd0[idx])

    # check number of valid pixels
    assert lat.size, "BC: no valid data in the selected data range."
    if lat.size < 1000:
        if info["verbose"]:
            print("warning: number of valid data in BC is less than 1000")

    # calculate amfgeo and VCD
    cossza = np.cos(np.radians(sza))
    cosvza = np.cos(np.radians(vza))
    amf_geo = (2 * __r__ + 1) / (
        np.sqrt((__r__ * cossza) ** 2 + 2 * __r__ + 1)
        + __r__ * cossza
    ) + 1 / cosvza
    ydata = scd / amf_geo

    # find xdata
    if info["bc_x_name"].lower() == "lat":
        xdata = lat
    elif info["bc_x_name"].lower() == "sza":
        xdata = sza
    else:
        xdata = cossza

    ypoint1s = np.full_like(xpoint1s, np.nan)
    for i in range(xpoints.size-ratio):
        idx = (xdata >= xpoints[i]) & (xdata < xpoints[i+ratio])
        ns = np.count_nonzero(idx)
        if ns >= numlim:
            data0 = ydata[idx]
            limit = np.percentile(data0, [5, 95])
            ypoint1s[i] = np.mean(
                data0[(data0 > limit[0]) & (data0 < limit[1])]
            )

    if info["bc_test_flag"]:
        plt.plot(xdata, ydata, ".", markersize=1)
        plt.plot(
            xpoint1s[::ratio], ypoint1s[::ratio], "or", lw=1, markersize=4
        )
        plt.plot(xpoint1s, ypoint1s, "-r", lw=1)
        plt.xlabel(info["bc_x_name"])
        plt.ylabel("initial VCD")

        # remove 1% of data as outliers
        ylimit = np.percentile(ydata, [0.1, 99.9])
        plt.ylim(ylimit)
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    result = {
        "x": xpoint1s,
        "y": ypoint1s,
    }
    return result
