import netCDF4 as nc
import numpy as np
from collections import namedtuple
import calendar
import bisect
import datetime as dt
from scipy.interpolate import RegularGridInterpolator
from . import BeAMF_function as amf_func


def read_prodat(info):
    """
    read profile data when (climatology, CTM data etc.)
    initialized variables from pro_file
    pro_pro/pro_tpro dimension should be time x layer x lat x lon
    pro_topopause dimension should be time x lat x lon
    pro_th dimension should be lat x lon
    pro_grid_mode should be 0 or 1.
    pro_pam/pro_pbm: midpoint of hybrid coeff. pa and pb
    pro_pai/pro_pbi: layer interface of hybrid coeff. pa and pb
    pam = (pai[:-1] + pai[1:]) / 2
    pbm = (pbi[:-1] + pbi[1:]) / 2
    tpro is only used when tcorr_flag or spcorr_flag is True
    tropopause is only used when amftrop_flag=True
    th is only used when spcorr_flag is True
    """
    assert info["pro_mode"] in [1, 2, 3], "pro_mode is not 1/2/3"

    # initialize variables from pro_file
    pro_pro = np.array([])  # layer x time x lat x lon
    pro_tpro = np.array([])  # layer x time x lat x lon
    pro_tropopause = np.array([])  # time x lat x lon
    pro_sp = np.array([])  # time x lat x lon
    pro_pres = np.array([])  # layer x time x lat x lon
    pro_time = np.array([])  # time
    pro_th = np.array([])  # lat x lon
    lat = np.array([])
    lon = np.array([])
    pam = np.array([])  # layer
    pbm = np.array([])  # layer
    pai = np.array([])  # layer+1
    pbi = np.array([])  # layer+1
    pro_grid_flag = False

    # if pro_mode = 2/3, then len(pro_file) = 1
    if (info["pro_mode"] == 2) | (info["pro_mode"] == 3):
        assert len(info["pro_file"]) == 1, "only 1 pro_file when pro_mode=2/3"
    elif info["pro_mode"] == 1:
        assert len(info["pro_file"]) in [
            1,
            2,
        ], "1 or 2 pro_file when pro_mode=1"
    # start read pro_file
    for pro_file in info["pro_file"]:
        with nc.Dataset(pro_file, "r") as fid:
            dimname_lon = fid[info["pro_lon_name"]].dimensions
            dimname_lat = fid[info["pro_lat_name"]].dimensions
            assert len(dimname_lon) == len(dimname_lat) == 1
            dimname_lon = dimname_lon[0]
            dimname_lat = dimname_lat[0]
            if info["pro_mode"] in [1, 2]:
                time = amf_func.masked(fid[info["pro_time_name"]][:])
                dimname_time = fid[info["pro_time_name"]].dimensions
                assert len(dimname_time) == 1
                dimname_time = dimname_time[0]
                # if pro_mode, only 12 months
                if info["pro_mode"] == 2:
                    assert (
                        time.size == 12
                    ), "time.size in profile data is not 12 when pro_mode=2"
            # read surface pressure
            # ensure that dimension of sp = time x lat x lon
            sp = amf_func.masked(fid[info["pro_sp_name"]][:])
            sp = amf_func.pres_convert(sp, info["pro_sp_units"])
            dimname = fid[info["pro_sp_name"]].dimensions
            dimorder = ()
            if len(dimname) == 2:
                dimorder += (dimname.index(dimname_lat),)
                dimorder += (dimname.index(dimname_lon),)
                sp = sp.transpose(dimorder)
                if (info["pro_mode"] == 1) | (info["pro_mode"] == 2):
                    sp = np.einsum("i,jk->ijk", np.ones(time.size), sp)
                elif info["pro_mode"] == 3:
                    sp = sp[np.newaxis, ...]
            elif len(dimname) == 3:
                dimorder += (dimname.index(dimname_time),)
                dimorder += (dimname.index(dimname_lat),)
                dimorder += (dimname.index(dimname_lon),)
                sp = sp.transpose(dimorder)
            else:
                assert False, "ndim of sp in pro_file is not 2 or 3"
            # read profile data
            pro = amf_func.masked(fid[info["pro_name"]][:])
            dimname = fid[info["pro_name"]].dimensions
            if (info["pro_mode"] == 1) | (info["pro_mode"] == 2):
                assert len(dimname) == 4, "ndim of pro in pro_file is not 4"
                # get the layer dimension name
                dimname_layer = list(dimname)
                dimname_layer.remove(dimname_time)
                dimname_layer.remove(dimname_lat)
                dimname_layer.remove(dimname_lon)
                dimname_layer = dimname_layer[0]
                # sort the dimensions to time x layer x lat x lon
                dimorder = ()
                dimorder += (dimname.index(dimname_layer),)
                dimorder += (dimname.index(dimname_time),)
                dimorder += (dimname.index(dimname_lat),)
                dimorder += (dimname.index(dimname_lon),)
                pro = pro.transpose(dimorder)
            elif info["pro_mode"] == 3:
                assert len(dimname) == 3, "ndim of pro in pro_file is not 3"
                # get the layer dimension name
                dimname_layer = list(dimname)
                dimname_layer.remove(dimname_lat)
                dimname_layer.remove(dimname_lon)
                dimname_layer = dimname_layer[0]
                # sort the dimensions to layer x lat x lon
                dimorder = ()
                dimorder += (dimname.index(dimname_layer),)
                dimorder += (dimname.index(dimname_lat),)
                dimorder += (dimname.index(dimname_lon),)
                pro = pro.transpose(dimorder)
                pro = pro[:, np.newaxis, ...]
            # number of layer/level
            nlayer = pro.shape[0]
            nlevel = nlayer + 1

            # read tropopause if any(amftrop_flag=True)
            if info["amftrop_flag"]:
                tropopause = amf_func.masked(fid[info["tropopause_name"]][:])
                if info["tropopause_mode"] == 1:
                    tropopause = amf_func.pres_convert(
                        tropopause, info["pro_sp_units"]
                    )
                dimname = fid[info["tropopause_name"]].dimensions
                dimorder = ()
                if len(dimname) == 2:
                    dimorder += (dimname.index(dimname_lat),)
                    dimorder += (dimname.index(dimname_lon),)
                    tropopause = tropopause.transpose(dimorder)
                    if (info["pro_mode"] == 1) | (info["pro_mode"] == 2):
                        tropopause = np.einsum(
                            "i,jk->ijk", np.ones(time.size), tropopause
                        )
                    elif info["pro_mode"] == 3:
                        tropopause = tropopause[np.newaxis, ...]
                elif len(dimname) == 3:
                    dimorder += (dimname.index(dimname_time),)
                    dimorder += (dimname.index(dimname_lat),)
                    dimorder += (dimname.index(dimname_lon),)
                    tropopause = tropopause.transpose(dimorder)
                else:
                    assert (
                        False
                    ), "ndim of tropopause in pro_file is not 2 or 3"
            # read pro_grid_name if pro_grid_mode=1
            if info["pro_grid_mode"] == 1:  # layer boundary
                pres = amf_func.masked(fid[info["pro_grid_name"][0]][:])
                pres = amf_func.pres_convert(pres, info["pro_sp_units"])
                dimname = fid[info["pro_grid_name"][0]].dimensions
                if (info["pro_mode"] == 1) | (info["pro_mode"] == 2):
                    # should be time x layer x lat x lon
                    assert (
                        len(dimname) == 4
                    ), "ndim of pro in pro_file is not 4"
                    # sort the dimensions to time x layer x lat x lon
                    dimorder = ()
                    dimorder += (dimname.index(dimname_layer),)
                    dimorder += (dimname.index(dimname_time),)
                    dimorder += (dimname.index(dimname_lat),)
                    dimorder += (dimname.index(dimname_lon),)
                    pres = pres.transpose(dimorder)
                # should be layer x lat x lon
                elif info["pro_mode"] == 3:
                    assert (
                        len(dimname) == 3
                    ), "ndim of pro in pro_file is not 3"
                    # get the layer dimension name
                    dimname_layer = list(dimname)
                    dimname_layer.remove(dimname_lat)
                    dimname_layer.remove(dimname_lon)
                    dimname_layer = dimname_layer[0]
                    # sort the dimensions to layer x lat x lon
                    dimorder = ()
                    dimorder += (dimname.index(dimname_layer),)
                    dimorder += (dimname.index(dimname_lat),)
                    dimorder += (dimname.index(dimname_lon),)
                    pro = pro.transpose(dimorder)
                    # add time dimensions
                    pro = pro[:, np.newaxis, ...]
            # read temperature profile data
            if info["tcorr_flag"] | info["spcorr_flag"]:
                tpro = amf_func.masked(fid[info["tpro_name"]][:])
                tpro = amf_func.temp_convert(tpro, info["tpro_units"])
                dimname = fid[info["tpro_name"]].dimensions
                if (info["pro_mode"] == 1) | (info["pro_mode"] == 2):
                    assert (
                        len(dimname) == 4
                    ), "ndim of tpro in pro_file is not 4"
                    dimorder = ()
                    dimorder += (dimname.index(dimname_layer),)
                    dimorder += (dimname.index(dimname_time),)
                    dimorder += (dimname.index(dimname_lat),)
                    dimorder += (dimname.index(dimname_lon),)
                    tpro = tpro.transpose(dimorder)
                elif info["pro_mode"] == 3:
                    assert (
                        len(dimname) == 3
                    ), "ndim of tpro in pro_file is not 3"
                    dimorder = ()
                    dimorder += (dimname.index(dimname_layer),)
                    dimorder += (dimname.index(dimname_lat),)
                    dimorder += (dimname.index(dimname_lon),)
                    tpro = tpro.transpose(dimorder)
                    tpro = tpro[:, np.newaxis, ...]
            # th/lon/lat/pa/pb only set once
            if lon.size == 0:
                lon = amf_func.masked(fid[info["pro_lon_name"]][:])
                lat = amf_func.masked(fid[info["pro_lat_name"]][:])
                if info["pro_grid_mode"] == 0:  # layer boundary
                    pam = amf_func.masked(fid[info["pro_grid_name"][0]][:])
                    pbm = amf_func.masked(fid[info["pro_grid_name"][1]][:])
                    pam = amf_func.pres_convert(pam, info["pro_sp_units"])
                    dimname = fid[info["pro_grid_name"][0]].dimensions
                    assert (
                        len(dimname) == 1
                    ), "ndim of pro_grid_name in pro_file is not 1"
                    assert (
                        dimname[0] == dimname_layer
                    ), "dimension of pro_grid_name in pro_file is incorrect"
                    dimname = fid[info["pro_grid_name"][1]].dimensions
                    assert (
                        len(dimname) == 1
                    ), "ndim of pro_grid_name in pro_file is not 1"
                    assert (
                        dimname[0] == dimname_layer
                    ), "dimension of pro_grid_name in pro_file is incorrect"
                    # check order of pressure grid
                    # should be decresing (the first is surface layer)
                    if amf_func.monotonic(pbm) in [2, 4]:
                        pam = pam[::-1]
                        pbm = pbm[::-1]
                        pro_grid_flag = True  # reverse vertical grid
                    else:
                        assert amf_func.monotonic(pbm) in [1, 3], (
                            "pbm in pro_file is not monotonically de/"
                            "increasing"
                        )
                    pai = np.full(nlevel, np.nan)
                    pbi = np.full(nlevel, np.nan)
                    pai[0] = 0.0
                    pbi[0] = 1.0
                    for i in range(nlayer):
                        pai[i + 1] = 2 * pam[i] - pai[i]
                        pbi[i + 1] = 2 * pbm[i] - pbi[i]
                # read th if spcorr_flag set.
                if info["spcorr_flag"]:
                    pro_th = amf_func.masked(fid[info["pro_th_name"]][:])
                    pro_th = amf_func.height_convert(
                        pro_th, info["pro_th_units"]
                    )
                    dimname = fid[info["pro_th_name"]].dimensions
                    assert len(dimname) == 2, "ndim of th in pro_file is not 2"
                    dimorder = ()
                    dimorder += (dimname.index(dimname_lat),)
                    dimorder += (dimname.index(dimname_lon),)
                    pro_th = pro_th.transpose(dimorder)
                    if not info["pro_region_flag"]:
                        pro_th1 = pro_th.copy()
                        pro_th = np.full(
                            np.array(pro_th1.shape) + np.array([2, 2]), np.nan
                        )
                        pro_th[1:-1, 1:-1] = pro_th1
                        pro_th[0, :] = pro_th[1, :]
                        pro_th[-1, :] = pro_th[-2, :]
                        pro_th[:, 0] = pro_th[:, -2]
                        pro_th[:, -1] = pro_th[:, 1]
                # add values to the variables
                # time
                pro_time = time.copy()
                # trace gas profile
                # if pro_region_flag=False, then it's global map,
                # and need to add lon/lat boundary for the maps
                if info["pro_region_flag"]:
                    pro_pro = pro.copy()
                else:
                    pro_pro = np.full(
                        np.array(pro.shape) + np.array([0, 0, 2, 2]), np.nan
                    )
                    pro_pro[:, :, 1:-1, 1:-1] = pro
                    pro_pro[:, :, 0, :] = pro_pro[:, :, 1, :]
                    pro_pro[:, :, -1, :] = pro_pro[:, :, -2, :]
                    pro_pro[:, :, :, 0] = pro_pro[:, :, :, -2]
                    pro_pro[:, :, :, -1] = pro_pro[:, :, :, 1]
                # surface pressure
                if info["pro_region_flag"]:
                    pro_sp = sp.copy()
                else:
                    pro_sp = np.full(
                        np.array(sp.shape) + np.array([0, 2, 2]), np.nan
                    )
                    pro_sp[:, 1:-1, 1:-1] = sp
                    pro_sp[:, 0, :] = pro_sp[:, 1, :]
                    pro_sp[:, -1, :] = pro_sp[:, -2, :]
                    pro_sp[:, :, 0] = pro_sp[:, :, -2]
                    pro_sp[:, :, -1] = pro_sp[:, :, 1]
                # pressure grid
                if info["pro_grid_mode"] == 1:  # layer boundary
                    if info["pro_region_flag"]:
                        pro_pres = pres.copy()
                    else:
                        pro_pres = np.full(
                            np.array(pres.shape) + np.array([0, 0, 2, 2]),
                            np.nan,
                        )
                        pro_pres[:, :, 1:-1, 1:-1] = pres
                        pro_pres[:, :, 0, :] = pro_pres[:, :, 1, :]
                        pro_pres[:, :, -1, :] = pro_pres[:, :, -2, :]
                        pro_pres[:, :, :, 0] = pro_pres[:, :, :, -2]
                        pro_pres[:, :, :, -1] = pro_pres[:, :, :, 1]
                # temperature profile
                if info["tcorr_flag"] | info["spcorr_flag"]:
                    if info["pro_region_flag"]:
                        pro_tpro = tpro.copy()
                    else:
                        pro_tpro = np.full(
                            np.array(tpro.shape) + np.array([0, 0, 2, 2]),
                            np.nan,
                        )
                        pro_tpro[:, :, 1:-1, 1:-1] = tpro
                        pro_tpro[:, :, 0, :] = pro_tpro[:, :, 1, :]
                        pro_tpro[:, :, -1, :] = pro_tpro[:, :, -2, :]
                        pro_tpro[:, :, :, 0] = pro_tpro[:, :, :, -2]
                        pro_tpro[:, :, :, -1] = pro_tpro[:, :, :, 1]
                # tropopause
                if info["amftrop_flag"]:
                    if info["pro_region_flag"]:
                        pro_tropopause = tropopause.copy()
                    else:
                        pro_tropopause = np.full(
                            np.array(tropopause.shape) + np.array([0, 2, 2]),
                            np.nan,
                        )
                        pro_tropopause[:, 1:-1, 1:-1] = tropopause
                        pro_tropopause[:, 0, :] = pro_tropopause[:, 1, :]
                        pro_tropopause[:, -1, :] = pro_tropopause[:, -2, :]
                        pro_tropopause[:, :, 0] = pro_tropopause[:, :, -2]
                        pro_tropopause[:, :, -1] = pro_tropopause[:, :, 1]
            else:
                # time
                pro_time = np.append(pro_time, time)
                # trace gas profile
                # if pro_region_flag=False, then it's global map,
                # and need to add lon/lat boundary for the maps
                if info["pro_region_flag"]:
                    pro_pro = np.append(pro_pro, pro, axis=1)
                else:
                    pro1 = np.full(
                        np.array(pro.shape) + np.array([0, 0, 2, 2]), np.nan
                    )
                    pro1[:, :, 1:-1, 1:-1] = pro
                    pro1[:, :, 0, :] = pro1[:, :, 1, :]
                    pro1[:, :, -1, :] = pro1[:, :, -2, :]
                    pro1[:, :, :, 0] = pro1[:, :, :, -2]
                    pro1[:, :, :, -1] = pro1[:, :, :, 1]
                    pro_pro = np.append(pro_pro, pro1, axis=1)
                # surface pressure
                if info["pro_region_flag"]:
                    pro_sp = np.append(pro_sp, sp, axis=0)
                else:
                    sp1 = np.full(
                        np.array(sp.shape) + np.array([0, 2, 2]), np.nan
                    )
                    sp1[:, 1:-1, 1:-1] = sp
                    sp1[:, 0, :] = sp1[:, 1, :]
                    sp1[:, -1, :] = sp1[:, -2, :]
                    sp1[:, :, 0] = sp1[:, :, -2]
                    sp1[:, :, -1] = sp1[:, :, 1]
                    pro_sp = np.append(pro_sp, sp1, axis=0)
                # pressure grid
                if info["pro_grid_mode"] == 1:  # layer boundary
                    if info["pro_region_flag"]:
                        pro_pres = np.append(pro_pres, pres, axis=1)
                    else:
                        pres1 = np.full(
                            np.array(pres.shape) + np.array([0, 0, 2, 2]),
                            np.nan,
                        )
                        pres1[:, :, 1:-1, 1:-1] = pres
                        pres1[:, :, 0, :] = pres1[:, :, 1, :]
                        pres1[:, :, -1, :] = pres1[:, :, -2, :]
                        pres1[:, :, :, 0] = pres1[:, :, :, -2]
                        pres1[:, :, :, -1] = pres1[:, :, :, 1]
                        pro_pres = np.append(pro_pres, pres1, axis=1)
                # temperature profile
                if info["tcorr_flag"] | info["spcorr_flag"]:
                    if info["pro_region_flag"]:
                        pro_tpro = np.append(pro_tpro, tpro, axis=1)
                    else:
                        tpro1 = np.full(
                            np.array(tpro.shape) + np.array([0, 0, 2, 2]),
                            np.nan,
                        )
                        tpro1[:, :, 1:-1, 1:-1] = tpro
                        tpro1[:, :, 0, :] = tpro1[:, :, 1, :]
                        tpro1[:, :, -1, :] = tpro1[:, :, -2, :]
                        tpro1[:, :, :, 0] = tpro1[:, :, :, -2]
                        tpro1[:, :, :, -1] = tpro1[:, :, :, 1]
                        pro_tpro = np.append(pro_tpro, tpro1, axis=1)
                # tropopause
                if info["amftrop_flag"]:
                    if info["pro_region_flag"]:
                        pro_tropopause = np.append(
                            pro_tropopause, tropopause, axis=0
                        )
                    else:
                        tropopause1 = np.full(
                            np.array(tropopause.shape) + np.array([0, 2, 2]),
                            np.nan,
                        )
                        tropopause1[:, 1:-1, 1:-1] = tropopause
                        tropopause1[:, 0, :] = tropopause1[:, 1, :]
                        tropopause1[:, -1, :] = tropopause1[:, -2, :]
                        tropopause1[:, :, 0] = tropopause1[:, :, -2]
                        tropopause1[:, :, -1] = tropopause1[:, :, 1]
                        pro_tropopause = np.append(
                            pro_tropopause, tropopause1, axis=0
                        )
    # if pro_mode=2, the time grid need to be set but later
    if info["pro_mode"] == 2:
        pro_pro = np.insert(pro_pro, 0, pro_pro[:, -1, ...], axis=1)
        pro_pro = np.insert(pro_pro, pro_pro[:, 1, ...], axis=1)
        pro_sp = np.insert(pro_sp, 0, pro_sp[:, -1, ...], axis=1)
        pro_sp = np.insert(pro_sp, pro_sp[:, 1, ...], axis=1)
        if info["pro_grid_mode"] == 1:
            pro_pres = np.insert(pro_pres, 0, pro_pres[:, -1, ...], axis=1)
            pro_pres = np.insert(pro_pres, pro_pres[:, 1, ...], axis=1)
        if info["tcorr_flag"] | info["spcorr_flag"]:
            pro_tpro = np.insert(pro_tpro, 0, pro_tpro[:, -1, ...], axis=1)
            pro_tpro = np.insert(pro_tpro, pro_tpro[:, 1, ...], axis=1)
        if info["amftrop_flag"]:
            pro_tropopause = np.insert(
                pro_tropopause, 0, pro_tropopause[:, -1, ...], axis=1
            )
            pro_tropopause = np.insert(
                pro_tropopause, pro_tropopause[:, 1, ...], axis=1
            )
    # check monotonicity of lat
    if amf_func.monotonic(lat) == 1:
        lat = lat[::-1]
        pro_pro = pro_pro[:, :, ::-1, :]
        if info["tcorr_flag"] | info["spcorr_flag"]:
            pro_tpro = pro_tpro[:, :, ::-1, :]
        if info["amftrop_flag"]:
            pro_tropopause = pro_tropopause[:, ::-1, :]
        if info["spcorr_flag"]:
            pro_th = pro_th[::-1, :]
        pro_sp = pro_sp[:, ::-1, :]
    else:
        assert (
            amf_func.monotonic(lat) == 2
        ), "array pro_lat is not monotonically de/increasing"
    # check monotonicity of lon
    if amf_func.monotonic(lon) == 1:
        lon = lon[::-1]
        pro_pro = pro_pro[:, :, :, ::-1]
        if info["tcorr_flag"] | info["spcorr_flag"]:
            pro_tpro = pro_tpro[:, :, :, ::-1]
        if info["amftrop_flag"]:
            pro_tropopause = pro_tropopause[:, :, ::-1]
        if info["spcorr_flag"]:
            pro_th = pro_th[:, ::-1]
        pro_sp = pro_sp[:, :, ::-1]
    else:
        assert (
            amf_func.monotonic(lon) == 2
        ), "array pro_lon is not monotonically increasing"
    # check monotonicity of pro_time
    if info["pro_mode"] == 1:
        assert (
            amf_func.monotonic(pro_time) == 2
        ), "array pro_time is not monotonically increasing"
    # if reversed the pressure profile
    if info["pro_grid_mode"] == 1:
        if amf_func.monotonic(pro_pres) == 2:
            pro_pres = pro_pres[::-1, ...]
            pro_grid_flag = True
        else:
            assert (
                amf_func.monotonic(pro_pres, axis=1) == 1
            ), "pres in pro_file is not monotonically de/increasing"
        pro_pres1 = np.copy(pro_pres)
        pro_pres = np.full((nlevel,) + pro_sp.shape, np.nan)
        pro_pres[0, :] = pro_sp
        for i in range(nlayer):
            pro_pres[:, i + 1] = 2 * pro_pres1[:, i] - pro_pres[:, i]
    assert (
        amf_func.monotonic(pro_pres) == 1
    ), "pres in pro_file is not monotonicity decreasing"
    if pro_grid_flag:
        pro_pro = pro_pro[::-1, :, :, :]
        if info["tcorr_flag"]:
            pro_tpro = pro_tpro[::-1, :, :, :]
    # final lat/lon grid
    if not info["pro_region_flag"]:
        lat = np.insert(lat, 0, -90.0)
        lat = np.append(lat, 90.0)
        lon = np.insert(lon, 0, lon[-1] - 360.0)
        lon = np.append(lon, lon[1] + 360.0)
    data = namedtuple(
        "data",
        "pro tpro tropopause sp pres time th lat "
        "lon pai pbi nlayer nlevel",
    )
    return data(
        pro=pro_pro,
        tpro=pro_tpro,
        tropopause=pro_tropopause,
        sp=pro_sp,
        pres=pro_pres,
        time=pro_time,
        th=pro_th,
        lat=lat,
        lon=lon,
        pai=pai,
        pbi=pbi,
        nlayer=nlayer,
        nlevel=nlevel,
    )


def cal_pro(info, inp, data):
    """
    calculate profile for corresponding satellite pixels based on
    CTM or climatology database when pro_mode>1
    data: gridded profile data (including trace gas/temperature profiles,
          tropopause, surface pressure, terrain height etc)
    pydate: date (datetime format) for the 1st pixel of the orbit
    time: fraction of day for measurement time of satellite pixels
    lat/lon: latitude/longitude of satellite pixel
    idx: valid pixels
    """
    pydate = inp["date"]
    size = inp["size"]
    lat = inp["lat"]
    lon = inp["lon"]
    time = inp["time"]
    idx = inp["idx"]
    pai = data.pai
    pbi = data.pbi
    nlayer = data.nlayer
    nlevel = data.nlevel

    pro = np.full(
        (
            size,
            nlayer,
        ),
        np.nan,
    )
    sp = np.full(size, np.nan)
    pres = np.full(
        (
            size,
            nlevel,
        ),
        np.nan,
    )
    if info["tcorr_flag"]:
        tpro = np.full(
            (
                size,
                nlayer,
            ),
            np.nan,
        )
    else:
        tpro = np.array([])
    if info["amftrop_flag"]:
        tropopause = np.full(size, np.nan)
    else:
        tropopause = np.array([])
    if info["spcorr_flag"]:
        th = np.full(size, np.nan)
        if not info["tcorr_flag"]:
            tpro = np.full(size, np.nan)
    else:
        th = np.array([])
    # set time grid when alb_time_mode=1/2
    if info["pro_mode"] == 1:
        # calculate fraction of day
        # 0: data.time is days since pro_time_reference
        if info["pro_time_mode"] == 0:
            pro_time0 = info["pro_time_reference"]
            if len(pro_time0) == 3:
                reference = dt.datetime(
                    pro_time0[0], pro_time0[1], pro_time0[2]
                )
            elif len(pro_time0) == 6:
                reference = dt.datetime(
                    pro_time0[0],
                    pro_time0[1],
                    pro_time0[2],
                    pro_time0[3],
                    pro_time0[4],
                    pro_time0[5],
                )
            else:
                assert False, "len(pro_time_reference) is incorrect (3/6)"
            pro_time = (
                data.time
                - (pydate - reference).total_seconds() / 24.0 / 3600.0
            )
    # if which depends on if it is leap year
    elif info["pro_mode"] == 2:
        if calendar.isleap(pydate.year):
            data.time = np.array(
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
            data.time = np.array(
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
        dayofyear = (pydate - dt.datetime(pydate.year, 1, 1)).days + 1
    # profile data is time series of data from CTM
    if info["pro_mode"] == 1:
        # interpolation into satellite pixels
        # trace gas profile
        for i in range(nlayer):
            # profile interpolation for each layer
            interp_pro = RegularGridInterpolator(
                (pro_time, data.lat, data.lon),
                data.pro[i, ...],
                bounds_error=False,
                fill_value=np.nan,
            )
            pro[idx, i] = interp_pro((time[idx], lat[idx], lon[idx]))
            if info["tcorr_flag"]:
                # temperature profile interpolation for each layer
                interp_tpro = RegularGridInterpolator(
                    (pro_time, data.lat, data.lon),
                    data.tpro[i, ...],
                    bounds_error=False,
                    fill_value=np.nan,
                )
                tpro[idx, i] = interp_tpro((time[idx], lat[idx], lon[idx]))
        # tropopause
        if info["amftrop_flag"]:
            if info["tropopause_mode"] == 0:
                interp_tropopause = RegularGridInterpolator(
                    (pro_time, data.lat, data.lon),
                    data.tropopause,
                    bounds_error=False,
                    fill_value=np.nan,
                    method="nearest",
                )
            elif info["tropopause_mode"] == 1:
                interp_tropopause = RegularGridInterpolator(
                    (pro_time, data.lat, data.lon),
                    data.tropopause,
                    bounds_error=False,
                    fill_value=np.nan,
                )
            tropopause[idx] = interp_tropopause(
                (time[idx], lat[idx], lon[idx])
            )
        # pressure profile
        if info["pro_grid_mode"] == 1:
            for i in range(nlayer):
                # profile interpolation for each layer
                interp_pres = RegularGridInterpolator(
                    (pro_time, data.lat, data.lon),
                    data.pres[i, ...],
                    bounds_error=False,
                    fill_value=np.nan,
                )
                pres[idx, i] = interp_pres((time[idx], lat[idx], lon[idx]))
        # surface pressure
        interp_sp = RegularGridInterpolator(
            (pro_time, data.lat, data.lon),
            data.sp,
            bounds_error=False,
            fill_value=np.nan
        )
        sp[idx] = interp_sp((time[idx], lat[idx], lon[idx]))
        # terrain height interpolation
        if info["spcorr_flag"]:
            interp_th = RegularGridInterpolator(
                (data.lat, data.lon), data.th, fill_value=np.nan
            )
            th[idx] = interp_th((lat[idx], lon[idx]))
            # spcorr_flag=True and tcorr_flag=False
            # then tpro only calculate for the first layer
            if not info["tcorr_flag"]:
                interp_tpro = RegularGridInterpolator(
                    (pro_time, data.lat, data.lon),
                    data.tpro[0, ...],
                    bounds_error=False,
                    fill_value=np.nan,
                )
                tpro[idx] = interp_tpro((time[idx], lat[idx], lon[idx]))
    # profile data is monthly climatology at satellite overpass time
    elif info["pro_mode"] == 2:
        dayofyear = (pydate.date() - dt.date(pydate.year, 1, 1)).days + 1
        idx1 = bisect.bisect_right(data.time, dayofyear)
        assert (idx1 > 0) & (idx1 < data.time.size)
        idx0 = idx1 - 1
        dis = (data.time[idx1] - dayofyear) / (
            data.time[idx1] - data.time[idx0]
        )
        # 1. get daily map, and then interpolation into satellite pixels
        # trace gas & temperature profile
        pro_daily = data.pro[:, idx0, :, :] * dis + data.pro[:, idx1, :, :] * (
            1 - dis
        )
        if info["tcorr_flag"] | info["spcorr_flag"]:
            tpro_daily = data.tpro[:, idx0, :, :] * dis + data.tpro[
                :, idx1, :, :
            ] * (1 - dis)
        for i in range(nlayer):
            interp_pro = RegularGridInterpolator(
                (data.lat, data.lon),
                pro_daily[i, ...],
                fill_value=np.nan
            )
            pro[idx, i] = interp_pro((lat[idx], lon[idx]))
            if info["tcorr_flag"]:
                interp_tpro = RegularGridInterpolator(
                    (data.lat, data.lon),
                    tpro_daily[i, ...],
                    fill_value=np.nan
                )
                tpro[idx, i] = interp_tpro((lat[idx], lon[idx]))
        # tropopause
        if info["amftrop_flag"]:
            if info["tropopause_mode"] == 0:
                idx2 = np.abs(dayofyear - data.time).argmin()
                tropopause_daily = data.tropopause[idx2, :, :]
                interp_tropopause = RegularGridInterpolator(
                    (data.lat, data.lon),
                    tropopause_daily,
                    fill_value=np.nan,
                    method="nearest",
                )
            elif info["tropopause_mode"] == 1:
                tropopause_daily = data.tropopause[
                    idx0, :, :
                ] * dis + data.tropopause[idx1, :, :] * (1 - dis)
                interp_tropopause = RegularGridInterpolator(
                    (data.lat, data.lon),
                    tropopause_daily,
                    fill_value=np.nan
                )
            tropopause[idx] = interp_tropopause((lat[idx], lon[idx]))
        # pressure profile
        if info["pro_grid_mode"] == 1:
            pres_daily = data.pres[:, idx0, :, :] * dis + data.pres[
                :, idx1, :, :
            ] * (1 - dis)
            for i in range(nlayer):
                # profile interpolation for each layer
                interp_pres = RegularGridInterpolator(
                    (pro_time, data.lat, data.lon),
                    pres_daily[i, ...],
                    bounds_error=False,
                    fill_value=np.nan,
                )
                pres[idx, i] = interp_pres((time[idx], lat[idx], lon[idx]))
        # surface pressure
        sp_daily = data.sp[idx0, :, :] * dis + data.sp[idx1, :, :] * (1 - dis)
        interp_sp = RegularGridInterpolator(
            (data.lat, data.lon),
            sp_daily,
            fill_value=np.nan
        )
        sp[idx] = interp_sp((lat[idx], lon[idx]))
        # terrain height interpolation
        if info["spcorr_flag"]:
            interp_th = RegularGridInterpolator(
                (data.lat, data.lon),
                data.th,
                fill_value=np.nan
            )
            th[idx] = interp_th((lat[idx], lon[idx]))
            # spcorr_flag=True and tcorr_flag=False
            # then tpro only calculate for the first layer
            if not info["tcorr_flag"]:
                interp_tpro = RegularGridInterpolator(
                    (data.lat, data.lon),
                    tpro_daily[0, ...],
                    fill_value=np.nan
                )
                tpro[idx] = interp_tpro((lat[idx], lon[idx]))
    # profile data without time dimension
    elif info["pro_mode"] == 3:
        # interpolation into satellite pixels
        for i in range(nlayer):
            # profile interpolation for each layer
            interp_pro = RegularGridInterpolator(
                (data.lat, data.lon),
                data.pro[i, ...],
                fill_value=np.nan
            )
            pro[idx, i] = interp_pro((lat[idx], lon[idx]))
            if info["tcorr_flag"]:
                # temperature profile interpolation for each layer
                interp_tpro = RegularGridInterpolator(
                    (data.lat, data.lon),
                    data.tpro[i, ...],
                    fill_value=np.nan
                )
                tpro[idx, i] = interp_tpro((lat[idx], lon[idx]))
        # tropopause
        if info["amftrop_flag"]:
            if info["tropopause_mode"] == 0:
                interp_tropopause = RegularGridInterpolator(
                    (data.lat, data.lon),
                    data.tropopause,
                    fill_value=np.nan,
                    method="nearest",
                )
            elif info["tropopause_mode"] == 1:
                interp_tropopause = RegularGridInterpolator(
                    (data.lat, data.lon),
                    data.tropopause,
                    fill_value=np.nan
                )
            tropopause[idx] = interp_tropopause((lat[idx], lon[idx]))
        # pressure profile
        if info["pro_grid_mode"] == 1:
            for i in range(nlayer):
                # profile interpolation for each layer
                interp_pres = RegularGridInterpolator(
                    (data.lat, data.lon),
                    data.pres[i, ...],
                    fill_value=np.nan
                )
                pres[idx, i] = interp_pres((lat[idx], lon[idx]))
        # surface pressure interpolation
        interp_sp = RegularGridInterpolator(
            (data.lat, data.lon),
            data.sp,
            fill_value=np.nan
        )
        sp[idx] = interp_sp((lat[idx], lon[idx]))
        # terrain height interpolation
        if info["spcorr_flag"]:
            interp_th = RegularGridInterpolator(
                (data.lat, data.lon),
                data.th,
                fill_value=np.nan
            )
            th[idx] = interp_th((lat[idx], lon[idx]))
            # spcorr_flag=True and tcorr_flag=False
            # then tpro only calculate for the first layer
            if not info["tcorr_flag"]:
                interp_tpro = RegularGridInterpolator(
                    (data.lat, data.lon),
                    data.tpro[0, ...],
                    fill_value=np.nan
                )
                tpro[idx] = interp_tpro((lat[idx], lon[idx]))
    # calculate pressure profile
    if info["pro_grid_mode"] == 0:
        pai[np.abs(pai) < 1e-3] = 0
        pbi[np.abs(pbi) < 1e-6] = 0
        pres = np.outer(np.ones(size), pai) + np.outer(sp, pbi)
    # tropopause index
    if info["amftrop_flag"] & (info["tropopause_mode"] == 1):
        tropopause[idx] = np.argmin(
            np.abs(
                pres[idx, :] - np.outer(tropopause[idx], np.ones(nlevel)),
                axis=-1,
            )
        )
        tropopause[np.isnan(tropopause)] = -1
        tropopause = np.int8(tropopause)
        assert (
            np.count_nonzero(tropopause == 0) == 0
        ), "tropopause index should be larger than 0"
    # convert tpro from 1 to 2 dimensions (for only spcorr cases)
    if len(tpro.shape) == 1:
        tpro = tpro[:, np.newaxis]

    # return the results
    return pro, pres, tpro, tropopause, sp, th
