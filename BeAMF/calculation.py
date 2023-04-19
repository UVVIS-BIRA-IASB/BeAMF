import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import function as amf_func

# define constant
# https://en.wikipedia.org/wiki/Avogadro_constant
# Avogadro's Number
__ra__ = 6.02214076e23
# https://www.sciencedirect.com/topics/earth-and-planetary-sciences/lapse-rate
# temperature lapse rate in the lower atmosphere (K/m)
__deltaT__ = 0.0065
# https://en.wikipedia.org/wiki/Gas_constant
# gas constant for dry air (J/kg/K)
__rdry__ = 287.053
# https://en.wikipedia.org/wiki/Density_of_air
# molar mass of dry air (g/mol)
__dryair__ = 28.9652
# gravitational acceleration (m/s^2)
__g__ = 9.80665
# exponent used for surface pressure converson following the hypsometric
# equation (Wallace and Hobbs, 1977).
__abc__ = -__g__ / __deltaT__ / __rdry__

# radius of Earth (km)
__R__ = 6371
# effective scale height of the atmosphere
__Hatm__ = 8.5
__r__ = __R__ / __Hatm__


def collect_cloud_variables(
    info,
    inp,
    lut
):
    # Input validation
    assert isinstance(inp, dict), "Input parameters are not a dictionary"
    assert isinstance(lut, dict), "LUT parameters are not a dictionary"

    # Constants
    CLOUD_ALBEDO_MAX = 1.0
    CLOUD_ALBEDO_FIX = 0.8
    CLOUD_PRESSURE_MIN = lut["var"][1][0]

    # cloud fraction (range between 0 and 1)
    cf = np.clip(np.float64(inp["cf"]), 0, 1)

    # cloud pressure (within [minimum pressure in LUT, sp])
    # cp = sp when cf = 0
    sp = np.float64(inp["sp"])
    cp = np.clip(np.float64(inp["cp"]), CLOUD_PRESSURE_MIN, sp)
    cp[cf == 0] = sp[cf == 0]

    # cloud top albedo (range between 0 and 1)
    # ca = 0.8 when cf = 0
    ca = np.clip(np.float64(inp["ca"]), 0, CLOUD_ALBEDO_MAX)
    ca[cf == 0] = CLOUD_ALBEDO_FIX

    # output
    cld = {
        "cf": cf,
        "cp": cp,
        "ca": ca,
        "cldcorr_cfunits": info["cldcorr_cfunits"],
        "cldcorr_cfthreshold": info["cldcorr_cfthreshold"]
    }
    return cld


def valid_pixels(
    idx0,
    var,
    lut_var,
    cld=[]
):
    # idx0: valid pixels within the selected lat/lon/SZA/VZA range
    # output:
    # idx1: idx0 & variable range from LUT variable &
    # valid cloud retrieval
    idx = idx0.copy()
    for i in range(len(var)):
        idx = (
            idx
            & (var[i] >= np.min(lut_var[i]))
            & (var[i] <= np.max(lut_var[i]))
        )
    if cld:
        idx = (
            idx
            & np.isfinite(cld["cf"])
            & np.isfinite(cld["cp"])
            & np.isfinite(cld["ca"])
        )
    assert (np.count_nonzero(np.isnan(var[0][idx])) == 0), (
        "some pressure values are NaN"
        )
    return idx


def cal_avk(
        idx,
        pres,
        sp,
        sza,
        vza,
        raa,
        alb,
        var,
        lut_var,
        lut_rad,
        lut_amf,
        cld=[]
):
    # idx: valid pixels
    # input variables:
    # pres: pressure profile
    # sp: surface pressure
    # SZA/VZA/RAA
    # alb: surface albedo (elements)
    # var: other variables
    # lut_var: LUT varibales, pres/sp/sza/vza/raa/alb/var
    # lut_rad: radiance in LUT sp/sza/vza/raa/alb/var
    # lut_amf: box-AMF in LUT pres/sp/sza/vza/raa/alb/var
    # cloud variables if cldcorr_flag is True

    dim = list(pres.shape)
    assert len(dim) == 2, "pres must be a 2D array"

    # pressure is given at layer bounadry
    # box-AMF is calculated at layer
    dim[1] = dim[1] - 1

    # collect valid variables (idx)
    pres = pres[idx]
    sp = sp[idx]
    sza = sza[idx]
    vza = vza[idx]
    raa = raa[idx]
    nalb = len(alb)  # number of parameters for surface albedo
    alb1 = [alb[i][idx] for i in range(nalb)]
    nvar = len(var)
    var1 = [var[i][idx] for i in range(nvar)]

    assert np.isnan(pres).sum() == 0, "pressure for valid pixels is NaN"

    # normalized pressure grid
    pres = pres / sp[:, None]

    assert 0 <= np.min(pres) < np.max(pres) <= 1, (
        "normalized pressure grid is not in range of [0, 1]"
    )

    # midpoint of pressure profile
    pres1 = (pres[:, :-1] + pres[:, 1:]) / 2
    # create variables for output
    avk_clr = np.full(dim, np.nan)
    avk_cld = np.full(dim, np.nan)
    avk_cld[idx, :] = 0
    avk = np.full(dim, np.nan)
    crf = np.full(dim[0], np.nan)

    # preparation for cloud scene
    if len(cld):
        cf = cld["cf"][idx]
        cp = cld["cp"][idx]
        ca = alb1.copy()
        ca[0] = cld["ca"][idx]
        for i in range(1, nalb):
            ca[i] = np.full_like(ca[0], 0)
        # when cf=0, setting cp=sp and ca=0.8
        idx1 = cf == 0
        cp[idx1] = sp[idx1]
        ca[0][idx1] = 0.8
        # forcing cp<=sp & cp>=0
        cp = np.clip(cp, lut_var[1][0], sp)
        # forcing ca<=1 & ca>=0.4
        ca[0] = np.clip(ca[0], 0, 1)

        # cloud in which profile layer
        cldidx = np.full(dim[0], np.nan)
        # factor to correct cloudy AMF at cloud layer
        amffact = np.full(dim[0], np.nan)
        # normalized pressure layer for cloudy scenes
        pres2 = np.full_like(pres1, np.nan)
        # intensity for clear scene
        iclr = np.full(dim[0], np.nan)
        # intensity for cloudy scene
        icld = np.full(dim[0], np.nan)
        icld[idx] = 0

    # extrapolation (only for pressure grid)
    # interpwf = RegularGridInterpolator(
    #     tuple(lut_var),
    #     lut_amf,
    #     bounds_error=False,
    #     fill_value=None
    # )
    if len(cld):
        interpi = RegularGridInterpolator(
            tuple(lut_var[1:]),
            lut_rad,
            bounds_error=False,
            fill_value=0
        )
    # wf for clear scene
    # for i in range(dim[1]):
    #     temp = (pres1[:, i], sp, sza, vza, raa, *alb1, *var1)
    #     avk_clr[idx, i] = interpwf(temp)
    #     avk[idx, i] = avk_clr[idx, i]
    avk_tmp = np.full((sp.size, len(lut_var[0])), np.nan)
    temp = (sp, sza, vza, raa, *alb1, *var1)
    for i in range(lut_var[0].size):
        interpwf_temp = RegularGridInterpolator(
            tuple(lut_var[1:]),
            lut_amf[i, ...],
            bounds_error=False,
            fill_value=None
        )
        avk_tmp[:, i] = interpwf_temp(temp)
    assert (np.nanmin(pres1) > 0) & (np.nanmax(pres1) < 1)
    weight = np.full_like(pres1, np.nan)
    avk1 = np.full_like(pres1, np.nan)
    avk2 = np.full_like(pres1, np.nan)
    for i in range(lut_var[0].size - 1):
        if i == 0:
            idx0 = np.where((pres1 > 0) & (pres1 <= lut_var[0][1]))
        elif i == lut_var[0].size - 2:
            idx0 = np.where((pres1 > lut_var[0][-2]) & (pres1 <= 1))
        else:
            idx0 = np.where(
                (pres1 > lut_var[0][i]) & (pres1 <= lut_var[0][i+1])
            )
        if idx0[0].size == 0:
            continue
        weight[idx0] = (pres1[idx0] - lut_var[0][i]) / (
            lut_var[0][i+1] - lut_var[0][i]
        )
        idx1 = (*idx0[:-1], np.full_like(idx0[0], i))
        avk1[idx0] = avk_tmp[idx1]
        idx2 = (*idx0[:-1], np.full_like(idx0[0], i+1))
        avk2[idx0] = avk_tmp[idx2]
    avk_clr[idx] = avk1 * (1 - weight) + avk2 * weight
    avk[idx] = avk_clr[idx]

    # cloud correction
    if len(cld):
        # intensity for clear scene
        temp = (sp, sza, vza, raa, *alb1, *var1)
        iclr[idx] = interpi(temp)
        # intensity for cloudy scene
        temp = (cp, sza, vza, raa, *ca, *var1)
        icld[idx] = interpi(temp)
        # cloud radiance fraction
        crf[idx] = icld[idx] * cf / (icld[idx] * cf + (iclr[idx] * (1 - cf)))
        # pressure grid for cloudy scene
        for i in range(dim[1]):
            index = (cp/sp > pres[:, i+1]) & (cp/sp <= pres[:, i])
            if np.count_nonzero(index):
                # cloud index in pressure grid
                temp = cldidx[idx]
                temp[index] = i
                cldidx[idx] = temp
                # fraction
                temp = amffact[idx]
                temp[index] = (cp[index] / sp[index] - pres[index, i+1]) / (
                    pres[index, i] - pres[index, i+1]
                )
                amffact[idx] = temp
                # pressure grid for cloud scene
                pres2[index, i+1:] = (
                    pres1[index, i+1:] / cp[index, None] * sp[index, None]
                )
                pres2[index, i] = (
                    (1 + pres[index, i+1] / cp[index] * sp[index]) / 2
                )
        assert (np.nanmin(pres2) >= 0) & (np.nanmax(pres2) <= 1), (
            "normalized pressure grid is not in range of [0, 1]"
        )
        # wf for cloudy scene
        # for i in range(dim[1]):
        #     temp = tuple([pres2[:, i], cp, sza, vza, raa] + ca + var1)
        #     avk_cld[idx, i] = interpwf(temp)
        avk_tmp = np.full((sp.size, len(lut_var[0])), np.nan)
        temp = (cp, sza, vza, raa, *ca, *var1)
        for i in range(lut_var[0].size):
            interpwf_temp = RegularGridInterpolator(
                tuple(lut_var[1:]),
                lut_amf[i, ...],
                bounds_error=False,
                fill_value=None
            )
            avk_tmp[:, i] = interpwf_temp(temp)
        assert (np.nanmin(pres2) > 0) & (np.nanmax(pres2) < 1)
        weight = np.full_like(pres2, np.nan)
        avk1 = np.full_like(pres2, np.nan)
        avk2 = np.full_like(pres2, np.nan)
        for i in range(lut_var[0].size - 1):
            if i == 0:
                idx0 = np.where((pres2 > 0) & (pres2 <= lut_var[0][1]))
            elif i == lut_var[0].size - 2:
                idx0 = np.where((pres2 > lut_var[0][-2]) & (pres2 <= 1))
            else:
                idx0 = np.where(
                    (pres2 > lut_var[0][i]) & (pres2 <= lut_var[0][i+1])
                )
            if idx0[0].size == 0:
                continue
            weight[idx0] = (pres2[idx0] - lut_var[0][i]) / (
                lut_var[0][i+1] - lut_var[0][i]
            )
            idx1 = (*idx0[:-1], np.full_like(idx0[0], i))
            avk1[idx0] = avk_tmp[idx1]
            idx2 = (*idx0[:-1], np.full_like(idx0[0], i+1))
            avk2[idx0] = avk_tmp[idx2]
        avk_cld[idx] = avk1 * (1 - weight) + avk2 * weight
        # correction for cloud layer
        for i in range(dim[1]):
            index = (cldidx == i)
            if np.count_nonzero(index):
                avk_cld[index, i] = avk_cld[index, i] * amffact[index]
                avk_cld[index, :i] = 0.0
        # calculate total wf
        if cld["cldcorr_cfunits"] == 0:
            index = (cld["cf"] >= cld["cldcorr_cfthreshold"])
        elif cld["cldcorr_cfunits"] == 1:
            index = (crf >= cld["cldcorr_cfthreshold"])
        else:
            assert False, "wrong cldcorr_cfunits value"
        if np.count_nonzero(index):
            avk[index] = (
                avk_clr[index, :] * (1 - crf[index, None])
                + avk_cld[index, :] * crf[index, None]
            )
    return avk_clr, avk_cld, avk, crf


def cal_amf(
    info,
    inp,
    lut
):
    # output from the calculation of the AMF
    # nwv is number of wavelength for AMF calculation
    # number of output wavelength dimension
    # (depends on wv_flag)
    nwv0 = len(info["wavelength"])
    # if wv_flag=True,
    # only calculate AMF for selected wavelength for each pixel
    if info["wv_flag"]:
        nwv = 1
    else:
        nwv = nwv0
    # dimensions for output date
    dim = (nwv, inp["size"])
    # dimensions for averaging kernel
    dim_avk = (nwv, inp["size"], inp["nlayer"])
    # initialize output variable
    avk_clr = np.full(dim_avk, np.nan)
    avk_cld = np.full(dim_avk, np.nan)
    avk = np.full(dim_avk, np.nan)
    tfactor = np.ones(dim_avk)
    crf = np.full(dim, np.nan)
    # if amftrop_flag=True, 3 elements for total/tropospheric/stratospheri AMF
    # otherwise, only 1 element for total AMF
    if info["amftrop_flag"]:
        amf = [
            np.full(dim, np.nan),
            np.full(dim, np.nan),
            np.full(dim, np.nan)
        ]
        amf_clr = [
            np.full(dim, np.nan),
            np.full(dim, np.nan),
            np.full(dim, np.nan)
        ]
        amf_cld = [
            np.full(dim, np.nan),
            np.full(dim, np.nan),
            np.full(dim, np.nan)
        ]
    else:
        amf = [np.full(dim, np.nan)]
        amf_clr = [np.full(dim, np.nan)]
        amf_cld = [np.full(dim, np.nan)]
    # cloud parameters
    if info["cldcorr_flag"]:
        cld = collect_cloud_variables(info, inp, lut)
    else:
        cld = []

    # geometric AMF
    amf_geo = np.array([])
    if info["amfgeo_flag"] | info["bc_flag"]:
        amf_geo = (2 * __r__ + 1) / (
            np.sqrt((__r__ * inp["cossza"]) ** 2 + 2 * __r__ + 1)
            + __r__ * inp["cossza"]
            ) + 1 / inp["cosvza"]
        # amf_geo = 1 / inp["cossza"] + 1 / inp["cosvza"]
        amf_geo[amf_geo.mask] = np.nan

    # temperature correction factor
    # if tcorr_flag = False, then all tfactor is 1.
    if info["tcorr_flag"]:
        for iwv in range(nwv0):
            if info["wv_flag"]:
                idx0 = inp["idx"] & (inp["wvidx"] == iwv + 1)
                if np.count_nonzero(idx0) > 0:
                    tfactor[0][idx0] = amf_func.cal_tfactor(
                        inp["tpro"][idx0],
                        info["tcorr_mode"][iwv],
                        info["tcorr_coeffs"][iwv],
                        info["tcorr_ref"][iwv],
                    )
            else:
                tfactor[iwv] = amf_func.cal_tfactor(
                    inp["tpro"],
                    info["tcorr_mode"][iwv],
                    info["tcorr_coeffs"][iwv],
                    info["tcorr_ref"][iwv],
                )
    # calculation of wf and crf
    for iwv in range(nwv0):
        # collect variables
        pres = inp["pres"]
        sp = inp["sp"]
        raa = inp["raa"]
        alb = [a[iwv, ...] for a in inp["alb"]]
        var = inp["var"]
        if info["geo_units"] == 0:
            sza = inp["sza"]
            vza = inp["vza"]
        else:
            sza = inp["cossza"]
            vza = inp["cosvza"]
        variables = [sp, sza, vza, raa] + alb + var
        # valid pixels
        idx = valid_pixels(inp["idx"], variables, lut["var"][2:], cld=cld)
        if np.count_nonzero(idx) == 0:
            if info["verbose"]:
                print("no valid pixels for data processing")
            continue

        # if no valid data
        if np.count_nonzero(idx) == 0:
            continue
        # if wv_flag=True, for each pixel, only calculate AMF for selected
        # wavelength
        # othewise, calculate all wavelengths for each pixel
        if info["wv_flag"]:
            idx1 = idx & (inp["wvidx"] == iwv + 1)
            if np.count_nonzero(idx1) > 0:
                avk_clr0, avk_cld0, avk0, crf0 = cal_avk(
                    idx1,
                    pres,
                    sp,
                    sza,
                    vza,
                    raa,
                    alb,
                    var,
                    lut["var"][1:],
                    lut["rad"][iwv, ...],
                    lut["bamf"][iwv, ...],
                    cld=cld,
                )
                avk[0, idx1] = avk0[idx1] * tfactor[0, idx1]
                avk_clr[0, idx1] = avk_clr0[idx1] * tfactor[0, idx1]
                avk_cld[0, idx1] = avk_cld0[idx1] * tfactor[0, idx1]
                crf[0, idx1] = crf0[idx1]
        else:
            avk_clr0, avk_cld0, avk0, crf0 = cal_avk(
                idx,
                pres,
                sp,
                sza,
                vza,
                raa,
                alb,
                var,
                lut["var"][1:],
                lut["rad"][iwv, ...],
                lut["bamf"][iwv, ...],
                cld=cld,
            )
            avk_clr[iwv, idx] = avk_clr0[idx] * tfactor[iwv, idx]
            avk_cld[iwv, idx] = avk_cld0[idx] * tfactor[iwv, idx]
            avk[iwv, idx] = avk0[idx] * tfactor[iwv, idx]
            crf[iwv, idx] = crf0[idx]

    # profile (partial column density)
    if info["pro_units"] == 0:  # VMR
        prof = inp["pro"] * -np.diff(inp["pres"], axis=-1)
    else:  # VCD
        prof = inp["pro"]
    # calculation of AMF
    # total AMF
    amf[0][:, idx] = (
        np.sum(avk[:, idx] * prof[idx], axis=-1)
        / np.sum(prof[idx], axis=-1)
    )
    amf_clr[0][:, idx] = (
        np.sum(avk_clr[:, idx] * prof[idx], axis=-1)
        / np.sum(prof[idx], axis=-1)
    )
    amf_cld[0][:, idx] = (
        np.sum(avk_cld[:, idx] * prof[idx], axis=-1)
        / np.sum(prof[idx], axis=-1)
    )
    # if amftrop_flag, then calculate tropospheric and stratospheric AMF
    if info["amftrop_flag"]:
        # tropospheric AMF
        trop_flag = np.zeros_like(prof)
        for i in range(1, inp["nlayer"]):
            trop_flag[inp["tropopause"] == i, 0:i] = 1
        trop_flag[np.isnan(inp["tropopause"])] = np.nan
        amf[1][:, idx] = (
            np.sum(avk[:, idx] * prof[idx] * trop_flag[idx], axis=-1) /
            np.sum(prof[idx] * trop_flag[idx], axis=-1)
            )
        amf_clr[1][:, idx] = (
            np.sum(avk_clr[:, idx] * prof[idx] * trop_flag[idx], axis=-1) /
            np.sum(prof[idx] * trop_flag[idx], axis=-1)
        )
        amf_cld[1][:, idx] = (
            np.sum(avk_cld[:, idx] * prof[idx] * trop_flag[idx], axis=-1) /
            np.sum(prof[idx] * trop_flag[idx], axis=-1)
        )
        # stratospheric AMF
        strat_flag = np.zeros_like(prof)
        for i in range(1, inp["nlayer"]):
            strat_flag[inp["tropopause"] == i, i:] = 1
        strat_flag[np.isnan(inp["tropopause"])] = np.nan
        amf[2][:, idx] = (
            np.sum(avk[:, idx] * prof[idx] * strat_flag[idx], axis=-1) /
            np.sum(prof[idx] * strat_flag[idx], axis=-1)
        )
        amf_clr[2][:, idx] = (
            np.sum(avk_clr[:, idx] * prof[idx] * strat_flag[idx], axis=-1) /
            np.sum(prof[idx] * strat_flag[idx], axis=-1)
        )
        amf_cld[2][:, idx] = (
            np.sum(avk_cld[:, idx] * prof[idx] * strat_flag[idx], axis=-1) /
            np.sum(prof[idx] * strat_flag[idx], axis=-1)
        )

    # calculate Averaging kernel
    # if amftrop_flag = True, calculate AK for tropospheric column
    # otherwise, calculate for total column
    if info["amftrop_flag"]:
        i = 1
    else:
        i = 0
    idx2 = amf[i] > 0.01
    avk[idx2] = avk[idx2] / amf[i][idx2, None]
    avk[~idx2] = np.nan
    idx = amf_clr[i] > 0.01
    avk_clr[idx2] = avk_clr[idx2] / amf_clr[i][idx2, None]
    avk_clr[~idx2] = np.nan

    output = {
        "amf_geo": amf_geo,
        "amf": amf,
        "amf_clr": amf_clr,
        "amf_cld": amf_cld,
        "cloud_radiance_fraction": crf,
        "averaging_kernel": avk,
        "averaging_kernel_clr": avk_clr,
    }
    return output
