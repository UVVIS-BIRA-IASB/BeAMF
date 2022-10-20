import numpy as np
from scipy.interpolate import RegularGridInterpolator
import datetime as dt

# declare list of constants used in the code
__deltaT__ = 0.0065
__rdry__ = 287.0
__g__ = 9.80665
__abc__ = -__g__ / __deltaT__ / __rdry__


# functions used in beamf tool
def monotonic(x, axis=-1):
    """
    check array monotonicity
    """
    dx = np.diff(x, axis=axis)
    if np.all(dx[np.isfinite(dx)] < 0):  # decrease
        return 1
    elif np.all(dx[np.isfinite(dx)] > 0):  # increase
        return 2
    elif np.all(dx[np.isfinite(dx)] <= 0):  # decrease
        return 3
    elif np.all(dx[np.isfinite(dx)] >= 0):  # increase
        return 4
    else:
        print("Array is not monotonicity")
        return 0


def masked(arr):
    """
    when using netcdf4 from python it will read data without masked values in
    it as numpy array, so far so good. When reading netcdf4 with masked values,
    we see that netcdf4 lib create a maskedarray, we do not want this,
    so in this function we change the masked values in a mask array to numpy
    arrays with nan, with makes our life a lot easier.
    The only reason why WE would use masked arrays is when we want to remove
    from time to time the mask, but this is not really the case.
    """
    if type(arr) == np.ma.core.MaskedArray:
        return np.where(arr.mask, np.nan, arr)
    else:
        return arr


def cal_tfactor(tpro, tmode, tcoeff, tref):
    """
    calculate temperature correction factor for AMF calculation
    temp: temperature array in K (an array of float)
    tcoeff: coefficients for temperature correction formula (a list)
    tref: temperature reference for temperature correction mode (a value)
    tmode: temperature correction mode (an integer)
    0: polynomial
    1: 1/polynomial
    2: (tref-tcoeff)/(temp-tcoeff)
    3: (temp-tcoeff)/(tref-tcoeff)
    """
    assert type(tref) in [
        int,
        float,
    ], "datatype of tref is incorrect! (integer or float)"

    tfactor = np.full_like(tpro, np.nan)
    if tmode in [0, 1]:
        tfactor[:] = 1.0
        for i in range(len(tcoeff)):
            tfactor = tfactor + tcoeff[i] * (tpro - tref) ** (i + 1)
        if tmode == 1:
            tfactor = 1.0 / tfactor
    elif tmode in [2, 3]:
        assert len(tcoeff) == 1, "size of tcoeff is incorrect! (not 1)"
        assert type(tcoeff[0]) == [
            int,
            float,
        ], "datatype of tcoeff is incorrect! (integer or float)"
        if tmode == 2:
            tfactor = (tref - tcoeff[0]) / (tpro - tcoeff[0])
        if tmode == 3:
            tfactor = (tpro - tcoeff[0]) / (tref - tcoeff[0])
    return tfactor


def spcorr(surf_pres, surf_t, pro_th, th):
    """
    Surface pressure correction based on Zhou et al. 2008
    """
    sp = (
        surf_pres * (surf_t / (surf_t + __deltaT__ * (pro_th - th))) ** __abc__
    )
    return sp


def temp_convert(temp, temp_units):
    """
    convert temperature units into Kelvin
    0: Kelvin
    1: Celsius
    2: Fahrenheit
    """
    if temp_units == 1:  # Celsius
        temp = temp + 273.15
    elif temp_units == 2:  # Fahrenheit
        temp = (temp + 459.67) * 5 / 9
    else:
        assert temp_units == 0, "temperature units is out of range"
    return temp


def pres_convert(pres, pres_units):
    """
    convert temperature units into Pa
    0: Pa
    1: hPa
    """
    if pres_units == 1:  # hPa
        pres = pres * 100.0
    else:
        assert pres_units == 0, "pressure units is out of range"
    return pres


def height_convert(height, height_units):
    """
    convert height units into m
    0: m
    1: km
    """
    if height_units == 1:  # km
        height = height * 1000.0
    else:
        assert height_units == 0, "pressure units is out of range"
    return height


def interpn(lut, lutvars, vars):
    """
    multidimensional interpolation for any dimension data
    lut(list): the data on the regular grid in n dimensions
    lutvars(list): the points defining the regular grid in n dimensions
    vars(array): the points at which to evaluate the interpolated values
    """
    # number of dimension and variables
    ndim, nvar = vars.shape
    dim_var = lut.shape
    assert (ndim >= 2) & (ndim <= 10)
    # check number of dimension for variables/LUT
    assert len(dim_var) == len(lutvars) == ndim
    for i in range(ndim):
        assert dim_var[i] == lutvars[i].size, print(
            "dimension " + str(i) + " is not consistent between lut and vars"
        )
        # lutvars is monotonically decreasing
        # then reverse the array lutvars
        if monotonic(lutvars[i]) == 1:
            lutvars[i] = lutvars[i][::-1]
            lut = np.flip(lut, axis=i)
        else:
            assert monotonic(lutvars[i]) == 2, print(
                "var " + str(i) + " in LUT is not monotonically "
                "in/decreasing"
            )
    # result
    res = np.full(nvar, np.nan)
    if ndim == 2:
        f2 = RegularGridInterpolator(
            (
                lutvars[0],
                lutvars[1]
            ),
            lut,
            fill_value=np.nan
        )
        res = f2(
            (
                vars[0, :],
                vars[1, :]
            )
        )
    elif ndim == 3:
        f3 = RegularGridInterpolator(
            (
                lutvars[0],
                lutvars[1],
                lutvars[2]
            ),
            lut,
            fill_value=np.nan
        )
        res = f3(
            (
                vars[0, :],
                vars[1, :],
                vars[2, :]
            )
        )
    elif ndim == 4:
        f4 = RegularGridInterpolator(
            (
                lutvars[0],
                lutvars[1],
                lutvars[2],
                lutvars[3]
            ),
            lut,
            fill_value=np.nan,
        )
        res = f4(
            (
                vars[0, :],
                vars[1, :],
                vars[2, :],
                vars[3, :]
            )
        )
    elif ndim == 5:
        f5 = RegularGridInterpolator(
            (
                lutvars[0],
                lutvars[1],
                lutvars[2],
                lutvars[3],
                lutvars[4]
            ),
            lut,
            fill_value=np.nan,
        )
        res = f5(
            (
                vars[0, :],
                vars[1, :],
                vars[2, :],
                vars[3, :],
                vars[4, :]
            )
        )
    elif ndim == 6:
        f6 = RegularGridInterpolator(
            (
                lutvars[0],
                lutvars[1],
                lutvars[2],
                lutvars[3],
                lutvars[4],
                lutvars[5],
            ),
            lut,
            fill_value=np.nan,
        )
        res = f6(
            (
                vars[0, :],
                vars[1, :],
                vars[2, :],
                vars[3, :],
                vars[4, :],
                vars[5, :],
            )
        )
    elif ndim == 7:
        f7 = RegularGridInterpolator(
            (
                lutvars[0],
                lutvars[1],
                lutvars[2],
                lutvars[3],
                lutvars[4],
                lutvars[5],
                lutvars[6],
            ),
            lut,
            fill_value=np.nan,
        )
        res = f7(
            (
                vars[0, :],
                vars[1, :],
                vars[2, :],
                vars[3, :],
                vars[4, :],
                vars[5, :],
                vars[6, :],
            )
        )
    elif ndim == 8:
        f8 = RegularGridInterpolator(
            (
                lutvars[0],
                lutvars[1],
                lutvars[2],
                lutvars[3],
                lutvars[4],
                lutvars[5],
                lutvars[6],
                lutvars[7],
            ),
            lut,
            fill_value=np.nan,
        )
        res = f8(
            (
                vars[0, :],
                vars[1, :],
                vars[2, :],
                vars[3, :],
                vars[4, :],
                vars[5, :],
                vars[6, :],
                vars[7, :],
            )
        )
    elif ndim == 9:
        f9 = RegularGridInterpolator(
            (
                lutvars[0],
                lutvars[1],
                lutvars[2],
                lutvars[3],
                lutvars[4],
                lutvars[5],
                lutvars[6],
                lutvars[7],
                lutvars[8],
            ),
            lut,
            fill_value=np.nan,
        )
        res = f9(
            (
                vars[0, :],
                vars[1, :],
                vars[2, :],
                vars[3, :],
                vars[4, :],
                vars[5, :],
                vars[6, :],
                vars[7, :],
                vars[8, :],
            )
        )
    elif ndim == 10:
        f10 = RegularGridInterpolator(
            (
                lutvars[0],
                lutvars[1],
                lutvars[2],
                lutvars[3],
                lutvars[4],
                lutvars[5],
                lutvars[6],
                lutvars[7],
                lutvars[8],
                lutvars[9],
            ),
            lut,
            fill_value=np.nan,
        )
        res = f10(
            (
                vars[0, :],
                vars[1, :],
                vars[2, :],
                vars[3, :],
                vars[4, :],
                vars[5, :],
                vars[6, :],
                vars[7, :],
                vars[8, :],
                vars[9, :],
            )
        )
    return res


def convert_time(time, time_units, time_reference=[]):
    """
    convert units of time into reference date and day shfit
    time_units = 0/1, time_reference should be a list/array with 3 elements
    """
    if (time_units == 0) | (time_units == 1):
        if len(time_reference) == 3:
            date_ref = dt.datetime(
                time_reference[0], time_reference[1], time_reference[2]
            )
        elif len(time_reference) == 6:
            date_ref = dt.datetime(
                time_reference[0],
                time_reference[1],
                time_reference[2],
                time_reference[3],
                time_reference[4],
                time_reference[5],
            )
        else:
            assert False, "len(time_reference) is not 3 or 6!"
    # 0:seconds since the reference time
    if time_units == 0:
        # datetime calculation based on timedelta(time) and reference date
        dates = date_ref + dt.timedelta(seconds=1) * time
        # date0 for the file
        date0 = dt.datetime(dates[0].year, dates[0].month, dates[0].day)
        # fraction of day since date0
        timedelta = (
            (dates - date0).astype("timedelta64[ms]").astype(int)
            / 1000
            / 86400
        )
        return date0, timedelta
    # 1:days since the reference time
    elif time_units == 1:
        # date calculation based on timedelta(time) and reference date
        dates = date_ref + dt.timedelta(days=1) * time
        # date0 for the file
        date0 = dt.datetime(dates[0].year, dates[0].month, dates[0].day)
        # fraction of day since date0
        timedelta = (
            (dates - date0).astype("timedelta64[ms]").astype(int)
            / 1000
            / 86400
        )
        return date0, timedelta
    # 2: yyyymmdd
    elif time_units == 2:
        year = np.int16(time / 10000)
        month = np.int16((time - year * 10000) / 100)
        day = time % 100
        # date0 for the file
        idx = np.where((year > 1990) & (year < 2100))[0]
        date0 = dt.datetime(year[idx[0]], month[idx[0]], day[idx[0]])
        # date calculation based on year/month/day
        dates = np.array(
            [dt.datetime(y, m, d) for y, m, d in zip(year, month, day)]
        )
        # fraction of day since date0
        timedelta = (
            (dates - date0).astype("timedelta64[ms]").astype(int)
            / 1000
            / 86400
        )
        return date0, timedelta
    # 3: year/month/day/hour/minute/second/microsecond
    elif time_units == 3:
        time0 = np.int32(time).reshape(-1, time.shape[-1])
        # date0 for the file
        idx = np.where((time0[..., 0] > 1990) & (time0[..., 0] < 2100))
        date0 = dt.datetime(
            time0[idx][0, 0], time0[idx][0, 1], time0[idx][0, 2]
        )
        # datetime calculation based on time
        dates = np.array([dt.datetime(*x) for x in time0])
        # fraction of day since date0
        timedelta = (
            (dates - date0).astype("timedelta64[ms]").astype(int)
            / 1000
            / 86400
        )
        timedelta = timedelta.reshape(time.shape[:-1])
        return date0, timedelta


def convert_timedelta(timedelta, timedelta_units):
    """
    convert units of timedelta into fraction of days
    """
    # timedelta_new = np.full_like(timedelta, np.nan)
    if timedelta_units == 0:  # milliseconds
        timedelta_new = timedelta / 1000 / 86400
    elif timedelta_units == 1:  # seconds
        timedelta_new = timedelta / 86400
    elif timedelta_units == 2:  # hhmmss
        hours = np.int16(timedelta / 10000)
        minutes = np.int16((timedelta - hours * 10000) / 100)
        seconds = timedelta % 100
        timedelta_new = hours / 24 + minutes / 1440 + seconds / 86400
    elif timedelta_units == 3:  # fractional time (days)
        timedelta_new = timedelta
    elif timedelta_units == 4:  # fractional time (hours)
        timedelta_new = timedelta / 24
    return timedelta_new
