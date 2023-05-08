import numpy as np
import os
import harp
import netCDF4 as nc


# main program
def output(filename, info, out_flag, out_name, out):
    # create a new file
    if info["out_file_mode"] == 1:
        if info["out_file_type"] == 9:
            create_file(filename, info, out_flag, out_name, out)
    # write into harp file
    if info["out_file_type"] == 0:
        write_harp(filename, info, out_flag, out_name, out)
    elif info["out_file_type"] == 1:
        write_tropomi(filename, info, out_flag, out_name, out)
    elif info["out_file_type"] == 2:
        write_qa4ecv(filename, info, out_flag, out_name, out)
    # elif info["out_file_type"] == 3:
    #     write_domino(filename, info, out_flag, out_name, out)
    # elif info["out_file_type"] == 4:
    #     write_gome2(filename, info, out_flag, out_name, out)
    elif info["out_file_type"] == 9:
        write_file(filename, info, out_flag, out_name, out)


# create variable in netCDF file
def create_variable(fid, name, dim, size, dtype):
    fid.createVariable(
        name,
        dtype,
        dim,
        zlib=True,
        shuffle=True,
        complevel=9,
        fletcher32=True,
        chunksizes=size,
        fill_value=nc.default_fillvals[dtype],
    )


# print error information
def write_error(error, var_name, name, verbose):
    if verbose:
        print(error)
        print("Can not ouput the varibale(" + var_name + ") to (" + name + ")")


# create file with customized netCDF4 format
def create_file(filename, info, flag, name, out):
    # split into filepath, filename
    [fpath, fname] = os.path.split(filename)
    if fpath == "":
        fpath = "."
    # if directory is not exist, then create it.
    if not os.path.isdir(fpath):
        os.makedirs(fpath)
    # start to create netCDF4 file
    with nc.Dataset(filename, "w") as fid:
        dimnames = ()
        # satellite data dim
        dim = out["dim"]
        # dimensions
        for i in range(len(dim)):
            dimname = "dim" + str(i)
            fid.createDimension(dimname, dim[i])
            dimnames = (*dimnames, dimname)
        fid.createDimension("corner", 4)
        fid.createDimension("layer", out["nlayer"])
        fid.createDimension("wavelength", out["nwv"])
        fid.createDimension("albedo_variable", out["nalb"])

        # 1. input information
        # 1.1 input information with dimension (dim)
        keys = [
            "latitude",
            "longitude",
            "solar_zenith_angle",
            "sensor_zenith_angle",
            "solar_azimuth_angle",
            "sensor_azimuth_angle",
            "relative_azimuth_angle",
            "surface_altitude",
            "cloud_fraction",
            "cloud_pressure",
            "cloud_albedo",
            "surface_pressure",
            "scd",
        ]
        var_dimname = dimnames
        var_dim = dim
        for key in keys:
            if flag[key] & (len(name[key]) > 0):
                create_variable(fid, name[key], var_dimname, var_dim, "f4")
        # 1.2 input information with dimension (dim,4)
        keys = ["latitudecorners", "longitudecorners"]
        var_dimname = (*dimnames, "corner")
        var_dim = (*dim, 4)
        for key in keys:
            if flag[key] & (len(name[key]) > 0):
                create_variable(fid, name[key], var_dimname, var_dim, "f4")
        # 1.3 input surface albedo information
        # for BRDF surface, save it in multi or single varibale
        # for multi wavelength calculation, save it in a single variable
        if flag["surface_albedo"]:
            if out["nalb"] > 1:
                if len(name["surface_albedo"]) == out["nalb"]:
                    var_dimname = dimnames
                    var_dim = dim
                    for var_name in name["surface_albedo"]:
                        create_variable(
                            fid, var_name, var_dimname, var_dim, "f4"
                        )
                elif len(name["surface_albedo"]) == 1:
                    var_dimname = ("albedo_variable", *dimnames)
                    var_dim = (out["nalb"], *dim)
                    var_name = name["surface_albedo"][0]
                    create_variable(fid, var_name, var_dimname, var_dim, "f4")
            elif out["nwv"] > 1:
                var_dimname = ("wavelength", *dimnames)
                var_dim = (out["nwv"], *dim)
                var_name = name["surface_albedo"][0]
                create_variable(
                    fid, var_name, var_dimname, var_dim, "f4"
                )
                if len(name["surface_albedo"]) > 1:
                    if info["verbose"]:
                        print(
                            "warning: more than one out_name[surface_albedo]"
                        )
            else:
                assert out["nwv"] == out["nalb"] == 1
                var_dimname = dimnames
                var_dim = dim
                var_name = name["surface_albedo"][0]
                create_variable(fid, var_name, var_dimname, var_dim, "f4")
        # 1.4 input information with dimension (dim,layer)
        keys = ["profile", "pressure"]
        var_dimname = (*dimnames, "layer")
        var_dim = (*dim, out["nlayer"])
        for key in keys:
            if flag[key] & (len(name[key]) > 0):
                create_variable(fid, name[key], var_dimname, var_dim, "f4")
        # 1.5 input information with dimension (dim,layer) or (dim)
        # temperature profile (or only surface temperature)
        if flag["temperature"] & (len(name["temperature"]) > 0):
            if out["temperature"].shape[-1] == 1:
                var_dimname = dimnames
                var_dim = dim
            else:
                var_dimname = (*dimnames, "layer")
                var_dim = (*dim, out["nlayer"])
            var_name = name["temperature"]
            create_variable(fid, var_name, var_dimname, var_dim, "f4")
        # 1.6 input information with dimension (dim) as integer
        # tropopause (index)
        if flag["tropopause"] & (len(name["tropopause"]) > 0):
            var_dimname = dimnames
            var_dim = dim
            var_name = name["tropopause"]
            create_variable(fid, var_name, var_dimname, var_dim, "i1")
        # 1.7 list of input information with dimension (dim)
        # other variable
        if flag["other_variable"]:
            var_dimname = dimnames
            var_dim = dim
            for i in range(len(name["other_variable"])):
                if len(name["other_variable"][i]) > 0:
                    var_name = name["other_variable"][i]
                    create_variable(fid, var_name, var_dimname, var_dim, "f4")
        # 2. AMF results
        # 2.1 AMF results without wavelength dimension (dim)
        var_dimname = dimnames
        var_dim = dim
        keys = ["amf_geo"]
        for key in keys:
            if flag[key] & (len(name[key]) > 0):
                var_name = name[key]
                create_variable(fid, var_name, var_dimname, var_dim, "f4")
        # 2.2 AMF results with wavelength dimension (wv, dim) or (dim)
        keys = ["cloud_radiance_fraction", "vcd", "vcdtrop"]
        if out["nwv"] == 1:
            var_dimname = dimnames
            var_dim = dim
        else:
            var_dimname = ("wavelength", *dimnames)
            var_dim = (out["nwv"], *dim)
        for key in keys:
            if flag[key] & (len(name[key]) > 0):
                var_name = name[key]
                create_variable(fid, var_name, var_dimname, var_dim, "f4")
        # 2.3 list of AMF results with wavelength dimension
        # (wv, dim) or (dim)
        # dimension is the same as 2.2
        keys = ["amf", "amf_clr", "amf_cld"]
        for key in keys:
            if flag[key]:
                for i in range(len(name[key])):
                    if len(name[key][i]) > 0:
                        var_name = name[key][i]
                        create_variable(
                            fid, var_name, var_dimname, var_dim, "f4"
                        )
        # 2.4 list of AMF results with wavelength/layer dimension
        # (wv, dim, layer) or (dim, layer)
        keys = ["averaging_kernel", "averaging_kernel_clr"]
        if out["nwv"] == 1:
            var_dimname = (*dimnames, "layer")
            var_dim = (*dim, out["nlayer"])
        else:
            var_dimname = ("wavelength", *dimnames, "layer")
            var_dim = (out["nwv"], *dim, out["nlayer"])
        for key in keys:
            if flag[key] & (len(name[key]) > 0):
                var_name = name[key]
                create_variable(fid, var_name, var_dimname, var_dim, "f4")


# (create) and write in harp format file
def write_harp(filename, info, flag, name, out):
    # split into filepath, filename and suffix
    [fpath, fname] = os.path.split(filename)
    [fname, fexten] = os.path.splitext(fname)
    if fpath == "":
        fpath = "."
    # if directory is not exist, then create it.
    if not os.path.isdir(fpath):
        os.makedirs(fpath)
    # For HARP format, output AMF result wavelength by wavelength
    nwv = len(info["wavelength"])
    for iwv in range(nwv):
        if nwv == 1:
            fullname = fpath + "/" + fname + ".nc"
        else:
            fullname = (
                fpath + "/" + fname + "_" + str(info["wavelength"][iwv])
                + "nm.nc"
            )
        # if out_file_mode=1, then create a new harp format data
        # otherwise, load the date from filename file first
        if info["out_file_mode"] == 1:
            data = harp.Product()
        else:
            data = harp.import_product(filename)
        keys = [
            "latitude",
            "longitude",
            "solar_zenith_angle",
            "sensor_zenith_angle",
            "solar_azimuth_angle",
            "sensor_azimuth_angle",
            "relative_azimuth_angle",
            "surface_altitude",
            "cloud_pressure",
            "cloud_albedo",
            "surface_pressure",
            "scd",
            "vcd",
            "vcdtrop",
        ]
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                data[name[key]] = harp.Variable(np.float32(out[key].ravel()), ["time"])
        # latitudecorners/longitudecorners
        keys = ["latitudecorners", "longitudecorners"]
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                data[name[key]] = harp.Variable(
                    np.float32(out[key]), ["time", None]
                )
        # tropopause
        # units from index to pressure
        if (
            flag["tropopause"]
            & (len(name["tropopause"]) > 0)
            & (out["tropopause"].size > 0)
        ):
            trop_index = np.int8(out["tropopause"] + 1)
            pres = out["pressure"]
            index = np.arange(trop_index.size)
            trop_pres = pres[index, trop_index]
            data[name["tropopause"]] = harp.Variable(
                np.float32(trop_pres), ["time"]
            )
        # surface albedo
        # for harp format, only 1 element accept (not for BRDF parameters)
        if flag["surface_albedo"]:
            if out["nalb"] > 1:
                nalb = np.max((out["nalb"], len(name["surface_albedo"])))
                for ialb in nalb:
                    if (len(name["surface_albedo"][ialb]) > 0) & (
                        out["surface_albedo"][ialb].size > 0
                    ):
                        data[name["surface_albedo"][ialb]] = harp.Variable(
                            np.float32(out["surface_albedo"][ialb][iwv, ...]),
                            ["time"],
                        )
                if info["verbose"]:
                    print(
                        "warning: BRDF surface albedo data is not standard "
                        "output for HARP format"
                    )
            else:
                if (len(name["surface_albedo"]) > 0) & (
                    out["surface_albedo"][0].size > 0
                ):
                    data[name["surface_albedo"][0]] = harp.Variable(
                        np.float32(out["surface_albedo"][0][iwv, ...]),
                        ["time"],
                    )
        # AMF
        if flag["amf"]:
            for i in range(len(name["amf"])):
                if (len(name["amf"][i]) > 0) & (out["amf"][i].size > 0):
                    data[name["amf"][i]] = harp.Variable(
                        np.float32(out["amf"][i][iwv, ...]), ["time"]
                    )
        # averaging kernel
        if (
            flag["averaging_kernel"]
            & (len(name["averaging_kernel"]) > 0)
            & (out["averaging_kernel"].size > 0)
        ):
            data[name["averaging_kernel"]] = harp.Variable(
                np.float32(out["averaging_kernel"][iwv, ...]),
                ["time", "vertical"],
            )
        # trace gas / temperature profile
        keys = ["profile", "temperature"]
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                data[name[key]] = harp.Variable(
                    np.float32(out[key]), ["time", "vertical"]
                )
        # pressure profile
        if (
            flag["pressure"]
            & (len(name["pressure"]) > 0)
            & (out["pressure"].size > 0)
        ):
            pres = out["pressure"]
            pres1 = (pres[:, :-1] + pres[:, 1:]) / 2.0
            data[name["pressure"]] = harp.Variable(
                np.float32(pres1), ["time", "vertical"]
            )
        # other variable
        if flag["other_variable"]:
            for i in range(len(name["other_variable"])):
                if (len(name["other_variable"][i]) > 0) & (
                    out["other_variable"][i].size > 0
                ):
                    data[name["other_variable"][i]] = harp.Variable(
                        np.float32(out["other_variable"][i]), ["time"]
                    )
        # cloud fractions
        # if cldcorr_flag=True, output one of cloud_fraction and
        # cloud_radiance_fraction as cloud fraction
        # otherwise, output cloud_fraction
        if info["cldcorr_flag"]:
            if (
                flag["cloud_fraction"]
                & (len(name["cloud_fraction"]) > 0)
                & (out["cloud_fraction"].size > 0)
            ):
                data[name["cloud_fraction"]] = harp.Variable(
                    np.float32(out["cloud_fraction"]), ["time"]
                )
            elif (
                flag["cloud_radiance_fraction"]
                & (len(name["cloud_radiance_fraction"]) > 0)
                & (out["cloud_radiance_fraction"].size > 0)
            ):
                data[name["cloud_radiance_fraction"]] = harp.Variable(
                    np.float32(out["cloud_radiance_fraction"][iwv, ...]),
                    ["time"],
                )
        elif info["cf_flag"]:
            if (
                flag["cloud_fraction"]
                & (len(name["cloud_fraction"]) > 0)
                & (out["cloud_fraction"].size > 0)
            ):
                data[name["cloud_fraction"]] = harp.Variable(
                    np.float32(out["cloud_fraction"]), ["time"]
                )
        # not standard HARP output variable
        if (
            flag["amf_geo"]
            & (len(name["amf_geo"]) > 0)
            & (out["amf_geo"].size > 0)
        ):
            data[name["amf_geo"]] = harp.Variable(
                np.float32(out["amf_geo"]), ["time"]
            )
            if info["verbose"]:
                print(
                    "warning: amf_geo is not standard output for HARP format"
                )
        if flag["amf_clr"]:
            for i in range(len(name["amf_clr"])):
                if (len(name["amf_clr"][i]) > 0) & (
                    out["amf_clr"][i].size > 0
                ):
                    data[name["amf_clr"][i]] = harp.Variable(
                        np.float32(out["amf_clr"][i][iwv, ...]), ["time"]
                    )
            if info["verbose"]:
                print(
                    "warning: amf_clr is not standard output for HARP format"
                )
        if flag["amf_cld"]:
            for i in range(len(name["amf_cld"])):
                if (len(name["amf_cld"][i]) > 0) & (
                    out["amf_cld"][i].size > 0
                ):
                    data[name["amf_cld"][i]] = harp.Variable(
                        np.float32(out["amf_cld"][i][iwv, ...]), ["time"]
                    )
            if info["verbose"]:
                print(
                    "warning: amf_cld is not standard output for HARP format"
                )
        if flag["averaging_kernel_clr"]:
            if (len(name["averaging_kernel_clr"]) > 0) & (
                out["averaing_kernel_clr"].size > 0
            ):
                data[name["averaging_kernel_clr"]] = harp.Variable(
                    np.float32(out["averaging_kernel_clr"][iwv, ...]),
                    ["time", "vertical"],
                )
            if info["verbose"]:
                print(
                    "warning: averaging_kernel_clr is not standard output for "
                    "HARP format"
                )
        # if the file exists, then first make sure the file can read and write
        if os.path.exists(fullname):
            os.chmod(fullname, 0o777)
        #    os.remove(fullname)
        # Export the product as a HARP compliant data product.
        harp.export_product(data, fullname,file_format='hdf5', hdf5_compression=6)


def write_tropomi(filename, info, flag, name, out):
    with nc.Dataset(filename, "a") as fid:
        # satellite data dim
        dim = out["dim"]
        # 1. input information
        # 1.1 input information with dimension (dim)
        keys = [
            "latitude",
            "longitude",
            "solar_zenith_angle",
            "sensor_zenith_angle",
            "solar_azimuth_angle",
            "sensor_azimuth_angle",
            "surface_altitude",
            "cloud_fraction",
            "cloud_pressure",
            "cloud_albedo",
            "surface_pressure",
            "scd"
        ]
        var_dim = dim
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                try:
                    fid[name[key]][:] = out[key].reshape(var_dim)
                except IndexError as err:
                    write_error(err, key, name[key], info["verbose"])
        # 1.2 input information with dimension (dim,4)
        keys = ["latitudecorners", "longitudecorners"]
        var_dim = (*dim, 4)
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                try:
                    fid[name[key]][:] = out[key].reshape(var_dim)
                except IndexError as err:
                    write_error(err, key, name[key], info["verbose"])
        # 1.3 input surface albedo information
        # for BRDF surface, save it in multi or single varibale
        # for multi wavelength calculation, save it in a single variable
        if (
            flag["surface_albedo"]
            & (len(name["surface_albedo"]) > 0)
            & (len(out["surface_albedo"]) > 0)
        ):
            # For TROPOMI product,
            # only for single wavelength AMF calculation except SO2
            # but wv_flag = True for SO2
            # For others, surface albedo will output first albedo element
            if out["nwv"] != 1:
                if info["verbose"]:
                    print("warning: nwv=1 for TROPOMI output data format")
            var_dim = (-1, *dim)
            if info["wv_flag"]:
                nwv = out["surface_albedo"][0].shape[0]
                # if len(name["surface_albedo"]) = nalb,
                # then save all albedo values for each wavelength
                if len(name["surface_albedo"]) == nwv:
                    for iwv in range(nwv):
                        # check if name["surface_albedo"] is not empty
                        if len(name["surface_albedo"][iwv]) > 0:
                            try:
                                fid[name["surface_albedo"][iwv]][:] = out[
                                    "surface_albedo"
                                ][0][iwv].reshape(var_dim)
                            except IndexError as err:
                                write_error(
                                    err,
                                    "surface_albedo",
                                    name["surface_albedo"][iwv],
                                    info["verbose"],
                                )
                # otherwise, assume len(name["surface_albedo"])=1
                # only save albedo values for the selected wavelength
                else:
                    # find albedo for the selected wavelength
                    alb = np.full(dim, np.nan).ravel()
                    for iwv in range(nwv):
                        idx = (out["wvidx"] == iwv + 1)
                        if np.count_nonzero(idx):
                            alb[idx] = out["surface_albedo"][0][iwv, idx]
                    try:
                        fid[name["surface_albedo"][0]][:] = alb.reshape(
                            var_dim
                        )
                    except IndexError as err:
                        write_error(
                            err,
                            "surface_albedo",
                            name["surface_albedo"][0],
                            info["verbose"],
                        )
            else:
                try:
                    fid[name["surface_albedo"][0]][:] = out["surface_albedo"][
                        0
                    ].reshape(var_dim)[0]
                except IndexError as err:
                    write_error(
                        err,
                        "surface_albedo",
                        name["surface_albedo"][0],
                        info["verbose"],
                    )
        # 1.4 input information with dimension (dim,layer)
        var_dim = (*dim, out["nlayer"])
        key = "profile"
        if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
            try:
                fid[name[key]][:] = out[key].reshape(var_dim)
            except IndexError as err:
                write_error(err, key, name[key], info["verbose"])
        # 1.6 input information with dimension (dim) as integer
        # tropopause (index)
        if (
            flag["tropopause"]
            & (len(name["tropopause"]) > 0)
            & (out["tropopause"].size > 0)
        ):
            var_dim = dim
            data0 = out["tropopause"]
            data0[np.isnan(data0)] = nc.default_fillvals["i4"]
            data0 = np.int32(data0)
            try:
                fid[name["tropopause"]][:] = data0.reshape(var_dim)
            except IndexError as err:
                write_error(
                    err, "tropopause", name["tropopause"], info["verbose"]
                )
        # 2. AMF results
        # 2.1 AMF results without wavelength dimension (dim)
        var_dim = dim
        keys = ["amf_geo"]
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                try:
                    fid[name[key]][:] = out[key].reshape(var_dim)
                except IndexError as err:
                    write_error(err, key, name[key], info["verbose"])
        # 2.2 AMF results with wavelength dimension (wv, dim) or (dim)
        if out["nwv"] == 1:
            var_dim = dim
        else:
            var_dim = (out["nwv"], *dim)
        keys = ["cloud_radiance_fraction", "vcd", "vcdtrop"]
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                if out["nwv"] == 1:
                    data = out[key].reshape(var_dim)
                else:
                    data = out[key].reshape(var_dim)[0]
                try:
                    fid[name[key]][:] = data
                except IndexError as err:
                    write_error(err, key, name[key], info["verbose"])
        # 2.3 list of AMF results with wavelength dimension
        # (wv, dim) or (dim)
        # dimension is the same as 2.2
        keys = ["amf", "amf_clr", "amf_cld"]
        for key in keys:
            if flag[key]:
                for i in range(len(name[key])):
                    if (len(name[key][i]) > 0) & (out[key][i].size > 0):
                        if out["nwv"] == 1:
                            data = out[key][i].reshape(var_dim)
                        else:
                            data = out[key][i].reshape(var_dim)[0]
                        try:
                            fid[name[key][i]][:] = data
                        except IndexError as err:
                            write_error(
                                err, key, name[key][i], info["verbose"]
                            )
        # 2.4 list of AMF results with wavelength/layer dimension
        # (wv, dim, layer) or (dim, layer)
        if out["nwv"] == 1:
            var_dim = (*dim, out["nlayer"])
        else:
            var_dim = (out["nwv"], *dim, out["nlayer"])
        keys = ["averaging_kernel", "averaging_kernel_clr"]
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                if out["nwv"] == 1:
                    data = out[key].reshape(var_dim)
                else:
                    data = out[key].reshape(var_dim)[0]
                try:
                    fid[name[key]][:] = data
                except IndexError as err:
                    write_error(err, key, name[key], info["verbose"])


def write_qa4ecv(filename, info, flag, name, out):
    with nc.Dataset(filename, "a") as fid:
        # satellite data dim
        dim = out["dim"]
        # 1. input information
        # 1.1 input information with dimension (dim)
        keys = [
            "latitude",
            "longitude",
            "solar_zenith_angle",
            "solar_azimuth_angle",
            "relative_azimuth_angle",
            "surface_altitude",
            "cloud_fraction",
            "cloud_pressure",
            "cloud_albedo",
            "surface_pressure",
            "scd",
        ]
        var_dim = dim
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                try:
                    fid[name[key]][:] = out[key].reshape(var_dim)
                except IndexError as err:
                    write_error(err, key, name[key], info["verbose"])
        # 1.2 input information with dimension (dim,4)
        keys = ["latitudecorners", "longitudecorners"]
        var_dim = (*dim, 4)
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                try:
                    fid[name[key]][:] = out[key].reshape(var_dim)
                except IndexError as err:
                    write_error(err, key, name[key], info["verbose"])
        # 1.3 input surface albedo information
        # for BRDF surface, save it in multi or single varibale
        # for multi wavelength calculation, save it in a single variable
        # wv_flag is not valid for QA4ECV format
        if (
            flag["surface_albedo"]
            & (len(name["surface_albedo"]) > 0)
            & (len(out["surface_albedo"]) > 0)
        ):
            if out["nwv"] == 1:
                var_dim = dim
                data = out["surface_albedo"][0].reshape(var_dim)
            else:
                var_dim = (out["nwv"], *dim)
                data = out["surface_albedo"][0].reshape(var_dim)[0]
            try:
                fid[name["surface_albedo"][0]][:] = data
            except IndexError as err:
                write_error(
                    err,
                    "surface albedo",
                    name["surface_albedo"][0],
                    info["verbose"],
                )
        # 1.4 input information with dimension (dim,layer)
        var_dim = (*dim, out["nlayer"])
        key = "profile"
        if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
            try:
                fid[name[key]][:] = out[key].reshape(var_dim)
            except IndexError as err:
                write_error(err, key, name[key], info["verbose"])
        # 1.6 input information with dimension (dim) as integer
        # tropopause (index)
        if (
            flag["tropopause"]
            & (len(name["tropopause"]) > 0)
            & (out["tropopause"].size > 0)
        ):
            var_dim = dim
            data0 = out["tropopause"]
            data0[np.isnan(data0)] = nc.default_fillvals["i4"]
            data0 = np.int32(data0)
            try:
                fid[name["tropopause"]][:] = data0.reshape(var_dim)
            except IndexError as err:
                write_error(
                    err, "tropopause", name["tropopause"], info["verbose"]
                )
        # 2. AMF results
        # 2.1 AMF results without wavelength dimension (dim)
        var_dim = dim
        keys = ["amf_geo"]
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                try:
                    fid[name[key]][:] = out[key].reshape(var_dim)
                except IndexError as err:
                    write_error(
                        err, "amf_geo", name["amf_geo"], info["verbose"]
                    )
        # 2.2 AMF results with wavelength dimension (wv, dim) or (dim)
        if out["nwv"] == 1:
            var_dim = dim
        else:
            var_dim = (out["nwv"], *dim)
            if info["verbose"]:
                print(
                    "warning: number of wavelength > 1 and only output the "
                    "AMF results for the first wavelength for QA4ECV format"
                )
        keys = ["cloud_radiance_fraction", "vcd", "vcdtrop"]
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                if out["nwv"] == 1:
                    data = out[key].reshape(var_dim)
                else:
                    data = out[key].reshape(var_dim)[0]
                try:
                    fid[name[key]][:] = data
                except IndexError as err:
                    write_error(err, key, name[key], info["verbose"])
        # 2.3 list of AMF results with wavelength dimension
        # (wv, dim) or (dim)
        # dimension is the same as 2.2
        keys = ["amf", "amf_clr", "amf_cld"]
        for key in keys:
            if flag[key]:
                for i in range(len(name[key])):
                    if (len(name[key][i]) > 0) & (out[key][i].size > 0):
                        if out["nwv"] == 1:
                            data = out[key][i].reshape(var_dim)
                        else:
                            data = out[key][i].reshape(var_dim)[0]
                        try:
                            fid[name[key][i]][:] = data
                        except IndexError as err:
                            write_error(err, key, name[key], info["verbose"])
        # 2.4 list of AMF results with wavelength/layer dimension
        # (wv, dim, layer) or (dim, layer)
        if out["nwv"] == 1:
            var_dim = (*dim, out["nlayer"])
        else:
            var_dim = (out["nwv"], *dim, out["nlayer"])
        keys = ["averaging_kernel", "averaging_kernel_clr"]
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                if out["nwv"] == 1:
                    data = out[key].reshape(var_dim)
                else:
                    data = out[key].reshape(var_dim)[0]
                try:
                    fid[name[key]][:] = data
                except IndexError as err:
                    write_error(err, key, name[key], info["verbose"])


def write_file(filename, info, flag, name, out):
    with nc.Dataset(filename, "a") as fid:
        # satellite data dim
        dim = out["dim"]
        # 1. input information
        # 1.1 input information with dimension (dim)
        keys = [
            "latitude",
            "longitude",
            "solar_zenith_angle",
            "sensor_zenith_angle",
            "solar_azimuth_angle",
            "sensor_azimuth_angle",
            "relative_azimuth_angle",
            "surface_altitude",
            "cloud_fraction",
            "cloud_pressure",
            "cloud_albedo",
            "surface_pressure",
            "scd",
        ]
        var_dim = dim
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                try:
                    fid[name[key]][:] = out[key].reshape(var_dim)
                except IndexError as err:
                    write_error(err, key, name[key], info["verbose"])
        # 1.2 input information with dimension (dim,4)
        keys = ["latitudecorners", "longitudecorners"]
        var_dim = (*dim, 4)
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                try:
                    fid[name[key]][:] = out[key].reshape(var_dim)
                except IndexError as err:
                    write_error(err, key, name[key], info["verbose"])
        # 1.3 input surface albedo information
        # for BRDF surface, save it in multi or single varibale
        # for multi wavelength calculation, save it in a single variable
        if (
            flag["surface_albedo"]
            & (len(name["surface_albedo"]) > 0)
            & (len(out["surface_albedo"]) > 0)
        ):
            if out["nalb"] > 1:
                if len(name["surface_albedo"]) == out["nalb"]:
                    var_dim = dim
                    for ialb in range(name["surface_albedo"]):
                        if out["surface_albedo"][ialb].size > 0:
                            try:
                                fid[name["surface_albedo"][ialb]][:] = out[
                                    "surface_albedo"
                                ][ialb].reshape(var_dim)
                            except IndexError as err:
                                write_error(
                                    err,
                                    "surface_albedo",
                                    name["surface_albedo"][ialb],
                                    info["verbose"],
                                )
                elif len(name["surface_albedo"]) == 1:
                    var_dim = (out["nalb"], *dim)
                    data1 = np.full(var_dim)
                    for ialb in range(name["surface_albedo"]):
                        data1[ialb, ...] = out["surface_albedo"][ialb].reshape(
                            var_dim
                        )
                    try:
                        fid[name["surface_albedo"][0]][:] = data1
                    except IndexError as err:
                        write_error(
                            err,
                            "surface_albedo",
                            name["surface_albedo"][0],
                            info["verbose"],
                        )
            elif out["nwv"] > 1:
                if out["surface_albedo"][0].size > 0:
                    var_dim = (out["nwv"], *dim)
                    try:
                        fid[name["surface_albedo"][0]][:] = out[
                            "surface_albedo"
                        ][0].reshape(var_dim)
                    except IndexError as err:
                        write_error(
                            err,
                            "surface_albedo",
                            name["surface_albedo"][0],
                            info["verbose"],
                        )
            else:
                assert out["nwv"] == out["nalb"] == 1
                if len(name["surface_albedo"][0]) > 0:
                    var_dim = dim
                    if info["wv_flag"]:
                        alb = np.full(dim, np.nan).ravel()
                        for iwv in range(out["surface_albedo"][0].shape[0]):
                            idx = out["wvidx"] == iwv + 1
                            if np.count_nonzero(idx):
                                alb[idx] = out["surface_albedo"][0][iwv, idx]
                        data = alb.reshape(var_dim)
                    else:
                        data = out["surface_albedo"][0].reshape(var_dim)
                    try:
                        fid[name["surface_albedo"][0]][:] = data
                    except IndexError as err:
                        write_error(
                            err,
                            "surface_albedo",
                            name["surface_albedo"][0],
                            info["verbose"],
                        )
        # 1.4 input information with dimension (dim,layer)
        var_dim = (*dim, out["nlayer"])
        key = "profile"
        if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
            try:
                fid[name[key]][:] = out[key].reshape(var_dim)
            except IndexError as err:
                write_error(err, key, name[key], info["verbose"])
        key = "pressure"
        if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
            data0 = (out[key][:, 1:] + out[key][:, :-1]) / 2.0
            try:
                fid[name[key]][:] = data0.reshape(var_dim)
            except IndexError as err:
                write_error(err, key, name[key], info["verbose"])
        # 1.5 input information with dimension (dim,layer) or (dim)
        # temperature profile (or only surface temperature)
        if (
            flag["temperature"]
            & (len(name["temperature"]) > 0)
            & (out["temperature"].size > 0)
        ):
            if out["temperature"].shape[-1] == 1:
                var_dim = dim
            else:
                var_dim = (*dim, out["nlayer"])
            try:
                fid[name["temperature"]][:] = out["temperature"].reshape(
                    var_dim
                )
            except IndexError as err:
                write_error(
                    err, "temperature", name["temperature"], info["verbose"]
                )
        # 1.6 input information with dimension (dim) as integer
        # tropopause (index)
        if (
            flag["tropopause"]
            & (len(name["tropopause"]) > 0)
            & (out["tropopause"].size > 0)
        ):
            var_dim = dim
            data0 = out["tropopause"]
            data0[np.isnan(data0)] = nc.default_fillvals["i1"]
            data0 = np.int8(data0)
            try:
                fid[name["tropopause"]][:] = data0.reshape(var_dim)
            except IndexError as err:
                write_error(
                    err, "tropopause", name["tropopause"], info["verbose"]
                )
        # 2. AMF results
        # 2.1 AMF results without wavelength dimension (dim)
        var_dim = dim
        keys = ["amf_geo"]
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                try:
                    fid[name[key]][:] = out[key].reshape(var_dim)
                except IndexError as err:
                    write_error(err, key, name[key], info["verbose"])
        # 2.2 AMF results with wavelength dimension (wv, dim) or (dim)
        if out["nwv"] == 1:
            var_dim = dim
        else:
            var_dim = (out["nwv"], *dim)
        keys = ["cloud_radiance_fraction", "vcd", "vcdtrop"]
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                try:
                    fid[name[key]][:] = out[key].reshape(var_dim)
                except IndexError as err:
                    write_error(err, key, name[key], info["verbose"])
        # 2.3 list of AMF results with wavelength dimension
        # (wv, dim) or (dim)
        # dimension is the same as 2.2
        keys = ["amf", "amf_clr", "amf_cld"]
        for key in keys:
            if flag[key]:
                for i in range(len(name[key])):
                    if (len(name[key][i]) > 0) & (out[key][i].size > 0):
                        try:
                            fid[name[key][i]][:] = out[key][i].reshape(var_dim)
                        except IndexError as err:
                            write_error(err, key, name[key], info["verbose"])
        # 2.4 list of AMF results with wavelength/layer dimension
        # (wv, dim, layer) or (dim, layer)
        if out["nwv"] == 1:
            var_dim = (*dim, out["nlayer"])
        else:
            var_dim = (out["nwv"], *dim, out["nlayer"])
        keys = ["averaging_kernel", "averaging_kernel_clr"]
        for key in keys:
            if flag[key] & (len(name[key]) > 0) & (out[key].size > 0):
                try:
                    fid[name[key]][:] = out[key].reshape(var_dim)
                except IndexError as err:
                    write_error(err, key, name[key], info["verbose"])
