import numpy as np
from netCDF4 import Dataset
from pathlib import Path
import os
import bz2
from importlib import resources

ARGOSTATS = os.environ["ARGOSTATS"]

minimal_depth = -10

MSKFILE = "msk_one10th.bin"


def get_mskfile():
    d = resources.files("argostats")
    p = Path(f"{d.as_posix()}/bathy/{MSKFILE}")
    return p


def visual_check():
    import matplotlib.pyplot as plt
    plt.clf()
    plt.pcolor(*read_topo())
    plt.colorbar()
    plt.show(block=False)


def findclosest(vector, value):
    return np.argmin(np.abs(vector-value))


class Topo:
    def __init__(self, src="bin"):
        if src == "bin":
            self.lon, self.lat, self.m = self.from_bin()

        else:
            self.lon, self.lat, self.h = read_topo(src)
            self.m = (self.h < minimal_depth).astype("i1")

    def get_idx(self, glon, glat):
        idx = np.asarray([findclosest(self.lon, lon) for lon in glon])
        jdx = np.asarray([findclosest(self.lat, lat) for lat in glat])
        Idx = jdx[:, np.newaxis]*len(self.lon)+idx[np.newaxis, :]
        return idx, jdx, Idx

    def msk(self, glon, glat):
        _, _, Idx = self.get_idx(glon, glat)
        return self.m.ravel()[Idx]

    def land(self, domain):
        lon1, lon2, lat1, lat2 = domain
        idx, jdx, _ = self.get_idx([lon1, lon2], [lat1, lat2])
        i0, i1 = idx
        j0, j1 = jdx
        return self.lon[i0:i1], self.lat[j0:j1], 1-self.m[j0:j1, i0:i1]

    def to_bin(self):
        assert hasattr(self, "h")
        m = (self.h < minimal_depth).astype("i1")
        o = bz2.compress(m)
        print(f"{MSKFILE} filesize: {len(o)}")
        with open(get_mskfile(), "bw") as fid:
            fid.write(o)

    def from_bin(self):
        f = get_mskfile()
        msg = f"Unable to setup the bathymetry mask, '{f}' is missing"
        assert f.is_file(), msg
        with open(f, "br") as fid:
            o = fid.read()
        reso = 1/10
        nlat, nlon = shape = self.shape(reso)
        # shape = (720, 1440)
        m = np.frombuffer(bz2.decompress(o), "i1").reshape(shape)
        lon = np.arange(nlon)*reso-180+reso/2
        lat = np.arange(nlat)*reso-90+reso/2
        return lon, lat, m

    def shape(self, reso):
        return (int(180/reso), int(360/reso))


def read_topo(etopofile):
    f = Path(etopofile)
    msg = f"Unable to setup the bathymetry mask, '{f}' is missing"
    assert f.is_file(), msg
    with Dataset(f.as_posix()) as nc:
        lon = nc.variables["lon"][:]
        lat = nc.variables["lat"][:]
        h = nc.variables["h"][:, :]
    return lon, lat, h
