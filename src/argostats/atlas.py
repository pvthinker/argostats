import numpy as np
import gsw
from netCDF4 import Dataset

from argostats.tools.parallel import get_pool, chrono, split_range_evenly
from argostats.toctools import ARGOSTATS
from argostats.lorenz.lorenz_state import gammat_analytic
from argostats.bathy.bathy import Topo

import warnings
warnings.filterwarnings("ignore")

R14 = 0
T25 = 1

EAPE_algo = {R14: "Roullet et al. 2014",
             T25: "Tailleux and Roullet 2025"}

DOMAINS = {
    "biscaye": [-15, 0, 35, 60],
    "subpolar": [-70, 0, 50, 70],
    "korea": [130, 140, 36, 44],
    "natl": [-90, 10, -10, 70],
    "satl": [-80, 0, -70, 10],
    "acc_indian": [0, 100, -60, -20],
    "acc": [-180, 180, -70, -30],
    "zapiola": [-60, -40, -50, -30],
    "atl": [-80, 20, -60, 60],
    "senegal": [-30, -10, 10, 30],
    "agulhas": [10, 30, -50, -30],
    "newzealand": [140, 180, -60, -30],
    "gulfstream": [-80, -50, 20, 50],
    "drake": [-80, -50, -70, -40],
    "gulfmexico": [-100, -60, 10, 30]
}

TOPO = Topo()

UNITS = {"CT": "°C",
         "SR": "$g\,kg^{-1}$",
         "SIGSTAR": "$kg\,m^{-3}$",
         "GAMMAT": "$kg\,m^{-3}$",
         "EAPE_R14": "$cm^2\,s^{-2}$",
         "EAPE_T25": "$cm^2\,s^{-2}$"
         }


def approximate_haversine(lon1, lat1, lon2, lat2):
    """ this function matches the haversine formula
    for small distances, corresponding to angles d<1.0 rad=57°
    """
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    deltalam = np.deg2rad(lon1-lon2)
    return 1-np.cos(phi1-phi2)+np.cos(phi1)*np.cos(phi2)*(1-np.cos(deltalam))


def get_data_near_pos(lon, lat, lon0, lat0):
    phi1 = np.deg2rad(lat)
    phi2 = np.deg2rad(lat0)
    deltalam = np.deg2rad(lon-lon0)
    return 1-np.cos(phi1-phi2)+np.cos(phi1)*np.cos(phi2)*(1-np.cos(deltalam))


def haversine_arg(lon1, lat1, lon2, lat2):
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    deltalam = np.deg2rad(lon1-lon2)/2
    deltaphi = np.deg2rad(lat1-lat2)/2

    arg = np.sin(deltaphi)**2+np.cos(phi1)*np.cos(phi2)*np.sin(deltalam)**2
    return arg


def haversine(lon1, lat1, lon2, lat2):
    """ exact Haversine formula
    https://en.wikipedia.org/wiki/Haversine_formula
    """
    arg = haversine_arg(lon1, lat1, lon2, lat2)
    return np.rad2deg(2*np.arcsin(np.sqrt(arg)))


def mylinspace(x1, x2, dx): return x1+np.arange(int((x2-x1)/dx))*dx


def get_gridpos(box, reso=1):
    lon1, lon2, lat1, lat2 = box
    lon = mylinspace(lon1, lon2, reso)
    lat = mylinspace(lat1, lat2, reso)
    return (lon, lat)


def mean(X, coef, N):
    res = (X*coef).sum(axis=0)
    res[N > 0] /= N[N > 0]
    return res


def select_data(data, lon0, lat0, dcritical):
    which = ["LATITUDE", "LONGITUDE", "IDX"]

    lat, lon, iidx = data[which]

    dist = haversine_arg(lon, lat, lon0, lat0)
    kdx, = np.where(dist < dcritical)

    idx = iidx[kdx, :]
    c = np.exp(-dist[kdx])
    coef = c[:, np.newaxis]*idx

    N = coef.sum(axis=0)

    return kdx, coef, N


def get_TS_at_pos(data, kdx, coef, N):
    """ Compute climatological average of CT and SR at a given location

    Parameters
    ----------
    data : AOS
        interpolated Argo profiles
    kdx : ndarray
        vector of index of data profiles to select
    coef : ndarray
        2-d weight of each measurement in the climatology
    N : ndarray
        sum of coefs along the data index axis

    Returns
    -------
    CT, SR : ndarray
        1-D vertical profiles at interpolated levels
    """

    which = ["CT", "SR"]
    T, S = data[which]

    CT = T[kdx, :]
    SR = S[kdx, :]

    return mean(CT, coef, N), mean(SR, coef, N)


def eape_unit():
    dbar = 1e4
    rho0 = 1e3
    cm2 = 1e4
    return dbar/rho0**2*cm2


def get_eape_roullet_at_pos(data, RHO, CF, kdx, coef, N):
    """Compute climatological EAPE at a given location

    Using Roullet et al (2014) algorithm.

    The computation of the isopycnal displacement has been improved in
    2020 (unpublished). It now uses the 'compensated' density RHOSTAR,
    to remove the isentropic compressibility. RHOSTAR is a sort of
    'neutral density'.

    Parameters
    ----------
    data : AOS
        interpolated Argo profiles
    kdx : ndarray
        vector of index of data profiles to select
    coef : ndarray
        2-d weight of each measurement in the climatology
    N : ndarray
        sum of coefs along the data index axis

    Returns
    -------
    EAPE : ndarray
        1-D vertical profile at interpolated levels

    """

    PREF = np.asarray(data.PREF)
    which = ["SR", "CT"]
    SR, CT = data[which]

    rho = gsw.density.rho(SR[kdx], CT[kdx], PREF[np.newaxis, :])

    # Roullet et al 2014 EAPE
    p = np.interp(rho*CF[np.newaxis, :], RHO*CF, PREF)

    EAPE = (0.5*eape_unit())*(rho-RHO)*(p-PREF)

    return mean(EAPE, coef, N)


def get_eape_tailleux_at_pos(data, RHO, kdx, coef, N):
    """Compute climatological EAPE at a given location

    Using Tailleux and Roullet (2025) algorithm.

    The computation of the isopycnal displacement uses the World Ocean
    Lorenz profile, that has been been fitted by Tailleux and Wolf (2020).

    Parameters
    ----------
    data : AOS
        interpolated Argo profiles
    kdx : ndarray
        vector of index of data profiles to select
    coef : ndarray
        2-d weight of each measurement in the climatology
    N : ndarray
        sum of coefs along the data index axis

    Returns
    -------
    EAPE : ndarray
        1-D vertical profile at interpolated levels

    """

    PREF = np.asarray(data.PREF)
    which = ["SR", "CT"]
    SR, CT = data[which]

    rho = gsw.density.rho(SR[kdx], CT[kdx], PREF[np.newaxis, :])

    # Tailleux and Roullet 2025 EAPE
    _, _, p, _ = gammat_analytic(SR[kdx], CT[kdx])

    EAPE = (0.5*eape_unit())*(rho-RHO)*(p-PREF)

    return mean(EAPE, coef, N)


def crop_data(data, glon, glat, width):
    lon1 = glon.min()
    lon2 = glon.max()
    lat1 = glat.min()
    lat2 = glat.max()

    lon0 = 0.5*(lon1+lon2)
    lat0 = 0.5*(lat1+lat2)

    d1 = haversine(lon0, lat0, lon1, lat1)
    d2 = haversine(lon0, lat0, lon2, lat2)

    radius = (max(d1, d2)+width)

    which = ["LATITUDE", "LONGITUDE", "FLAG", "DATA_MODE"]
    lat, lon, flag, data_mode = data[which]

    d = haversine(lon0, lat0, lon, lat)
    # TODO: make customizable the condition (flag == 1) & (data_mode == 1)
    return data.crop((d <= radius) & (flag == 1) & (data_mode == 1))


def proceed_TS_tile(k, data, glon, glat, dcritical, unused):

    print(f"\r  tile #{k:3}", end="")

    nz = len(data.PREF)
    nlon = len(glon)
    nlat = len(glat)
    T = np.zeros((nlat, nlon, nz), dtype="f4")
    S = np.zeros((nlat, nlon, nz), dtype="f4")

    msk = TOPO.msk(glon, glat)

    for j, i in np.ndindex(nlat, nlon):
        if msk[j, i]:
            kdx, coef, N = select_data(data, glon[i], glat[j], dcritical)
            *_, T[j, i], S[j, i] = get_TS_at_pos(data, kdx, coef, N)

    return T, S


def proceed_eape_tile(k, data, glon, glat, dcritical, algo):
    print(f"\r  tile #{k}", end="")
    PREF = np.asarray(data.PREF)
    nz = len(PREF)
    nlon = len(glon)
    nlat = len(glat)
    rho = np.zeros((nlat, nlon, nz), dtype="f4")
    EAPE = np.zeros((nlat, nlon, nz), dtype="f4")

    msk = TOPO.msk(glon, glat)

    for j, i in np.ndindex(nlat, nlon):
        if msk[j, i]:
            kdx, coef, N = select_data(data, glon[i], glat[j], dcritical)
            *_, Tm, Sm = get_TS_at_pos(data, kdx, coef, N)
            RHO = gsw.density.rho(Sm, Tm, PREF)

            if algo == R14:
                # rho is SIGMASTAR
                CF = compute_CF(Sm, Tm, RHO, PREF)
                rho[j, i, :] = RHO*CF-1000
                EAPE[j, i, :] = get_eape_roullet_at_pos(
                    data, RHO, CF, kdx, coef, N)

            elif algo == T25:
                # rho is GAMMAT
                rho[j, i, :], *_ = gammat_analytic(Sm, Tm)
                EAPE[j, i, :] = get_eape_tailleux_at_pos(
                    data, RHO, kdx, coef, N)
            else:
                raise ValueError("unknown EAPE algo")

    return EAPE, rho


def dist_threshold(reso, smoothing_factor):
    d0 = np.deg2rad(smoothing_factor*reso)
    return np.sin(d0/2)**2


def mid(x):
    return 0.5*(x[1:]+x[:-1])


def compute_CF(SR, CT, RHO, PRES):
    dbar = 1e4
    csound = gsw.sound_speed(mid(SR), mid(CT), mid(PRES))
    integrand = np.hstack([0, dbar*np.diff(PRES)/(mid(RHO)*csound**2)])
    return np.exp(-np.cumsum(integrand))


def setup_tile(k, data, gglon, gglat, j, i, nlat, nlon, width):
    """ Crop the data (of AOS type) that are needed for the tile

    This is an essential feature for speed
    """
    print(f"\rset up tiles {j:2}x{i:2}", end="")
    jdx = split_range_evenly(len(gglat), nlat)[j]
    idx = split_range_evenly(len(gglon), nlon)[i]
    glat = gglat[jdx]
    glon = gglon[idx]
    tiledata = crop_data(data, glon, glat, width)
    return (k, tiledata, glon, glat), (jdx, idx, slice(None))


def get_subdomains(domain, reso):
    """Define sub-domains for Large Atlas

    A small atlas is attached on each subdomain. The computation of
    the whole domain is delegated by each small atlas.

    """

    glon, glat = get_gridpos(domain, reso=reso)

    lon1, lon2, lat1, lat2 = domain

    dlon, dlat = lon2-lon1, lat2-lat1
    nlon, nlat = int(dlon/15), int(dlat/15)

    idx = split_range_evenly(len(glon), nlon)
    jdx = split_range_evenly(len(glat), nlat)

    i0 = [s.start for s in idx]
    j0 = [s.start for s in jdx]

    lon = [float(glon[i]) for i in i0]+[lon2]
    lat = [float(glat[j]) for j in j0]+[lat2]

    domains = [(lon[i], lon[i+1], lat[j], lat[j+1])
               for j, i in np.ndindex((nlat, nlon))]

    index = [(jdx[j], idx[i], slice(None))
             for j, i in np.ndindex((nlat, nlon))]

    return domains, index


def is_small(domain):
    lon1, lon2, lat1, lat2 = domain
    return ((lon2-lon1) < 30) & ((lat2-lat1) < 30)


def Atlas(domain, reso, globalaos, pool=None):
    if is_small(domain):
        return SmallAtlas(domain, reso, globalaos, pool=None)
    else:
        return LargeAtlas(domain, reso, globalaos, pool=None)


class LargeAtlas:
    def __init__(self, domain, reso, globalaos, pool=None):
        self.domain = domain
        self.reso = reso
        self.algo = R14
        self.smoothing_factor = 4
        self.aos = crop_data(globalaos, *self.lonlat,
                             self.reso*self.smoothing_factor)

        self.PREF = np.asarray(globalaos.PREF)
        self.set_shape()
        if pool is None:
            self.pool = get_pool()
        else:
            self.pool = pool
        self.setup_atlases(globalaos)

    @property
    def lonlat(self):
        return get_gridpos(self.domain, reso=self.reso)

    def set_shape(self):
        nz = len(self.PREF)
        lon, lat = self.lonlat
        nlon = len(lon)
        nlat = len(lat)
        self.shape = (nlat, nlon, nz)

    def setup_atlases(self, aos):
        self.subdomains, self.idx = get_subdomains(self.domain, self.reso)
        print(f"Setup sub-atlases: #{len(self.subdomains)}")
        self.atlases = []
        for k, d in enumerate(self.subdomains):
            print(f"\rsub-atlas {k}")
            self.atlases += [SmallAtlas(d, self.reso, aos, pool=self.pool)]
            print("\x1b[1A", end="")
        print()

    def _allocate_clim_array(self):
        return np.zeros(self.shape, dtype="f4")

    def clim_TS(self):
        self.CT = np.zeros(self.shape, dtype="f4")
        self.SR = np.zeros(self.shape, dtype="f4")

        self._process_subatlases(self.CT, self.SR, "clim_TS")

        return self.CT, self.SR

    def clim_EAPE(self, algo=None):
        if algo is not None:
            self.algo = algo

        if algo == R14:
            self.EAPE_R14 = self._allocate_clim_array()
            self.SIGSTAR = self._allocate_clim_array()
            eape = self.EAPE_R14
            rho = self.SIGSTAR
        elif algo == T25:
            self.EAPE_T25 = self._allocate_clim_array()
            self.GAMMAT = self._allocate_clim_array()
            eape = self.EAPE_T25
            rho = self.GAMMAT

        self._process_subatlases(eape, rho, "clim_EAPE")

        return eape, rho

    def _process_subatlases(self, X, Y, method):
        for k, (atlas, idx) in enumerate(zip(self.atlases, self.idx)):
            print(f"sub-atlas {k}")
            func = getattr(atlas, method)
            Xloc, Yloc = func(algo=self.algo)
            print("\x1b[3A", end="")
            X[idx] = Xloc
            Y[idx] = Yloc

    def to_netcdf(self):
        write_atlas_to_netcdf(self)


class SmallAtlas:
    def __init__(self, domain, reso, globalaos, pool=None):
        self.domain = domain
        self.reso = reso
        self.smoothing_factor = 4
        self.algo = R14
        self.dcritical = dist_threshold(self.reso, self.smoothing_factor)
        self.PREF = np.asarray(globalaos.PREF)
        self.aos = crop_data(globalaos, *self.lonlat,
                             self.reso*self.smoothing_factor)

        self.parallel = True
        if pool is None:
            self.pool = get_pool()
        else:
            self.pool = pool
        self.set_tiles()
        self.set_shape()

    @property
    def lonlat(self):
        return get_gridpos(self.domain, reso=self.reso)

    def set_shape(self):
        nz = len(self.PREF)
        lon, lat = self.lonlat
        nlon = len(lon)
        nlat = len(lat)
        self.shape = (nlat, nlon, nz)

    def set_tiles(self):
        self.n = n = 10
        gglon, gglat = self.lonlat
        self.nlontiles, self.nlattiles = len(gglon)//self.n, len(gglat)//self.n

        width = 3  # self.dcritical*3
        self.tile_definition = [(self.aos, gglon, gglat, j, i,
                                 self.nlattiles, self.nlontiles, width)
                                for j, i in np.ndindex(self.nlattiles, self.nlontiles)]

        self.tiles = []
        self.idx = []
        for k, (j, i) in enumerate(np.ndindex(self.nlattiles, self.nlontiles)):
            tile, idx = setup_tile(k, self.aos, gglon, gglat, j, i,
                                   self.nlattiles, self.nlontiles, width)

            self.tiles += [tile]
            self.idx += [idx]

    def _average_tiles(self, method):
        tasks = (tile+(self.dcritical, self.algo)
                 for tile in self.tiles)

        print(f"average tiles: #{len(self.tiles)}")
        if self.parallel:
            res = self.pool.starmap(method, tasks)
        else:
            res = [method(*task) for task in tasks]
        print()
        return res

    def _allocate_clim_array(self):
        return np.zeros(self.shape, dtype="f4")

    def clim_TS(self, **kwargs):
        res = self._average_tiles(proceed_TS_tile)

        self.CT = self._allocate_clim_array()
        self.SR = self._allocate_clim_array()

        for idx, (Tloc, Sloc) in zip(self.idx, res):
            self.CT[idx] = Tloc
            self.SR[idx] = Sloc

        return self.CT, self.SR

    def clim_EAPE(self, algo=None):
        if algo is not None:
            self.algo = algo
        res = self._average_tiles(proceed_eape_tile)

        if algo == R14:
            self.EAPE_R14 = self._allocate_clim_array()
            self.SIGSTAR = self._allocate_clim_array()
            eape = self.EAPE_R14
            rho = self.SIGSTAR
        elif algo == T25:
            self.EAPE_T25 = self._allocate_clim_array()
            self.GAMMAT = self._allocate_clim_array()
            eape = self.EAPE_T25
            rho = self.GAMMAT

        for idx, (Eloc, Rloc) in zip(self.idx, res):
            eape[idx] = Eloc
            rho[idx] = Rloc

        return eape, rho

    def to_netcdf(self):
        write_atlas_to_netcdf(self)


def infer_domain_name(domain):
    if domain in DOMAINS.values():
        return next(k for k, v in DOMAINS.items() if domain == v)
    else:
        return "someregion"


def write_atlas_to_netcdf(atlas):
    name = infer_domain_name(atlas.domain)
    ncfile = f"{ARGOSTATS}/atlas_{name}.nc"
    atts = {"name": "Atlas of Argo statistics",
            "domain": atlas.domain,
            "resolution": atlas.reso,
            "latest profile": str(atlas.aos.JULD.max())
            }
    lon, lat = atlas.lonlat
    pref = atlas.PREF
    dimensions = {"lon": {"size": len(lon),
                          "value": lon,
                          "specs": ("lon", "f4", ("lon",))},
                  "lat": {"size": len(lat),
                          "value": lat,
                          "specs": ("lat", "f4", ("lat",))},
                  "pres": {"size": len(pref),
                           "value": pref,
                           "specs": ("pres", "f4", ("pres",))}
                  }

    names = ["CT", "SR", "EAPE_R14", "EAPE_T25", "SIGSTAR", "GAMMAT"]
    variables = {name: {"specs": (name, "f4", ("pres", "lat", "lon")),
                        "atts": {"units": UNITS[name]},
                        "value": getattr(atlas, name).transpose((2, 0, 1))
                        }
                 for name in names
                 if name in atlas.__dict__
                 }
    print(f"store atlas in {ncfile}")
    print(f"   variables: {list(variables)}")
    write_netcdf(ncfile, atts, dimensions, variables)


def write_netcdf(ncfile, atts, dimensions, variables):

    with Dataset(ncfile, "w", format='NETCDF4') as nc:

        nc.setncatts(atts)

        for dim in dimensions:
            size = dimensions[dim]["size"]
            nc.createDimension(dim, size)

        for dim in dimensions:
            name, dtype, dims = dimensions[dim]["specs"]
            nc.createVariable(dim, dtype, dims)

        for var in variables:
            name, dtype, dims = variables[var]["specs"]
            nc.createVariable(name, dtype, dims)

        for dim in dimensions:
            nc.variables[dim][:] = dimensions[dim]["value"]

        for var in variables:
            nc.variables[var][:] = variables[var]["value"]
            if "atts" in variables[var]:
                atts = variables[var]["atts"]
                nc.variables[var].setncatts(atts)
