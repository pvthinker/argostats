# Argostats

Argostats is a set of Python tools to perform statistics with the Argo
database. As of October 2025, the database includes more than 3
millions profiles. The tools are the following

1. **build a Pandas DataFrame** that contains the key parameters of
   each profile: `DAC, WMO, I_PROF, N_LEVS, LATITUDE, LONGITUDE, JULD,
   DATA_MODE, PLATFORM_TYPE, FLAG`. `FLAG` is derived from the
   `QC`. `FLAG==1` means all `QC` are 1.
2. **interpolate all the Argo profiles** on selected pressure points
   `PREF` and store them in a single binary file. The variables are
   `CT` and `SR`, computed from native `TEMP` and `PSAL` using TEOS-10.
3. **compute climatological statistics** on gridded regions. The
   statistics include mean CT, mean SR, mean SIGMA and EAPE. EAPE can
   be computed using either Roullet et al. (2014) or Tailleux and
   Roullet (2025) method. The resolution of the grid is a user
   parameter. With the current coverage, global atlas at 1/4° are
   possible. This resolution can be increased in regions where the
   profiles density is large.


## Installation

1. Define in your shell two variables
```shell
export GDAC=/path/to/Argo/GDAC
export ARGOSTATS=/your/work/directory/to/store/the/generated/files
```
2. Create a virtual environement

3. Test everything works fine


## Usage

1. Open an interactive Python session, either an Ipython terminal or a
   notebook.

2. Build the Pandas DataFrame
```python
import argostats as argo
argo.build_summary()
```

or load it if you have already build one

```python
import argostats as argo
df = argo.load_summary()
```

It takes about 5' to build it (with 28 cores).

3. Explore the dataframe. You can already do a lot of interesting
   things with this dataframe. All the 3M+ profiles locations, from
   the ~20k profilers, are now are your fingertips. For instance

```python
def plot_profiles_location(df):
    plt.figure()
    plt.plot(df.LONGITUDE, df.LATITUDE, '.', markersize=1, alpha=0.01)

def count_profiles_per_year(df):
    year = argo.year(np.asarray(df.JULD))
    count, bins = np.histogram(year, bins=2000+np.arange(26))
    plt.figure()
    plt.bar(bins[:-1], count)

plot_profiles_location(df)
count_profiles_per_year(df)
```



4. Interpolate the profiles

```python
import argostats as argo

df = argo.load_summary()
ai = argo.ArgoInterp(df)
ai.proceed_all()
```

This takes about 10' to complete (with 28 cores).

5. Load the profiles

```python
import argostats as argo

aop = argo.load_profiles()
print(aop)
```

and explore them. The variable `aop` contains all the interpolated profiles along with their informations from the dataframe. Technically `aop` is an `ArrayOfStruct`, a central `class` of `argostats`.

```python
lon,lat,ct = aop[["LONGITUDE", "LATITUDE", "CT"]]
print(lon)
print(lon.shape, lat.shape, ct.shape)
```

You can filter the profiles and plot them

```python
smallaop = aop.crop((-10<lon)&(lon<-9)&(48<lat)&(lat<49))
ct, idx = smallaop[["CT", "IDX"]]
print(ct.shape)
plt.clf()
for k in range(len(ct)):
    plt.plot(ct[k][idx[k]==1], aop.PREF[idx[k]], "k", alpha=0.2)
```

The array `idx[k,j]` flags whether the profile `k` has an interpolated value at level `j`.

5. Compute an atlas.

```python
aop = argo.load_profiles()
domain = argo.DOMAINS["agulhas"]
at = argo.Atlas(domain, 1/4, aop)
CT, SR = at.clim_TS()
EAPE, SIGSTAR = at.clim_EAPE(algo=argo.R14)
at.to_netcdf()

lon, lat = at.lonlat
print(lon.shape, lat.shape, CT.shape)
```

The NetCDF file is stored in the `ARGOSTATS` directory. Two methods are possible forEAPE `R14` or `T25`.

and plot it

```Python
argo.figures.map_atlas(at, "EAPE", vmin=0, vmax=2000, kz=43)
```

## Under the hood
The **computations** with Argostats **are fast** for several reasons

1. the computations are **multithreaded**. For simply reason, it uses
   the basic `multiprocessing.Pool` instead of the fancier `dask`
   method.
2. the interpolated profiles are stored in a **single array**. It
   contains 3M+ rows and is about 2GB in size. This array contains
   heteregeneous variables of various different types. Data for each
   profile are contiguous in memory, despite their heterogeneity. Data
   for each WMO are also contiguous in memory. Each variable is
   accessed transparently. This array is stored in a **self-documented
   binary file**. These two features are implemented in
   `binaryfile.py`. In your shell, do a `more` of this binary file to
   see its structure.
3. the interpolation is done per WMO, each bunch of profiles is then
   immediately stored in the binary file. This computation can be
   multithreaded without dead-locks because each thread works on its
   own data, and each thread writes directly in the binary file.
4. the array of **profiles are cropped to local sub-domains** when
   computing statistics, this allows to work with a reduced amount of
   profiles, which mean **smaller memory exchange and fewer FLOPS**.

## Numerics
1. Vertical interpolations are done using either a basic cubic
   interpolation (default choice) or `gsw.interpolation.sa_ct_interp`
2. Climatological means are computed using a gaussian weight on the
   distance of the profile to the grid point (like in R14).
3. Grid points on land are skipped. The `bathymetry` modules loads a
   1/10° msk, extracted from ETOPO, that flags whether a point is over land.
4. `JULD` are stored in the basic `numpy.datetime` format, instead of
   the more powerful `pandas.TimeStamp` format. The reason is the
   `ArrayOfStruct` that requires a numpy native format for all variables.
5. All floats are stored in 32 bits format ("f4"), like in the native
   Argo database.

## Data quality

1. Argo data are read from the `*_proj.nc` files.
2. When available we use the ajusted values
3. For the interpolation we use only data with all the `QC==1`: qc on
   location, date, pressure, temperature and salinity.


## Outliers

The climatology have outliers here and there. There is still some work
ahead to remove them.
