import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
from argostats.atlas import UNITS, TOPO
from argostats.tools.colormaps import precip16


def map_atlas(atlas, field, kz=43, plottopo=False, **kwargs):
    """Produce a horizontal map of an atlas variable

    Parameters
    ----------
    atlas: argo Atlas

    field: str
        name of the variables

    Keywords
    --------
    kz: int
        vertical level

    **kwargs are passed to pcolor

    """
    plt.ion()

    glon, glat = atlas.lonlat
    pres = atlas.PREF[kz]
    T = getattr(atlas, field)

    if "EAPE" in field:
        cmap = precip16
    else:
        cmap = "YlGnBu_r"

    if "cmap" not in kwargs:
        kwargs["cmap"] = cmap

    plt.clf()

    lon1, lon2, lat1, lat2 = atlas.domain

    projection = ccrs.PlateCarree()

    ax = plt.axes(projection=projection)

    im = ax.pcolor(glon, glat, T[:, :, kz],  **kwargs)

    ax.set_title(f"depth = {pres:.0f} dbar")

    cb = plt.colorbar(im, location="bottom")
    cb.set_label(rf"{field} [{UNITS[field]}]", rotation="horizontal")

    land = TOPO.land(atlas.domain)
    ax.contourf(*land, [0.5, 1], colors="#CCCCCC")
    ax.contour(*land, [0.5], color="k")
    # ax.coastlines()

    if plottopo and hasattr(TOPO, "h"):
        ax.contour(TOPO.lon, TOPO.lat, TOPO.h,
                   [-4000, -2000, -100], colors="k")
    ax.axis(atlas.domain)

    gl = ax.gridlines(crs=projection, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_left = False

    plt.tight_layout()
