import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from argostats.toctools import load_argo_summary


def plot_first_profiles():
    plt.ion()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    infos = load_argo_summary()

    q = infos.groupby(by="WMO").first()
    ax.plot(q.LONGITUDE, q.LATITUDE, ".", markersize=1)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_left = False

    plt.tight_layout()
    plt.savefig("argo_firstprofiles.pdf")


def time(t): return f"iprof={t}"


class Animation:
    def __init__(self, df):
        self.t0 = 0.0
        self.nframes = 500
        self.plist = []
        self.maxpoints = 5
        self.df = df
        self.setup()

    def decrease_alpha(self, p):
        s = p.get_alpha()
        p.set_alpha(s-1/self.maxpoints)

    def get_lonlat(self, iprof):
        q = self.df[self.df.IPROF == iprof]
        return q.LONGITUDE, q.LATITUDE, iprof

    def add_points(self, lon, lat):
        self.plist += [self.ax.plot(lon, lat, "b.", markersize=1, alpha=1.0)]

    def setup(self):
        # self.fig, self.ax = plt.subplots()
        self.fig = plt.figure(figsize=(20, 12))
        self.ax = plt.axes(projection=ccrs.PlateCarree())
        self.ax.coastlines()
        gl = self.ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                               linewidth=2, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_left = False

        plt.tight_layout()

        lon, lat, t = self.get_lonlat(0)
        self.add_points(lon, lat)
        self.ti = self.ax.set_title(time(t))

    def update(self, frame):
        lon, lat, t = self.get_lonlat(frame)
        if len(self.plist) == self.maxpoints:
            p = self.plist.pop(0)
            p[0].remove()
        for p in self.plist:
            # decrease_markersize(p[0])
            self.decrease_alpha(p[0])
        self.add_points(lon, lat)

        self.ti.set_text(time(t))

    def create(self):
        self.ani = animation.FuncAnimation(
            fig=self.fig, func=self.update, frames=self.nframes, interval=30)


def domovie():
    df = load_argo_summary()
    print("setting up the plot")
    ani = Animation(df)
    ani.create()
    print("saving into movie.mp4")
    ani.ani.save("movie.mp4")


def fig_wmo_n_prof():
    df = load_argo_summary()

    df.groupby("WMO", group_keys=True).first(
    ).N_PROF.hist(bins=10*np.arange(71))

    plt.xlabel("#profiles")
    plt.ylabel("#wmos (binsize=10)")
    plt.savefig("argo_wmos_n_prof.pdf")
