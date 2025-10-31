import datetime
import numpy as np

# https://stackoverflow.com/questions/13648774/get-year-month-or-day-from-numpy-datetime64


def year(dates):
    """ Return the year of a datetime64 array
    """
    return dates.astype('datetime64[Y]').astype("i")+1970


def month(dates):
    """ Return the month of a datetime64 array (between 1 and 12)
    """
    return dates.astype('datetime64[M]').astype("i") % 12+1


def day(dates):
    """ Return the day of a datetime64 array (between 1 and 31)
    """
    return dates.astype('datetime64[D]') - dates.astype('datetime64[M]') + 1


def dayofyear(dates):
    """ Return the day of year of a datetime64 array (between 1 and 366)
    """
    return dates.astype('datetime64[D]') - dates.astype('datetime64[Y]') + 1


def hour(dates):
    """ Return the hour of a datetime64 array (between 0 and 23)
    """
    return dates.astype('datetime64[h]') - dates.astype('datetime64[D]')


def minute(dates):
    """ Return the minute of a datetime64 array (between 0 and 59)
    """
    return dates.astype('datetime64[m]') - dates.astype('datetime64[h]')


def second(dates):
    """ Return the second of a datetime64 array (between 0 and 59)
    """
    return dates.astype('datetime64[s]') - dates.astype('datetime64[m]')


def YMDhms(dates):
    """ Return (Y,M,D,h,m,s) of a datetime64 array
    """
    Y = dates.astype('datetime64[Y]')
    M = dates.astype('datetime64[M]')
    D = dates.astype('datetime64[D]')
    h = dates.astype('datetime64[h]')
    m = dates.astype('datetime64[m]')
    s = dates.astype('datetime64[s]')
    return toint(Y+1970, M-Y+1, D-M+1, h-D, m-h, s-m)


def toint(*args):
    return [a.astype("i") for a in args]


def test_datetools():
    d = np.arange(np.datetime64('2000-01-01'), np.datetime64('2010-01-01'))
    e = np.datetime64("2025-10-10T08:37:25")
    y = year(d)
    m = month(d)
    d0 = d[31+28]
    assert year(d0) == 2000
    assert month(d0) == 2
    assert day(d0) == 29
    assert dayoftheyear(d0) == 60
    assert len(d[y == 2000]) == 366
    assert len(d[y == 2001]) == 365
    assert len(d[(y == 2000) & (m == 2)]) == 29
    assert len(d[(y == 2010)]) == 0
    assert month(e) == 10
    assert day(e) == 10
