import gsw
import numpy as np


def check():
    """Check that `gammat_analytic` (rewritten by Roullet)
    matches `gsw_gammat_analytic_CT_exact` (written by Tailleux & Wolfe)
    """
    import time

    N = 10_000
    sr = np.linspace(34, 36, N, dtype="f4")
    ct = np.linspace(15, 20, N, dtype="f4")
    t0 = [time.time()]

    g0, z0, p0, s0 = gsw_gammat_analytic_CT_exact(sr, ct)

    t0 += [time.time()]

    g1, z1, p1, s1 = gammat_analytic(sr, ct)

    assert np.allclose(g0, g1)
    assert np.allclose(z0, z1)
    assert np.allclose(p0, p1)
    assert np.allclose(s0, s1)
    t0 += [time.time()]
    t0 = np.asarray(t0)
    print(np.diff(t0))
    print("functions in this module are ok")


# Set values of coefficients
# --------------------------
# parameter sets based on ID 8 and 9
a = 4.56016575
b = -1.24898501
c = 0.00439778209
d = 1030.99373
e = 8.32218903

# Set polynomial corrections
#     Linear model Poly8:
#      f(x) = p1*x^8 + p2*x^7 + p3*x^6 + p4*x^5 +
#                     p5*x^4 + p6*x^3 + p7*x^2 + p8*x + p9
#        where x is normalized by mean 1440 and std 1470
#      Coefficients (with 95% confidence bounds):
p1 = 0.0007824  # (0.0007792, 0.0007855)
p2 = -0.008056  # (-0.008082, -0.008031)
p3 = 0.03216  # (0.03209, 0.03223)
p4 = -0.06387  # (-0.06393, -0.06381)
p5 = 0.06807  # (0.06799, 0.06816)
p6 = -0.03696  # (-0.03706, -0.03687)
p7 = -0.08414  # (-0.08419, -0.0841)
p8 = 6.677  # (6.677, 6.677)
p9 = 6.431  # (6.431, 6.431)

# Set value of gravity
# --------------------
grav = 9.81
grav2 = grav/1e4


def Lorenz_reference_density(zref):
    """Compute Lorenz reference density

    based on an analytical expression fitted from the World Ocean Atlas

    Parameters
    ----------
    zref : ndarray
        depth (>=0), m

    Returns
    -------
    rho : ndarray
        Lorenz reference density, kg/m^3

    """
    return a/(b+1)*(zref+e)**(b+1)+c*zref+d


rhoref = Lorenz_reference_density
rho_eos10 = gsw.density.rho


A = a/((b+1)*(b+2))
D = a/((b+1)*(b+2))*e**(b+2)
C = c/2

# def pr(zref):
#     return (grav2) * (a/((b+1)*(b+2))*((zref+e)**(b+2)) + c/2.*zref**2 + d*zref - a/((b+1)*(b+2))*e**(b+2))


def pr(zref):
    return (grav2) * (A*((zref+e)**(b+2)) + C*zref**2 + d*zref - D)


# Polynomial correction
# --------------------
f = np.polynomial.Polynomial([p9, p8, p7, p6, p5, p4, p3, p2, p1])


def zref_new_ev(ss, ztop, zbot, zref):
    return 0.25*(1.+ss)*(ztop+zref) + 0.25*(1.-ss)*(zbot+zref)


def ztop_ev(ss, ztop, zref):
    return 0.5*(1+ss)*ztop + 0.5*(1-ss)*zref


def zbot_ev(ss, zbot, zref):
    return 0.5*(1-ss)*zbot + 0.5*(1+ss)*zref


def gammat_analytic(sr, ct):
    """Compute the thermodynamic neutral density (gammat) and the
    reference depth (zref) of a water parcel (sr, ct)


    based on an analytical expression of Lorenz reference density
    (Tailleux 2020)

    Parameters
    ----------
    sr : ndarray
        Reference composition Salinity, g/kg
    ct : ndarray
        Conservative Temperature, degrees C

    Returns
    -------
    gammat : ndarray
        Thermodynamic neutral density
    zref : ndarray
        Reference depth
    pref : ndarray
        Reference pressure
    sigref : ndarray
        Reference density

    """

    # AUTHOR OF ORIGINAL MATLAB CODE:
    #   Remi Tailleux, University of Reading, 8 July 2020

    # CHANGES TO ORIGINAL CODE:
    #   25.Apr.2022: Converted into Python (Gabriel Wolf)
    #   29.Oct.2025: Replace numexpr expressions with function (Guillaume Roullet)
    # ==========================================================================

    # Compute the reference positions
    # -------------------------------
    zmin = ztop = 0.
    zmax = zbot = 6000.

    zref = np.ones_like(sr) * 2000.

    # Compute zref as solution of
    #
    # rhoref(zref) = rho_eos10(sr, ct, pr(zref))
    #

    n_c = 30
    for i_c in range(0, n_c):
        # Compute buoyancy
        buoyancy = rhoref(zref) - rho_eos10(sr, ct, pr(zref))
        # Compute sign of buoyancy
        ss = np.sign(buoyancy)
        # Redefine zref depending on sign of buoyancy
        zref_new = 0.25*(1.+ss)*(ztop+zref) + 0.25*(1.-ss)*(zbot+zref)
        ztop = 0.5*(1+ss)*ztop + 0.5*(1-ss)*zref
        zbot = 0.5*(1-ss)*zbot + 0.5*(1+ss)*zref
        zref = zref_new

    # Compute analytic gammat
    # -----------------------
    pmean = 1440.
    pstd = 1470.

    pref = pr(zref)
    x = (pref-pmean)/pstd
    sigref = rho_eos10(sr, ct, pref) - 1000.
    gammat = sigref - f(x)

    return gammat, zref, pref, sigref


def gsw_gammat_analytic_CT_exact(sr, ct):
    import numexpr as ne
    # gsw_gammat_analytic_CT_exact: Compute thermodynamic neutral density based on an
    # analytical expression of Lorenz reference density
    #
    # INPUT:
    #
    #   sr          : reference composition salinity (g/kg)
    #   ct          : Conservative Temperature (deg C)
    #
    # OUTPUT:
    #   zref        : Reference position
    #   pref        : Reference pressure
    #   sigref      : Reference density
    #   gammat      : Thermodynamic neutral density
    #
    # DEPENDENCIES:
    #   - gsw_rho_CT_exact(sr,ct,p): equation of state for seawater as a function
    #     of reference composition salinity (sr), conservative temperature (ct)
    #     and pressure (p, in dbars)
    #   - gsw
    #     git repository cloned from https://github.com/TEOS-10/GSW-Python (29.Oct.2020)
    #     function rho_CT_exact not available in this repository, included here as function
    #     gsw_rho_CT_exact from original python-gsw (https://github.com/TEOS-10/python-gsw/) which will be
    #     replaced by GSW-Python after a brief overlap period (https://teos-10.github.io/GSW-Python/)
    #
    # AUTHOR OF ORIGINAL MATLAB CODE:
    #   Remi Tailleux, University of Reading, 8 July 2020
    #
    # CHANGES TO ORIGINAL CODE:
    #   25.Apr.2022: Converted into Python (Gabriel Wolf)
    # ==========================================================================

    # check input format
    # ------------------
    if np.isscalar(sr):
        output_c = 'scalar'
    elif isinstance(sr, list):
        output_c = 'list'
    else:
        output_c = 'array'

    # Set values of coefficients
    # --------------------------
    # parameter sets based on ID 8 and 9
    a = 4.56016575
    b = -1.24898501
    c = 0.00439778209
    d = 1030.99373
    e = 8.32218903

    # Set polynomial corrections
    #     Linear model Poly8:
    #      f(x) = p1*x^8 + p2*x^7 + p3*x^6 + p4*x^5 +
    #                     p5*x^4 + p6*x^3 + p7*x^2 + p8*x + p9
    #        where x is normalized by mean 1440 and std 1470
    #      Coefficients (with 95% confidence bounds):
    p1 = 0.0007824  # (0.0007792, 0.0007855)
    p2 = -0.008056  # (-0.008082, -0.008031)
    p3 = 0.03216  # (0.03209, 0.03223)
    p4 = -0.06387  # (-0.06393, -0.06381)
    p5 = 0.06807  # (0.06799, 0.06816)
    p6 = -0.03696  # (-0.03706, -0.03687)
    p7 = -0.08414  # (-0.08419, -0.0841)
    p8 = 6.677  # (6.677, 6.677)
    p9 = 6.431  # (6.431, 6.431)

    # Set value of gravity
    # --------------------
    grav = 9.81

    # Define the different analytical functions
    # -----------------------------------------
    # drhordz = @(z) a.*(z+e).^b + c;
    rhor = 'a/(b+1)*(zref+e)**(b+1) + c*zref + d'
    pr = 'grav * (a/((b+1)*(b+2))*((zref+e)**(b+2)) + c/2.*zref**2 + d*zref - a/((b+1)*(b+2))*e**(b+2))/1e4'

    # Polynomial correction
    # --------------------
    f = 'p9 + x*( p8 + x*( p7 + x*( p6 + x*( p5 + x*( p4 + x*( p3 + x*( p2 + x*p1)))))))'

    # Compute the reference positions
    # -------------------------------
    zmin = 0.
    zmax = 6000.
    zref = np.ones_like(sr) * 2000.
    ztop = np.ones_like(sr) * zmin
    zbot = np.ones_like(sr) * zmax

    # Valid points
    # -------------
    zref_new_ev = '0.25*(1.+ss)*(ztop+zref) + 0.25*(1.-ss)*(zbot+zref)'
    ztop_ev = '0.5*(1+ss)*ztop + 0.5*(1-ss)*zref'
    zbot_ev = '0.5*(1-ss)*zbot + 0.5*(1+ss)*zref'
    n_c = 30
    for i_c in range(0, n_c):
        # Compute buoyancy
        buoyancy = ne.evaluate(
            rhor) - rho_eos10(sr, ct, ne.evaluate(pr))
        # Compute sign of buoyancy
        ss = np.sign(buoyancy)
        # Redefine zref depending on sign of buoyancy
        zref_new = ne.evaluate(zref_new_ev)
        ztop = ne.evaluate(ztop_ev)
        zbot = ne.evaluate(zbot_ev)
        zref = zref_new

    # Compute analytic gammat
    # -----------------------
    pmean = 1440.
    pstd = 1470.

    pref = ne.evaluate(pr)
    x = (pref-pmean)/pstd
    sigref = rho_eos10(sr, ct, pref) - 1000.
    gammat = sigref - ne.evaluate(f)

    # Return values
    # -----------------------
    if output_c == 'scalar':
        return gammat, np.float(zref), np.float(pref), sigref
    elif output_c == 'list':
        return gammat.tolist(), zref.tolist(), pref.tolist(), sigref.tolist()
    else:
        return gammat, zref, pref, sigref


def gsw_rho_CT_exact(SA, CT, p):
    """
    Calculates in-situ density from Absolute Salinity and Conservative
    Temperature.
    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure [dbar]
    Returns
    -------
    rho_CT_exact : array_like
                   in-situ density [kg/m**3]
    Notes
    -----
    The potential density with respect to reference pressure, p_ref, is
    obtained by calling this function with the pressure argument being p_ref
    (i.e. "rho_CT_exact(SA, CT, p_ref)").  This function uses the full Gibbs
    function.  There is an alternative to calling this function, namely
    rho_CT(SA, CT, p), which uses the computationally efficient 48-term
    expression for density in terms of SA, CT and p (McDougall et al., 2011).
    Examples
    --------
    TODO
    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
       of seawater - 2010: Calculation and use of thermodynamic properties.
       Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
       UNESCO (English), 196 pp. See Eqn. (2.8.2).
    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
       computationally efficient 48-term expression for the density of
       seawater in terms of Conservative Temperature, and related properties
       of seawater.
    """

    Remi_method = True
    if Remi_method:
        t = gsw.t_from_CT(SA, CT, p)
        return gsw.rho_t_exact(SA, t, p)
    else:
        return gsw.density.rho(SA, CT, p)
