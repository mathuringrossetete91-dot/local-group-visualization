import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from astropy.coordinates import SkyCoord, Galactocentric, Galactic, Supergalactic, CartesianRepresentation
import astropy.units as u

# ── Données ───────────────────────────────────────────────────────────────────
_BASE   = r"C:\Users\Etudiant\OneDrive - Université Paris-Saclay\L3\STAGE"
df_pm   = pd.read_csv(os.path.join(_BASE, "avec_pm.csv"))
df_nopm = pd.read_csv(os.path.join(_BASE, "sans_pm.csv")).copy()

df_pm = df_pm.dropna(subset=[
    "sg_xx","sg_yy","sg_zz","ll","bb","ra","dec","distance","name","velocity_gsr","pmra","pmdec"
]).copy()

for df in (df_pm, df_nopm):
    df["host_clean"] = df["host"].fillna("Unknown").astype(str).str.replace("_", " ")
    df["r"] = np.sqrt(df["sg_xx"]**2 + df["sg_yy"]**2 + df["sg_zz"]**2)
rmax    = 1600
df_pm   = df_pm[df_pm["r"]     <= rmax].reset_index(drop=True).copy()
df_nopm = df_nopm[df_nopm["r"] <= rmax].reset_index(drop=True).copy()

# ── Coordonnées galactiques cartésiennes ──────────────────────────────────────
def _galactic_xyz(df):
    l, b = np.radians(df["ll"].values), np.radians(df["bb"].values)
    d, ok = df["distance"].values, df["distance"].notna().values
    df["gx"] = np.where(ok, d*np.cos(b)*np.cos(l), np.nan)
    df["gy"] = np.where(ok, d*np.cos(b)*np.sin(l), np.nan)
    df["gz"] = np.where(ok, d*np.sin(b),            np.nan)
    sg_ok = ~ok & df["sg_xx"].notna().values
    if sg_ok.any():
        gal = Supergalactic(CartesianRepresentation(
            x=df.loc[sg_ok,"sg_xx"].values*u.kpc,
            y=df.loc[sg_ok,"sg_yy"].values*u.kpc,
            z=df.loc[sg_ok,"sg_zz"].values*u.kpc,
        )).transform_to(Galactic())
        df.loc[sg_ok,"gx"] = gal.cartesian.x.to(u.kpc).value
        df.loc[sg_ok,"gy"] = gal.cartesian.y.to(u.kpc).value
        df.loc[sg_ok,"gz"] = gal.cartesian.z.to(u.kpc).value

_galactic_xyz(df_pm); _galactic_xyz(df_nopm)

# ── Taille (masse dynamique) ────────────────────────────────────────────
_log_dyn_pm   = np.log10(df_pm["mass_dynamical_wolf"].replace(0, np.nan))
_log_dyn_nopm = np.log10(df_nopm["mass_dynamical_wolf"].replace(0, np.nan))
_gmin = min(_log_dyn_pm.min(skipna=True), _log_dyn_nopm.min(skipna=True))
_gmax = max(_log_dyn_pm.max(skipna=True), _log_dyn_nopm.max(skipna=True))
_fb   = 2 + 7 * ((_log_dyn_pm.median(skipna=True) + _log_dyn_nopm.median(skipna=True)) / 2 - _gmin) / (_gmax - _gmin)

def _size(s): return (2 + 7 * (s - _gmin) / (_gmax - _gmin)).fillna(_fb)

df_pm["size_plot"]   = _size(_log_dyn_pm)
df_nopm["size_plot"] = _size(_log_dyn_nopm)

# ── Classification HI ─────────────────────────────────────────────────────
def _hi_type(df):
    m = df["mass_HI"]
    return pd.Series(np.where(m.isna() | (m <= 0.5), "HI-poor/No data", "HI-rich"), index=df.index)

df_pm["HI_type"]   = _hi_type(df_pm)
df_nopm["HI_type"] = _hi_type(df_nopm)
hi_colors = {"HI-rich": "cyan", "HI-poor/No data": "lightgray"}

# ── Vitesses tangentielles ────────────────────────────────────────────
coords_icrs = SkyCoord(
    ra=df_pm["ra"].values * u.deg,
    dec=df_pm["dec"].values * u.deg,
    distance=df_pm["distance"].values * u.kpc,
    pm_ra_cosdec=df_pm["pmra"].values * u.mas / u.yr,
    pm_dec=df_pm["pmdec"].values * u.mas / u.yr,
    radial_velocity=df_pm["velocity_gsr"].values * u.km / u.s,
    frame="icrs",
)
gal_obs    = coords_icrs.transform_to("galactic")
gc         = coords_icrs.transform_to(Galactocentric())
zeros      = np.zeros(gc.x.shape) * u.km / u.s
gal_reflex = Galactocentric(x=gc.x, y=gc.y, z=gc.z, v_x=zeros, v_y=zeros, v_z=zeros).transform_to(Galactic())
mu_l_cosb  = (gal_obs.pm_l_cosb - gal_reflex.pm_l_cosb).to(u.mas / u.yr).value
mu_b       = (gal_obs.pm_b      - gal_reflex.pm_b     ).to(u.mas / u.yr).value
d_kpc      = df_pm["distance"].values
V_l        = 4.74 * mu_l_cosb * d_kpc
V_b        = 4.74 * mu_b * d_kpc
V_tan      = np.sqrt(V_l**2 + V_b**2)
df_pm["V_tan"] = V_tan

# ── Vecteurs 3D ───────────────────────────────────────────────────────────────────
_gal_corrected = Galactic(
    l=df_pm["ll"].values * u.deg, b=df_pm["bb"].values * u.deg,
    distance=df_pm["distance"].values * u.kpc,
    pm_l_cosb=mu_l_cosb * u.mas / u.yr, pm_b=mu_b * u.mas / u.yr,
    radial_velocity=df_pm["velocity_gsr"].values * u.km / u.s,
)
_dv = _gal_corrected.cartesian.differentials["s"]
df_pm["vx_gal"] = _dv.d_x.to(u.km / u.s).value
df_pm["vy_gal"] = _dv.d_y.to(u.km / u.s).value
df_pm["vz_gal"] = _dv.d_z.to(u.km / u.s).value
df_pm["V3d"]    = np.sqrt(df_pm["vx_gal"]**2 + df_pm["vy_gal"]**2 + df_pm["vz_gal"]**2)

# ── Pôles orbitaux : direction du moment angulaire L = r × v (galactocentrique) ──
_r_gc = np.vstack([
    gc.x.to(u.kpc).value, gc.y.to(u.kpc).value, gc.z.to(u.kpc).value
]).T
_v_gc = np.vstack([
    gc.v_x.to(u.km / u.s).value, gc.v_y.to(u.km / u.s).value, gc.v_z.to(u.km / u.s).value
]).T
_Lvec  = np.cross(_r_gc, _v_gc)
_Lnorm = np.linalg.norm(_Lvec, axis=1)
_Lunit = np.divide(_Lvec, _Lnorm[:, None],
                   out=np.zeros_like(_Lvec), where=_Lnorm[:, None] > 0)
df_pm["pole_lon"] = np.degrees(np.arctan2(_Lunit[:, 1], _Lunit[:, 0]))
df_pm["pole_lat"] = np.degrees(np.arcsin(np.clip(_Lunit[:, 2], -1.0, 1.0)))
# Longitude tracée sur l'Aitoff (X = -l) avec décalage de 180°, repliée dans [-180, 180]
df_pm["pole_lon_plot"] = ((-df_pm["pole_lon"] - 180.0 + 180.0) % 360.0) - 180.0

fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scene"}, {"type": "geo"}]],
    column_widths=[0.72, 0.28],
    horizontal_spacing=0.04,
)

# ======================================================
# DISQUES GALACTIQUES : Voie Lactée, M31, M33
# ======================================================

def _disk_normal_from_pa(l_deg, b_deg, dist_kpc, ra_deg, dec_deg, inc_deg, pa_eq_deg):
    """Normal au disque et axe majeur dans les coords galactiques cartésiennes."""
    l, b = np.radians(l_deg), np.radians(b_deg)
    e_los   = np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])
    e_gal_n = np.array([-np.sin(b)*np.cos(l), -np.sin(b)*np.sin(l), np.cos(b)])
    e_gal_e = np.array([-np.sin(l), np.cos(l), 0.0])

    # Nord équatorial projeté dans le plan du ciel galactique
    sc0 = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg,
                   distance=dist_kpc*u.kpc, frame='icrs')
    scn = SkyCoord(ra=ra_deg*u.deg, dec=(dec_deg+0.01)*u.deg,
                   distance=dist_kpc*u.kpc, frame='icrs')
    g0, gn = sc0.galactic, scn.galactic
    dl = np.radians(gn.l.deg - g0.l.deg)
    db = np.radians(gn.b.deg - g0.b.deg)
    eq_n = db*e_gal_n + dl*np.cos(b)*e_gal_e
    nrm = np.linalg.norm(eq_n)
    eq_n = eq_n/nrm if nrm > 1e-10 else e_gal_n

    eq_e = np.cross(e_los, eq_n)
    nrm_e = np.linalg.norm(eq_e)
    eq_e = eq_e/nrm_e if nrm_e > 1e-10 else e_gal_e

    pa = np.radians(pa_eq_deg)
    e_maj = np.cos(pa)*eq_n + np.sin(pa)*eq_e
    e_maj /= np.linalg.norm(e_maj)

    e_perp = np.cross(e_maj, e_los)
    nrm_p = np.linalg.norm(e_perp)
    e_perp = e_perp/nrm_p if nrm_p > 1e-10 else eq_e

    inc = np.radians(inc_deg)
    normal = np.cos(inc)*e_los + np.sin(inc)*e_perp
    return normal/np.linalg.norm(normal), e_maj


def _disk_mesh(cx, cy, cz, R, normal, e_major, color, name, opacity=0.28, nt=80, thick=0.0):
    nv = np.array(normal, dtype=float); nv /= np.linalg.norm(nv)
    uv = np.array(e_major, dtype=float); uv /= np.linalg.norm(uv)
    vv = np.cross(nv, uv); vv /= np.linalg.norm(vv)
    th = np.linspace(0, 2*np.pi, nt, endpoint=False)
    rim = np.stack([np.cos(th), np.sin(th)], axis=1)  # (nt, 2)

    if thick <= 0:
        xs = [cx] + list(cx + R*(rim[:,0]*uv[0] + rim[:,1]*vv[0]))
        ys = [cy] + list(cy + R*(rim[:,0]*uv[1] + rim[:,1]*vv[1]))
        zs = [cz] + list(cz + R*(rim[:,0]*uv[2] + rim[:,1]*vv[2]))
        ii = [0]*nt
        jj = list(range(1, nt+1))
        kk = list(range(2, nt+1)) + [1]
    else:
        h = thick / 2.0
        def _face(offset):
            cx2 = cx + offset*nv[0]
            cy2 = cy + offset*nv[1]
            cz2 = cz + offset*nv[2]
            xs_ = [cx2] + list(cx2 + R*(rim[:,0]*uv[0] + rim[:,1]*vv[0]))
            ys_ = [cy2] + list(cy2 + R*(rim[:,0]*uv[1] + rim[:,1]*vv[1]))
            zs_ = [cz2] + list(cz2 + R*(rim[:,0]*uv[2] + rim[:,1]*vv[2]))
            return xs_, ys_, zs_

        xb, yb, zb = _face(-h)
        xt, yt, zt = _face(+h)
        xs = xb + xt; ys = yb + yt; zs = zb + zt
        n0 = nt + 1

        ii, jj, kk = [], [], []
        for k in range(nt):
            ii.append(0); jj.append(k+1); kk.append((k+1)%nt + 1)
        for k in range(nt):
            ii.append(n0); jj.append(n0+(k+1)%nt+1); kk.append(n0+k+1)
        for k in range(nt):
            b0, b1 = k+1, (k+1)%nt+1
            t0, t1 = n0+k+1, n0+(k+1)%nt+1
            ii += [b0, b0]; jj += [b1, t0]; kk += [t0, t1]

    return go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=ii, j=jj, k=kk,
        color=color, opacity=opacity,
        name=name, showlegend=True,
        hoverinfo="name", flatshading=True,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
    )


fig.add_trace(
    _disk_mesh(0, 0, 0, R=15,
               normal=[0, 0, 1], e_major=[1, 0, 0],
               color="gold", name="Milky Way disk", opacity=0.18, thick=2.0),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="text", text=["Milky Way"],
        textposition="top center",
        textfont=dict(size=11, color="goldenrod", family="Arial Black"),
        hoverinfo="skip", showlegend=False,
    ),
    row=1, col=1,
)

# M31
_m31_l, _m31_b, _m31_d = 121.17, -21.57, 785.0
_m31_gx = _m31_d*np.cos(np.radians(_m31_b))*np.cos(np.radians(_m31_l))
_m31_gy = _m31_d*np.cos(np.radians(_m31_b))*np.sin(np.radians(_m31_l))
_m31_gz = _m31_d*np.sin(np.radians(_m31_b))
# ── Repère M31 : base orthonormée (eZ = M31→MW, eY = pôle nord ⊥ eZ) ────────
_m31_vec  = np.array([_m31_gx, _m31_gy, _m31_gz])
_e_Z_m31  = -_m31_vec / np.linalg.norm(_m31_vec)
_e_Y_m31  = np.array([0., 0., 1.]) - np.dot(np.array([0., 0., 1.]), _e_Z_m31) * _e_Z_m31
_e_Y_m31 /= np.linalg.norm(_e_Y_m31)
_e_X_m31  = np.cross(_e_Y_m31, _e_Z_m31)
_e_X_m31 /= np.linalg.norm(_e_X_m31)
_R_m31    = np.array([_e_X_m31, _e_Y_m31, _e_Z_m31])  # lignes = vecteurs de la base
_n31, _e31 = _disk_normal_from_pa(
    _m31_l, _m31_b, _m31_d, 10.68, 41.27, inc_deg=77.5, pa_eq_deg=38.0)
fig.add_trace(
    _disk_mesh(_m31_gx, _m31_gy, _m31_gz, R=30,
               normal=_n31, e_major=_e31,
               color="lightsalmon", name="M31 Andromeda", opacity=0.30, thick=3.0),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter3d(
        x=[_m31_gx], y=[_m31_gy], z=[_m31_gz],
        mode="text", text=["M31 Andromeda"],
        textposition="top center",
        textfont=dict(size=11, color="lightsalmon", family="Arial Black"),
        hoverinfo="skip", showlegend=False,
    ),
    row=1, col=1,
)

# M33
_m33_l, _m33_b, _m33_d = 133.61, -31.33, 840.0
_m33_gx = _m33_d*np.cos(np.radians(_m33_b))*np.cos(np.radians(_m33_l))
_m33_gy = _m33_d*np.cos(np.radians(_m33_b))*np.sin(np.radians(_m33_l))
_m33_gz = _m33_d*np.sin(np.radians(_m33_b))
_n33, _e33 = _disk_normal_from_pa(
    _m33_l, _m33_b, _m33_d, 23.46, 30.66, inc_deg=56.0, pa_eq_deg=23.0)
fig.add_trace(
    _disk_mesh(_m33_gx, _m33_gy, _m33_gz, R=8,
               normal=_n33, e_major=_e33,
               color="lightsteelblue", name="M33 Triangulum", opacity=0.30, thick=1.5),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter3d(
        x=[_m33_gx], y=[_m33_gy], z=[_m33_gz],
        mode="text", text=["M33 Triangulum"],
        textposition="top center",
        textfont=dict(size=11, color="lightsteelblue", family="Arial Black"),
        hoverinfo="skip", showlegend=False,
    ),
    row=1, col=1,
)

# ── Labels Aitoff : MW, M31, M33 ──────────────────────────────────────────
fig.add_trace(
    go.Scattergeo(
        lon=[0], lat=[0],
        mode="markers+text",
        marker=dict(size=10, color="gold", symbol="star",
                    line=dict(color="goldenrod", width=1)),
        text=["Milky Way"],
        textposition="top center",
        textfont=dict(size=11, color="goldenrod", family="Arial Black"),
        hoverinfo="skip", showlegend=False,
    ),
    row=1, col=2,
)
fig.add_trace(
    go.Scattergeo(
        lon=[-_m31_l], lat=[_m31_b],
        mode="markers+text",
        marker=dict(size=10, color="lightsalmon", symbol="star",
                    line=dict(color="salmon", width=1)),
        text=["M31 Andromeda"],
        textposition="top center",
        textfont=dict(size=11, color="lightsalmon", family="Arial Black"),
        hoverinfo="skip", showlegend=False,
    ),
    row=1, col=2,
)
fig.add_trace(
    go.Scattergeo(
        lon=[-_m33_l], lat=[_m33_b],
        mode="markers+text",
        marker=dict(size=8, color="lightsteelblue", symbol="star",
                    line=dict(color="steelblue", width=1)),
        text=["M33 Triangulum"],
        textposition="top center",
        textfont=dict(size=11, color="lightsteelblue", family="Arial Black"),
        hoverinfo="skip", showlegend=False,
    ),
    row=1, col=2,
)

for label, color in hi_colors.items():
    fig.add_trace(
        go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            name=label,
            legendgroup=label,
            legend="legend",
            meta={"panel": "3d", "is_ghost": True, "hi_type": label},
            showlegend=True,
        ),
        row=1, col=1,
    )

fig.add_trace(
    go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(size=10, color="black", symbol="circle"),
        name="with proper motion",
        legend="legend",
        meta={"panel": "3d", "is_ghost": True},
        showlegend=True,
    ),
    row=1, col=1,
)

fig.add_trace(
    go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(size=10, color="black", symbol="circle-open", line=dict(width=1)),
        name="without proper motion",
        legend="legend",
        meta={"panel": "3d", "is_ghost": True},
        showlegend=True,
    ),
    row=1, col=1,
)

fig.add_trace(
    go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="lines",
        line=dict(color="#444444", width=3),
        name="Velocity arrows (3D)",
        legendgroup="velocity_3d",
        legend="legend",
        meta={"panel": "3d", "is_ghost": True},
        showlegend=True,
    ),
    row=1, col=1,
)

fig.add_trace(
    go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="text",
        text=["Aa"],
        textfont=dict(size=9, color="#333333"),
        name="Galaxy names",
        legendgroup="galaxy_names",
        legend="legend",
        meta={"panel": "3d", "is_ghost": True},
        showlegend=True,
    ),
    row=1, col=1,
)

# ======================================================
# LÉGENDE AITOFF  (traces fantômes — panneau droit)
# ======================================================

for label, color in hi_colors.items():
    fig.add_trace(
        go.Scattergeo(
            lat=[None], lon=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            name=label,
            legendgroup="aitoff_" + label,
            legend="legend2",
            meta={"panel": "aitoff", "is_ghost": True, "hi_type": label},
            showlegend=True,
        ),
        row=1, col=2,
    )

fig.add_trace(
    go.Scattergeo(
        lat=[None], lon=[None],
        mode="markers",
        marker=dict(size=10, color="black", symbol="circle"),
        name="with proper motion",
        legend="legend2",
        meta={"panel": "aitoff", "is_ghost": True},
        showlegend=True,
    ),
    row=1, col=2,
)

fig.add_trace(
    go.Scattergeo(
        lat=[None], lon=[None],
        mode="markers",
        marker=dict(size=10, color="rgba(0,0,0,0)", symbol="circle",
                    line=dict(width=1.5, color="black")),
        name="without proper motion",
        legend="legend2",
        meta={"panel": "aitoff", "is_ghost": True},
        showlegend=True,
    ),
    row=1, col=2,
)

fig.add_trace(
    go.Scattergeo(
        lat=[None], lon=[None],
        mode="lines",
        line=dict(color="rgba(180,40,40,0.85)", width=2),
        name="Velocity arrows (Aitoff)",
        legendgroup="velocity_aitoff",
        legend="legend2",
        meta={"panel": "aitoff", "is_ghost": True},
        showlegend=True,
    ),
    row=1, col=2,
)

fig.add_trace(
    go.Scattergeo(
        lat=[None], lon=[None],
        mode="text",
        text=["Aa"],
        textfont=dict(size=8, color="#333333"),
        name="Galaxy names",
        legendgroup="galaxy_names_aitoff",
        legend="legend2",
        meta={"panel": "aitoff", "is_ghost": True},
        showlegend=True,
    ),
    row=1, col=2,
)

# ======================================================
# DIMINUTIFS DE NOMS
# ======================================================

_WORD_ABBREV = {
    "andromeda": "And", "triangulum": "Tri", "fornax": "Fnx",
    "sculptor": "Scl", "carina": "Car", "sextans": "Sex",
    "sagittarius": "Sgr", "draco": "Dra", "ursa": "UMi",
    "minor": "", "major": "Maj", "large": "L", "small": "S",
    "magellanic": "MC", "cloud": "", "dwarf": "dw",
    "galaxy": "", "irregular": "irr", "spheroidal": "Sph",
    "elliptical": "Ell", "the": "", "of": "",
    "leo": "Leo", "phoenix": "Phx", "tucana": "Tuc",
    "antlia": "Ant", "cetus": "Cet", "pegasus": "Peg",
    "aquarius": "Aqr", "pisces": "Psc", "eridanus": "Eri",
    "hercules": "Her", "bootes": "Boo", "virgo": "Vir",
    "cassiopeia": "Cas", "andromede": "And",
}

def _abbrev(name):
    """Retourne un diminutif du nom de galaxie."""
    # Déjà court ou catalogue (NGC, IC, UGC, AGC, LMC, SMC, ESO…)
    if len(name) <= 7:
        return name
    parts = name.replace("_", " ").split()
    result = []
    for p in parts:
        low = p.lower().rstrip(".,;")
        if low in _WORD_ABBREV:
            v = _WORD_ABBREV[low]
            if v:
                result.append(v)
        else:
            result.append(p)
    abbr = " ".join(result).strip()
    return abbr if abbr else name

# Pré-calculer les colonnes de diminutifs
df_pm["name_short"]   = df_pm["name"].apply(_abbrev)
df_nopm["name_short"] = df_nopm["name"].apply(_abbrev)

# ======================================================
# HOVER
# ======================================================

def hover_pm(s):

    return (

        s["name"]

        + "<br>Host : "

        + s["host_clean"]

        + "<br>Type : "

        + s["HI_type"]

        + "<br>X : "

        + s["gx"].round(1).astype(str)

        + " kpc"

        + "<br>Y : "

        + s["gy"].round(1).astype(str)

        + " kpc"

        + "<br>Z : "

        + s["gz"].round(1).astype(str)

        + " kpc"

        + "<br>velocity_gsr : "

        + s["velocity_gsr"].round(1).astype(str)

        + " km/s"

        + "<br>V_tan : "

        + s["V_tan"].round(1).astype(str)

        + " km/s"

        + "<br>log(Mdyn) : "

        + _log_dyn_pm.loc[s.index].round(2).astype(str).where(
            _log_dyn_pm.loc[s.index].notna(), "N/A"
        )

        + "<br>log(M*) : "

        + s["mass_stellar"].replace(0, np.nan).round(2).astype(str).where(
            s["mass_stellar"].notna(), "N/A"
        )

        + "<br>pmRA : "

        + s["pmra"].round(3).astype(str)

        + "<br>pmDEC : "

        + s["pmdec"].round(3).astype(str)

        + "<br><br>\u25cf with proper motion"
    )

def hover_nopm(s):

    return (

        s["name"]

        + "<br>Host : "

        + s["host_clean"]

        + "<br>Type : "

        + s["HI_type"]

        + "<br>X : "

        + s["gx"].round(1).astype(str)

        + " kpc"

        + "<br>Y : "

        + s["gy"].round(1).astype(str)

        + " kpc"

        + "<br>Z : "

        + s["gz"].round(1).astype(str)

        + " kpc"

        + "<br>velocity_gsr : "

        + s["velocity_gsr"].fillna(float("nan")).round(1).astype(str).where(
            s["velocity_gsr"].notna(), "N/A"
        )

        + " km/s"

        + "<br>log(Mdyn) : "

        + _log_dyn_nopm.loc[s.index].round(2).astype(str).where(
            _log_dyn_nopm.loc[s.index].notna(), "N/A"
        )

        + "<br>log(M*) : "

        + s["mass_stellar"].replace(0, np.nan).round(2).astype(str).where(
            s["mass_stellar"].notna(), "N/A"
        )

        + "<br><br>\u25cb without proper motion"
    )

# ======================================================
# PM
# ======================================================

for hi_type in hi_colors.keys():

    subset_pm = df_pm[
        df_pm["HI_type"] == hi_type
    ]

    if len(subset_pm) == 0:
        continue

    text_pm = hover_pm(subset_pm)
    _nm_pm  = subset_pm["name"].tolist()

    # ==================================================
    # 3D
    # ==================================================

    fig.add_trace(

        go.Scatter3d(

            x=subset_pm["gx"],
            y=subset_pm["gy"],
            z=subset_pm["gz"],

            mode="markers",

            marker=dict(

                size=subset_pm["size_plot"],

                color=hi_colors[hi_type],

                symbol="circle",

                opacity=0.95,
            ),

            text=text_pm,

            hovertemplate="%{text}<extra></extra>",

            legendgroup=hi_type,

            meta={"group": "pm", "names": _nm_pm},

            showlegend=False,
        ),

        row=1,
        col=1,
    )

    # étiquettes noms 3D (PM)
    fig.add_trace(
        go.Scatter3d(
            x=subset_pm["gx"],
            y=subset_pm["gy"],
            z=subset_pm["gz"],
            mode="text",
            text=subset_pm["name_short"].tolist(),
            textposition="top center",
            textfont=dict(size=9, color="#333333"),
            hoverinfo="skip",
            legendgroup=hi_type,
            meta={"group": "pm", "is_label": True, "names": _nm_pm},
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ==================================================
    # AITOFF
    # ==================================================

    fig.add_trace(

        go.Scattergeo(

            lon=-subset_pm["ll"],
            lat=subset_pm["bb"],

            mode="markers",

            marker=dict(

                size=subset_pm["size_plot"],

                color=hi_colors[hi_type],

                symbol="circle",

                opacity=0.95,

                line=dict(
                    width=0.5,
                    color="black"
                ),
            ),

            text=text_pm,

            hovertemplate="%{text}<extra></extra>",

            customdata=list(zip(
                subset_pm["name"].tolist(),
                subset_pm["gx"].round(2).tolist(),
                subset_pm["gy"].round(2).tolist(),
                subset_pm["gz"].round(2).tolist(),
            )),

            legendgroup=hi_type,

            meta={"group": "pm", "names": _nm_pm},

            showlegend=False,
        ),

        row=1,
        col=2,
    )

    # étiquettes noms Aitoff (PM)
    fig.add_trace(
        go.Scattergeo(
            lon=-subset_pm["ll"],
            lat=subset_pm["bb"],
            mode="text",
            text=subset_pm["name_short"].tolist(),
            textposition="top center",
            textfont=dict(size=8, color="#333333"),
            hoverinfo="skip",
            legendgroup=hi_type,
            meta={"group": "pm", "is_label": True, "names": _nm_pm},
            showlegend=False,
        ),
        row=1, col=2,
    )

    # ==================================================
    # AITOFF PÔLES ORBITAUX  (geo2, carte du haut)
    # ==================================================

    text_pole = (
        subset_pm["name"]
        + "<br>Orbital pole (L = r×v)"
        + "<br>l = " + subset_pm["pole_lon"].round(1).astype(str) + "°"
        + "<br>b = " + subset_pm["pole_lat"].round(1).astype(str) + "°"
    )

    fig.add_trace(
        go.Scattergeo(
            geo="geo2",
            lon=subset_pm["pole_lon_plot"],
            lat=subset_pm["pole_lat"],
            mode="markers",
            marker=dict(
                size=subset_pm["size_plot"],
                color=hi_colors[hi_type],
                symbol="circle",
                opacity=0.95,
                line=dict(width=0.5, color="black"),
            ),
            text=text_pole,
            hovertemplate="%{text}<extra></extra>",
            customdata=list(zip(
                subset_pm["name"].tolist(),
                subset_pm["gx"].round(2).tolist(),
                subset_pm["gy"].round(2).tolist(),
                subset_pm["gz"].round(2).tolist(),
            )),
            legendgroup=hi_type,
            meta={"group": "pm", "is_pole": True, "names": _nm_pm},
            showlegend=False,
        ),
    )

    # étiquettes noms pôles orbitaux (geo2)
    fig.add_trace(
        go.Scattergeo(
            geo="geo2",
            lon=subset_pm["pole_lon_plot"],
            lat=subset_pm["pole_lat"],
            mode="text",
            text=subset_pm["name_short"].tolist(),
            textposition="top center",
            textfont=dict(size=8, color="#333333"),
            hoverinfo="skip",
            legendgroup=hi_type,
            meta={"group": "pm", "is_label": True, "is_pole": True, "names": _nm_pm},
            showlegend=False,
        ),
    )

# ======================================================
# noPM
# ======================================================

for hi_type in hi_colors.keys():

    subset_nopm = df_nopm[
        df_nopm["HI_type"] == hi_type
    ]

    if len(subset_nopm) == 0:
        continue

    text_nopm = hover_nopm(subset_nopm)
    _nm_nopm  = subset_nopm["name"].tolist()

    # ==================================================
    # 3D
    # ==================================================

    # Cercle extérieur coloré (anneau)
    fig.add_trace(
        go.Scatter3d(
            x=subset_nopm["gx"],
            y=subset_nopm["gy"],
            z=subset_nopm["gz"],
            mode="markers",
            marker=dict(
                size=subset_nopm["size_plot"],
                color=hi_colors[hi_type],
                symbol="circle",
                opacity=0.85,
            ),
            text=text_nopm,
            hovertemplate="%{text}<extra></extra>",
            legendgroup=hi_type,
            meta={"group": "nopm", "names": _nm_nopm},
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Cercle intérieur blanc (crée le trou — ratio 0.45 = anneau ~25% d'épaisseur)
    fig.add_trace(
        go.Scatter3d(
            x=subset_nopm["gx"],
            y=subset_nopm["gy"],
            z=subset_nopm["gz"],
            mode="markers",
            marker=dict(
                size=subset_nopm["size_plot"] * 0.45,
                color="white",
                symbol="circle",
                opacity=1.0,
            ),
            hoverinfo="skip",
            legendgroup=hi_type,
            meta={"group": "nopm", "names": _nm_nopm},
            showlegend=False,
        ),
        row=1, col=1,
    )

    # étiquettes noms 3D (noPM)
    fig.add_trace(
        go.Scatter3d(
            x=subset_nopm["gx"],
            y=subset_nopm["gy"],
            z=subset_nopm["gz"],
            mode="text",
            text=subset_nopm["name_short"].tolist(),
            textposition="top center",
            textfont=dict(size=9, color="#333333"),
            hoverinfo="skip",
            legendgroup=hi_type,
            meta={"group": "nopm", "is_label": True, "names": _nm_nopm},
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ==================================================
    # AITOFF
    # ==================================================

    fig.add_trace(

        go.Scattergeo(

            lon=-subset_nopm["ll"],
            lat=subset_nopm["bb"],

            mode="markers",

            marker=dict(

                size=subset_nopm["size_plot"],

                color="rgba(0,0,0,0)",

                symbol="circle",

                line=dict(
                    width=1.3,
                    color=hi_colors[hi_type]
                ),

                opacity=0.85,
            ),

            text=text_nopm,

            hovertemplate="%{text}<extra></extra>",

            customdata=list(zip(
                subset_nopm["name"].tolist(),
                subset_nopm["gx"].round(2).tolist(),
                subset_nopm["gy"].round(2).tolist(),
                subset_nopm["gz"].round(2).tolist(),
            )),

            legendgroup=hi_type,

            meta={"group": "nopm", "names": _nm_nopm},

            showlegend=False,
        ),

        row=1,
        col=2,
    )

    # étiquettes noms Aitoff (noPM)
    fig.add_trace(
        go.Scattergeo(
            lon=-subset_nopm["ll"],
            lat=subset_nopm["bb"],
            mode="text",
            text=subset_nopm["name_short"].tolist(),
            textposition="top center",
            textfont=dict(size=8, color="#333333"),
            hoverinfo="skip",
            legendgroup=hi_type,
            meta={"group": "nopm", "is_label": True, "names": _nm_nopm},
            showlegend=False,
        ),
        row=1, col=2,
    )

# ======================================================
# FLÈCHES PM
# ======================================================

seuil_vt = 0.0
arrow_scale = 7.5e-4
max_arc = 0.20

l_rad = np.radians(df_pm["ll"].values)
b_rad = np.radians(df_pm["bb"].values)

V_safe = np.where(V_tan > 0, V_tan, 1.0)

t_l = V_l / V_safe
t_b = V_b / V_safe

s_max = np.minimum(
    V_tan * arrow_scale,
    max_arc
)

def great_circle_arc(
    l,
    b,
    t_l,
    t_b,
    s_max,
    n=30
):

    p = np.array([
        np.cos(b) * np.cos(l),
        np.cos(b) * np.sin(l),
        np.sin(b)
    ])

    e_l = np.array([
        -np.sin(l),
        np.cos(l),
        0.0
    ])

    e_b = np.array([
        -np.sin(b) * np.cos(l),
        -np.sin(b) * np.sin(l),
        np.cos(b)
    ])

    t_vec = t_l * e_l + t_b * e_b

    s = np.linspace(
        0.0,
        s_max,
        n
    )

    q = (
        p[:, None] * np.cos(s)
        + t_vec[:, None] * np.sin(s)
    )

    b_arc = np.arcsin(
        np.clip(q[2], -1.0, 1.0)
    )

    l_arc = np.arctan2(
        q[1],
        q[0]
    )

    return (
        np.degrees(l_arc),
        np.degrees(b_arc)
    )

# ======================================================
# FLÈCHES PAR TYPE HI  (pour le toggle de légende)
# ======================================================
# On regroupe les arcs par type HI afin que le clic sur la
# légende masque simultanément points ET flèches.

for hi_type in hi_colors.keys():

    idx_hi = df_pm.index[
        (df_pm["HI_type"] == hi_type)
        & (V_tan >= seuil_vt)
        & (s_max > 0)
    ]

    lon_hi = []
    lat_hi = []
    names_hi = []

    for i in idx_hi:

        lon_a, lat_a = great_circle_arc(
            l_rad[i], b_rad[i],
            t_l[i], t_b[i],
            s_max[i], n=30,
        )

        if np.any(np.abs(np.diff(lon_a)) > 180):
            continue

        lon_hi.extend(list(-lon_a) + [None])
        lat_hi.extend(list(lat_a)  + [None])
        names_hi.extend([df_pm.loc[i, "name"]] * len(lon_a) + [None])

    if not lon_hi:
        continue

    fig.add_trace(
        go.Scattergeo(
            lon=lon_hi,
            lat=lat_hi,
            mode="lines",
            line=dict(color="rgba(180,40,40,0.85)", width=1.4),
            hoverinfo="skip",
            legendgroup=hi_type,
            meta={"group": "pm", "is_velocity": True, "hi_type": hi_type, "names": names_hi},
            showlegend=False,
        ),
        row=1, col=2,
    )

# ======================================================
# AXES 3D
# ======================================================

axis_length = 1100

axis_specs = [
    ("X", axis_length, "#c0392b"),   # rouge sobre
    ("Y", axis_length, "#27ae60"),   # vert sobre
    ("Z", axis_length, "#2980b9"),   # bleu sobre
]

for axis_name, L, color in axis_specs:

    x0 = [-L, L] if axis_name == "X" else [0, 0]
    y0 = [-L, L] if axis_name == "Y" else [0, 0]
    z0 = [-L, L] if axis_name == "Z" else [0, 0]

    # ligne d'axe
    fig.add_trace(
        go.Scatter3d(
            x=x0, y=y0, z=z0,
            mode="lines",
            line=dict(color=color, width=3),
            hoverinfo="skip",
            showlegend=False,
            meta={"is_axis": True, "axis": axis_name, "L": axis_length},
        ),
        row=1, col=1,
    )

    # étiquettes aux deux extrémités (légèrement décalées vers l'extérieur)
    tip = 1.06
    fig.add_trace(
        go.Scatter3d(
            x=[x0[0] * tip, x0[1] * tip],
            y=[y0[0] * tip, y0[1] * tip],
            z=[z0[0] * tip, z0[1] * tip],
            mode="text",
            text=[f"\u2212{axis_name}", f"+{axis_name}"],
            textfont=dict(size=14, color=color, family="Arial Black"),
            textposition="middle center",
            hoverinfo="skip",
            showlegend=False,
            meta={"is_axis_label": True, "axis": axis_name, "L": axis_length},
        ),
        row=1, col=1,
    )

# ======================================================
# TRACE DE SURBRILLANCE 3D (mise à jour par JS au clic Aitoff)
# ======================================================

fig.add_trace(
    go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers+text",
        marker=dict(
            size=12,
            color="yellow",
            symbol="diamond",
            line=dict(color="black", width=2),
            opacity=1.0,
        ),
        text=[""],
        textposition="top center",
        textfont=dict(size=13, color="black", family="Arial Black"),
        name="__highlight__",
        showlegend=False,
        hoverinfo="skip",
    ),
    row=1, col=1,
)

# ======================================================
# FLÈCHES DE VITESSE 3D (toggleable via légende)
# ======================================================
# Une trace par type HI (couleur cohérente), toutes dans
# le même legendgroup "velocity_3d".

_arrow_scale_3d = 0.15   # kpc / (km/s)
_max_arrow_3d   = 150.0  # cap en kpc
_seuil_v3d      = 0.0   # pas de seuil minimum

for hi_type in hi_colors.keys():

    _sub = df_pm[
        (df_pm["HI_type"] == hi_type)
        & (df_pm["V3d"] >= _seuil_v3d)
    ]

    if len(_sub) == 0:
        continue

    _xs, _ys, _zs = [], [], []
    _nms_v = []

    for _i in _sub.index:
        _v  = _sub.loc[_i, "V3d"]
        _sc = min(_arrow_scale_3d, _max_arrow_3d / _v)
        _x0 = _sub.loc[_i, "gx"]
        _y0 = _sub.loc[_i, "gy"]
        _z0 = _sub.loc[_i, "gz"]
        _xs.extend([_x0, _x0 + _sc * _sub.loc[_i, "vx_gal"], None])
        _ys.extend([_y0, _y0 + _sc * _sub.loc[_i, "vy_gal"], None])
        _zs.extend([_z0, _z0 + _sc * _sub.loc[_i, "vz_gal"], None])
        _nms_v.extend([_sub.loc[_i, "name"], _sub.loc[_i, "name"], None])

    fig.add_trace(
        go.Scatter3d(
            x=_xs, y=_ys, z=_zs,
            mode="lines",
            line=dict(color=hi_colors[hi_type], width=2),
            opacity=0.75,
            hoverinfo="skip",
            legendgroup="velocity_3d",
            meta={"is_velocity": True, "hi_type": hi_type, "names": _nms_v},
            showlegend=False,
        ),
        row=1, col=1,
)

# ======================================================
# CONSTANTES DE DOMAINE
# ======================================================
# On réserve une bande en haut de chaque panneau pour la barre de titre.
# Les "paper coordinates" vont de 0 à 1 (l→r, b→t).
#
#   Panneau 3D   : x [0.00, 0.63]  y [0.02, 0.97]
#   Panneau Geo  : x [0.67, 0.99]  y [0.02, 0.46]
#   Légende      : x [0.67, 0.99]  y [0.50, 0.97]

X0_3D,  X1_3D  = 0.00, 0.63
X0_GEO, X1_GEO = 0.67, 0.99

# espace pour la barre de titre (hauteur = 0.04 en coords paper)
_BAR = 0.04

Y0_3D,  Y1_3D  = 0.02, 0.97          # corps du panneau 3D (sous la barre)

# Deux cartes Aitoff empilées dans la colonne de droite :
#   geo  (bas)  = positions (l, b)
#   geo2 (haut) = pôles orbitaux (L = r × v)
Y0_GEO,  Y1_GEO  = 0.02, 0.43         # corps Aitoff bas  (positions)
Y0_GEO2, Y1_GEO2 = 0.52, 0.93         # corps Aitoff haut (pôles orbitaux)
Y0_PANELS = Y0_3D                    # alias conservé pour scene domain

# barres de titre (bande haute de chaque panneau)
_Y0_BAR_3D   = Y1_3D   + 0.001
_Y1_BAR_3D   = Y1_3D   + _BAR + 0.001
_Y0_BAR_GEO  = Y1_GEO  + 0.001
_Y1_BAR_GEO  = Y1_GEO  + _BAR + 0.001
_Y0_BAR_GEO2 = Y1_GEO2 + 0.001
_Y1_BAR_GEO2 = Y1_GEO2 + _BAR + 0.001

# zone légende
_Y0_LEG = Y1_GEO + _BAR + 0.025
_Y1_LEG = Y1_3D  + _BAR + 0.001

fig.update_layout(

    width=2000,
    height=950,

    clickmode="event",

    paper_bgcolor="#f0f2f5",   # fond gris très clair façon dashboard

    margin=dict(l=20, r=20, t=20, b=20),

    font=dict(color="#1a1a2e", family="Arial"),

    # --------------------------------------------------
    # Légende 3D — overlay dans le panneau 3D (gauche)
    # --------------------------------------------------
    legend=dict(
        title=dict(text="3D VIEW", font=dict(size=11, color="#1a1a2e", family="Arial Black")),
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#8899aa",
        borderwidth=1,
        font=dict(size=10, color="#1a1a2e"),
        x=0.01,
        y=0.96,
        xanchor="left",
        yanchor="top",
        groupclick="toggleitem",
        tracegroupgap=2,
    ),
    # --------------------------------------------------
    # Légende Aitoff — panneau dédié (droite, haut)
    # --------------------------------------------------
    legend2=dict(
        title=dict(text="AITOFF", font=dict(size=11, color="#1a1a2e", family="Arial Black")),
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#8899aa",
        borderwidth=1,
        font=dict(size=10, color="#1a1a2e"),
        x=X1_GEO - 0.005,
        y=Y1_GEO2 - 0.005,
        xanchor="right",
        yanchor="top",
        groupclick="toggleitem",
        tracegroupgap=2,
    ),

    # --------------------------------------------------
    # 3D  — panneau gauche
    # --------------------------------------------------
    scene=dict(
        bgcolor="white",
        domain=dict(x=[X0_3D, X1_3D], y=[Y0_3D, Y1_3D]),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
    ),

    # --------------------------------------------------
    # Aitoff  — panneau bas-droit (positions)
    # --------------------------------------------------
    geo=dict(
        projection_type="aitoff",
        domain=dict(x=[X0_GEO, X1_GEO], y=[Y0_GEO, Y1_GEO]),
        showland=False,
        showcoastlines=False,
        showcountries=False,
        showframe=True,
        framecolor="#8899aa",
        showocean=False,
        bgcolor="white",
        lonaxis=dict(
            showgrid=True,
            gridcolor="rgba(80,100,130,0.18)",
            gridwidth=0.6,
            dtick=30,
        ),
        lataxis=dict(
            showgrid=True,
            gridcolor="rgba(80,100,130,0.18)",
            gridwidth=0.6,
            dtick=15,
        ),
    ),

    # --------------------------------------------------
    # Aitoff pôles orbitaux  — panneau haut-droit
    # --------------------------------------------------
    geo2=dict(
        projection_type="aitoff",
        domain=dict(x=[X0_GEO, X1_GEO], y=[Y0_GEO2, Y1_GEO2]),
        showland=False,
        showcoastlines=False,
        showcountries=False,
        showframe=True,
        framecolor="#8899aa",
        showocean=False,
        bgcolor="white",
        lonaxis=dict(
            showgrid=True,
            gridcolor="rgba(80,100,130,0.18)",
            gridwidth=0.6,
            dtick=30,
        ),
        lataxis=dict(
            showgrid=True,
            gridcolor="rgba(80,100,130,0.18)",
            gridwidth=0.6,
            dtick=15,
        ),
    ),

    # --------------------------------------------------
    # Formes : cadres + barres de titre façon dashboard
    # --------------------------------------------------
    shapes=[

        # ── Cadre panneau 3D ──────────────────────────
        dict(
            type="rect",
            xref="paper", yref="paper",
            x0=X0_3D  - 0.005, x1=X1_3D  + 0.005,
            y0=Y0_3D  - 0.005, y1=_Y1_BAR_3D + 0.002,
            line=dict(color="#8899aa", width=1.5),
            fillcolor="white",
            layer="below",
        ),
        # barre de titre 3D
        dict(
            type="rect",
            xref="paper", yref="paper",
            x0=X0_3D  - 0.005, x1=X1_3D  + 0.005,
            y0=_Y0_BAR_3D - 0.001, y1=_Y1_BAR_3D + 0.002,
            line=dict(width=0),
            fillcolor="#1a3a5c",
            layer="above",
        ),

        # ── Cadre panneau Aitoff ──────────────────────
        dict(
            type="rect",
            xref="paper", yref="paper",
            x0=X0_GEO - 0.005, x1=X1_GEO + 0.005,
            y0=Y0_GEO - 0.005, y1=_Y1_BAR_GEO + 0.002,
            line=dict(color="#8899aa", width=1.5),
            fillcolor="white",
            layer="below",
        ),
        # barre de titre Aitoff
        dict(
            type="rect",
            xref="paper", yref="paper",
            x0=X0_GEO - 0.005, x1=X1_GEO + 0.005,
            y0=_Y0_BAR_GEO - 0.001, y1=_Y1_BAR_GEO + 0.002,
            line=dict(width=0),
            fillcolor="#1a3a5c",
            layer="above",
        ),

        # ── Cadre panneau Aitoff pôles orbitaux ───────
        dict(
            type="rect",
            xref="paper", yref="paper",
            x0=X0_GEO - 0.005, x1=X1_GEO + 0.005,
            y0=Y0_GEO2 - 0.005, y1=_Y1_BAR_GEO2 + 0.002,
            line=dict(color="#8899aa", width=1.5),
            fillcolor="white",
            layer="below",
        ),
        # barre de titre Aitoff pôles orbitaux
        dict(
            type="rect",
            xref="paper", yref="paper",
            x0=X0_GEO - 0.005, x1=X1_GEO + 0.005,
            y0=_Y0_BAR_GEO2 - 0.001, y1=_Y1_BAR_GEO2 + 0.002,
            line=dict(width=0),
            fillcolor="#1a3a5c",
            layer="above",
        ),
    ],

    # --------------------------------------------------
    # Annotations : titres dans les barres bleues
    # --------------------------------------------------
    annotations=[
        dict(
            text=(
                f"3D LOCAL GROUP"
            ),
            x=(X0_3D + X1_3D) / 2,
            y=(_Y0_BAR_3D + _Y1_BAR_3D) / 2 + 0.001,
            xref="paper", yref="paper",
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=13, color="white",
                      family="Arial Black"),
        ),
        dict(
            text="AITOFF — POSITIONS (l, b)",
            x=(X0_GEO + X1_GEO) / 2,
            y=(_Y0_BAR_GEO + _Y1_BAR_GEO) / 2 + 0.001,
            xref="paper", yref="paper",
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=13, color="white",
                      family="Arial Black"),
        ),
        dict(
            text="AITOFF — ORBITAL POLES (L = r×v)",
            x=(X0_GEO + X1_GEO) / 2,
            y=(_Y0_BAR_GEO2 + _Y1_BAR_GEO2) / 2 + 0.001,
            xref="paper", yref="paper",
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=13, color="white",
                      family="Arial Black"),
        ),
    ],
)

# ======================================================
# ÉCHELLE DE VITESSE V_tan (Aitoff)
# ======================================================
# Barre de référence placée dans la bande vide sous l'ovale Aitoff
# (coordonnées "paper", hors projection → ne perturbe pas la figure).
# Longueur de la barre = longueur angulaire d'une flèche de V_REF km/s,
# convertie à l'échelle de la projection Aitoff dans son domaine.
# La barre se redimensionne avec le zoom Aitoff (voir JS, via _geoScale).

_V_REF = 200.0                                   # vitesse de référence (km/s)
_s_ref = min(_V_REF * arrow_scale, max_arc)      # longueur angulaire (rad)

# Échelle de la projection Aitoff dans son domaine (px par unité-radian)
_geo_w_px = (X1_GEO - X0_GEO) * fig.layout.width
_geo_h_px = (Y1_GEO - Y0_GEO) * fig.layout.height
_aitoff_w_units, _aitoff_h_units = 2 * np.pi, np.pi   # ovale Aitoff : 2π × π
_px_per_unit = min(_geo_w_px / _aitoff_w_units, _geo_h_px / _aitoff_h_units)

_bar_paper = (_s_ref * _px_per_unit) / fig.layout.width   # longueur barre (paper-x)

# Position : bande vide entre le bas de l'ovale et le bas du domaine geo
_oval_h_paper = (_aitoff_h_units * _px_per_unit) / fig.layout.height
_oval_bottom  = (Y0_GEO + Y1_GEO) / 2 - _oval_h_paper / 2
_bar_y  = (Y0_GEO + _oval_bottom) / 2
_bar_xc = (X0_GEO + X1_GEO) / 2
_bar_half = _bar_paper / 2
_bar_x0 = _bar_xc - _bar_half
_bar_x1 = _bar_xc + _bar_half
_tick_h = 0.004
_arrow_col = "rgb(180,40,40)"   # même rouge que les flèches Aitoff

# Indices des shapes de l'échelle (après les 4 shapes cadres/barres déjà présentes)
_n_shapes_before = len(fig.layout.shapes)
_SCALE_MAIN = _n_shapes_before        # barre principale
_SCALE_T0   = _n_shapes_before + 1    # tick gauche
_SCALE_T1   = _n_shapes_before + 2    # tick droit

# Barre principale
fig.add_shape(
    type="line", xref="paper", yref="paper",
    x0=_bar_x0, x1=_bar_x1, y0=_bar_y, y1=_bar_y,
    line=dict(color=_arrow_col, width=2.2),
)
# Petits "ticks" aux extrémités
for _xt in (_bar_x0, _bar_x1):
    fig.add_shape(
        type="line", xref="paper", yref="paper",
        x0=_xt, x1=_xt, y0=_bar_y - _tick_h, y1=_bar_y + _tick_h,
        line=dict(color=_arrow_col, width=2.2),
    )
# Étiquette
fig.add_annotation(
    xref="paper", yref="paper",
    x=_bar_xc, y=_bar_y - _tick_h - 0.004,
    xanchor="center", yanchor="top",
    showarrow=False,
    text=f"V<sub>tan</sub> = {_V_REF:.0f} km s<sup>-1</sup>",
    font=dict(size=10, color=_arrow_col, family="Arial"),
)

# ======================================================
# AFFICHER
# ======================================================

# ======================================================
# EXPORT HTML + JAVASCRIPT POUR L'INTERACTION AITOFF → 3D
# ======================================================

import os

_JS = """
<script>
(function() {
    function setup() {
        var divs = document.querySelectorAll('.plotly-graph-div');
        if (!divs.length) { setTimeout(setup, 300); return; }
        var gd = divs[0];
        var _lastHighlight = null;  // nom du point surligné (clic Aitoff)
        var _geoScale = 1.0;        // niveau de zoom Aitoff courant
        var _geoCenterLon = 0, _geoCenterLat = 0; // centre courant de la projection Aitoff
        var _panStart = null;       // état drag-to-pan
        var _m31OrigCache = null;   // cache coordonnées originales (mode M31)

        // Rectangle réel (pixels client) du panneau Aitoff tel qu'affiché.
        // On le lit directement dans le DOM (fond blanc du sous-graphe geo) :
        // la zone de pan/zoom suit donc toujours exactement le panneau Aitoff,
        // quelle que soit la taille / le ratio de la page, et n'empiète jamais
        // sur la vue 3D.
        // Deux cartes Aitoff sont empilées (geo = positions en bas,
        // geo2 = pôles orbitaux en haut). Le pan/zoom ne pilote que la
        // carte du bas (geo) : on sélectionne donc le fond le plus bas à
        // l'écran (plus grand "top" en coords client).
        function _geoClientRect() {
            var bgs = gd.querySelectorAll('.geolayer .bg');
            if (bgs.length) {
                var best = null;
                for (var i = 0; i < bgs.length; i++) {
                    var r = bgs[i].getBoundingClientRect();
                    if (!best || r.top > best.top) best = r;
                }
                return best;
            }
            var g = gd.querySelector('.geo');
            return g ? g.getBoundingClientRect() : null;
        }

        // Met à jour la longueur de la barre d'échelle V_tan selon le zoom Aitoff.
        // La barre est en coords "paper" ; on borne pour ne pas déborder du panneau.
        function _updateScaleBar(scale) {
            var half = _BAR_HALF * scale;
            var maxHalf = 0.155;                 // largeur max ~ moitié du panneau geo
            if (half > maxHalf) half = maxHalf;
            var x0 = _BAR_XC - half, x1 = _BAR_XC + half;
            var upd = {};
            upd['shapes[' + _SCALE_MAIN + '].x0'] = x0;
            upd['shapes[' + _SCALE_MAIN + '].x1'] = x1;
            upd['shapes[' + _SCALE_T0   + '].x0'] = x0;
            upd['shapes[' + _SCALE_T0   + '].x1'] = x0;
            upd['shapes[' + _SCALE_T1   + '].x0'] = x1;
            upd['shapes[' + _SCALE_T1   + '].x1'] = x1;
            Plotly.relayout(gd, upd);
        }

        gd.on('plotly_click', function(ev) {
            if (!ev || !ev.points || !ev.points.length) return;
            var pt = ev.points[0];

            // ── Clic sur Aitoff → surbrillance 3D ──────────────────
            if (pt.data && pt.data.type === 'scattergeo') {
                var cd = pt.customdata;
                if (!cd || cd.length < 4) return;
                var name = cd[0], hx = cd[1], hy = cd[2], hz = cd[3];
                var hi = -1;
                for (var i = 0; i < gd.data.length; i++) {
                    if (gd.data[i].name === '__highlight__') { hi = i; break; }
                }
                if (hi < 0) return;
                if (_lastHighlight === name) {
                    _lastHighlight = null;
                    Plotly.restyle(gd, {x: [[null]], y: [[null]], z: [[null]], text: [['']]}, [hi]);
                } else {
                    _lastHighlight = name;
                    Plotly.restyle(gd, {x: [[hx]], y: [[hy]], z: [[hz]], text: [[name]]}, [hi]);
                }
                return;
            }
        });

        // Gestion complète des clics de légende (3D et Aitoff)
        gd.on('plotly_legendclick', function(ev) {
            var clicked = gd.data[ev.curveNumber];
            var name = clicked.name;
            var meta = clicked.meta || {};
            var panel = meta.panel;   // "3d" ou "aitoff"
            var isVisible = (clicked.visible === true || clicked.visible === undefined || clicked.visible === null);
            var newVis = isVisible ? 'legendonly' : true;
            var indices = [ev.curveNumber];

            // --- with / without proper motion ---
            if (name === 'with proper motion' || name === 'without proper motion') {
                var targetGroup = name === 'with proper motion' ? 'pm' : 'nopm';
                var targetType  = panel === '3d' ? 'scatter3d' : 'scattergeo';
                for (var i = 0; i < gd.data.length; i++) {
                    if (i === ev.curveNumber) continue;
                    var d = gd.data[i];
                    if (d.meta && d.meta.is_ghost) continue;
                    if (d.meta && d.meta.group === targetGroup && d.type === targetType) {
                        indices.push(i);
                    }
                }
                Plotly.restyle(gd, {visible: newVis}, indices);
                return false;
            }

            // --- Velocity arrows (3D) ---
            if (name === 'Velocity arrows (3D)') {
                for (var i = 0; i < gd.data.length; i++) {
                    if (i === ev.curveNumber) continue;
                    var d = gd.data[i];
                    if (d.meta && d.meta.is_velocity && d.type === 'scatter3d') {
                        indices.push(i);
                    }
                }
                Plotly.restyle(gd, {visible: newVis}, indices);
                return false;
            }

            // --- Velocity arrows (Aitoff) ---
            if (name === 'Velocity arrows (Aitoff)') {
                for (var i = 0; i < gd.data.length; i++) {
                    if (i === ev.curveNumber) continue;
                    var d = gd.data[i];
                    if (d.meta && d.meta.is_velocity && d.type === 'scattergeo') {
                        indices.push(i);
                    }
                }
                Plotly.restyle(gd, {visible: newVis}, indices);
                return false;
            }

            // --- Galaxy names ---
            if (name === 'Galaxy names') {
                var targetType = panel === '3d' ? 'scatter3d' : 'scattergeo';
                for (var i = 0; i < gd.data.length; i++) {
                    if (i === ev.curveNumber) continue;
                    var d = gd.data[i];
                    if (d.meta && d.meta.is_label && d.type === targetType) {
                        indices.push(i);
                    }
                }
                Plotly.restyle(gd, {visible: newVis}, indices);
                return false;
            }

            // --- Type HI (indépendant par panneau) ---
            var hiTypes = ['HI-rich', 'HI-poor/No data'];
            if (hiTypes.indexOf(name) >= 0) {
                var hiType     = meta.hi_type || name;
                var targetType = panel === '3d' ? 'scatter3d' : 'scattergeo';
                for (var i = 0; i < gd.data.length; i++) {
                    if (i === ev.curveNumber) continue;
                    var d = gd.data[i];
                    if (d.meta && d.meta.is_ghost) continue;
                    // points + labels + inner white circles du même type HI, même panneau
                    if (d.legendgroup === hiType && d.type === targetType) {
                        indices.push(i);
                    }
                    // flèches de vitesse du même type HI, même panneau
                    if (d.meta && d.meta.is_velocity && d.meta.hi_type === hiType && d.type === targetType) {
                        indices.push(i);
                    }
                }
                Plotly.restyle(gd, {visible: newVis}, indices);
                return false;
            }
        });

        // Double-clic → réinitialise surbrillance + zoom + pan Aitoff
        gd.on('plotly_doubleclick', function() {
            var hi = -1;
            for (var i = 0; i < gd.data.length; i++) {
                if (gd.data[i].name === '__highlight__') { hi = i; break; }
            }
            if (hi >= 0) Plotly.restyle(gd, {x: [[null]], y: [[null]], z: [[null]], text: [['']]}, [hi]);
            _lastHighlight = null;
            if (gd._fullLayout && gd._fullLayout.geo) {
                _geoScale = 1.0;
                _geoCenterLon = 0; _geoCenterLat = 0;
                Plotly.relayout(gd, {
                    'geo.projection.scale': 1.0,
                    'geo.center.lon': 0,
                    'geo.center.lat': 0,
                });
                _updateScaleBar(1.0);
            }
        });

        // ── Pan drag sur la projection Aitoff ───────────────────────────────
        // Phase de capture (3e arg = true) : s'exécute AVANT les handlers Plotly
        // sur les éléments enfants (markers), qui appellent stopPropagation().
        gd.addEventListener('mousedown', function(e) {
            if (e.button !== 0) return;
            if (!gd._fullLayout || !gd._fullLayout.geo) return;
            var gr = _geoClientRect();
            if (!gr) return;
            if (e.clientX < gr.left || e.clientX > gr.right ||
                e.clientY < gr.top  || e.clientY > gr.bottom) return;
            // Pas de preventDefault ici → les clics sur les points Aitoff restent actifs
            _panStart = {x: e.clientX, y: e.clientY, lon: _geoCenterLon, lat: _geoCenterLat, moved: false};
        }, true);  // capture phase

        window.addEventListener('mousemove', function(e) {
            if (!_panStart) return;
            var dx = e.clientX - _panStart.x;
            var dy = e.clientY - _panStart.y;
            if (Math.abs(dx) < 3 && Math.abs(dy) < 3) return;  // seuil : ne pas déclencher sur simple clic
            _panStart.moved = true;
            var gr = _geoClientRect();
            if (!gr) return;
            var domW = gr.width, domH = gr.height;
            // pixels → degrés (ajusté par le facteur de zoom)
            _geoCenterLon = _panStart.lon - dx * (360 / (domW * _geoScale));
            _geoCenterLat = Math.max(-85, Math.min(85,
                _panStart.lat + dy * (180 / (domH * _geoScale))));
            Plotly.relayout(gd, {
                'geo.center.lon': _geoCenterLon,
                'geo.center.lat': _geoCenterLat,
            });
        });

        window.addEventListener('mouseup', function() { _panStart = null; });

        // ── Zoom molette sur la projection Aitoff ──────────────────────────
        gd.addEventListener('wheel', function(e) {
            if (!gd._fullLayout || !gd._fullLayout.geo) return;
            var gr = _geoClientRect();
            if (!gr) return;
            if (e.clientX < gr.left || e.clientX > gr.right ||
                e.clientY < gr.top  || e.clientY > gr.bottom) return;
            e.preventDefault();
            e.stopPropagation();
            var factor = e.deltaY > 0 ? 0.85 : 1.15;
            _geoScale  = Math.max(1.0, Math.min(20, _geoScale * factor));
            Plotly.relayout(gd, {'geo.projection.scale': _geoScale});
            _updateScaleBar(_geoScale);
        }, {passive: false});

        // ── Boutons de navigation 3D ───────────────────────────────────────────
        var _btnSt = 'padding:5px 12px;font-size:11px;font-family:Arial;cursor:pointer;' +
            'border:1px solid #8899aa;border-radius:4px;';

        // Applique l'apparence active (bleu) ou inactive (blanc) aux 2 boutons.
        function _setActiveNav(active) {
            if (active === 'm31') {
                _btnM31.style.background = '#1a3a5c'; _btnM31.style.color = 'white';
                _btnReset.style.background = 'white'; _btnReset.style.color = '#1a3a5c';
            } else {
                _btnReset.style.background = '#1a3a5c'; _btnReset.style.color = 'white';
                _btnM31.style.background = 'white'; _btnM31.style.color = '#1a3a5c';
            }
        }

        var _btnM31 = document.createElement('button');
        _btnM31.textContent = 'M31 center';
        _btnM31.title = 'Centrer la vue 3D sur M31 Andromeda';
        _btnM31.style.cssText = _btnSt + 'background:white;color:#1a3a5c;';
        _btnM31.onclick = function() {
            _setActiveNav('m31');
            var L = 1100, tip = 1.06;
            var r = _R_M31, ox = _M31_GX, oy = _M31_GY, oz = _M31_GZ;
            // Construction du cache des coordonnées originales (une seule fois)
            if (!_m31OrigCache) {
                _m31OrigCache = [];
                for (var i = 0; i < gd.data.length; i++) {
                    var d = gd.data[i];
                    if (d.type === 'scatter3d' || d.type === 'mesh3d') {
                        _m31OrigCache[i] = {
                            x: _arr(d.x).slice(),
                            y: _arr(d.y).slice(),
                            z: _arr(d.z).slice(),
                            text: Array.isArray(d.text) ? d.text.slice() : d.text
                        };
                    }
                }
            }
            // Applique p' = R * (p - M31) à chaque trace 3D
            for (var i = 0; i < gd.data.length; i++) {
                var d = gd.data[i];
                if (d.type !== 'scatter3d' && d.type !== 'mesh3d') continue;
                var orig = _m31OrigCache[i];
                if (!orig) continue;
                var meta = d.meta || {};
                if (meta.is_axis) {
                    var ax = meta.axis;
                    Plotly.restyle(gd, {
                        x: [ax==='X' ? [-L, L] : [0, 0]],
                        y: [ax==='Y' ? [-L, L] : [0, 0]],
                        z: [ax==='Z' ? [-L, L] : [0, 0]],
                    }, [i]);
                } else if (meta.is_axis_label) {
                    var ax2 = meta.axis;
                    Plotly.restyle(gd, {
                        x: [ax2==='X' ? [-L*tip, L*tip] : [0, 0]],
                        y: [ax2==='Y' ? [-L*tip, L*tip] : [0, 0]],
                        z: [ax2==='Z' ? [-L*tip, L*tip] : [0, 0]],
                    }, [i]);
                } else {
                    var xs = orig.x, ys = orig.y, zs = orig.z;
                    var nx = [], ny = [], nz = [];
                    for (var k = 0; k < xs.length; k++) {
                        if (xs[k] == null || isNaN(+xs[k])) {
                            nx.push(null); ny.push(null); nz.push(null); continue;
                        }
                        var dx = xs[k]-ox, dy = ys[k]-oy, dz = zs[k]-oz;
                        nx.push(r[0]*dx + r[1]*dy + r[2]*dz);
                        ny.push(r[3]*dx + r[4]*dy + r[5]*dz);
                        nz.push(r[6]*dx + r[7]*dy + r[8]*dz);
                    }
                    var _restyle = {x: [nx], y: [ny], z: [nz]};
                    // Hover : afficher les coordonnées dans le repère M31.
                    // On réécrit le bloc « X/Y/Z » du texte d'origine avec les
                    // valeurs M31 (nx, ny, nz) déjà calculées ci-dessus.
                    if (Array.isArray(orig.text)) {
                        var nt = orig.text.slice();
                        for (var t = 0; t < nt.length; t++) {
                            if (typeof nt[t] !== 'string' || nx[t] == null) continue;
                            var bloc = '<br>X : ' + nx[t].toFixed(1) + ' kpc (M31)' +
                                       '<br>Y : ' + ny[t].toFixed(1) + ' kpc' +
                                       '<br>Z : ' + nz[t].toFixed(1) + ' kpc';
                            nt[t] = nt[t].replace(
                                /<br>X : [^<]*<br>Y : [^<]*<br>Z : [^<]*kpc/, bloc);
                        }
                        _restyle.text = [nt];
                    }
                    Plotly.restyle(gd, _restyle, [i]);
                }
            }
            Plotly.relayout(gd, {
                'scene.xaxis.autorange': true,
                'scene.yaxis.autorange': true,
                'scene.zaxis.autorange': true,
                'scene.aspectmode': 'data',
            });
        };

        var _btnReset = document.createElement('button');
        _btnReset.textContent = 'MW center';
        _btnReset.title = 'Revenir à la vue complète centrée sur la Voie Lactée';
        _btnReset.style.cssText = _btnSt + 'background:#1a3a5c;color:white;';
        _btnReset.onclick = function() {
            _setActiveNav('mw');
            var L = 1100, tip = 1.06;
            if (_m31OrigCache) {
                // Restaurer toutes les traces depuis le cache
                for (var i = 0; i < gd.data.length; i++) {
                    var orig = _m31OrigCache[i];
                    if (!orig) continue;
                    var d = gd.data[i];
                    var meta = d.meta || {};
                    if (meta.is_axis) {
                        var ax = meta.axis;
                        Plotly.restyle(gd, {
                            x: [ax==='X' ? [-L, L] : [0, 0]],
                            y: [ax==='Y' ? [-L, L] : [0, 0]],
                            z: [ax==='Z' ? [-L, L] : [0, 0]],
                        }, [i]);
                    } else if (meta.is_axis_label) {
                        var ax2 = meta.axis;
                        Plotly.restyle(gd, {
                            x: [ax2==='X' ? [-L*tip, L*tip] : [0, 0]],
                            y: [ax2==='Y' ? [-L*tip, L*tip] : [0, 0]],
                            z: [ax2==='Z' ? [-L*tip, L*tip] : [0, 0]],
                        }, [i]);
                    } else {
                        var _rs = {
                            x: [orig.x.slice()], y: [orig.y.slice()], z: [orig.z.slice()]
                        };
                        // Restaurer aussi le hover d'origine (coords Voie Lactée)
                        if (Array.isArray(orig.text)) _rs.text = [orig.text.slice()];
                        Plotly.restyle(gd, _rs, [i]);
                    }
                }
            } else {
                // Pas encore de cache : juste réinitialiser les axes
                for (var i = 0; i < gd.data.length; i++) {
                    var d = gd.data[i];
                    if (!d.meta) continue;
                    if (d.meta.is_axis) {
                        var ax = d.meta.axis;
                        Plotly.restyle(gd, {
                            x: [ax==='X' ? [-L, L] : [0, 0]],
                            y: [ax==='Y' ? [-L, L] : [0, 0]],
                            z: [ax==='Z' ? [-L, L] : [0, 0]],
                        }, [i]);
                    } else if (d.meta.is_axis_label) {
                        var ax2 = d.meta.axis;
                        Plotly.restyle(gd, {
                            x: [ax2==='X' ? [-L*tip, L*tip] : [0, 0]],
                            y: [ax2==='Y' ? [-L*tip, L*tip] : [0, 0]],
                            z: [ax2==='Z' ? [-L*tip, L*tip] : [0, 0]],
                        }, [i]);
                    }
                }
            }
            Plotly.relayout(gd, {
                'scene.xaxis.autorange': true,
                'scene.yaxis.autorange': true,
                'scene.zaxis.autorange': true,
                'scene.aspectmode': 'data',
            });
        };

        var _nav = document.createElement('div');
        _nav.style.cssText = 'display:flex;flex-direction:row;gap:5px;';
        _nav.appendChild(_btnM31);
        _nav.appendChild(_btnReset);

        // ── Barre de recherche de galaxie ──────────────────────────────────
        // Indexe tous les noms depuis les customdata des traces Aitoff
        // (chaque entrée = [name, gx, gy, gz]). Réutilise la trace
        // __highlight__ pour mettre en évidence la galaxie trouvée en 3D.
        var _nameIndex = {};   // nom (minuscule) → {name, x, y, z}
        var _allNames = [];
        for (var i = 0; i < gd.data.length; i++) {
            var d = gd.data[i];
            if (d.type === 'scattergeo' && d.customdata) {
                for (var j = 0; j < d.customdata.length; j++) {
                    var cd = d.customdata[j];
                    if (!cd || cd.length < 4 || !cd[0]) continue;
                    var key = String(cd[0]).toLowerCase();
                    if (!(key in _nameIndex)) {
                        _nameIndex[key] = {name: cd[0], x: cd[1], y: cd[2], z: cd[3]};
                        _allNames.push(cd[0]);
                    }
                }
            }
        }
        _allNames.sort(function(a, b) { return a.toLowerCase() < b.toLowerCase() ? -1 : 1; });

        function _highlightIndex() {
            for (var i = 0; i < gd.data.length; i++) {
                if (gd.data[i].name === '__highlight__') return i;
            }
            return -1;
        }

        // Surligne une galaxie par son nom (exact, sinon 1ère correspondance partielle).
        function _searchGalaxy(query) {
            var key = (query || '').toLowerCase().trim();
            if (!key) return false;
            var rec = _nameIndex[key];
            if (!rec) {
                for (var k = 0; k < _allNames.length; k++) {
                    if (_allNames[k].toLowerCase().indexOf(key) >= 0) {
                        rec = _nameIndex[_allNames[k].toLowerCase()];
                        break;
                    }
                }
            }
            var hi = _highlightIndex();
            if (hi < 0 || !rec) return false;
            _lastHighlight = rec.name;
            Plotly.restyle(gd, {x: [[rec.x]], y: [[rec.y]], z: [[rec.z]], text: [[rec.name]]}, [hi]);
            return true;
        }

        // Liste d'autocomplétion (noms de galaxies)
        var _dl = document.createElement('datalist');
        _dl.id = '_galaxyList';
        for (var n = 0; n < _allNames.length; n++) {
            var _opt = document.createElement('option');
            _opt.value = _allNames[n];
            _dl.appendChild(_opt);
        }
        document.body.appendChild(_dl);

        var _searchInput = document.createElement('input');
        _searchInput.type = 'text';
        _searchInput.placeholder = 'Search galaxy\u2026';
        _searchInput.setAttribute('list', '_galaxyList');
        _searchInput.style.cssText = 'font-size:11px;font-family:Arial;padding:4px 6px;' +
            'border:1px solid #8899aa;border-radius:3px;width:150px;outline:none;';

        var _searchBtn = document.createElement('button');
        _searchBtn.textContent = 'Find';
        _searchBtn.title = 'Surligner la galaxie en 3D';
        _searchBtn.style.cssText = _btnSt + 'background:#1a3a5c;color:white;';

        var _searchMsg = document.createElement('span');
        _searchMsg.style.cssText = 'font-size:10px;font-family:Arial;color:#c0392b;min-width:55px;';

        function _doSearch() {
            var ok = _searchGalaxy(_searchInput.value);
            _searchMsg.textContent = ok ? '' : 'not found';
        }
        _searchBtn.onclick = _doSearch;
        _searchInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') { e.preventDefault(); _doSearch(); }
        });
        _searchInput.addEventListener('change', _doSearch);

        var _searchWrap = document.createElement('div');
        _searchWrap.style.cssText = 'display:flex;gap:5px;align-items:center;';
        _searchWrap.appendChild(_searchInput);
        _searchWrap.appendChild(_searchBtn);
        _searchWrap.appendChild(_searchMsg);

        // ── Filtre par rayon autour d'un objet de référence ──────────────
        // Cache des coordonnées d'origine de chaque trace filtrable
        // (toute trace portant meta.names : marqueurs, étiquettes, flèches,
        //  pôles, en 3D comme en Aitoff). Le filtre masque, point par point,
        //  les entrées dont la galaxie est au-delà du rayon ; les repères
        //  MW / M31 / M33 (sans meta.names) restent toujours visibles.
        var _origCache = null;
        // Plotly encode souvent les coordonnées numériques en tableaux typés
        // base64 ({dtype, bdata}). _arr() renvoie toujours un tableau JS simple,
        // quel que soit le format de stockage (array, typed-array, base64).
        function _arr(v) {
            if (v == null) return [];
            if (Array.isArray(v)) return v.slice();
            if (ArrayBuffer.isView(v)) return Array.from(v);
            if (v._inputArray) return Array.from(v._inputArray);
            if (v.bdata !== undefined && v.dtype) {
                var bin = atob(v.bdata), n = bin.length;
                var bytes = new Uint8Array(n);
                for (var i = 0; i < n; i++) bytes[i] = bin.charCodeAt(i);
                var TA = {f8: Float64Array, f4: Float32Array, i4: Int32Array,
                          i2: Int16Array, i1: Int8Array, u1: Uint8Array,
                          u2: Uint16Array, u4: Uint32Array}[v.dtype] || Float64Array;
                return Array.from(new TA(bytes.buffer));
            }
            return [];
        }
        function _buildOrigCache() {
            _origCache = [];
            for (var i = 0; i < gd.data.length; i++) {
                var d = gd.data[i];
                if (!d.meta || !d.meta.names) continue;
                if (d.type === 'scattergeo') {
                    _origCache.push({i: i, geo: true,
                        lon: _arr(d.lon), lat: _arr(d.lat),
                        names: d.meta.names});
                } else {
                    _origCache.push({i: i, geo: false,
                        x: _arr(d.x), y: _arr(d.y), z: _arr(d.z),
                        names: d.meta.names});
                }
            }
        }

        function _applyRadiusFilter(refName, radius) {
            if (!_origCache) _buildOrigCache();
            var ref = _POS[refName];
            if (!ref) return;
            for (var c = 0; c < _origCache.length; c++) {
                var e = _origCache[c], nm = e.names;
                if (e.geo) {
                    var lon = e.lon.slice(), lat = e.lat.slice();
                    for (var k = 0; k < nm.length; k++) {
                        var p = nm[k] ? _POS[nm[k]] : null;
                        if (!p) continue;
                        var dx = p[0]-ref[0], dy = p[1]-ref[1], dz = p[2]-ref[2];
                        if (Math.sqrt(dx*dx+dy*dy+dz*dz) > radius) { lon[k] = null; lat[k] = null; }
                    }
                    Plotly.restyle(gd, {lon: [lon], lat: [lat]}, [e.i]);
                } else {
                    var X = e.x.slice(), Y = e.y.slice(), Z = e.z.slice();
                    for (var k2 = 0; k2 < nm.length; k2++) {
                        var p2 = nm[k2] ? _POS[nm[k2]] : null;
                        if (!p2) continue;
                        var ax = p2[0]-ref[0], ay = p2[1]-ref[1], az = p2[2]-ref[2];
                        if (Math.sqrt(ax*ax+ay*ay+az*az) > radius) { X[k2] = null; Y[k2] = null; Z[k2] = null; }
                    }
                    Plotly.restyle(gd, {x: [X], y: [Y], z: [Z]}, [e.i]);
                }
            }
        }

        function _resetRadiusFilter() {
            if (!_origCache) return;
            for (var c = 0; c < _origCache.length; c++) {
                var e = _origCache[c];
                if (e.geo) Plotly.restyle(gd, {lon: [e.lon.slice()], lat: [e.lat.slice()]}, [e.i]);
                else       Plotly.restyle(gd, {x: [e.x.slice()], y: [e.y.slice()], z: [e.z.slice()]}, [e.i]);
            }
        }

        // Contrôles du filtre par rayon (objet de référence + distance kpc).
        var _filterLabel = document.createElement('span');
        _filterLabel.textContent = 'Within';
        _filterLabel.style.cssText = 'font-size:11px;font-family:Arial;color:#1a3a5c;font-weight:bold;';

        var _radInput = document.createElement('input');
        _radInput.type = 'number'; _radInput.min = '0'; _radInput.step = '10';
        _radInput.value = '500';
        _radInput.title = 'Rayon en kpc';
        _radInput.style.cssText = 'font-size:11px;font-family:Arial;padding:4px 6px;' +
            'border:1px solid #8899aa;border-radius:3px;width:70px;outline:none;';

        var _radUnit = document.createElement('span');
        _radUnit.textContent = 'kpc of';
        _radUnit.style.cssText = 'font-size:11px;font-family:Arial;color:#1a3a5c;';

        var _refSelect = document.createElement('select');
        _refSelect.title = 'Objet de référence du filtre';
        _refSelect.style.cssText = 'font-size:11px;font-family:Arial;padding:4px 6px;' +
            'border:1px solid #8899aa;border-radius:3px;max-width:150px;outline:none;';
        for (var r = 0; r < _REF_NAMES.length; r++) {
            var _o = document.createElement('option');
            _o.value = _REF_NAMES[r]; _o.textContent = _REF_NAMES[r];
            _refSelect.appendChild(_o);
        }

        var _filterBtn = document.createElement('button');
        _filterBtn.textContent = 'Filter';
        _filterBtn.title = 'Masquer les galaxies au-delà du rayon';
        _filterBtn.style.cssText = _btnSt + 'background:#1a3a5c;color:white;';
        _filterBtn.onclick = function() {
            var rad = parseFloat(_radInput.value);
            if (isNaN(rad) || rad <= 0) return;
            _applyRadiusFilter(_refSelect.value, rad);
        };

        var _filterClear = document.createElement('button');
        _filterClear.textContent = 'All';
        _filterClear.title = 'Afficher toutes les galaxies';
        _filterClear.style.cssText = _btnSt + 'background:white;color:#1a3a5c;';
        _filterClear.onclick = function() { _resetRadiusFilter(); };

        _radInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') { e.preventDefault(); _filterBtn.onclick(); }
        });

        var _filterWrap = document.createElement('div');
        _filterWrap.style.cssText = 'display:flex;gap:5px;align-items:center;flex-wrap:wrap;';
        _filterWrap.appendChild(_filterLabel);
        _filterWrap.appendChild(_radInput);
        _filterWrap.appendChild(_radUnit);
        _filterWrap.appendChild(_refSelect);
        _filterWrap.appendChild(_filterBtn);
        _filterWrap.appendChild(_filterClear);

        // ── Upload de fichier (CSV) : afficher des objets externes ──────────
        // Lit un fichier CSV/TXT côté navigateur, demande les noms des colonnes
        // X / Y / Z (positions galactiques cartésiennes, en kpc) et ajoute les
        // objets dans la vue 3D et sur la carte Aitoff.
        var _uploadInput = document.createElement('input');
        _uploadInput.type = 'file';
        _uploadInput.accept = '.csv,.txt,.tsv';
        _uploadInput.style.display = 'none';

        var _uploadBtn = document.createElement('button');
        _uploadBtn.textContent = 'Upload file';
        _uploadBtn.title = 'Charger un fichier (CSV/TXT) et afficher ses objets';
        _uploadBtn.style.cssText = _btnSt + 'background:#1a3a5c;color:white;';
        _uploadBtn.onclick = function() { _uploadInput.click(); };

        var _uploadClear = document.createElement('button');
        _uploadClear.textContent = 'Clear import';
        _uploadClear.title = 'Retirer les objets importés';
        _uploadClear.style.cssText = _btnSt + 'background:white;color:#1a3a5c;';
        _uploadClear.style.display = 'none';

        var _uploadMsg = document.createElement('span');
        _uploadMsg.style.cssText = 'font-size:11px;font-family:Arial;color:#1a3a5c;';

        var _uploadedTraces = [];  // indices des traces importées

        // Découpe une ligne en champs : détecte virgule, point-virgule,
        // tabulation, sinon espaces multiples ; retire les guillemets.
        function _splitLine(line) {
            var sep = null;
            if (line.indexOf(',') >= 0) sep = ',';
            else if (line.indexOf(';') >= 0) sep = ';';
            else if (line.indexOf('\\t') >= 0) sep = '\\t';
            var parts = sep ? line.split(sep) : line.trim().split(/\\s+/);
            return parts.map(function(s) { return s.trim().replace(/^"|"$/g, ''); });
        }

        function _parseCSV(text) {
            var lines = text.split(/\\r\\n|\\r|\\n/).filter(function(l) { return l.trim().length; });
            if (!lines.length) return null;
            var header = _splitLine(lines[0]);
            var rows = [];
            for (var i = 1; i < lines.length; i++) rows.push(_splitLine(lines[i]));
            return {header: header, rows: rows};
        }

        _uploadInput.onchange = function(ev) {
            var file = ev.target.files && ev.target.files[0];
            if (!file) return;
            var reader = new FileReader();
            reader.onload = function(e) {
                var parsed = _parseCSV(e.target.result);
                if (!parsed || !parsed.header.length) {
                    _uploadMsg.textContent = 'fichier illisible'; _uploadInput.value = ''; return;
                }
                var hdr = parsed.header;

                // ── Modal de sélection des colonnes ──────────────────────────
                var _modal = document.createElement('div');
                _modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;' +
                    'background:rgba(0,0,0,0.45);z-index:9999;display:flex;' +
                    'align-items:center;justify-content:center;';

                var _box = document.createElement('div');
                _box.style.cssText = 'background:white;border-radius:6px;padding:18px 20px;' +
                    'min-width:320px;box-shadow:0 4px 20px rgba(0,0,0,0.35);' +
                    'font-family:Arial;font-size:12px;';

                var _mtitle = document.createElement('div');
                _mtitle.textContent = 'Colonnes — ' + file.name;
                _mtitle.style.cssText = 'font-weight:bold;font-size:12px;color:#1a3a5c;' +
                    'margin-bottom:14px;word-break:break-all;';
                _box.appendChild(_mtitle);

                function _makeRow(labelTxt, isOptional) {
                    var row = document.createElement('div');
                    row.style.cssText = 'display:flex;align-items:center;gap:8px;margin-bottom:9px;';
                    var lbl = document.createElement('span');
                    lbl.textContent = labelTxt + (isOptional ? ' (opt.)' : '') + ' :';
                    lbl.style.cssText = 'width:110px;color:#1a3a5c;flex-shrink:0;' +
                        'font-weight:' + (isOptional ? 'normal' : 'bold') + ';';
                    var sel = document.createElement('select');
                    sel.style.cssText = 'flex:1;font-size:11px;font-family:Arial;' +
                        'padding:4px 6px;border:1px solid #8899aa;border-radius:3px;outline:none;';
                    if (isOptional) {
                        var eOpt = document.createElement('option');
                        eOpt.value = ''; eOpt.textContent = '— aucun —';
                        sel.appendChild(eOpt);
                    }
                    for (var h = 0; h < hdr.length; h++) {
                        var opt = document.createElement('option');
                        opt.value = hdr[h]; opt.textContent = hdr[h];
                        sel.appendChild(opt);
                    }
                    row.appendChild(lbl); row.appendChild(sel);
                    _box.appendChild(row);
                    return sel;
                }

                var selX = _makeRow('Colonne X', false);
                var selY = _makeRow('Colonne Y', false);
                var selZ = _makeRow('Colonne Z', false);
                var selN = _makeRow('Nom des objets', true);

                // Sélecteur de repère des coordonnées du fichier.
                //  • mw  : galactocentrique (origine = Voie Lactée), affiché tel quel.
                //  • m31 : base propre à M31 → converti via p_MW = R_M31^T · p_M31 + r_M31.
                var _frameRow = document.createElement('div');
                _frameRow.style.cssText = 'display:flex;align-items:center;gap:8px;margin-bottom:9px;';
                var _frameLbl = document.createElement('span');
                _frameLbl.textContent = 'Repère :';
                _frameLbl.style.cssText = 'width:110px;color:#1a3a5c;flex-shrink:0;font-weight:bold;';
                var selFrame = document.createElement('select');
                selFrame.style.cssText = 'flex:1;font-size:11px;font-family:Arial;' +
                    'padding:4px 6px;border:1px solid #8899aa;border-radius:3px;outline:none;';
                var _frOpts = [
                    {v: 'mw',  t: 'Voie Lactée '},
                    {v: 'm31', t: 'M31 '}
                ];
                for (var fo = 0; fo < _frOpts.length; fo++) {
                    var _foOpt = document.createElement('option');
                    _foOpt.value = _frOpts[fo].v; _foOpt.textContent = _frOpts[fo].t;
                    selFrame.appendChild(_foOpt);
                }
                _frameRow.appendChild(_frameLbl); _frameRow.appendChild(selFrame);
                _box.appendChild(_frameRow);

                // Pré-sélection automatique par mot-clé
                function _trySelect(sel, keys) {
                    for (var k = 0; k < keys.length; k++) {
                        for (var h = 0; h < hdr.length; h++) {
                            if (hdr[h].toLowerCase().indexOf(keys[k]) >= 0) {
                                sel.value = hdr[h]; return;
                            }
                        }
                    }
                }
                _trySelect(selX, ['gx','_x','xpos','pos_x','x_kpc']);
                _trySelect(selY, ['gy','_y','ypos','pos_y','y_kpc']);
                _trySelect(selZ, ['gz','_z','zpos','pos_z','z_kpc']);
                _trySelect(selN, ['name','nom','id','label','objet']);

                var _btnRow = document.createElement('div');
                _btnRow.style.cssText = 'display:flex;gap:8px;justify-content:flex-end;margin-top:14px;';

                var _cancelBtn = document.createElement('button');
                _cancelBtn.textContent = 'Annuler';
                _cancelBtn.style.cssText = 'padding:5px 12px;font-size:11px;font-family:Arial;' +
                    'cursor:pointer;border:1px solid #8899aa;border-radius:4px;' +
                    'background:white;color:#1a3a5c;';
                _cancelBtn.onclick = function() {
                    document.body.removeChild(_modal); _uploadInput.value = '';
                };

                var _loadBtn = document.createElement('button');
                _loadBtn.textContent = 'Charger';
                _loadBtn.style.cssText = 'padding:5px 12px;font-size:11px;font-family:Arial;' +
                    'cursor:pointer;border:1px solid #8899aa;border-radius:4px;' +
                    'background:#1a3a5c;color:white;';
                _loadBtn.onclick = function() {
                    document.body.removeChild(_modal);
                    var cx = selX.value, cy = selY.value, cz = selZ.value, cn = selN.value;
                    var frame = selFrame.value;

                    // Convertit des coordonnées centrées M31 (base propre à M31)
                    // vers le repère galactique affiché : p_MW = R_M31^T · p_M31 + r_M31.
                    function _m31ToMW(px, py, pz) {
                        var R = _R_M31;
                        return [
                            R[0]*px + R[3]*py + R[6]*pz + _M31_GX,
                            R[1]*px + R[4]*py + R[7]*pz + _M31_GY,
                            R[2]*px + R[5]*py + R[8]*pz + _M31_GZ
                        ];
                    }

                    function _idx(name) {
                        if (!name) return -1;
                        var t = name.trim().toLowerCase();
                        for (var k = 0; k < hdr.length; k++) {
                            if (hdr[k].trim().toLowerCase() === t) return k;
                        }
                        return -1;
                    }
                    var ix = _idx(cx), iy = _idx(cy), iz = _idx(cz), iname = _idx(cn);
                    if (ix < 0 || iy < 0 || iz < 0) {
                        _uploadMsg.textContent = 'colonne introuvable'; _uploadInput.value = ''; return;
                    }

                    var X = [], Y = [], Z = [], lon = [], lat = [], names = [], hov = [];
                    for (var r = 0; r < parsed.rows.length; r++) {
                        var row = parsed.rows[r];
                        var x = parseFloat(row[ix]), y = parseFloat(row[iy]), z = parseFloat(row[iz]);
                        if (isNaN(x) || isNaN(y) || isNaN(z)) continue;
                        if (frame === 'm31') {
                            var _mw = _m31ToMW(x, y, z);
                            x = _mw[0]; y = _mw[1]; z = _mw[2];
                        }
                        var nm = (iname >= 0 && row[iname]) ? row[iname] : ('object ' + (r + 1));
                        X.push(x); Y.push(y); Z.push(z); names.push(nm);
                        var rr = Math.sqrt(x*x + y*y + z*z);
                        lon.push(Math.atan2(y, x) * 180 / Math.PI);
                        lat.push(rr > 0 ? Math.asin(z / rr) * 180 / Math.PI : 0);
                        hov.push(nm + '<br>X : ' + x.toFixed(1) + ' kpc<br>Y : ' +
                                 y.toFixed(1) + ' kpc<br>Z : ' + z.toFixed(1) +
                                 ' kpc<br><br>\\u25c6 imported');
                    }
                    if (!X.length) {
                        _uploadMsg.textContent = 'aucune donnée valide'; _uploadInput.value = ''; return;
                    }

                    // Nom court pour la légende : retire l'extension et tronque.
                    var _base = file.name.replace(/\\.[^.]+$/, '');
                    if (_base.length > 14) _base = _base.slice(0, 13) + '\\u2026';
                    var _label = '\\u25c6 ' + _base;
                    var trace3d = {
                        type: 'scatter3d', mode: 'markers',
                        x: X, y: Y, z: Z, scene: 'scene',
                        marker: {size: 0.4, color: 'magenta', symbol: 'diamond',
                                 line: {color: '#600060', width: 0.1}},
                        name: _label, text: hov, hoverinfo: 'text', showlegend: true
                    };
                    var traceGeo = {
                        type: 'scattergeo', mode: 'markers',
                        lon: lon, lat: lat, geo: 'geo',
                        marker: {size: 0.5, color: 'magenta', symbol: 'diamond',
                                 line: {color: '#600060', width: 0.1}},
                        name: _label, text: hov, hoverinfo: 'text', showlegend: false
                    };
                    var before = gd.data.length;
                    Plotly.addTraces(gd, [trace3d, traceGeo]).then(function() {
                        _uploadedTraces.push(before, before + 1);
                    });
                    _uploadMsg.textContent = X.length + ' objets importés';
                    _uploadClear.style.display = '';
                    _uploadInput.value = '';
                };

                _btnRow.appendChild(_cancelBtn);
                _btnRow.appendChild(_loadBtn);
                _box.appendChild(_btnRow);
                _modal.appendChild(_box);
                document.body.appendChild(_modal);
            };
            reader.readAsText(file);
        };

        _uploadClear.onclick = function() {
            if (_uploadedTraces.length) {
                var idx = _uploadedTraces.slice().sort(function(a, b) { return b - a; });
                Plotly.deleteTraces(gd, idx);
                _uploadedTraces = [];
            }
            _uploadMsg.textContent = '';
            _uploadClear.style.display = 'none';
        };

        var _uploadWrap = document.createElement('div');
        _uploadWrap.style.cssText = 'display:flex;gap:5px;align-items:center;flex-wrap:wrap;';
        _uploadWrap.appendChild(_uploadBtn);
        _uploadWrap.appendChild(_uploadClear);
        _uploadWrap.appendChild(_uploadMsg);
        _uploadWrap.appendChild(_uploadInput);

        // Panneau unique (recherche + navigation), ancré en bas à gauche,
        // immobile (position:fixed) et hors de la légende 3D (en haut à gauche).
        var _ctrlPanel = document.createElement('div');
        _ctrlPanel.style.cssText = 'position:fixed;bottom:12px;left:12px;z-index:1001;' +
            'display:flex;flex-direction:column;gap:6px;background:rgba(255,255,255,0.92);' +
            'padding:8px;border:1px solid #8899aa;border-radius:4px;';
        _ctrlPanel.appendChild(_searchWrap);
        _ctrlPanel.appendChild(_filterWrap);
        _ctrlPanel.appendChild(_uploadWrap);
        _ctrlPanel.appendChild(_nav);
        document.body.appendChild(_ctrlPanel);
    }

    if (document.readyState === 'complete') setup();
    else window.addEventListener('load', setup);
})();
</script>
"""

_out = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "3D aitoff combined.html"
)

fig.write_html(
    _out,
    include_plotlyjs=True,
    full_html=True,
    config={"responsive": True},     # Plotly réajuste le tracé à la taille du conteneur
    default_width="100%",
    default_height="100vh",
)

with open(_out, "r", encoding="utf-8") as f:
    _html = f.read()

# CSS : la page occupe toute la fenêtre, quelle que soit la résolution de l'écran.
# → évite le débordement / les panneaux décalés sur les écrans plus petits.
_RESPONSIVE_CSS = (
    "<style>\n"
    "  html, body { margin:0; padding:0; width:100%; height:100%;\n"
    "               background:#f0f2f5; overflow:auto; }\n"
    "  .plotly-graph-div { width:100% !important; height:100vh !important; }\n"
    "</style>\n"
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
)
_html = _html.replace("</head>", _RESPONSIVE_CSS + "</head>")

# ── Table des positions (kpc) pour le filtre par rayon ────────────────────
# name -> [gx, gy, gz] ; inclut MW / M31 / M33 et toutes les galaxies.
import json as _json

_pos_map = {}
for _df in (df_pm, df_nopm):
    for _, _row in _df.iterrows():
        _gx, _gy, _gz = _row["gx"], _row["gy"], _row["gz"]
        if pd.isna(_gx) or pd.isna(_gy) or pd.isna(_gz):
            continue
        _pos_map[str(_row["name"])] = [round(float(_gx), 2),
                                       round(float(_gy), 2),
                                       round(float(_gz), 2)]

_pos_map["Milky Way"]      = [0.0, 0.0, 0.0]
_pos_map["M31 Andromeda"]  = [round(float(_m31_gx), 2), round(float(_m31_gy), 2), round(float(_m31_gz), 2)]
_pos_map["M33 Triangulum"] = [round(float(_m33_gx), 2), round(float(_m33_gy), 2), round(float(_m33_gz), 2)]

# Objets de référence : repères majeurs en tête, puis toutes les galaxies triées.
_anchors   = ["Milky Way", "M31 Andromeda", "M33 Triangulum"]
_others    = sorted([n for n in _pos_map if n not in _anchors], key=str.lower)
_ref_names = _anchors + _others

_FILTER_VARS = (
    "<script>var _POS=" + _json.dumps(_pos_map)
    + ";var _REF_NAMES=" + _json.dumps(_ref_names) + ";</script>\n"
)

_M31_VARS = (f'<script>var _M31_GX={_m31_gx:.1f},_M31_GY={_m31_gy:.1f},_M31_GZ={_m31_gz:.1f};</script>\n')
_SCALE_VARS = (
    f'<script>var _SCALE_MAIN={_SCALE_MAIN},_SCALE_T0={_SCALE_T0},_SCALE_T1={_SCALE_T1},'
    f'_BAR_XC={_bar_xc:.5f},_BAR_HALF={_bar_half:.5f},_BAR_Y={_bar_y:.5f},_TICK_H={_tick_h:.5f};</script>\n'
)
_M31_ROT_VARS = (
    '<script>var _R_M31=['
    + ','.join(f'{v:.8f}' for v in _R_m31.flatten())
    + '];</script>\n'
)
_html = _html.replace("</body>", _FILTER_VARS + _M31_VARS + _SCALE_VARS + _M31_ROT_VARS + _JS + "\n</body>")

with open(_out, "w", encoding="utf-8") as f:
    f.write(_html)

os.startfile(_out)
print(f"Figure saved and opened: {_out}")