# app_v2_2_2.py
# Coastal Listings Map – robust build with root/data CSV loading, popup sanitization,
# no KeyError on filters, and optional heatmap / near‑shore nudge.

import os, html
import pandas as pd
import numpy as np
import streamlit as st
import folium
from folium.plugins import MarkerCluster, HeatMap, MiniMap
from streamlit_folium import st_folium

st.set_page_config(page_title="Coastal Listings Map — v2.2.2", layout="wide")

# --------------------------- Utilities ---------------------------

@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_csv_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ascii_sea_band(v):
    if pd.isna(v): return np.nan
    if v <= 0.5: return "0-0.5 km"
    if v <= 1.0: return "0.5-1 km"
    if v <= 2.0: return "1-2 km"
    if v <= 5.0: return "2-5 km"
    return ">5 km"

def fmt_inr(v):
    if pd.isna(v): return "NA"
    if v >= 1e7: return f"₹{v/1e7:.2f} Cr"
    if v >= 1e5: return f"₹{v/1e5:.2f} L"
    return f"₹{v:,.0f}"

# Where to look for the CSV (in this order)
CANDIDATES = [
    "data/NO_Url_export.csv",   # repo data/ folder
    "NO_Url_export.csv",        # repo root
]
DEFAULT_URL = os.environ.get("DATA_CSV_URL", "").strip()  # optional raw GitHub CSV

# --------------------------- Data source ---------------------------

st.sidebar.title("Data source")
use_uploaded = st.sidebar.checkbox("Upload CSV now", value=False)

df = None
if use_uploaded:
    uploaded = st.sidebar.file_uploader("Upload combined CSV", type=["csv"], key="uploader")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
else:
    if DEFAULT_URL:
        try:
            df = load_csv_from_url(DEFAULT_URL)
            st.sidebar.success("Loaded CSV from DATA_CSV_URL.")
        except Exception as e:
            st.sidebar.error(f"URL load failed: {e}")
    if df is None:
        for p in CANDIDATES:
            if os.path.exists(p):
                try:
                    df = load_csv_from_path(p)
                    st.sidebar.success(f"Loaded CSV from {p}.")
                    break
                except Exception as e:
                    st.sidebar.error(f"Load failed for {p}: {e}")

if df is None or df.empty:
    st.warning(
        "No data loaded. Upload a CSV, place it at **data/combined_full_listings_clean_v9.csv** "
        "or at repo root as **combined_full_listings_clean_v9.csv**, "
        "or set **DATA_CSV_URL** to a raw CSV URL."
    )
    st.stop()

# --------------------------- Standardize ---------------------------

expected = [
    "id","location","region","property_type","latitude","longitude","price",
    "area_cent","price_per_cent","orientation","nearest_beach_name",
    "distance_to_nearest_beach_km","sea_distance_band_fine",
    "distance_to_city_center","source_url"
]
for c in expected:
    if c not in df.columns:
        df[c] = np.nan

num_cols = ["price","area_cent","price_per_cent","latitude","longitude",
            "distance_to_nearest_beach_km","distance_to_city_center"]
df = ensure_numeric(df, num_cols)

# Clean sea band (or derive it)
if "sea_distance_band_fine" not in df.columns or df["sea_distance_band_fine"].isna().all():
    df["sea_distance_band_fine"] = df["distance_to_nearest_beach_km"].apply(ascii_sea_band)
else:
    df["sea_distance_band_fine"] = (
        df["sea_distance_band_fine"].astype(str).str.replace("–","-", regex=False)
    )

# Remove NBSPs and fill
for c in ["id","location","region","property_type","nearest_beach_name"]:
    df[c] = df[c].astype(str).str.replace("\xa0"," ", regex=False)

df["region"] = df["region"].fillna("Unknown")
df["property_type"] = df["property_type"].fillna("unknown")
df["location"] = df["location"].fillna("")
df["sea_distance_band_fine"] = df["sea_distance_band_fine"].fillna("Unspecified")

# --------------------------- Filters ---------------------------

st.sidebar.title("Filters")
regions = sorted(df["region"].dropna().unique().tolist())
types   = sorted(df["property_type"].dropna().unique().tolist())
bands   = [b for b in ["0-0.5 km","0.5-1 km","1-2 km","2-5 km",">5 km","Unspecified"]
           if b in df["sea_distance_band_fine"].unique()]

sel_regions = st.sidebar.multiselect("Region", regions, default=regions, key="f_regions")
sel_types   = st.sidebar.multiselect("Property Type", types, default=types, key="f_types")
sel_bands   = st.sidebar.multiselect("Sea Distance Band", bands, default=bands, key="f_bands")

pmax  = float(np.nanmax(df["price"]))            if np.isfinite(df["price"]).any() else 1.0
ppcmax= float(np.nanmax(df["price_per_cent"]))   if np.isfinite(df["price_per_cent"]).any() else 1.0
amax  = float(np.nanmax(df["area_cent"]))        if np.isfinite(df["area_cent"]).any() else 1.0

price_range = st.sidebar.slider("Price (INR)", 0.0, max(pmax,1.0), (0.0, max(pmax,1.0)), key="price_rng")
ppc_range   = st.sidebar.slider("₹/cent", 0.0, max(ppcmax,1.0), (0.0, max(ppcmax,1.0)), key="ppc_rng")
area_range  = st.sidebar.slider("Area (cent)", 0.0, max(amax,1.0), (0.0, max(amax,1.0)), key="area_rng")

st.sidebar.title("Map options")
size_by    = st.sidebar.selectbox("Point size by", ["price","price_per_cent","area_cent"], index=0, key="sizeby")
use_cluster= st.sidebar.checkbox("Cluster markers", value=True, key="cluster")
show_heat  = st.sidebar.checkbox("Show ₹/cent heatmap", value=False, key="heat")
base_tiles = st.sidebar.selectbox("Base map", ["Carto (light)","OpenStreetMap","Satellite"], index=0, key="tiles")
nudge      = st.sidebar.checkbox("Nudge near-shore points inland (visual)", value=True, key="nudge")

# Apply filters
f = df.copy()
f = f[f["region"].isin(sel_regions)]
f = f[f["property_type"].isin(sel_types)]
f = f[f["sea_distance_band_fine"].isin(sel_bands)]
f = f[f["price"].between(price_range[0], price_range[1])]
f = f[f["price_per_cent"].between(ppc_range[0], ppc_range[1])]
f = f[f["area_cent"].between(area_range[0], area_range[1])]

st.sidebar.success(f"{len(f):,} listings match filters")

# --------------------------- Map ---------------------------

# Center
if f[["latitude","longitude"]].dropna().empty:
    center_lat, center_lon = 8.5639, 76.8443  # St Andrews fallback
else:
    center_lat = float(f["latitude"].median())
    center_lon = float(f["longitude"].median())

tiles_map = {
    "Carto (light)": ("cartodbpositron", None),
    "OpenStreetMap": ("OpenStreetMap", None),
    "Satellite": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                  "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"),
}
tiles, attr = tiles_map[base_tiles]
m = folium.Map(location=[center_lat, center_lon], tiles=tiles, attr=attr,
               zoom_start=11, control_scale=True)

# Legend (points + cluster colors)
legend_html = '''
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
     background: white; padding: 10px 12px; border: 1px solid #888; border-radius: 6px; font-size: 13px;">
<b>Legend</b><br>
<span style="display:inline-block;width:12px;height:12px;background:#00A8FF;border-radius:50%;margin-right:6px;"></span> Plot (land)<br>
<span style="display:inline-block;width:12px;height:12px;background:#DC2626;border-radius:50%;margin-right:6px;"></span> Property (built)<br>
<hr style="margin:6px 0;">
<b>Cluster colors</b><br>
<span style="display:inline-block;width:12px;height:12px;background:#90ee90;border-radius:50%;margin-right:6px;border:1px solid #666;"></span> ≤ 10 points<br>
<span style="display:inline-block;width:12px;height:12px;background:#ffd166;border-radius:50%;margin-right:6px;border:1px solid #666;"></span> 11–50 points<br>
<span style="display:inline-block;width:12px;height:12px;background:#f94144;border-radius:50%;margin-right:6px;border:1px solid #666;"></span> > 50 points<br>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Work with only rows we will plot (avoids KeyError on indexing)
to_plot = f.dropna(subset=["latitude","longitude"]).reset_index(drop=True)

# Optional visual nudge for near‑shore points (avoids markers in the water)
if nudge and "distance_to_nearest_beach_km" in to_plot.columns:
    near = (to_plot["distance_to_nearest_beach_km"] <= 0.08) | (to_plot["sea_distance_band_fine"]=="0-0.5 km")
    to_plot.loc[near, "longitude"] = to_plot.loc[near, "longitude"] + 0.0015  # ~160 m inland

# Marker size scaling aligned to the plotted rows
def scale_radius(series):
    s = pd.to_numeric(series, errors="coerce")
    if np.isnan(s).all():
        return np.full(len(series), 8.0)
    q1, q3 = np.nanquantile(s, 0.1), np.nanquantile(s, 0.9)
    s = np.clip(s, q1, q3)
    mn, mx = np.nanmin(s), np.nanmax(s)
    if mx - mn == 0:
        return np.full(len(series), 8.0)
    return 6 + 18 * (s - mn) / (mx - mn)

radii = np.asarray(scale_radius(to_plot[size_by]))
cluster = MarkerCluster(disableClusteringAtZoom=15) if use_cluster else None

for i, row in to_plot.iterrows():
    color = "#00A8FF" if row.get("property_type") == "plot" else "#DC2626"

    # Sanitize popup text to avoid odd component GETs
    def esc(x): return html.escape(str(x)).replace("\xa0", " ")

    popup_html = f"""
    <b>{esc(row.get('id',''))}</b> — {esc(row.get('location',''))}<br>
    <b>Type:</b> {esc(row.get('property_type',''))} | <b>Region:</b> {esc(row.get('region',''))}<br>
    <b>Price:</b> {fmt_inr(row.get('price'))} | <b>Area:</b> {esc(row.get('area_cent'))} cent |
    <b>₹/cent:</b> {fmt_inr(row.get('price_per_cent'))}<br>
    <b>Sea dist:</b> {esc(row.get('distance_to_nearest_beach_km'))} km<br>
    <b>Beach:</b> {esc(row.get('nearest_beach_name',''))}<br>
    <a href="{esc(row.get('source_url',''))}" target="_blank" rel="noopener">Open listing</a>
    """

    radius = float(radii[i]) if i < len(radii) else 8.0
    cm = folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=radius, color=color, fill=True, fill_opacity=0.85, weight=1
    )
    cm.add_child(folium.Popup(popup_html, max_width=350))
    if cluster:
        cluster.add_child(cm)
    else:
        cm.add_to(m)

if cluster:
    cluster.add_to(m)

# Optional heatmap of ₹/cent
if show_heat:
    heat_df = to_plot.dropna(subset=["price_per_cent"]).copy()
    if not heat_df.empty:
        w = heat_df["price_per_cent"]
        w = (w - np.nanmin(w)) / (np.nanmax(w) - np.nanmin(w) + 1e-9)
        HeatMap(list(zip(heat_df["latitude"], heat_df["longitude"], w)), name="₹/cent heat").add_to(m)

MiniMap(toggle_display=True, minimized=True).add_to(m)
folium.LayerControl(position="topleft").add_to(m)

st.markdown("### Interactive Map (click to open listing)")
st_folium(m, width=1100, height=650, key="map_component")

# --------------------------- Metrics & Table ---------------------------

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Listings", f"{len(to_plot):,}")
with c2: st.metric("Median Price", fmt_inr(np.nanmedian(to_plot["price"])) if len(to_plot) else "NA")
with c3: st.metric("Median ₹/cent", fmt_inr(np.nanmedian(to_plot["price_per_cent"])) if len(to_plot) else "NA")
with c4: st.metric("Median Sea Dist", f"{np.nanmedian(to_plot['distance_to_nearest_beach_km']):.2f} km" if len(to_plot) else "NA")

with st.expander("What do the symbols and colors mean?"):
    st.markdown("""
**Points (blue/red):** individual listings. Blue = plots (land). Red = built properties.  
**Marker size:** the field you chose (price / ₹ per cent / area).  
**Cluster circles (green/yellow/red):** listing counts when zoomed out (≤10 / 11–50 / >50). Click to zoom in.  
**HeatMap (optional):** highlights where ₹/cent is higher (hotter glow).  
**Nudge near‑shore points:** nudges very near‑shore markers ~160 m inland for readability; it does **not** change your data.
""")

st.markdown("### Filtered Listings")
show_cols = ["id","location","region","property_type","price","area_cent","price_per_cent","sea_distance_band_fine",
             "distance_to_nearest_beach_km","nearest_beach_name","source_url","latitude","longitude"]
st.dataframe(to_plot[show_cols], use_container_width=True)

csv = to_plot[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv, file_name="filtered_listings.csv", mime="text/csv")
