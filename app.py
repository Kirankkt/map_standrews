
import math
import pandas as pd
import numpy as np
import streamlit as st

# Folium + Streamlit-Folium for clickable popups
import folium
from folium.plugins import MarkerCluster, HeatMap, MiniMap
from streamlit_folium import st_folium

st.set_page_config(page_title="Coastal Listings Map — v2", layout="wide")

# --------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        return pd.DataFrame()

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

# --------------- Data input ----------------
st.sidebar.title("Data")
uploaded = st.sidebar.file_uploader("Upload the combined CSV", type=["csv"])

if uploaded is None:
    st.info("Upload your latest combined CSV (e.g., combined_full_listings_clean_v9.csv).")
    st.stop()

df = load_csv(uploaded)

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

if "sea_distance_band_fine" not in df.columns or df["sea_distance_band_fine"].isna().all():
    df["sea_distance_band_fine"] = df["distance_to_nearest_beach_km"].apply(ascii_sea_band)
else:
    df["sea_distance_band_fine"] = df["sea_distance_band_fine"].astype(str).str.replace("–","-", regex=False)

df["region"] = df["region"].fillna("Unknown")
df["property_type"] = df["property_type"].fillna("unknown")
df["location"] = df["location"].fillna("")
df["sea_distance_band_fine"] = df["sea_distance_band_fine"].fillna("Unspecified")

# --------------- Filters ----------------
st.sidebar.title("Filters")
regions = sorted(df["region"].dropna().unique().tolist())
types   = sorted(df["property_type"].dropna().unique().tolist())
bands   = [b for b in ["0-0.5 km","0.5-1 km","1-2 km","2-5 km",">5 km","Unspecified"] if b in df["sea_distance_band_fine"].unique()]

sel_regions = st.sidebar.multiselect("Region", regions, default=regions)
sel_types   = st.sidebar.multiselect("Property Type", types, default=types)
sel_bands   = st.sidebar.multiselect("Sea Distance Band", bands, default=bands)

pmax = float(np.nanmax(df["price"])) if np.isfinite(df["price"]).any() else 1.0
ppcmax = float(np.nanmax(df["price_per_cent"])) if np.isfinite(df["price_per_cent"]).any() else 1.0
amax = float(np.nanmax(df["area_cent"])) if np.isfinite(df["area_cent"]).any() else 1.0

price_range = st.sidebar.slider("Price (INR)", 0.0, max(pmax,1.0), (0.0, max(pmax,1.0)))
ppc_range   = st.sidebar.slider("₹/cent", 0.0, max(ppcmax,1.0), (0.0, max(ppcmax,1.0)))
area_range  = st.sidebar.slider("Area (cent)", 0.0, max(amax,1.0), (0.0, max(amax,1.0)))

f = df.copy()
f = f[f["region"].isin(sel_regions)]
f = f[f["property_type"].isin(sel_types)]
f = f[f["sea_distance_band_fine"].isin(sel_bands)]
f = f[f["price"].between(price_range[0], price_range[1])]
f = f[f["price_per_cent"].between(ppc_range[0], ppc_range[1])]
f = f[f["area_cent"].between(area_range[0], area_range[1])]

st.sidebar.success(f"{len(f):,} listings match filters")

# --------------- Map config ----------------
st.sidebar.title("Map Options")
size_by = st.sidebar.selectbox("Point size by", ["price","price_per_cent","area_cent"], index=0)
use_cluster = st.sidebar.checkbox("Cluster markers", value=True)
show_heat = st.sidebar.checkbox("Show ₹/cent heatmap", value=False)

# center
if f[["latitude","longitude"]].dropna().empty:
    center_lat, center_lon = 8.5639, 76.8443
else:
    center_lat, center_lon = float(f["latitude"].median()), float(f["longitude"].median())

m = folium.Map(location=[center_lat, center_lon], tiles="cartodbpositron", zoom_start=11, control_scale=True)

# Legend
legend_html = '''
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
     background: white; padding: 10px 12px; border: 1px solid #888; border-radius: 6px; font-size: 13px;">
<b>Legend</b><br>
<span style="display:inline-block;width:12px;height:12px;background:#00A8FF;border-radius:50%;margin-right:6px;"></span> Plot<br>
<span style="display:inline-block;width:12px;height:12px;background:#DC2626;border-radius:50%;margin-right:6px;"></span> Property<br>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Size scaling
def scale_radius(series):
    s = pd.to_numeric(series, errors="coerce")
    q1, q3 = np.nanquantile(s, 0.1), np.nanquantile(s, 0.9)
    s = np.clip(s, q1, q3)
    mn, mx = np.nanmin(s), np.nanmax(s)
    if mx - mn == 0: 
        return np.full_like(s, 8, dtype=float)
    return 6 + 18 * (s - mn) / (mx - mn)  # radius in pixels

radii = scale_radius(f[size_by])

# Clusters
cluster = MarkerCluster(disableClusteringAtZoom=15) if use_cluster else None

# Add points
for idx, row in f.dropna(subset=["latitude","longitude"]).reset_index(drop=True).iterrows():
    color = "#00A8FF" if row.get("property_type")=="plot" else "#DC2626"
    popup_html = f"""
    <b>{row.get('id','')}</b> — {row.get('location','')}<br>
    <b>Type:</b> {row.get('property_type','')} | <b>Region:</b> {row.get('region','')}<br>
    <b>Price:</b> {fmt_inr(row.get('price'))} | <b>Area:</b> {row.get('area_cent')} cent | 
    <b>₹/cent:</b> {fmt_inr(row.get('price_per_cent'))}<br>
    <b>Sea dist:</b> {row.get('distance_to_nearest_beach_km')} km<br>
    <b>Beach:</b> {row.get('nearest_beach_name','')}<br>
    <a href="{row.get('source_url','')}" target="_blank">Open listing</a>
    
    cm = folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=float(radii[idx]) if not pd.isna(radii[idx]) else 8,
        color=color, fill=True, fill_opacity=0.85, weight=1
    )
    cm.add_child(folium.Popup(popup_html, max_width=350))
    if cluster: 
        cluster.add_child(cm)
    else:
        cm.add_to(m)

if cluster:
    cluster.add_to(m)

# Optional heatmap (₹/cent)
if show_heat:
    heat_data = f.dropna(subset=["latitude","longitude","price_per_cent"])[["latitude","longitude","price_per_cent"]].values.tolist()
    if heat_data:
        HeatMap(heat_data, name="₹/cent heat", radius=18, blur=20, max_zoom=16).add_to(m)

MiniMap(toggle_display=True, minimized=True).add_to(m)
folium.LayerControl(position="topleft").add_to(m)

st.markdown("### Interactive Map (clickable)")
st_data = st_folium(m, width=1100, height=650)

# Metrics & table
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Listings", f"{len(f):,}")
with c2: st.metric("Median Price", fmt_inr(np.nanmedian(f["price"])) if len(f) else "NA")
with c3: st.metric("Median ₹/cent", fmt_inr(np.nanmedian(f["price_per_cent"])) if len(f) else "NA")
with c4: st.metric("Median Sea Dist", f"{np.nanmedian(f['distance_to_nearest_beach_km']):.2f} km" if len(f) else "NA")

st.markdown("### Filtered Listings")
show_cols = ["id","location","region","property_type","price","area_cent","price_per_cent","sea_distance_band_fine",
             "distance_to_nearest_beach_km","nearest_beach_name","source_url","latitude","longitude"]
st.dataframe(f[show_cols])

csv = f[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv, file_name="filtered_listings.csv", mime="text/csv")

st.caption("Colors: blue = plot (land), red = property (built). Marker size scales by the chosen field. Heatmap shows ₹/cent intensity.")
