
import math
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Coastal Listings Map", layout="wide")

# -------------------- Helpers --------------------
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        return pd.DataFrame()

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

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

# -------------------- Data input --------------------
st.sidebar.title("Data")
uploaded = st.sidebar.file_uploader("Upload the combined CSV", type=["csv"])

if uploaded is not None:
    df = load_csv(uploaded)
else:
    st.sidebar.info("No file uploaded yet. Upload your latest combined CSV to begin.")
    df = pd.DataFrame()

if df.empty:
    st.stop()

# -------------------- Standardize fields --------------------
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

TECHNOPARK = (8.5581, 76.8816)
VARKALA_STN = (8.74453, 76.719692)

if df[["latitude","longitude"]].notna().all(axis=1).any():
    df["distance_to_technopark_km"] = haversine_km(df["latitude"], df["longitude"], TECHNOPARK[0], TECHNOPARK[1])
    df["distance_to_varkala_station_km"] = haversine_km(df["latitude"], df["longitude"], VARKALA_STN[0], VARKALA_STN[1])
    df["distance_to_city_proxy_km"] = np.where(df["region"].astype(str)=="Varkala",
                                               df["distance_to_varkala_station_km"],
                                               df["distance_to_technopark_km"])
else:
    df["distance_to_technopark_km"] = np.nan
    df["distance_to_varkala_station_km"] = np.nan
    df["distance_to_city_proxy_km"] = np.nan

df["region"] = df["region"].fillna("Unknown")
df["property_type"] = df["property_type"].fillna("unknown")
df["location"] = df["location"].fillna("")
df["sea_distance_band_fine"] = df["sea_distance_band_fine"].fillna("Unspecified")

# -------------------- Sidebar filters --------------------
st.sidebar.title("Filters")

regions = sorted(df["region"].dropna().unique().tolist())
types = sorted(df["property_type"].dropna().unique().tolist())
bands = [b for b in ["0-0.5 km","0.5-1 km","1-2 km","2-5 km",">5 km","Unspecified"] if b in df["sea_distance_band_fine"].unique()]

sel_regions = st.sidebar.multiselect("Region", regions, default=regions)
sel_types   = st.sidebar.multiselect("Property Type", types, default=types)
sel_bands   = st.sidebar.multiselect("Sea Distance Band", bands, default=bands)

pmin, pmax = float(np.nanmin(df["price"])) if np.isfinite(df["price"]).any() else 0.0, float(np.nanmax(df["price"])) if np.isfinite(df["price"]).any() else 1.0
ppcmin, ppcmax = float(np.nanmin(df["price_per_cent"])) if np.isfinite(df["price_per_cent"]).any() else 0.0, float(np.nanmax(df["price_per_cent"])) if np.isfinite(df["price_per_cent"]).any() else 1.0
amin, amax = float(np.nanmin(df["area_cent"])) if np.isfinite(df["area_cent"]).any() else 0.0, float(np.nanmax(df["area_cent"])) if np.isfinite(df["area_cent"]).any() else 1.0

price_range = st.sidebar.slider("Price (INR)", min_value=0.0, max_value=max(pmax,1.0), value=(0.0, max(pmax,1.0)))
ppc_range   = st.sidebar.slider("₹/cent", min_value=0.0, max_value=max(ppcmax,1.0), value=(0.0, max(ppcmax,1.0)))
area_range  = st.sidebar.slider("Area (cent)", min_value=0.0, max_value=max(amax,1.0), value=(0.0, max(amax,1.0)))

sea_max = float(np.nanmax(df["distance_to_nearest_beach_km"])) if np.isfinite(df["distance_to_nearest_beach_km"]).any() else 5.0
city_max = float(np.nanmax(df["distance_to_city_proxy_km"])) if np.isfinite(df["distance_to_city_proxy_km"]).any() else 30.0

sea_range = st.sidebar.slider("Distance to Sea (km)", 0.0, max(sea_max,0.5), (0.0, max(sea_max,0.5)))
city_range = st.sidebar.slider("Distance to City Proxy (km)", 0.0, max(city_max,1.0), (0.0, max(city_max,1.0)))

# -------------------- Apply filters --------------------
f = df.copy()
f = f[f["region"].isin(sel_regions)]
f = f[f["property_type"].isin(sel_types)]
f = f[f["sea_distance_band_fine"].isin(sel_bands)]
f = f[f["price"].between(price_range[0], price_range[1])]
f = f[f["price_per_cent"].between(ppc_range[0], ppc_range[1])]
f = f[f["area_cent"].between(area_range[0], area_range[1])]
if np.isfinite(f["distance_to_nearest_beach_km"]).any():
    f = f[f["distance_to_nearest_beach_km"].between(sea_range[0], sea_range[1])]
if np.isfinite(f["distance_to_city_proxy_km"]).any():
    f = f[f["distance_to_city_proxy_km"].between(city_range[0], city_range[1])]

st.sidebar.success(f"{len(f):,} listings match filters")

# -------------------- Map controls --------------------
st.sidebar.title("Map")
layer_kind = st.sidebar.selectbox("Layer", ["Points (by property type)","Hexagon (₹/cent intensity)"])
size_by = st.sidebar.selectbox("Point Size By", ["price","price_per_cent","area_cent"], index=0)

# center map
if f[["latitude","longitude"]].dropna().empty:
    center = {"lat": 8.5639, "lon": 76.8443}
else:
    center = {"lat": float(f["latitude"].median()), "lon": float(f["longitude"].median())}

# Colors
type_colors = {
    "plot": [0, 168, 255],
    "property": [220, 38, 38],
    "unknown": [120, 120, 120]
}
f["_color"] = f["property_type"].apply(lambda t: type_colors.get(str(t), [120,120,120]))

def normalize_size(s):
    arr = pd.to_numeric(s, errors="coerce")
    q1, q3 = np.nanquantile(arr, 0.1), np.nanquantile(arr, 0.9)
    arr = np.clip(arr, q1, q3)
    if np.nanmax(arr) - np.nanmin(arr) == 0:
        return np.full_like(arr, 200, dtype=float)
    return 200 + 800 * (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))

f["_size"] = normalize_size(f[size_by])

def build_tooltip(row):
    return (f"<b>{row.get('id','')}</b> — {row.get('location','')}<br/>"
            f"<b>Type:</b> {row.get('property_type','')} | <b>Region:</b> {row.get('region','')}<br/>"
            f"<b>Price:</b> {fmt_inr(row.get('price'))} | <b>Area:</b> {row.get('area_cent')} cent | "
            f"<b>₹/cent:</b> {fmt_inr(row.get('price_per_cent'))}<br/>"
            f"<b>Sea dist:</b> {row.get('distance_to_nearest_beach_km')} km | "
            f"<b>City proxy dist:</b> {row.get('distance_to_city_proxy_km')} km<br/>"
            f"<b>Beach:</b> {row.get('nearest_beach_name','')}<br/>"
            f"<a href='{row.get('source_url','')}' target='_blank'>Link</a>")

f["_tooltip"] = f.apply(build_tooltip, axis=1)

st.markdown("### Interactive Map")
if layer_kind.startswith("Points"):
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=f.dropna(subset=["latitude","longitude"]),
        get_position='[longitude, latitude]',
        get_radius="_size",
        get_fill_color="_color",
        pickable=True,
        opacity=0.8,
        auto_highlight=True,
    )
    tooltip = {"html": "{_tooltip}", "style": {"backgroundColor": "white", "color": "black"}}
    view_state = pdk.ViewState(latitude=center["lat"], longitude=center["lon"], zoom=11, pitch=30)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="mapbox://styles/mapbox/light-v9"))
else:
    data_hex = f.dropna(subset=["latitude","longitude","price_per_cent"])[["latitude","longitude","price_per_cent"]]
    layer = pdk.Layer(
        "HexagonLayer",
        data=data_hex,
        get_position='[longitude, latitude]',
        elevation_scale=50,
        pickable=True,
        extruded=True,
        radius=150,
        elevation_range=[0, 4000],
    )
    view_state = pdk.ViewState(latitude=center["lat"], longitude=center["lon"], zoom=11, pitch=30)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v9"))

# -------------------- Metrics & Table --------------------
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Listings", f"{len(f):,}")
with c2: st.metric("Median Price", fmt_inr(np.nanmedian(f["price"])) if len(f) else "NA")
with c3: st.metric("Median ₹/cent", fmt_inr(np.nanmedian(f["price_per_cent"])) if len(f) else "NA")
with c4: st.metric("Median Sea Dist", f"{np.nanmedian(f['distance_to_nearest_beach_km']):.2f} km" if len(f) else "NA")

st.markdown("### Filtered Listings")
show_cols = ["id","location","region","property_type","price","area_cent","price_per_cent","sea_distance_band_fine",
             "distance_to_nearest_beach_km","distance_to_city_proxy_km","nearest_beach_name","source_url","latitude","longitude"]
st.dataframe(f[show_cols])

csv = f[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv, file_name="filtered_listings.csv", mime="text/csv")

st.markdown('''---
**Tips**
- Use the layer switch to view **points** by type or a **hexagon intensity** by ₹/cent.
- The “City proxy” is **Technopark** for North‑of‑Puthenthope and **Varkala Station** for Varkala.
- The map uses straight‑line distances. For drive time, we’d need a routing backend.
''')
