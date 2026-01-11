import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.title("🔍 Laptop Recommendation")

# Load dataset
df = pd.read_csv("laptop.csv")
df_display = df.copy()

# Preprocessing
df["Price"] = df["Price"].str.replace("₹","").str.replace(",","").astype(int)
df["Ram_GB"] = df["Ram"].str.extract(r"(\d+)").fillna(0).astype(int)
df["SSD_GB"] = df["SSD"].str.extract(r"(\d+)").fillna(0).astype(int)

def graphics_flag(x):
    if "Intel" in str(x) or "UHD" in str(x):
        return 0
    return 1

df["Graphics_Flag"] = df["Graphics"].apply(graphics_flag)
df["Rating"] = df["Rating"].fillna(df["Rating"].mean())

X = df[["Price","Ram_GB","SSD_GB","Rating","Graphics_Flag"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_scaled)

# Sidebar Search (TOP LEFT 🔥)
st.sidebar.header("🔎 Search Laptop")

budget = st.sidebar.slider("Budget (₹)", 30000, 200000, 60000)
ram = st.sidebar.selectbox("RAM (GB)", [4, 8, 16, 32])
ssd = st.sidebar.selectbox("SSD (GB)", [256, 512, 1024])
rating = st.sidebar.slider("Minimum Rating", 50, 100, 60)
graphics = st.sidebar.selectbox("Graphics", ["Integrated", "Dedicated"])

graphics_val = 1 if graphics == "Dedicated" else 0

if st.sidebar.button("🚀 Recommend"):
    user_input = [[budget, ram, ssd, rating, graphics_val]]
    user_scaled = scaler.transform(user_input)
    distances, indices = knn.kneighbors(user_scaled)

    result = df_display.iloc[indices[0]][
        ["Model","Price","Ram","SSD","Graphics","Display","Rating"]
    ]

    st.subheader("✅ Recommended Laptops")
    st.dataframe(result, use_container_width=True)

