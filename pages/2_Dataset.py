import streamlit as st
import pandas as pd

st.title("📊 Laptop Dataset")

df = pd.read_csv("laptop.csv")

st.markdown("### Preview of Dataset Used for ML")
st.dataframe(df.head(50), use_container_width=True)

st.info(f"Total laptops in dataset: {df.shape[0]}")
