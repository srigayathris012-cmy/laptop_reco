import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Smart Laptop Recommender",
    page_icon="💻",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("laptops.csv")

# ---------------- ML PREPROCESS ----------------
le_usage = LabelEncoder()
le_ram = LabelEncoder()
le_storage = LabelEncoder()

df["Usage_enc"] = le_usage.fit_transform(df["Usage"])
df["RAM_enc"] = le_ram.fit_transform(df["RAM"])
df["Storage_enc"] = le_storage.fit_transform(df["Storage"])

features = df[["Usage_enc", "RAM_enc", "Storage_enc"]]
similarity = cosine_similarity(features)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🤖 Smart Recommendation", "🔄 Compare Laptops"]
)

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.markdown(
        """
        <h1 style="text-align:center;color:#1F618D;">
        Smart Laptop Recommendation System
        </h1>
        <p style="text-align:center;font-size:18px;">
        AI-powered laptop recommendations with comparison & buy links
        </p>
        <hr>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    col1.info("🎓 Students")
    col2.success("💼 Professionals")
    col3.warning("🎮 Gamers")

    st.markdown(
        """
        ### 🔥 Features
        ✔ Machine Learning based recommendation  
        ✔ Real e-commerce style UI  
        ✔ Laptop comparison  
        ✔ Amazon & Flipkart redirect  
        """
    )

# ---------------- SMART RECOMMENDATION ----------------
elif page == "🤖 Smart Recommendation":
    st.header("🤖 AI Laptop Recommendation")

    usage = st.selectbox("Usage", df["Usage"].unique())
    ram = st.selectbox("RAM", df["RAM"].unique())
    storage = st.selectbox("Storage", df["Storage"].unique())

    if st.button("🔍 Recommend"):
        u = le_usage.transform([usage])[0]
        r = le_ram.transform([ram])[0]
        s = le_storage.transform([storage])[0]

        user_vector = [[u, r, s]]
        scores = cosine_similarity(user_vector, features)[0]

        df["Score"] = scores
        results = df.sort_values("Score", ascending=False).head(3)

        for _, row in results.iterrows():
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(row["Image"], width=180)

            with col2:
                st.subheader(row["Model"])
                st.write(f"💰 Price: ₹{row['Price']}")
                st.write(f"🧠 Usage: {row['Usage']}")
                st.write(f"💾 RAM: {row['RAM']} | Storage: {row['Storage']}")

                st.markdown(
                    f"""
                    <a href="{row['Amazon']}" target="_blank">
                    <button style="background:#FF9900;color:white;padding:8px;border:none;border-radius:5px;">
                    Amazon
                    </button></a>
                    &nbsp;
                    <a href="{row['Flipkart']}" target="_blank">
                    <button style="background:#2874F0;color:white;padding:8px;border:none;border-radius:5px;">
                    Flipkart
                    </button></a>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("---")

# ---------------- COMPARE ----------------
elif page == "🔄 Compare Laptops":
    st.header("🔄 Compare Laptops")

    l1 = st.selectbox("Laptop 1", df["Model"])
    l2 = st.selectbox("Laptop 2", df["Model"], index=1)

    if st.button("⚖ Compare"):
        a = df[df["Model"] == l1].iloc[0]
        b = df[df["Model"] == l2].iloc[0]

        col1, col2 = st.columns(2)

        for col, lap in zip([col1, col2], [a, b]):
            with col:
                st.image(lap["Image"], width=220)
                st.subheader(lap["Model"])
                st.write(f"💰 ₹{lap['Price']}")
                st.write(f"🎯 {lap['Usage']}")
                st.write(f"💾 {lap['RAM']} | {lap['Storage']}")

                st.markdown(
                    f"""
                    <a href="{lap['Amazon']}" target="_blank">
                    <button style="background:#FF9900;color:white;padding:8px;border:none;border-radius:5px;">
                    Amazon
                    </button></a>
                    &nbsp;
                    <a href="{lap['Flipkart']}" target="_blank">
                    <button style="background:#2874F0;color:white;padding:8px;border:none;border-radius:5px;">
                    Flipkart
                    </button></a>
                    """,
                    unsafe_allow_html=True
                )
