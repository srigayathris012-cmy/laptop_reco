import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Laptop Recommendation System",
    page_icon="💻",
    layout="wide"
)

# ---------- CUSTOM CSS (COLORED THEME - NOT BLACK) ----------
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #f5f7ff, #eef2ff);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1e3a8a;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Headings */
h1 {
    color: #1e3a8a;
}
h2, h3 {
    color: #334155;
}

/* Info box */
.custom-box {
    background-color: #e0f2fe;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #2563eb;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HOME PAGE CONTENT ----------
st.markdown("## 💻 Laptop Recommendation System")
st.markdown("### Smart Laptop Suggestions using Machine Learning")

st.markdown("""
🚀 **What this app does:**
- Recommends laptops using **Machine Learning (KNN Algorithm)**
- Uses **real Kaggle laptop dataset**
- Helps users choose laptops easily
- Works based on **budget & specifications**
""")

st.markdown("""
<div class="custom-box">
👉 <b>Use the left sidebar</b> to navigate between pages:<br>
🔍 Recommend Laptop<br>
📊 Dataset View<br>
ℹ️ About Project
</div>
""", unsafe_allow_html=True)

st.success("Designed with ❤️ | Colored UI | Real website feel")
