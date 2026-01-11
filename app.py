import streamlit as st

st.set_page_config(
    page_title="Laptop Recommendation System",
    layout="wide",
    page_icon="💻"
)

# Custom CSS (Colored Theme)
st.markdown("""
<style>
body {
    background-color: #f4f6ff;
}
.sidebar .sidebar-content {
    background-color: #1f2c56;
}
h1, h2, h3 {
    color: #1f2c56;
}
</style>
""", unsafe_allow_html=True)

st.title("💻 Laptop Recommendation System")
st.subheader("Smart Laptop Suggestions using Machine Learning")

st.markdown("""
### 🚀 What this app does:
- Recommends laptops using **ML (KNN Algorithm)**
- Uses **real Kaggle dataset**
- Helps users choose laptops easily
- Works based on **budget & specs**

👈 Use the **left sidebar** to navigate.
""")

st.success("Designed with ❤️ | No black theme | Real website feel")
