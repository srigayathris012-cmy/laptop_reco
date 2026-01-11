import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="LaptopFinder AI",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for unique styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6C63FF;
        --secondary-color: #4CAF50;
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .laptop-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .laptop-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Search bar styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #667eea;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    /* Price badge */
    .price-badge {
        background: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    /* Spec badge */
    .spec-badge {
        background: #f0f0f0;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("laptop.csv")
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    
    df_display = df.copy()
    
    # Clean price
    df["Price"] = df["Price"].str.replace("₹", "").str.replace(",", "").astype(int)
    df["Ram_GB"] = df["Ram"].str.extract(r"(\d+)").fillna(0).astype(int)
    df["SSD_GB"] = df["SSD"].str.extract(r"(\d+)").fillna(0).astype(int)
    
    # Graphics flag
    def graphics_flag(x):
        if "Intel" in str(x) or "UHD" in str(x):
            return 0
        else:
            return 1
    
    df["Graphics_Flag"] = df["Graphics"].apply(graphics_flag)
    df["Rating"] = df["Rating"].fillna(df["Rating"].mean())
    
    return df, df_display

df, df_display = load_data()

# Train KNN model
@st.cache_resource
def train_model(df):
    X = df[["Price", "Ram_GB", "SSD_GB", "Rating", "Graphics_Flag"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn.fit(X_scaled)
    
    return knn, scaler

knn, scaler = train_model(df)

# Sidebar navigation
with st.sidebar:
    st.markdown("<h1 style='color: white; text-align: center;'>💻 Navigation</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "",
        ["🏠 Home", "🔍 Find Laptop", "📊 Analytics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<p style='color: white; text-align: center;'>Powered by AI 🤖</p>", unsafe_allow_html=True)

# Page: Home
if page == "🏠 Home":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💻 LaptopFinder AI</h1>
        <p>Find Your Perfect Laptop with Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{len(df)}</h2>
            <p>Laptops Available</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>₹{df['Price'].min():,}</h2>
            <p>Starting Price</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{df['Ram_GB'].max()} GB</h2>
            <p>Max RAM</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rating = df['Rating'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h2>{avg_rating:.1f}⭐</h2>
            <p>Avg Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features
    st.markdown("## ✨ Why Choose LaptopFinder AI?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Personalized
        Get recommendations based on your specific needs and budget
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 AI-Powered
        Machine learning algorithm finds the best matches for you
        """)
    
    with col3:
        st.markdown("""
        ### ⚡ Fast & Easy
        Find your perfect laptop in seconds, not hours
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # How it works
    st.markdown("## 🔧 How It Works")
    
    steps_col1, steps_col2, steps_col3 = st.columns(3)
    
    with steps_col1:
        st.info("**Step 1: Set Preferences**\nTell us your budget, RAM, storage, and graphics needs")
    
    with steps_col2:
        st.info("**Step 2: AI Analysis**\nOur algorithm analyzes thousands of laptops")
    
    with steps_col3:
        st.info("**Step 3: Get Results**\nReceive top 5 personalized recommendations")

# Page: Find Laptop
elif page == "🔍 Find Laptop":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Find Your Perfect Laptop</h1>
        <p>Enter your preferences and let AI do the rest</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Search bar
    search_query = st.text_input("🔎 Quick Search (Search by model name)", placeholder="e.g., HP Pavilion, Lenovo IdeaPad...")
    
    if search_query:
        filtered = df_display[df_display['Model'].str.contains(search_query, case=False, na=False)]
        st.success(f"Found {len(filtered)} laptops matching '{search_query}'")
        
        for idx, row in filtered.head(10).iterrows():
            st.markdown(f"""
            <div class="laptop-card">
                <h3>{row['Model']}</h3>
                <div class="price-badge">{row['Price']}</div>
                <p>
                    <span class="spec-badge">💾 {row['Ram']}</span>
                    <span class="spec-badge">💿 {row['SSD']}</span>
                    <span class="spec-badge">🎮 {row['Graphics']}</span>
                    <span class="spec-badge">⭐ {row['Rating']}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommendation System
    st.markdown("## 🎯 AI-Powered Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        budget = st.slider("💰 Budget (₹)", 
                          min_value=int(df['Price'].min()), 
                          max_value=int(df['Price'].max()), 
                          value=60000, 
                          step=5000)
        
        ram = st.selectbox("💾 RAM (GB)", [4, 8, 16, 32, 64], index=1)
        
        ssd = st.selectbox("💿 SSD Storage (GB)", [128, 256, 512, 1024, 2048], index=2)
    
    with col2:
        rating = st.slider("⭐ Minimum Rating", 
                          min_value=0, 
                          max_value=100, 
                          value=60, 
                          step=5)
        
        graphics = st.radio("🎮 Graphics Card", 
                           ["Integrated (Intel/UHD)", "Dedicated (NVIDIA/AMD)"],
                           index=1)
        
        graphics_flag = 0 if graphics == "Integrated (Intel/UHD)" else 1
    
    if st.button("🚀 Find My Laptop", use_container_width=True):
        # Get recommendations
        user_input = [[budget, ram, ssd, rating, graphics_flag]]
        user_scaled = scaler.transform(user_input)
        distances, indices = knn.kneighbors(user_scaled)
        
        recommended = df_display.iloc[indices[0]]
        
        st.markdown("### 🎉 Top 5 Recommendations for You")
        
        for i, (idx, row) in enumerate(recommended.iterrows(), 1):
            match_score = max(0, 100 - (distances[0][i-1] * 10))
            
            st.markdown(f"""
            <div class="laptop-card">
                <h3>#{i} {row['Model']}</h3>
                <div class="price-badge">{row['Price']}</div>
                <p style="color: #667eea; font-weight: bold;">Match Score: {match_score:.1f}%</p>
                <p>
                    <span class="spec-badge">💾 {row['Ram']}</span>
                    <span class="spec-badge">💿 {row['SSD']}</span>
                    <span class="spec-badge">🎮 {row['Graphics']}</span>
                    <span class="spec-badge">🖥️ {row['Display']}</span>
                    <span class="spec-badge">⭐ {row['Rating']}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

# Page: Analytics
elif page == "📊 Analytics":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Market Analytics</h1>
        <p>Insights and trends from our laptop database</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Price distribution
    st.markdown("### 💰 Price Distribution")
    fig_price = px.histogram(df, x='Price', nbins=30, 
                             title='Laptop Price Distribution',
                             color_discrete_sequence=['#667eea'])
    fig_price.update_layout(showlegend=False)
    st.plotly_chart(fig_price, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RAM distribution
        st.markdown("### 💾 RAM Distribution")
        ram_counts = df['Ram_GB'].value_counts().sort_index()
        fig_ram = px.pie(values=ram_counts.values, names=ram_counts.index,
                        title='RAM Configuration',
                        color_discrete_sequence=px.colors.sequential.Purples_r)
        st.plotly_chart(fig_ram, use_container_width=True)
    
    with col2:
        # Graphics distribution
        st.markdown("### 🎮 Graphics Card Type")
        graphics_counts = df['Graphics_Flag'].value_counts()
        fig_graphics = px.pie(values=graphics_counts.values, 
                             names=['Integrated', 'Dedicated'],
                             title='Graphics Card Distribution',
                             color_discrete_sequence=['#FFA07A', '#667eea'])
        st.plotly_chart(fig_graphics, use_container_width=True)
    
    # Price vs Rating
    st.markdown("### 📈 Price vs Rating Analysis")
    fig_scatter = px.scatter(df, x='Price', y='Rating', 
                            size='Ram_GB', color='Graphics_Flag',
                            title='Price vs Rating (Size = RAM)',
                            color_discrete_sequence=['#FFA07A', '#667eea'],
                            labels={'Graphics_Flag': 'Graphics Type'})
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Top brands
    st.markdown("### 🏆 Top Laptop Brands")
    df_display['Brand'] = df_display['Model'].str.split().str[0]
    brand_counts = df_display['Brand'].value_counts().head(10)
    
    fig_brands = px.bar(x=brand_counts.index, y=brand_counts.values,
                       title='Top 10 Laptop Brands',
                       labels={'x': 'Brand', 'y': 'Count'},
                       color=brand_counts.values,
                       color_continuous_scale='Purples')
    st.plotly_chart(fig_brands, use_container_width=True)
