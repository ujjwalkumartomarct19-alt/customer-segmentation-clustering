import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Title
st.title("Customer Segmentation using Clustering")

# -------------------------------
# DATA UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# Load uploaded data
df = pd.read_csv(uploaded_file)

# -------------------------------
# BASIC PREPROCESSING
# -------------------------------
# Select numeric columns only
numeric_df = df.select_dtypes(include='number').dropna()

# Feature engineering (safe checks)
if 'Year_Birth' in numeric_df.columns:
    numeric_df['Age'] = 2026 - numeric_df['Year_Birth']

if all(col in numeric_df.columns for col in [
    'MntWines','MntFruits','MntMeatProducts',
    'MntFishProducts','MntSweetProducts','MntGoldProds'
]):
    numeric_df['TotalSpend'] = numeric_df[
        ['MntWines','MntFruits','MntMeatProducts',
         'MntFishProducts','MntSweetProducts','MntGoldProds']
    ].sum(axis=1)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("Navigation")
option = st.sidebar.selectbox(
    "Select Section",
    ["Dataset Overview", "EDA", "Cluster Analysis"]
)

# -------------------------------
# DATASET OVERVIEW
# -------------------------------
if option == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write("Number of Rows:", df.shape[0])
    st.write("Number of Columns:", df.shape[1])
    st.dataframe(df.head())

# -------------------------------
# EDA
# -------------------------------
elif option == "EDA":
    st.subheader("Exploratory Data Analysis")

    st.write("### Distribution of Numerical Features")
    fig, ax = plt.subplots()
    numeric_df.hist(bins=20, ax=ax)
    st.pyplot(fig)

    st.write("### Boxplot for Outlier Detection")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(data=numeric_df, ax=ax)
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# -------------------------------
# CLUSTER ANALYSIS
# -------------------------------
elif option == "Cluster Analysis":
    st.subheader("Cluster Analysis")

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    # Select number of clusters
    k = st.slider("Select number of clusters", 2, 8, 4)

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df['Final_Cluster'] = clusters

    st.write("### Cluster-wise Summary")
    cluster_profile = df.groupby('Final_Cluster').mean(numeric_only=True)
    st.dataframe(cluster_profile)

    selected_cluster = st.selectbox(
        "Select Cluster to View Customers",
        df['Final_Cluster'].unique()
    )

    st.write(f"Customers in Cluster {selected_cluster}")
    st.dataframe(df[df['Final_Cluster'] == selected_cluster])
