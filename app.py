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
# SELECT KEY FEATURES FOR EDA
# -------------------------------
key_features = [
    col for col in [
        'Income',
        'Recency',
        'Age',
        'TotalSpend',
        'NumWebPurchases',
        'NumStorePurchases'
    ] if col in numeric_df.columns
]

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
# EDA (CLEAN VERSION)
# -------------------------------
elif option == "EDA":
    st.subheader("Exploratory Data Analysis")

    # ---------------- Distribution ----------------
    st.write("### Distribution of Key Numerical Features")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, col in enumerate(key_features):
        axes[i].hist(numeric_df[col], bins=20)
        axes[i].set_title(col)

    # Remove empty plots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

    # ---------------- Boxplot ----------------
    st.write("### Boxplot for Outlier Detection")

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=numeric_df[key_features])
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

    # ---------------- Heatmap ----------------
    st.write("### Correlation Heatmap")

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        numeric_df[key_features].corr(),
        annot=True,
        cmap='coolwarm'
    )
    st.pyplot(plt.gcf())
    plt.clf()

# -------------------------------
# CLUSTER ANALYSIS
# -------------------------------
elif option == "Cluster Analysis":
    st.subheader("Cluster Analysis")

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df[key_features])

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
