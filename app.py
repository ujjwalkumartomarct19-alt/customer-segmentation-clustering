import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Title
st.title("Customer Segmentation using Clustering")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("marketing_campaign.csv")

df = load_data()

# Sidebar
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
    df[['Income','Age','Recency','TotalSpend']].hist(bins=20, ax=ax)
    st.pyplot(fig)

    st.write("### Boxplot for Outlier Detection")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(data=df[['Income','TotalSpend','Recency']], ax=ax)
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(
        df[['Income','Age','Recency','TotalSpend',
            'NumWebPurchases','NumStorePurchases',
            'NumCatalogPurchases','NumDealsPurchases',
            'Children']].corr(),
        cmap='coolwarm',
        ax=ax
    )
    st.pyplot(fig)

# -------------------------------
# CLUSTER ANALYSIS
# -------------------------------
elif option == "Cluster Analysis":
    st.subheader("Cluster Analysis")

    st.write("### Cluster-wise Summary")
    cluster_profile = df.groupby('Final_Cluster').mean()
    st.dataframe(cluster_profile)

    selected_cluster = st.selectbox(
        "Select Cluster to View Customers",
        df['Final_Cluster'].unique()
    )

    st.write(f"Customers in Cluster {selected_cluster}")
    st.dataframe(df[df['Final_Cluster'] == selected_cluster])

