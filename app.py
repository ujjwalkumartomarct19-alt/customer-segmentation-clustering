import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Generic Clustering App", layout="wide")
st.title("Generic Customer Segmentation using Clustering")

# Upload file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write("Rows:", df.shape[0], "| Columns:", df.shape[1])
    st.dataframe(df.head())

    # Select numeric columns
    numeric_df = df.select_dtypes(include='number')

    if numeric_df.shape[1] < 2:
        st.error("Dataset must contain at least 2 numeric columns for clustering.")
        st.stop()

    st.subheader("Using Numeric Features Only")
    st.write(numeric_df.columns.tolist())

    # Drop missing values
    numeric_df = numeric_df.dropna()

    # Distribution
    st.subheader("Distribution of Numeric Features")
    numeric_df.hist(bins=20, figsize=(10,6))
    st.pyplot(plt.gcf())
    plt.clf()

    # Boxplot
    st.subheader("Boxplot (Outlier Detection)")
    sns.boxplot(data=numeric_df)
    st.pyplot(plt.gcf())
    plt.clf()

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    sns.heatmap(numeric_df.corr(), cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    # Number of clusters
    k = st.slider("Select number of clusters (K)", 2, 8, 4)

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled)
    kmeans_score = silhouette_score(X_scaled, labels_kmeans)

    # Hierarchical
    hierarchical = AgglomerativeClustering(n_clusters=k)
    labels_hier = hierarchical.fit_predict(X_scaled)
    hier_score = silhouette_score(X_scaled, labels_hier)

    # Comparison
    st.subheader("Model Comparison")
    comparison = pd.DataFrame({
        "Model": ["KMeans", "Hierarchical"],
        "Silhouette Score": [kmeans_score, hier_score]
    })
    st.dataframe(comparison)

    # Final model selection
    if kmeans_score >= hier_score:
        st.success("KMeans selected as Final Model")
        df["Final_Cluster"] = labels_kmeans
    else:
        st.success("Hierarchical selected as Final Model")
        df["Final_Cluster"] = labels_hier

    # Show clustered data
    st.subheader("Clustered Dataset Preview")
    st.dataframe(df.head())

else:
    st.info("Please upload a CSV file to begin.")
