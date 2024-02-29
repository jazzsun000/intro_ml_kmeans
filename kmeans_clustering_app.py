import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.title("K-means Clustering Demonstrator")

# Adjusted text to reflect changes
st.write("""
    Welcome to the K-means Clustering Demonstrator. This app lets you visualize how the K-means algorithm 
    groups data into clusters with varying degrees of variance and demonstrates its limitations with 
    non-convex cluster shapes.
""")

# Sidebar settings with explanations
st.sidebar.header("K-means Settings")
num_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300, help="Adjust the total number of data points.")
cluster_std = st.sidebar.slider("Cluster Std Dev", 0.5, 2.5, 1.0, help="Standard deviation of clusters. Higher values make clusters more spread out.")
dataset_type = st.sidebar.selectbox("Dataset Type", ["Blobs", "Moons"], help="Choose the type of dataset to generate.")

num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3, help="Select the number of clusters for 'Blobs' dataset only.")
max_iter = st.sidebar.slider("Max Iterations", 10, 300, 100, help="Maximum number of iterations for the K-means algorithm.")

# Generate synthetic data based on selected dataset type
if dataset_type == "Blobs":
    X, _ = make_blobs(n_samples=num_samples, centers=num_clusters, cluster_std=cluster_std, n_features=2, random_state=42)
elif dataset_type == "Moons":
    X, _ = make_moons(n_samples=num_samples, noise=cluster_std, random_state=42)
    num_clusters = 2  # Moons dataset naturally forms 2 clusters

# Run K-means
kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iter, random_state=42).fit(X)
labels = kmeans.labels_

# Plotting the K-means clustering result
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', label='Data Points')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x', label='Centroids')
ax.legend()
ax.grid(True)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
st.pyplot(fig)

# Function to calculate WCSS and silhouette scores
def calculate_metrics(X, range_n_clusters):
    wcss = []
    silhouette_scores = []
    for n in range_n_clusters:
        kmeans = KMeans(n_clusters=n, max_iter=max_iter, random_state=42).fit(X)
        wcss.append(kmeans.inertia_)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X, labels))
    return wcss, silhouette_scores

# Determine the optimal number of clusters based on silhouette score
range_n_clusters = range(2, 11)
wcss, silhouette_scores = calculate_metrics(X, range_n_clusters)
optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2

# Plotting Elbow Method
st.subheader("Elbow Method")
fig, ax = plt.subplots()
ax.plot(range_n_clusters, wcss, marker='o')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('WCSS')
ax.axvline(x=optimal_clusters, linestyle='--', color='r', label='Optimal Clusters')
ax.legend()
st.pyplot(fig)

# Plotting Silhouette Analysis
st.subheader("Silhouette Analysis")
fig, ax = plt.subplots()
ax.plot(range_n_clusters, silhouette_scores, marker='o')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Silhouette Score')
ax.axvline(x=optimal_clusters, linestyle='--', color='r', label='Optimal Clusters')
ax.legend()
st.pyplot(fig)

st.write(f"Optimal Number of Clusters based on Silhouette Score: {optimal_clusters}")
