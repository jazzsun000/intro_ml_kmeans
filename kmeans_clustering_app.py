import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.title("K-means Clustering Demonstrator")

# Sidebar settings
st.sidebar.header("K-means Settings")
num_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300)
num_features = st.sidebar.slider("Number of Features", 2, 5, 2)
num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
max_iter = st.sidebar.slider("Max Iterations", 10, 300, 100)

# Generate synthetic data
X, _ = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, random_state=42)

# Run K-means
kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iter, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Plotting the K-means clustering
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x', label='Centroids')
plt.legend()
plt.grid(True)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
st.pyplot(fig)

# Function to calculate the Within-Cluster Sum of Square (WCSS)
def calculate_wcss(data):
    wcss = []
    for n in range(2, 11):
        kmeans = KMeans(n_clusters=n, max_iter=max_iter, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss

# Elbow method to determine the optimal number of clusters
st.subheader("Elbow Method")
wcss = calculate_wcss(X)
fig, ax = plt.subplots()
ax.plot(range(2, 11), wcss)
ax.set_xlabel('Number of clusters')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Silhouette analysis
st.subheader("Silhouette Analysis")
range_n_clusters = range(2, 11)
silhouette_avg = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg.append(silhouette_score(X, cluster_labels))

fig, ax = plt.subplots()
ax.plot(range_n_clusters, silhouette_avg)
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Average Silhouette Score')
st.pyplot(fig)

st.write("The silhouette score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. "
         "Scores around zero indicate overlapping clusters.")
