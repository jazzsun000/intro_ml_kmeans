import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.title("K-means Clustering Demonstrator")

st.write("""
    Welcome to the K-means Clustering Demonstrator. This app lets you visualize how the K-means algorithm 
    groups data into clusters. Use the sliders to change the number of samples, features, clusters, 
    and iterations, then observe how these changes affect the clustering process and the resulting visualizations.
""")

# Sidebar settings with explanations
st.sidebar.header("K-means Settings")
st.sidebar.markdown("""
    **Number of Samples**: Adjust the total number of data points in the dataset.
""")
num_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300)

st.sidebar.markdown("""
    **Number of Features**: Set the number of features that each data point will have.
""")
num_features = st.sidebar.slider("Number of Features", 2, 5, 2)

st.sidebar.markdown("""
    **Number of Clusters**: Select the number of clusters to form.
""")
num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)

st.sidebar.markdown("""
    **Max Iterations**: Define the maximum number of iterations for the K-means algorithm.
""")
max_iter = st.sidebar.slider("Max Iterations", 10, 300, 100)

st.markdown("""
    **Data Generation**  
    The dataset is synthetically generated with the selected number of samples and features. 
    The `make_blobs` function from `sklearn` is used to create a dataset for clustering.
""")

# Generate synthetic data
X, _ = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, random_state=42)

# Run K-means
kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iter, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Plotting the K-means clustering
st.subheader("K-means Clustering Result")
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x', label='Centroids')
plt.legend()
plt.grid(True)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
st.pyplot(fig)

st.write("""
    The plot above displays the results of K-means clustering. Each point represents a data sample, 
    and the color indicates the cluster it belongs to. The red 'X' marks represent the centroids 
    of each cluster.
""")

# Function to calculate the Within-Cluster Sum of Square (WCSS)
def calculate_wcss(data):
    wcss = []
    for n in range(2, 11):
        kmeans = KMeans(n_clusters=n, max_iter=max_iter, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss

# Elbow method to determine the optimal number of clusters
st.subheader("Elbow Method for Optimal Number of Clusters")
st.write("""
    The elbow method plot below shows the Within-Cluster Sum of Square (WCSS) against 
    the number of clusters. The 'elbow' point is often considered as an indicator of the 
    optimal number of clusters.
""")
wcss = calculate_wcss(X)
fig, ax = plt.subplots()
ax.plot(range(2, 11), wcss)
ax.set_xlabel('Number of clusters')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Silhouette analysis
st.subheader("Silhouette Analysis")
st.write("""
    Silhouette analysis measures how similar an object is to its own cluster compared to other clusters.
    The silhouette score ranges from -1 to +1, where a high value indicates that the object is well matched 
    to its own cluster and poorly matched to neighboring clusters.
""")
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

# Show current settings used to create the data
st.sidebar.markdown("""
    **Current Settings:**  
    - Number of Samples: {}
    - Number of Features: {}
    - Number of Clusters: {}
    - Max Iterations: {}
""".format(num_samples, num_features, num_clusters, max_iter))
