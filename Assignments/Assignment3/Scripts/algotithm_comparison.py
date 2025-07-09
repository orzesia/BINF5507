### import the importables ###
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.datasets import make_moons, make_blobs, make_circles
import matplotlib.pyplot as plt


### preview datasets ###
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X_blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=42)
X_circles, _ = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)

def subplot(no, X, title):
    axs[no].scatter(X[:,0], X[:, 1], c='black')
    axs[no].set_title(title)
    axs[no].set_xticks([])
    axs[no].set_yticks([])


fig, axs = plt.subplots(1, 3, figsize=(15, 4))
subplot(0,X_moons, "make_moons")
subplot(1,X_blobs, "make_blobs")
subplot(2,X_circles, "make_circles")

plt.tight_layout()
plt.show()

### Clustering visualization ###

## dbscan

dbscan_moons = DBSCAN(eps=0.3, min_samples=5).fit_predict(X_moons)
dbscan_blobs = DBSCAN(eps=1, min_samples=5).fit_predict(X_blobs)
dbscan_circles = DBSCAN(eps=0.3, min_samples=3).fit_predict(X_circles)

# dbscan moons
plt.figure(figsize=(8, 6))
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=dbscan_moons, cmap = 'coolwarm')
plt.title('DBSCAN make_moons')
plt.show()

# dbscan blobs
mask = dbscan_blobs != -1
# Plot clustered points
plt.scatter(X_blobs[mask, 0], X_blobs[mask, 1], c=dbscan_blobs[mask], cmap='coolwarm', label='Clusters')
# Plot outliers
plt.scatter(X_blobs[~mask, 0], X_blobs[~mask, 1], c='black', label='Outliers', marker='x')
plt.legend()
plt.title("DBSCAN on make_blobs")
plt.show()

# dbscan circles
plt.figure(figsize=(8, 6))
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=dbscan_circles, cmap = 'coolwarm')
plt.title('DBSCAN make_circles')
plt.show()

# k-means

kmeans_moons = KMeans(n_clusters=2, random_state=42).fit_predict(X_moons)
kmeans_blobs = KMeans(n_clusters=3, random_state=42).fit_predict(X_blobs)
kmeans_circles = KMeans(n_clusters=2, random_state=42).fit_predict(X_circles)

# kmeans moons
plt.figure(figsize=(8, 6))
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=kmeans_moons, cmap = 'coolwarm')
plt.title('k-Means make_moons')
plt.show()

# kmeans blobs
plt.figure(figsize=(8, 6))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=kmeans_blobs, cmap = 'tab10')
plt.title('k-Means make_blobs')
plt.show()

# kmeans circles
plt.figure(figsize=(8, 6))
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=kmeans_circles, cmap = 'coolwarm')
plt.title('k-Means make_circles')
plt.show()


# agglomerative clustering

agglo_moons = AgglomerativeClustering(n_clusters=2).fit_predict(X_moons)
agglo_blobs = AgglomerativeClustering(n_clusters=3).fit_predict(X_blobs)
agglo_circles = AgglomerativeClustering(n_clusters=2).fit_predict(X_circles)

# agglomerative moons
plt.figure(figsize=(8, 6))
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=agglo_moons, cmap = 'coolwarm')
plt.title('Agglomerative make_moons')
plt.show()

# agglomerative blobs
plt.figure(figsize=(8, 6))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=agglo_blobs, cmap = 'coolwarm')
plt.title('Agglomerative make_blobs')
plt.show()

# agglomerative circles
plt.figure(figsize=(8, 6))
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=agglo_circles, cmap = 'coolwarm')
plt.title('Agglomerative make_circles')
plt.show()