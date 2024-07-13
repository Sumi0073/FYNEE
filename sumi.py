import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


file_path = 'D:/one/Desktop/in/mcdonalds.csv'

# Reading the data
mcdonalds = pd.read_csv(file_path)
print("Columns in the dataset:")
print(mcdonalds.columns)
print("\nShape of the dataset:")
print(mcdonalds.shape)
print("\nFirst 3 rows of the dataset:")
print(mcdonalds.head(3))

# Selecting the first 11 columns and converting "Yes" to 1 and others to 0
SS = mcdonalds.iloc[:, 0:11]
SS = (SS == "Yes").astype(int)
print("\nMean values rounded to 2 decimal places:")
print(SS.mean().round(2))

# Performing PCA
pca = PCA()
SS_pca = pca.fit_transform(SS)
print("\nCumulative explained variance ratio:")
print(pca.explained_variance_ratio_.cumsum())
print("\nPCA components rounded to 1 decimal place:")
print(np.round(pca.components_, 1))

# Plotting PCA components with arrows and labels
plt.scatter(SS_pca[:, 0], SS_pca[:, 1], color='grey')
for i, (comp1, comp2) in enumerate(zip(pca.components_[0], pca.components_[1])):
    plt.arrow(0, 0, comp1*0.1, comp2*0.1, color='r', alpha=0.5)
    plt.text(comp1*0.15, comp2*0.15, SS.columns[i], color='g', ha='center', va='center')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Components with Feature Labels')
plt.show()

# K-means clustering and inertia plot
kmeans_models = [KMeans(n_clusters=i, n_init=10, random_state=1234).fit(SS) for i in range(2, 9)]
inertia = [model.inertia_ for model in kmeans_models]

plt.plot(range(2, 9), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of within cluster distances')
plt.title('Elbow Method for Optimal k')
plt.show()

# Bootstrap example
np.random.seed(1234)
nboot = 100  
nrep = 10  

# Generate bootstrap samples
bootstrap_samples = []
for _ in range(nboot):
    bootstrap_sample = resample(SS, random_state=1234)
    bootstrap_samples.append(bootstrap_sample)

# Calculate adjusted Rand index for different numbers of clusters
adjusted_rand_index = []
num_clusters = range(2, 9)
for k in num_clusters:
    stability_scores = []
    for bootstrap_sample in bootstrap_samples:
        kmeans = KMeans(n_clusters=k, n_init=nrep, random_state=1234)
        kmeans.fit(bootstrap_sample)
        cluster_labels = kmeans.predict(bootstrap_sample)
        true_labels = kmeans.predict(SS)
        stability_score = adjusted_rand_score(true_labels, cluster_labels)
        stability_scores.append(stability_score)
    adjusted_rand_index.append(stability_scores)

# Transpose the adjusted_rand_index list
adjusted_rand_index = np.array(adjusted_rand_index).T

# Create boxplot of adjusted Rand index
plt.boxplot(adjusted_rand_index, labels=num_clusters)
plt.xlabel("Number of clusters")
plt.ylabel("Adjusted Rand Index")
plt.title("Bootstrap Flexclust")
plt.show()

# Information Criteria Plot
k_values = range(2, 9)
SS_m28 = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=1234)
    model.fit(SS.values)
    iter_val = model.n_iter_
    converged = True
    k_val = k
    k0_val = k
    log_likelihood = -model.inertia_
    n_samples, _ = SS.shape
    aic = -2 * log_likelihood + 2 * k
    bic = -2 * log_likelihood + np.log(n_samples) * k
    labels = model.labels_
    counts = np.bincount(labels)
    probs = counts / float(counts.sum())
    class_entropy = entropy(probs)
    icl = bic - class_entropy
    
    SS_m28.append((iter_val, converged, k_val, k0_val, log_likelihood, aic, bic, icl))

SS_m28 = pd.DataFrame(SS_m28, columns=['iter', 'converged', 'k', 'k0', 'logLik', 'AIC', 'BIC', 'ICL'])

num_segments = SS_m28["k"]
AIC_values = SS_m28["AIC"]
BIC_values = SS_m28["BIC"]
ICL_values = SS_m28["ICL"]

plt.plot(num_segments, AIC_values, marker='o', label='AIC')
plt.plot(num_segments, BIC_values, marker='o', label='BIC')
plt.plot(num_segments, ICL_values, marker='o', label='ICL')

plt.xlabel('Number of Segments')
plt.ylabel('Value of Information Criteria')
plt.title('Information Criteria (AIC, BIC, ICL)')
plt.legend()
plt.grid(True)

plt.show()

# Gaussian Mixture Model
k = 4
kmeans = KMeans(n_clusters=k, random_state=1234)
kmeans.fit(SS)
kmeans_clusters = kmeans.predict(SS)

gmm = GaussianMixture(n_components=k, random_state=1234)
gmm.fit(SS)
gmm_clusters = gmm.predict(SS)

results = pd.DataFrame({'kmeans': kmeans_clusters, 'mixture': gmm_clusters})

SS_m4 = SS[results['mixture'] == 3]

k4_m4 = KMeans(n_clusters=k, random_state=1234)
k4_m4.fit(SS_m4)
k4_m4_clusters = k4_m4.predict(SS_m4)

results_m4 = pd.DataFrame({'kmeans': k4_m4_clusters, 'mixture': 3})

print(pd.crosstab(results['kmeans'], results['mixture']))
print(pd.crosstab(results['kmeans'], results_m4['kmeans']))

# PCA Plot of KMeans Clusters
pca = PCA(n_components=2)
SS_pca = pca.fit_transform(SS)

plt.figure(figsize=(8, 6))
plt.scatter(SS_pca[:, 0], SS_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('PCA Plot of KMeans Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()
