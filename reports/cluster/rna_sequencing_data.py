import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

# Set parameters
sc.settings.verbosity = 3
sc.logging.print_versions()
results_file = './write/pbmc3k.h5ad'

# Load data
adata = sc.read_10x_mtx('./data/filtered_gene_bc_matrices/hg19/',
                        var_names='gene_symbols', cache=True)
adata.var_names_make_unique()

############# Preprocessing #############

sc.pl.highest_expr_genes(adata, n_top=20)

# filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

mito_genes = adata.var_names.str.startswith('MT-')
adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X,
                                                                               axis=1).A1
adata.obs['n_counts'] = adata.X.sum(axis=1).A1

adata = adata[adata.obs['n_genes'] < 2500, :]
adata = adata[adata.obs['percent_mito'] < 0.05, :]
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)
adata.raw = adata
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(adata)

adata = adata[:, adata.var['highly_variable']]
sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])
sc.pp.scale(adata, max_value=10)

############# PCA to 40 components #################

sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color='CST3')
sc.pl.pca_variance_ratio(adata, log=True)
adata.write(results_file)

# neighborhood graph
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
# sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP'])
# sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP'], use_raw=False)

############# clustering with Louvian method ###########

sc.tl.louvain(adata)
sc.pl.umap(adata, color=['louvain', 'CST3', 'NKG7'])
adata.write(results_file)

############# Extracting 2D data ###############

data = adata.obsm.X_umap
fig = plt.figure()
plt.scatter(data[:, 0], data[:, 1], s=15, alpha=0.1)
plt.xlabel('UMAP1', fontsize=15)
plt.ylabel('UMAP2', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# Number of samples
N = len(data)


############# quantum clustering for 2D data ######

def norm(data):
    mean = np.ones(np.shape(data)) * np.average(data, axis=0)
    sd = np.ones(np.shape(data)) * np.std(data, axis=0)
    return np.divide(data - mean, sd)


# The wavefunction
def Psi(x):
    return -np.sum(np.exp(
        -np.sum(np.square(np.repeat([x], N, axis=0) - data), axis=1) / (
                    2 * sigma ** 2)))


# The potential
def V(x):
    s = np.sum(np.square(np.repeat([x], N, axis=0) - data), axis=1)
    return np.sum(np.multiply(s, np.exp(-s / (2 * sigma ** 2)))) / abs(Psi(x))


def dV(x):
    dx2 = np.sum(np.square(np.repeat([x], N, axis=0) - data), axis=1)
    exp = np.exp(-dx2 / (2 * sigma ** 2))
    dv1 = np.multiply(dx2, exp).dot(np.repeat([x], N, axis=0) - data)
    dv2 = exp.dot(np.repeat([x], N, axis=0) - data)
    return -(1.0 / sigma ** 2) * dv1 + (2 + V(x) / sigma ** 2) * dv2


# clustering algorithm
def classify(func):
    predict = np.zeros(N)
    for i in range(N):
        predict[i] = np.sum(
            np.around(fmin_bfgs(func, data[i], gtol=0.001, disp=False, fprime=dV),
                      decimals=1))
        print(i)
    return predict


# Add labels to the data in PD format
def add_labels(adata, text, labels):
    adata.obs[text] = pd.Categorical(values=labels.astype('U'),
                                     categories=np.unique(labels).astype('U'))


# perform clustering
data = adata.obsm.X_umap
sigma = 1
N = len(data)
predict = classify(V)
predicted_classes = np.sort(np.unique(predict))

colors = np.zeros(N)
for i in range(N):
    colors[i] = np.where(predicted_classes == predict[i])[0]

fig = plt.figure()
plt.scatter(data[:, 0], data[:, 1], cmap='tab10', c=colors, alpha=1)
plt.colorbar()
plt.xlabel('UMAP1', fontsize=15)
plt.ylabel('UMAP2', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

########## other clustering methods ######

# Plot the data with K Means Labels
from sklearn.cluster import KMeans

X = data
kmeans = KMeans(8, random_state=0)
labels = kmeans.fit(X).predict(X)
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
plt.show()

# Gaussian mixture
from sklearn import mixture

gmm = mixture.GaussianMixture(n_components=8, covariance_type='full').fit(X)
labels = gmm.predict(X)
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
plt.show()

# Bayesian Gaussian Mixture
from sklearn import mixture

gmm = mixture.BayesianGaussianMixture(n_components=30, n_init=10,
                                      covariance_type='diag').fit(X)
labels = gmm.predict(X)
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
plt.show()

# Louvian graph
import anndata as ad

a = ad.AnnData(data)
sc.pp.neighbors(a)
sc.tl.louvain(a)

c = np.zeros(N)
for i in range(N):
    c[i] = int(a.obs.louvain[i])

fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
plt.show()

