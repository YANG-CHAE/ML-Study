#9

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X, y = mnist["data"], mnist["target"]

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 10000, random_state = 42)

train_x.shape

print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

import time
start = time.time()

clf = RandomForestClassifier(n_estimators=20, max_depth=5,random_state=0)
clf.fit(train_x,train_y)

print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

predict1 = clf.predict(test_x)
print(accuracy_score(test_y,predict1))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

pca_a = PCA(n_components = d)
X_reduced = pca_a.fit_transform(X)
cumsum_reduced = np.cumsum(pca_a.explained_variance_ratio_)

import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(cumsum)
plt.plot(cumsum_reduced)

#10
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]


y = y.astype(int)  

random_indices = np.random.choice(X.shape[0],  5000 , replace=False)
X_sample, y_sample = X[random_indices], y[random_indices]


tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X_sample)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sample)


mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X_sample)

def plot_scatter(X_transformed, title, ax):
    scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_sample, cmap='tab10', alpha=0.6, edgecolors='k')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return scatter

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

plot_scatter(X_tsne, "t-SNE", axes[0, 0])
plot_scatter(X_pca, "PCA", axes[0, 1])
plot_scatter(X_lle, "LLE", axes[1, 0])
plot_scatter(X_mds, "MDS", axes[1, 1])

plt.tight_layout()
plt.show()










