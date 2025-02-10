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
