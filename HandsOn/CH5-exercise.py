#8
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # 꽃잎 길이, 꽃잎 넓이
y = iris["target"]

lin_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
])
lin_clf.fit(X, y)

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="linear",C=1)),
])
svm_clf.fit(X, y)

sgd_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDClassifier(loss="hinge", learning_rate="constant", eta0=0.001, alpha=alpha,
                        max_iter=1000, tol=1e-3, random_state=42)),
])
sgd_clf.fit(X, y)

lin_model = lin_clf[-1]  
svm_model = svm_clf[-1]  
sgd_model = sgd_clf[-1] 


lin_scaler = lin_clf.named_steps["scaler"]
svm_scaler = svm_clf.named_steps["scaler"]
sgd_scaler = sgd_clf.named_steps["scaler"]


w1 = -lin_model.coef_[0, 0] / lin_model.coef_[0, 1]
b1 = -lin_model.intercept_[0] / lin_model.coef_[0, 1]


w2 = -svm_model.coef_[0, 0] / svm_model.coef_[0, 1]
b2 = -svm_model.intercept_[0] / svm_model.coef_[0, 1]


w3 = -sgd_model.coef_[0, 0] / sgd_model.coef_[0, 1]
b3 = -sgd_model.intercept_[0] / sgd_model.coef_[0, 1]


line1 = lin_scaler.inverse_transform([
    [-10, -10 * w1 + b1],
    [ 10,  10 * w1 + b1]
])
line2 = svm_scaler.inverse_transform([
    [-10, -10 * w2 + b2],
    [ 10,  10 * w2 + b2]
])
line3 = sgd_scaler.inverse_transform([
    [-10, -10 * w3 + b3],
    [ 10,  10 * w3 + b3]
])


plt.figure(figsize=(11, 4))

# 결정 경계
plt.plot(line1[:, 0], line1[:, 1], "k:", label="LinearSVC")
plt.plot(line2[:, 0], line2[:, 1], "b--", linewidth=2, label="SVC")
plt.plot(line3[:, 0], line3[:, 1], "r-", label="SGDClassifier")


plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")

plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper center", fontsize=14)
plt.axis([0, 5.5, 0, 2])  # 범위는 상황에 따라 조정
plt.show()
