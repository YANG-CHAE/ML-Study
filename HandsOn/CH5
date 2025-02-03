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

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # 꽃잎 길이, 꽃잎 넓이
y = iris["target"]

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
])
svm_clf.fit(X, y)

svm_clf.predict([[5.5, 1.7]])

y_pred_svm = svm_clf.predict(X_test)  # 테스트 세트에 대한 예측
accuracy_svm = accuracy_score(y_test, y_pred_svm)

def plot_dataset(X, y):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")   # 파랑 네모: 음성 데이터
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")   # 초록 세모: 양성 데이터

    plt.grid(True, which='both')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=20, loss="hinge"))
])
polynomial_svm_clf.fit(X,y)


plot_dataset(X, y)
plt.show()

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)  
    X = np.c_[x0.ravel(), x1.ravel()] 
    
    y_pred = clf.predict(X).reshape(x0.shape)

    y_decision = clf.decision_function(X).max(axis=1).reshape(x0.shape)  
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)      
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)  # 등고선 그리기


# 그래프 축 범위 정의
axes = [0, 7, 0, 3] 


plot_predictions(polynomial_svm_clf, axes) 
plot_dataset(X, y)


plt.show()


y_pred_poly = polynomial_svm_clf.predict(X_test)
accuracy_poly = accuracy_score(y_test, y_pred_poly)

print("LinearSVC 정확도:", accuracy_svm)
print("Polynomial SVC 정확도:", accuracy_poly)
