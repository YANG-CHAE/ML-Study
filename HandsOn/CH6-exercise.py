#7
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import numpy as np
from scipy.stats import mode
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)

grid_search_cv.fit(X_train, y_train)

grid_search_cv.best_estimator_

y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)

#8
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

rs = ShuffleSplit(n_splits=1000, train_size = 100, test_size=None, random_state=0)

trees = []
best_params_ = grid_search_cv.best_params_

for train_idx, _ in rs.split(X_train):
    # train_idx: (훈련세트 내) 무작위로 추출된 100개의 샘플 인덱스
    X_subset = X_train[train_idx]
    y_subset = y_train[train_idx]
    
    # 3) 최적 파라미터로 결정트리를 훈련
    tree_clf = DecisionTreeClassifier(**best_params_, random_state=42)
    tree_clf.fit(X_subset, y_subset)
    
    trees.append(tree_clf)
  
all_predictions = []


for tree in trees:
    pred = tree.predict(X_test)
    all_predictions.append(pred)

all_predictions = np.array(all_predictions)


majority_votes = mode(all_predictions, axis=0)
final_predictions = majority_votes.mode.ravel()  # shape: (테스트샘플 개수,)


accuracy = accuracy_score(y_test, final_predictions)
print("최종 앙상블 정확도:", accuracy)
