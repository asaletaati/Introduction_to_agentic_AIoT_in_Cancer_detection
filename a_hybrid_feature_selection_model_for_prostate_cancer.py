import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import f_classif, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# load data
def load_data(file_path):
    data = pd.read_csv(file_path, sep=r'\s+', header=None)
    print("loaded data with spaces")
    
    data = data.iloc[:, 1:]
    
    print(f"data shape: {data.shape}")
    print("first few rows:")
    print(data.head())
    print("labels (last row):")
    print(data.iloc[-1, :])
    
    X = data.iloc[:-1, :].T.to_numpy()
    X = X.astype(float)
    
    y = data.iloc[-1, :].values
    print(f"labels we got: {np.unique(y)}")
    
    y = np.where(y == 'tumor', 1, 0)
    
    print(f"samples: {X.shape[0]}")
    print(f"genes: {X.shape[1]}")
    print(f"tumor count: {np.sum(y)}")
    print(f"normal count: {len(y) - np.sum(y)}")
    
    return X, y

# fisher to rank features (using sklearn's f_classif)
def calc_fisher(X, y):
    F, p = f_classif(X, y)
    ranked = np.argsort(F)[::-1]
    return F, ranked

# SFS to pick the best features (using LogisticRegression for speed)
def do_sfs(X, y, ranked, max_feats=10, top_n=30):
    best_feats = ranked[:top_n]
    X_small = X[:, best_feats]
    
    clf = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
    
    sfs = SequentialFeatureSelector(clf, n_features_to_select=max_feats, scoring='f1', cv=3)
    sfs.fit(X_small, y)
    
    selected_mask = sfs.get_support()
    selected_indices = np.where(selected_mask)[0]
    
    final = best_feats[selected_indices]
    
    scores = cross_val_score(clf, X_small[:, selected_indices], y, cv=3, scoring='f1')
    best_score = np.mean(scores)
    
    print(f"Selected features indices: {final}")
    print(f"Best CV F1 score on train: {best_score:.4f}")
    
    return final

# main
file = 'prostate_preprocessed.txt'
X, y = load_data(file)

# scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# feature ranking on train set
_, ranked_train = calc_fisher(X_train, y_train)

# feature selection on train set
feats_selected = do_sfs(X_train, y_train, ranked_train, max_feats=10, top_n=30)

# Hyperparameter tuning for Random Forest on train set (with selected features)
param_grid = {
    'max_depth': [5, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
grid_search = GridSearchCV(clf_rf, param_grid, cv=3, scoring='f1')
grid_search.fit(X_train[:, feats_selected], y_train)

print("Best params from GridSearchCV:", grid_search.best_params_)
print(f"Best CV F1 score from GridSearchCV: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_rf.predict(X_test[:, feats_selected])
test_f1 = f1_score(y_test, y_pred)
print(f"Test set F1 score: {test_f1:.4f}")
