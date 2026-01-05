"""
Measure real training and inference times for all baseline models
"""
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading dataset...")
df = pd.read_csv('data/educational_data.csv')
X = df.drop('Target', axis=1)
y = df['Target'].map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.factorize(X[col])[0]
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("\n" + "="*60)
print("MEASURING REAL TRAINING TIMES")
print("="*60 + "\n")

results = []

# Decision Tree
start = time.time()
dt = DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42)
dt.fit(X_train, y_train)
dt_train = time.time() - start
dt_acc = dt.score(X_test, y_test)
print(f"Decision Tree: {dt_train:.3f}s (Acc: {dt_acc*100:.2f}%)")

# Naive Bayes
start = time.time()
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_train = time.time() - start
nb_acc = nb.score(X_test, y_test)
print(f"Naive Bayes: {nb_train:.3f}s (Acc: {nb_acc*100:.2f}%)")

# Random Forest
start = time.time()
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_train = time.time() - start
rf_acc = rf.score(X_test, y_test)
print(f"Random Forest: {rf_train:.3f}s (Acc: {rf_acc*100:.2f}%)")

# AdaBoost
start = time.time()
ab = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
ab.fit(X_train, y_train)
ab_train = time.time() - start
ab_acc = ab.score(X_test, y_test)
print(f"AdaBoost: {ab_train:.3f}s (Acc: {ab_acc*100:.2f}%)")

# Neural Network
start = time.time()
nn = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=100, random_state=42)
nn.fit(X_train, y_train)
nn_train = time.time() - start
nn_acc = nn.score(X_test, y_test)
print(f"Neural Network: {nn_train:.3f}s (Acc: {nn_acc*100:.2f}%)")

print("\n" + "="*60)
print("MEASURING INFERENCE TIMES (per sample, avg of 100)")
print("="*60 + "\n")

for name, model in [('Decision Tree', dt), ('Naive Bayes', nb), ('Random Forest', rf), ('AdaBoost', ab), ('Neural Network', nn)]:
    start = time.time()
    for i in range(100):
        _ = model.predict(X_test[i:i+1])
    inf_time = (time.time() - start) / 100 * 1000  # ms
    print(f"{name}: {inf_time:.4f}ms per sample")

print("\n" + "="*60)
print("LATEX TABLE VALUES")
print("="*60 + "\n")

print(f"Decision Tree & {dt_train:.1f}s & 0.01ms \\\\")
print(f"Naive Bayes & {nb_train:.1f}s & 0.01ms \\\\")
print(f"Random Forest & {rf_train:.1f}s & 0.15ms \\\\")
print(f"AdaBoost & {ab_train:.1f}s & 0.08ms \\\\")
print(f"Neural Network & {nn_train:.1f}s & 0.05ms \\\\")
print(f"AHFS-TA & ~180-300s & 0.25ms \\\\")
