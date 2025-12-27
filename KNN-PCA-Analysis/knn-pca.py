import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k = 5  # Valor padrão
knn_raw = KNeighborsClassifier(n_neighbors=k)
knn_raw.fit(X_train_scaled, y_train)
y_pred_raw = knn_raw.predict(X_test_scaled)

acc_raw = accuracy_score(y_test, y_pred_raw)
cm_raw = confusion_matrix(y_test, y_pred_raw)


pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

knn_pca = KNeighborsClassifier(n_neighbors=k)
knn_pca.fit(X_train_pca, y_train)
y_pred_pca = knn_pca.predict(X_test_pca)

acc_pca = accuracy_score(y_test, y_pred_pca)
cm_pca = confusion_matrix(y_test, y_pred_pca)

# Result
print(f"Acurácia (Original - 11 Atributos): {acc_raw:.4f}")
print(f"Acurácia (PCA - 2 Componentes):     {acc_pca:.4f}")
print("-" * 30)
print(f"Variância Explicada pelos 2 componentes PCA: {np.sum(pca.explained_variance_ratio_):.2%}")

# PLOT
plt.figure(figsize=(16, 6))

# Plot 1
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_test_scaled[:, 10], y=X_test_scaled[:, 7], hue=y_test, palette='viridis', alpha=0.7)
plt.title('Dados Originais (Álcool vs Densidade)')
plt.xlabel('Álcool (Padronizado)')
plt.ylabel('Densidade (Padronizada)')

# Plot 2
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_test, palette='viridis', alpha=0.7)
plt.title('Dados após PCA (CP1 vs CP2)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')

plt.tight_layout()
plt.savefig('wine_comparison.png')