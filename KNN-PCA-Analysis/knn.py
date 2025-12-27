import numpy as np
import pandas as pd
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Dados de Treino 
data = {
    'Fruta': ['A', 'B', 'C', 'D', 'E'],
    'Peso_x1': [100, 120, 180, 200, 150],
    'Cor_x2': [2, 3, 8, 9, 7],
    'Classe': [0, 0, 1, 1, 1] 
}

df = pd.DataFrame(data)

# Nova Fruta X para teste
X_new = np.array([[140, 6]]) 

print("--- CONJUNTO DE DADOS ---")
print(df)
print(f"\nNova fruta X para classificar: {X_new[0]}")
print("-" * 30)

# Distância Euclidiana Bruta (Sem padronização)
fruta_A = df.iloc[0][['Peso_x1', 'Cor_x2']].values
dist_raw = sqrt((X_new[0][0] - fruta_A[0])**2 + (X_new[0][1] - fruta_A[1])**2)

print(f"\nVALIDAÇÃO MANUAL (Sem Padronização)")
print(f"Distância d(X, A) esperada = 40.2")
print(f"Cálculo Python: {dist_raw:.2f}")

# Padronização (Standardization)
mu_peso, sigma_peso = 150, 37.4 
mu_cor, sigma_cor = 6, 3.0      

# Transformação Manual da Fruta A (100, 2)
z_peso_A = (100 - mu_peso) / sigma_peso
z_cor_A = (2 - mu_cor) / sigma_cor

print(f"\nVALIDAÇÃO DA PADRONIZAÇÃO (Slide 21)")
print(f"Fruta A padronizada ≈ (-1.336, -1.333)")
print(f"Python Manual:     Fruta A padronizada = ({z_peso_A:.3f}, {z_cor_A:.3f})")

# Transformação Manual da Nova Fruta X (140, 6)
z_peso_X = (140 - mu_peso) / sigma_peso
z_cor_X = (6 - mu_cor) / sigma_cor

print(f"\nFruta X padronizada = (-0.267, 0)")
print(f"Python Manual:     Fruta X padronizada = ({z_peso_X:.3f}, {z_cor_X:.3f})")

print("-" * 30)

print("\n--- CLASSIFICADOR KNN COM SCIKIT-LEARN ---")

# Preparar features (X) e labels (y)
X_train = df[['Peso_x1', 'Cor_x2']].values
y_train = df['Classe'].values

# Padronização
scaler = StandardScaler()
# os valores podem variar infinitesimalmente dos slides manuais.
X_train_scaled = scaler.fit_transform(X_train)
X_new_scaled = scaler.transform(X_new)

# Treinar o KNN
k = 3 # Escolhendo k=3 como exemplo de teste [cite: 345]
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Predição
prediction = knn.predict(X_new_scaled)
distances, indices = knn.kneighbors(X_new_scaled)

# Resultado
classes_map = {0: 'Kiwi', 1: 'Maçã'} # [cite: 149, 150]
vizinhos_proximos = df.iloc[indices[0]]

print(f"K escolhido: {k}")
print(f"Vizinhos mais próximos (índices): {indices[0]}")
print(f"Distâncias calculadas (padronizadas): {distances[0]}")
print(f"\n>>> PREDIÇÃO FINAL: A nova fruta é uma {classes_map[prediction[0]]} (Classe {prediction[0]})")