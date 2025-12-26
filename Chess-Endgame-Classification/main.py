import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- CONFIGURAÇÃO DO ARQUIVO ---
filename = 'data/krkopt.data' 

print(f"Tentando ler o arquivo '{filename}'...")

try:
    data = pd.read_csv(filename, header=None)
    print("Arquivo carregado com sucesso!")
    print(f"Dimensões do dataset: {data.shape}")
except FileNotFoundError:
    print(f"\n[ERRO CRÍTICO] O arquivo '{filename}' não foi encontrado.")
    print("Verifique se o código e o arquivo estão na mesma pasta e se o nome está correto.")
    exit()

# --- PRÉ-PROCESSAMENTO ---
le = LabelEncoder()

for col in data.columns:
    data[col] = le.fit_transform(data[col])

# Separar Features (X) e Target (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Normalização (Essencial para SVM e k-NN calcularem distâncias corretamente)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --- DIVISÃO DOS DADOS ---
# 70% Treino, 30% Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- CONFIGURAÇÃO DOS MODELOS ---
models = {
    'SVM (RBF Kernel)': SVC(kernel='rbf', C=1.0, random_state=42),
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5)
}

# --- EXECUÇÃO E GRÁFICOS ---
plt.figure(figsize=(14, 6))
results_text = ""

for i, (name, model) in enumerate(models.items()):
    # Treinamento
    model.fit(X_train, y_train)
    
    # Predição
    y_pred = model.predict(X_test)
    
    # Cálculo da Acurácia
    acc = accuracy_score(y_test, y_pred)
    results_text += f"\n--- {name} ---\nAcurácia: {acc:.4f}\n"
    
    # Gerar Matriz de Confusão
    plt.subplot(1, 2, i+1)
    cm = confusion_matrix(y_test, y_pred)
    
    # Plotar o Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens' if 'SVM' in name else 'Blues', cbar=False)
    plt.title(f'{name}\nAcurácia: {acc:.2%}')
    plt.ylabel('Real')
    plt.xlabel('Predito')

# Exibir resultados no terminal e plotar gráficos
print(results_text)
plt.tight_layout()
plt.show()
