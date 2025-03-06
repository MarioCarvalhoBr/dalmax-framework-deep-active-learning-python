import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys
args = sys.argv
if len(args) != 2:
    print("Usage: python validate_predictions.py")
    sys.exit(1)


def calcular_metricas(arquivo_csv):
    # Carregar os dados
    df = pd.read_csv(arquivo_csv)

    # Extrair classes reais e preditas
    y_true = df["Real Class"]
    y_pred = df["Predicted Class"]

    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Exibir os resultados
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# Exemplo de uso
arquivo = args[1]
calcular_metricas(arquivo)
