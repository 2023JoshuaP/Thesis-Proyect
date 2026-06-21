import os
import sys
from pathlib import Path

# Añadir ruta del proyecto
base_dir = Path(__file__).parent.parent.parent
sys.path.append(str(base_dir))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Importar lógicas de Dual-Stream
from pipeline.phases.training import _build_dataset as create_dataset_dual, _build_model as _build_model_dual, CLASSES

# Importar lógicas de Single-Stream
from pipeline.phases.training_single import create_dataset as create_dataset_single, _build_single_model

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def plot_and_save_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Clase Predicha', fontsize=12)
    plt.ylabel('Clase Verdadera', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Gráfico guardado: {filename}")
    plt.close()

def evaluate_dual(base_dir):
    print("\n--- Evaluando Modelo Dual-Stream ---")
    weights_path = base_dir / "models" / "drowsiness_v1" / "final_model_weights.weights.h5"
    if not weights_path.exists():
        print(f"No se encontraron pesos en {weights_path}")
        return

    test_ds, _ = create_dataset_dual(base_dir / "data" / "processed" / "nitymed_augmented" / "test")
    model = _build_model_dual()
    model.load_weights(str(weights_path))

    y_true, y_pred = [], []
    print("Calculando predicciones Dual-Stream...")
    for (eyes, mouth), labels in test_ds:
        preds = model.predict([eyes, mouth], verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    plot_and_save_matrix(
        y_true, y_pred, 
        title="Matriz de Confusión - Dual-Stream", 
        filename="matriz_confusion_dual.png"
    )

def evaluate_single(base_dir):
    print("\n--- Evaluando Modelo Single-Stream ---")
    weights_path = base_dir / "models" / "drowsiness_single_v1" / "final_model_weights_single.weights.h5"
    if not weights_path.exists():
        print(f"No se encontraron pesos en {weights_path}")
        return

    test_ds, _ = create_dataset_single(base_dir / "data" / "processed" / "nitymed_augmented" / "test")
    model = _build_single_model()
    model.load_weights(str(weights_path))

    y_true, y_pred = [], []
    print("Calculando predicciones Single-Stream...")
    for batch, labels in test_ds:
        preds = model.predict(batch, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    plot_and_save_matrix(
        y_true, y_pred, 
        title="Matriz de Confusión - Single-Stream (Early Fusion)", 
        filename="matriz_confusion_single.png"
    )

if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent
    
    evaluate_dual(base)
    evaluate_single(base)
    
    print("\n¡Proceso Finalizado! Revisa las imágenes generadas en la raíz del proyecto.")
