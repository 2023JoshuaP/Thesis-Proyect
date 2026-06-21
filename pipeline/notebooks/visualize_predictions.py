import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Añadir ruta del proyecto
base_dir = Path(__file__).parent.parent.parent
sys.path.append(str(base_dir))

# Importar lógicas de Dual-Stream
from pipeline.phases.training import _build_dataset as create_dataset_dual, _build_model as _build_model_dual, CLASSES

# Importar lógicas de Single-Stream
from pipeline.phases.training_single import create_dataset as create_dataset_single, _build_single_model

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def visualize_predictions(base_dir):
    print("\n--- Generando Cuadrícula de Evidencia Visual ---")
    weights_path = base_dir / "models" / "drowsiness_v1" / "final_model_weights.weights.h5"
    if not weights_path.exists():
        print(f"No se encontraron pesos en {weights_path}")
        return

    # Cargar Dataset de Test SIN shuffle para ver siempre las mismas, 
    # o CON shuffle para que cada vez salga un panel diferente.
    # Usaremos shuffle para que puedas generar varias y elegir la que más te guste.
    test_ds, _ = create_dataset_dual(base_dir / "data" / "processed" / "nitymed_augmented" / "test", shuffle=True)
    
    model = _build_model_dual()
    model.load_weights(str(weights_path))

    # Extraer 1 batch (32 imágenes)
    for (eyes_batch, mouth_batch), labels_batch in test_ds.take(1):
        preds = model.predict([eyes_batch, mouth_batch], verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = labels_batch.numpy()
        break # Solo necesitamos 1 batch

    # Configurar el gráfico 3x3
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    # Determinar cuántas imágenes tenemos (máximo 9)
    num_samples = min(9, eyes_batch.shape[0])
    
    for i in range(9):
        ax = axes[i]
        if i >= num_samples:
            ax.axis('off')
            continue

        # Ojos (Redimensionamos visualmente para que no parezcan estirados de miedo)
        eye_img = eyes_batch[i].numpy()
        eye_img = (eye_img - eye_img.min()) / (eye_img.max() - eye_img.min() + 1e-5)
        eye_img = tf.image.resize(eye_img, (80, 224)).numpy() # Aspecto natural
        
        # Boca
        mouth_img = mouth_batch[i].numpy()
        mouth_img = (mouth_img - mouth_img.min()) / (mouth_img.max() - mouth_img.min() + 1e-5)
        mouth_img = tf.image.resize(mouth_img, (80, 224)).numpy() # Aspecto natural
        
        # Las pegamos verticalmente con un pequeño separador negro
        separator = np.zeros((10, 224, 3))
        combined_img = np.vstack([eye_img, separator, mouth_img])
        
        ax.imshow(combined_img)
        ax.axis('off')
        
        true_label = CLASSES[true_classes[i]].upper()
        pred_label = CLASSES[pred_classes[i]].upper()
        confidence = preds[i][pred_classes[i]] * 100
        
        color = 'green' if true_label == pred_label else 'red'
        
        title = f"Real: {true_label}\nPred: {pred_label} ({confidence:.1f}%)"
        ax.set_title(title, color=color, fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path = base_dir / "predicciones_muestra_dual.png"
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    print(f"Imagen de evidencia DUAL-STREAM generada y guardada en: {output_path}")
    plt.close()

def visualize_single_stream(base_dir):
    print("\n--- Generando Cuadrícula de Evidencia Visual (Single-Stream) ---")
    weights_path = base_dir / "models" / "drowsiness_single_v1" / "final_model_weights_single.weights.h5"
    if not weights_path.exists():
        print(f"No se encontraron pesos en {weights_path}")
        return

    test_ds, _ = create_dataset_single(base_dir / "data" / "processed" / "nitymed_augmented" / "test")
    
    model = _build_single_model()
    model.load_weights(str(weights_path))

    for batch, labels_batch in test_ds.take(1):
        preds = model.predict(batch, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = labels_batch.numpy()
        combined_batch = batch
        break

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    num_samples = min(9, combined_batch.shape[0])
    
    for i in range(9):
        ax = axes[i]
        if i >= num_samples:
            ax.axis('off')
            continue

        img = combined_batch[i].numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        
        eye_part = tf.image.resize(img[:112, :], (80, 224)).numpy()
        mouth_part = tf.image.resize(img[112:, :], (80, 224)).numpy()
        separator = np.zeros((10, 224, 3))
        img_disp = np.vstack([eye_part, separator, mouth_part])
        
        ax.imshow(img_disp)
        ax.axis('off')
        
        true_label = CLASSES[true_classes[i]].upper()
        pred_label = CLASSES[pred_classes[i]].upper()
        confidence = preds[i][pred_classes[i]] * 100
        
        color = 'green' if true_label == pred_label else 'red'
        
        title = f"Real: {true_label}\nPred: {pred_label} ({confidence:.1f}%)"
        ax.set_title(title, color=color, fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path = base_dir / "predicciones_muestra_single.png"
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    print(f"Imagen de evidencia SINGLE-STREAM generada y guardada en: {output_path}")
    plt.close()

if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent
    visualize_predictions(base)
    visualize_single_stream(base)
