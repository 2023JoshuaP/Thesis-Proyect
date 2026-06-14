import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Configuración ────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 8
EPOCHS      = 30
NUM_CLASSES = 2
CLASSES     = ["bostezo", "microsueno"]
RANDOM_SEED = 42

# ── Dataloader ───────────────────────────────────────────────────────────────
"""
Carga y preprocesa un par de imágenes (ojos + boca) junto con su etiqueta.
Para cada imagen: lee el archivo, lo decodifica a tensor RGB,
lo redimensiona a 224x224 (requerido por ResNet50V2) y aplica preprocess_input
que normaliza los píxeles al rango esperado por ImageNet.
Retorna la tupla ((img_ojos, img_boca), label).
"""
def _load_pair(eyes_path: str, mouth_path: str, label: int):
    def _load_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = preprocess_input(img)
        return img
    return (_load_img(eyes_path), _load_img(mouth_path)), label

"""
Construye un tf.data.Dataset a partir de un directorio de train/, val/ o test/.
Recorre cada clase buscando pares válidos: empareja archivos eyes mouth_*.jpg 
usando el nombre base como clave.
Solo incluye pares completos (ambas ROIs presentes).
Luego arma el dataset aplicando _load_pair en paralelo,
con batching y prefetch para mayor eficiencia en el entrenamiento.
"""
def _build_dataset(split_dir: Path, shuffle: bool = False):
    eyes_paths, mouth_paths, labels = [], [], []

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            print(f"[WARN] No existe: {class_dir}")
            continue

        pairs: dict[str, dict] = {}
        for img_path in class_dir.glob("*.jpg"):
            name = img_path.name
            if name.startswith("eyes_"):
                base = name[len("eyes_"):]
                pairs.setdefault(base, {})["eyes"] = img_path
            elif name.startswith("mouth_"):
                base = name[len("mouth_"):]
                pairs.setdefault(base, {})["mouth"] = img_path

        complete = {k: v for k, v in pairs.items() if "eyes" in v and "mouth" in v}
        print(f"  [{class_name}] {len(complete)} pares")

        for pair in complete.values():
            eyes_paths.append(str(pair["eyes"]))
            mouth_paths.append(str(pair["mouth"]))
            labels.append(class_idx)

    dataset = tf.data.Dataset.from_tensor_slices((eyes_paths, mouth_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels), seed=RANDOM_SEED)

    dataset = dataset.map(
        lambda e, m, l: _load_pair(e, m, l),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset, len(labels)

# ── Modelo ───────────────────────────────────────────────────────────────────
"""
Construye la arquitectura dual-stream para la clasificación.
Instancia dos redes independientes (una para ojos, otra para boca),
ambas congeladas inicialmente (trainable=False) usando pesos preentrenados de ImageNet.
Cada rama extrae un vector de features mediante Global Average Pooling.
Esos dos vectores se concatenan y pasan por dos capas Dense con Dropout
antes del clasificador softmax de 2 clases (bostezo, microsueño).
El renombrado explícito de capas (eyes_*, mouth_*) evita conflictos
de nombres entre las dos ramas al compartir la misma arquitectura base.
"""
def _build_model() -> Model:
    base_eyes = ResNet50V2(include_top=False, weights="imagenet", pooling="avg")
    base_eyes._name = "resnet_eyes"
    for layer in base_eyes.layers:
        layer._name = f"eyes_{layer.name}"
        layer.trainable = False

    base_mouth = ResNet50V2(include_top=False, weights="imagenet", pooling="avg")
    base_mouth._name = "resnet_mouth"
    for layer in base_mouth.layers:
        layer._name = f"mouth_{layer.name}"
        layer.trainable = False

    input_eyes  = layers.Input(shape=(*IMG_SIZE, 3), name="input_eyes")
    input_mouth = layers.Input(shape=(*IMG_SIZE, 3), name="input_mouth")

    features_eyes  = base_eyes(input_eyes)
    features_mouth = base_mouth(input_mouth)

    combined = layers.Concatenate(name="concat_features")([features_eyes, features_mouth])
    x = layers.Dense(256, activation="relu", name="dense_1")(combined)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    output = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    return Model(inputs=[input_eyes, input_mouth], outputs=output, name="drowsiness_model")

# ── Callback para guardar mejores pesos ────────────────────────
"""
Callback para guardar los pesos del modelo
únicamente cuando val_loss mejora respecto a la mejor época anterior.
Es equivalente a ModelCheckpoint pero guardando solo pesos (.h5),
lo cual resulta más ligero que persistir el modelo completo.
Mantiene best_val_loss=inf como baseline inicial para la primera comparación.
"""
class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath    = filepath
        self.best_val_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.model.save_weights(self.filepath)
            print(f"\nEpoch {epoch+1}: val_loss mejoró a {val_loss:.5f} → pesos guardados")

# ── Entrenamiento ─────────────────────────────────────────────────────────────
"""
Función principal de orquestación del entrenamiento.
Carga los tres splits (train, val, test), construye y compila el modelo
con Adam (lr=1e-3) y sparse categorical crossentropy.
Entrena con tres callbacks:
    - EarlyStopping: detiene si val_loss no mejora en 5 épocas y restaura mejores pesos.
    - SaveBestModel: persiste en disco los mejores pesos ante un posible crash.
    - ReduceLROnPlateau: reduce el learning rate a la mitad si val_loss
    no mejora en 3 épocas consecutivas.
Al finalizar, evalúa en test generando predicciones batch a batch para
calcular el classification_report y F1 weighted. Guarda también
los pesos del último epoch como referencia.
"""
def train(dataset_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nCargando datos...")
    train_ds, train_n = _build_dataset(dataset_dir / "train", shuffle=True)
    val_ds,   val_n   = _build_dataset(dataset_dir / "val")
    test_ds,  test_n  = _build_dataset(dataset_dir / "test")

    print(f"\nTrain: {train_n} | Val: {val_n} | Test: {test_n}")

    print("\nConstruyendo modelo...")
    model = _build_model()
    model.summary()

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"]
    )

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        SaveBestModel(filepath=str(output_dir / "best_model_weights.h5")),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
    ]

    print("\nEntrenando...\n")
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = EPOCHS,
        callbacks       = callbacks
    )

    # ── Evaluación final en test ──────────────────────────────────────────────
    print("\nEvaluando en test set...")
    y_true, y_pred = [], []

    for (eyes_batch, mouth_batch), labels in test_ds:
        preds = model.predict([eyes_batch, mouth_batch], verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    print("\n" + "─" * 45)
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"F1 Score (weighted): {f1:.4f}")

    # Guardar solo pesos del modelo final
    model.save_weights(str(output_dir / "final_model_weights.h5"))
    print(f"\nPesos guardados en: {output_dir}")


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent

    train(
        dataset_dir = base / "data" / "processed" / "nitymed_augmented",
        output_dir  = base / "models" / "drowsiness_v1"
    )