import os
import tensorflow as tf
from pathlib import Path
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, f1_score

# ── Configuración Global ─────────────────────────────────────────────────────
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 30
CLASSES = ["alerta", "bostezo", "microsueno"]
NUM_CLASSES = len(CLASSES)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ── Dataloader Custom (Early Fusion) ──────────────────────────────────────────
def _load_image_pair(eyes_path, mouth_path, label):
    def process_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    eyes = process_img(eyes_path)
    mouth = process_img(mouth_path)

    # Aplastar a 112x224 (Alto x Ancho) para cada uno
    eyes = tf.image.resize(eyes, (112, 224))
    mouth = tf.image.resize(mouth, (112, 224))

    # Pegarlas verticalmente: (112+112, 224) -> (224, 224, 3)
    combined = tf.concat([eyes, mouth], axis=0)
    
    # Preprocesar
    combined = preprocess_input(combined)

    return combined, label

def create_dataset(dataset_dir: Path, batch_size=BATCH_SIZE) -> tuple[tf.data.Dataset, int]:
    eyes_paths, mouth_paths, labels = [], [], []

    for label_idx, class_name in enumerate(CLASSES):
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            continue

        eye_files = sorted(class_dir.glob("eyes_*"))
        for eye_path in eye_files:
            basename = eye_path.name.replace("eyes_", "")
            mouth_path = class_dir / f"mouth_{basename}"

            if mouth_path.exists():
                eyes_paths.append(str(eye_path))
                mouth_paths.append(str(mouth_path))
                labels.append(label_idx)

    ds = tf.data.Dataset.from_tensor_slices((eyes_paths, mouth_paths, labels))
    ds = ds.map(_load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds, len(labels)

# ── Modelo Single-Stream ──────────────────────────────────────────────────────
def _build_single_model() -> Model:
    # 1 sola red en vez de 2
    base_net = ResNet50V2(include_top=False, weights="imagenet", pooling="avg", name="resnet_base")
    base_net.trainable = False

    input_combined = layers.Input(shape=(*IMG_SIZE, 3), name="input_combined")
    
    features = base_net(input_combined)

    x = layers.Dense(256, activation="relu", name="dense_1")(features)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    output = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    return Model(inputs=input_combined, outputs=output, name="drowsiness_single_stream_model")

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
            print(f"\nÉpoca {epoch+1}: val_loss mejoró a {val_loss:.5f}, guardando pesos.")

# ── Bucle de Entrenamiento ───────────────────────────────────────────────────
def train(dataset_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Construyendo datasets...")
    train_ds, train_len = create_dataset(dataset_dir / "train")
    val_ds,   val_len   = create_dataset(dataset_dir / "val")
    test_ds,  test_len  = create_dataset(dataset_dir / "test")

    print(f"\nTrain: {train_len} | Val: {val_len} | Test: {test_len}\n")

    print("Construyendo modelo Single-Stream (Early Fusion)...")
    model = _build_single_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        SaveBestModel(filepath=str(output_dir / "best_model_weights_single.weights.h5")),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
    ]

    print("\nEntrenando modelo de 1 sola red...\n")
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = EPOCHS,
        callbacks       = callbacks
    )

    print("\nEvaluando en test set...")
    y_true, y_pred = [], []

    for batch, labels in test_ds:
        preds = model.predict(batch, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    print("\n" + "─" * 45)
    print("RESULTADOS SINGLE-STREAM (Early Fusion):")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"F1 Score (weighted): {f1:.4f}")

    model.save_weights(str(output_dir / "final_model_weights_single.weights.h5"))
    print(f"\nPesos single-stream guardados en: {output_dir}")

if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent

    train(
        dataset_dir = base / "data" / "processed" / "nitymed_augmented",
        output_dir  = base / "models" / "drowsiness_single_v1"
    )
