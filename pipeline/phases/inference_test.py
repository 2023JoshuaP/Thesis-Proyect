import cv2
import numpy as np
from pathlib import Path
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# ── Configuración ────────────────────────────────────────────────────────────
IMG_SIZE        = (224, 224)
CLASSES         = ["bostezo", "microsueno"]
DETECTION_SCALE = 0.5
MIN_FACE_CONF   = 0.95
PAD_EYES_X, PAD_EYES_Y   = 30, 20
PAD_MOUTH_X, PAD_MOUTH_Y = 25, 25

# Mapeo de nombres de carpeta a clases del modelo
FOLDER_CLASS_MAP = {
    "Yawning":    "bostezo",
    "Microsleep": "microsueno",
}

COLORS = {
    "bostezo":    (0, 165, 255),   # naranja
    "microsueno": (0, 0, 255),     # rojo
    "normal":     (0, 255, 0),     # verde
}

# ── Modelo ───────────────────────────────────────────────────────────────────
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
    output = layers.Dense(len(CLASSES), activation="softmax", name="output")(x)

    return Model(inputs=[input_eyes, input_mouth], outputs=output, name="drowsiness_model")

# ── Helpers ROI ───────────────────────────────────────────────────────────────
def _scale_keypoints(keypoints, scale):
    return {k: (int(v[0] / scale), int(v[1] / scale)) for k, v in keypoints.items()}

def _crop_eyes(image, keypoints, width, height):
    le, re = keypoints['left_eye'], keypoints['right_eye']
    x_min = max(0, min(le[0], re[0]) - PAD_EYES_X)
    y_min = max(0, min(le[1], re[1]) - PAD_EYES_Y)
    x_max = min(width,  max(le[0], re[0]) + PAD_EYES_X)
    y_max = min(height, max(le[1], re[1]) + PAD_EYES_Y)
    return image[y_min:y_max, x_min:x_max]

def _crop_mouth(image, keypoints, width, height):
    ml, mr = keypoints['mouth_left'], keypoints['mouth_right']
    x_min = max(0, min(ml[0], mr[0]) - PAD_MOUTH_X)
    y_min = max(0, min(ml[1], mr[1]) - PAD_MOUTH_Y)
    x_max = min(width,  max(ml[0], mr[0]) + PAD_MOUTH_X)
    y_max = min(height, max(ml[1], mr[1]) + PAD_MOUTH_Y + 20)
    return image[y_min:y_max, x_min:x_max]

def _preprocess_roi(crop):
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = preprocess_input(img.astype(np.float32))
    return tf.expand_dims(img, 0)

# ── Inferencia sobre un video ─────────────────────────────────────────────────
def process_video(video_path: Path, model, detector, output_path: Path, real_class: str):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] No se pudo abrir: {video_path.name}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    frame_count = correct = 0
    pred_counts = {c: 0 for c in CLASSES}

    print(f"  Procesando: {video_path.name} ({total} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated = frame.copy()
        pred_label = "sin_deteccion"

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small     = cv2.resize(image_rgb, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        results   = detector.detect_faces(small)

        if results and results[0]['confidence'] >= MIN_FACE_CONF:
            keypoints = _scale_keypoints(results[0]['keypoints'], DETECTION_SCALE)
            bbox      = results[0]['box']

            # Escalar bounding box
            bx = int(bbox[0] / DETECTION_SCALE)
            by = int(bbox[1] / DETECTION_SCALE)
            bw = int(bbox[2] / DETECTION_SCALE)
            bh = int(bbox[3] / DETECTION_SCALE)

            eyes_crop  = _crop_eyes(frame, keypoints, width, height)
            mouth_crop = _crop_mouth(frame, keypoints, width, height)

            if eyes_crop.size > 0 and mouth_crop.size > 0:
                eyes_t  = _preprocess_roi(eyes_crop)
                mouth_t = _preprocess_roi(mouth_crop)

                probs      = model.predict([eyes_t, mouth_t], verbose=0)[0]
                pred_idx   = np.argmax(probs)
                pred_label = CLASSES[pred_idx]
                confidence = probs[pred_idx]

                pred_counts[pred_label] += 1
                if pred_label == real_class:
                    correct += 1

                # Anotar frame
                color = COLORS.get(pred_label, (255, 255, 255))
                cv2.rectangle(annotated, (bx, by), (bx+bw, by+bh), color, 2)
                cv2.putText(annotated, f"{pred_label} {confidence:.2f}",
                            (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Info en esquina superior izquierda
        cv2.putText(annotated, f"Real: {real_class}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Frame: {frame_count}/{total}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(annotated)

    cap.release()
    writer.release()

    accuracy = correct / frame_count * 100 if frame_count > 0 else 0
    print(f"    Accuracy: {accuracy:.1f}% ({correct}/{frame_count} frames correctos)")
    print(f"    Predicciones: {pred_counts}")
    return accuracy

# ── Main ──────────────────────────────────────────────────────────────────────
def run_video_inference(videos_dir: Path, weights_path: Path, output_dir: Path):
    print("\nCargando modelo...")
    model    = _build_model()
    model.load_weights(str(weights_path))
    detector = MTCNN()
    print("Listo.\n")

    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.MP4"]
    results = []

    for folder_name, real_class in FOLDER_CLASS_MAP.items():
        class_dir = videos_dir / folder_name
        if not class_dir.exists():
            continue

        videos = [
            p for gender_dir in class_dir.iterdir() if gender_dir.is_dir()
            for ext in video_extensions
            for p in gender_dir.rglob(ext)
        ]

        print(f"[{folder_name}] {len(videos)} videos encontrados")

        for video_path in videos:
            out_path = output_dir / folder_name / video_path.parent.name / f"pred_{video_path.name}"
            acc = process_video(video_path, model, detector, out_path, real_class)
            if acc is not None:
                results.append({"video": video_path.name, "clase": real_class, "accuracy": acc})

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("RESUMEN GENERAL")
    print("─" * 50)
    for r in results:
        print(f"  {r['video']:40} | {r['clase']:10} | {r['accuracy']:.1f}%")

    if results:
        avg = np.mean([r["accuracy"] for r in results])
        print(f"\nAccuracy promedio sobre todos los videos: {avg:.1f}%")


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent

    run_video_inference(
        videos_dir   = base / "data" / "raw" / "NITYMED_videos",
        weights_path = base / "models" / "drowsiness_v1" / "best_model_weights.h5",
        output_dir   = base / "models" / "drowsiness_v1" / "video_inference"
    )