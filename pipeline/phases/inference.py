import argparse
import sys
from pathlib import Path

# Asegurar que Python reconozca la carpeta raíz del proyecto
base_dir = Path(__file__).parent.parent.parent
sys.path.append(str(base_dir))

import cv2
import numpy as np
from tqdm import tqdm

import tensorflow as tf
# Configurar VRAM antes de cargar Keras/MTCNN para no acaparar toda la memoria de la GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.applications.resnet_v2 import preprocess_input

from pipeline.phases.roi_extraction import (
    _detect_face_and_landmarks,
    _extract_rois,
    RIGHT_EYE_REGION,
    LEFT_EYE_REGION,
    MOUTH_REGION,
    EYE_MARGIN,
    MOUTH_MARGIN
)
from pipeline.phases.training import _build_model, IMG_SIZE, CLASSES

from mtcnn import MTCNN
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

def run_inference(input_video: Path, output_video: Path, weights_path: Path, model_asset_path: Path):
    print("Cargando detector facial (MTCNN)...")
    mtcnn_detector = MTCNN()

    print("Cargando detector de landmarks (MediaPipe)...")
    base_options = mp_python.BaseOptions(
        model_asset_path=str(model_asset_path),
        delegate=mp_python.BaseOptions.Delegate.GPU
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    print("Cargando red neuronal Dual-Stream ResNet50V2...")
    model = _build_model()
    model.load_weights(str(weights_path))

    print(f"Abriendo video de entrada: {input_video}")
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {input_video}")
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    eye_region = RIGHT_EYE_REGION + LEFT_EYE_REGION

    print(f"Procesando {frames} frames...")
    pbar = tqdm(total=frames, desc="Inferencia Frame a Frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Por defecto etiqueta vacía si no detecta rostro
        label = "No Face"
        confidence = 0.0
        color = (128, 128, 128)

        face_crop, lm_px = _detect_face_and_landmarks(frame, mtcnn_detector, face_landmarker)

        if face_crop is not None and lm_px is not None:
            eye_roi = _extract_rois(face_crop, lm_px, eye_region, EYE_MARGIN)
            mouth_roi = _extract_rois(face_crop, lm_px, MOUTH_REGION, MOUTH_MARGIN)

            if eye_roi.size > 0 and mouth_roi.size > 0:
                eye_img = cv2.resize(eye_roi, IMG_SIZE)
                eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
                eye_img = preprocess_input(eye_img.astype(np.float32))

                mouth_img = cv2.resize(mouth_roi, IMG_SIZE)
                mouth_img = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2RGB)
                mouth_img = preprocess_input(mouth_img.astype(np.float32))

                eye_batch = np.expand_dims(eye_img, axis=0)
                mouth_batch = np.expand_dims(mouth_img, axis=0)

                pred = model.predict([eye_batch, mouth_batch], verbose=0)[0]
                class_idx = np.argmax(pred)
                label = CLASSES[class_idx]
                confidence = pred[class_idx]

                if label == "alerta":
                    color = (0, 255, 0) # Verde
                elif label == "bostezo":
                    color = (0, 255, 255) # Amarillo (BGR)
                elif label == "microsueno":
                    color = (0, 0, 255) # Rojo

        # Dibujar cuadro de texto en la esquina superior izquierda
        cv2.rectangle(frame, (10, 10), (380, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"{label.upper()}: {confidence*100:.1f}%", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

        out.write(frame)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"\n¡Video guardado con éxito en: {output_video}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia en video Dual-Stream ResNet50V2")
    parser.add_argument("--input", type=str, required=True, help="Ruta al video de entrada")
    parser.add_argument("--output", type=str, default="data/processed/inference_result.mp4", help="Ruta al video de salida")
    
    args = parser.parse_args()
    
    base = Path(__file__).parent.parent.parent
    
    run_inference(
        input_video=Path(args.input),
        output_video=Path(args.output),
        weights_path=base / "models" / "drowsiness_v1" / "final_model_weights.weights.h5",
        model_asset_path=base / "models" / "face_landmarker.task"
    )
