"""
calibrate_thresholds.py

Utilidad para calibrar EAR_THRESH y MAR_THRESH antes de correr el pipeline
completo (preprocess_nitymed.py).

Procesa UN video y exporta:
  - Un CSV con (frame_idx, ear, mar) por cada frame muestreado.
  - Una carpeta "samples/" con recortes de rostro cada SAMPLE_EVERY frames,
    con el EAR/MAR escrito encima, para revisión visual.

Con el CSV se pueden graficar histogramas/series de EAR y MAR para elegir
EAR_THRESH y MAR_THRESH a partir de la distribución real de tus videos, en
lugar de usar solo valores de la literatura.

Uso:
    python calibrate_thresholds.py <video.mp4> <output_dir> [model_path]

Si no se indica model_path, se usa "models/face_landmarker.task" (ver
preprocess_nitymed.py para el link de descarga del modelo).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Añadir la raíz del proyecto al sys.path para poder importar el paquete `pipeline`
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # sube: notebooks/ -> pipeline/ -> raíz
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from mtcnn import MTCNN

from pipeline.phases.roi_extraction import (
    FRAME_STEP,
    LEFT_EYE_EAR,
    MOUTH_MAR_INDEX,
    RIGHT_EYE_EAR,
    _detect_face_and_landmarks,
    _eye_aspect_ratio,
    _mouth_aspect_ratio,
)

SAMPLE_EVERY = 30  # guardar 1 imagen de muestra cada N frames muestreados


def calibrate(video_path: Path, output_dir: Path, model_path: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    mtcnn_detector = MTCNN()
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(str(video_path))
    rows = []
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STEP == 0:
            face_crop, lm_px = _detect_face_and_landmarks(frame, mtcnn_detector, face_landmarker)
            if face_crop is not None and lm_px is not None:
                ear = (_eye_aspect_ratio(lm_px, RIGHT_EYE_EAR) +
                       _eye_aspect_ratio(lm_px, LEFT_EYE_EAR)) / 2.0
                mar = _mouth_aspect_ratio(lm_px, MOUTH_MAR_INDEX)
                rows.append({"frame_idx": frame_idx, "ear": round(ear, 4), "mar": round(mar, 4)})

                if frame_idx % SAMPLE_EVERY == 0:
                    vis = face_crop.copy()
                    cv2.putText(
                        vis, f"EAR:{ear:.3f} MAR:{mar:.3f}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                    )
                    cv2.imwrite(str(samples_dir / f"f{frame_idx:06d}.jpg"), vis)
                    saved += 1

        frame_idx += 1

    cap.release()
    face_landmarker.close()

    csv_path = output_dir / f"{video_path.stem}_ear_mar.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "ear", "mar"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV guardado en: {csv_path} ({len(rows)} frames)")
    print(f"Muestras visuales guardadas en: {samples_dir} ({saved} imágenes)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python calibrate_thresholds.py <video.mp4> <output_dir> [model_path]")
        sys.exit(1)

    _video_path = Path(sys.argv[1])
    _output_dir = Path(sys.argv[2])
    _model_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("models/face_landmarker.task")

    calibrate(_video_path, _output_dir, _model_path)