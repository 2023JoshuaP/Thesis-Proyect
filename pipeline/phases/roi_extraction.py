"""
Procesamiento del dataset de videos NITYMED para generar las Regiones de
Interés (ROIs) correspondientes a ojos y boca, etiquetándolas a nivel de
FRAME para 3 clases: alerta, bostezo y microsueño, mediante el uso de 
MTCNN, MediaPipe, EAR y MAR.
"""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mtcnn import MTCNN

"""
Configuracion de mediapipe para la deteccion de rostros y ojos
"""

FRAME_STEP = 2
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.60
MICROSLEEP_CONSEC = 15
EYE_MARGIN = 0.40
MOUTH_MARGIN = 0.40
FACE_PAD_RATIO = 0.20
IMG_OUT_SIZE = None

CLASSES = ["alerta", "bostezo", "microsueno"]

"""
Indices de los puntos de referencia de mediapipe para ojos y boca
"""
def _indices_from_connections(connections) -> list[int]:
    index = set()
    for connection in connections:
        index.add(connection.start)
        index.add(connection.end)
    return sorted(index)

_FLC = vision.FaceLandmarksConnections
RIGHT_EYE_REGION = _indices_from_connections(_FLC.FACE_LANDMARKS_RIGHT_EYE)
LEFT_EYE_REGION = _indices_from_connections(_FLC.FACE_LANDMARKS_LEFT_EYE)
MOUTH_REGION = _indices_from_connections(_FLC.FACE_LANDMARKS_LIPS)

"""
Puntos de referencia para el calculo del EAR (MediaPipe Facemesh) y MAR (comisuras - centro de labio)
"""
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
MOUTH_MAR_INDEX = [61, 291, 13, 14]

def _eucledian(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def _eye_aspect_ratio(landmarks_px, eye_indices) -> float:
    """
    Calculo del Eye Aspect Ratio para medir el nivel de apertura ocular.
    (EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||))
    p1, p4: puntos horizontales del ojo
    p2, p3, p5, p6: puntos verticales del ojo
    Umbral clave: EAR < 0.15 indica posible microsueño.
    """
    p = [landmarks_px[i] for i in eye_indices]
    vertical_1 = _eucledian(p[1], p[5])
    vertical_2 = _eucledian(p[2], p[4])
    horizontal = _eucledian(p[0], p[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def _mouth_aspect_ratio(landmarks_pixel, mouth_index) -> float:
    """
    Calculo del Mouth Aspect Ratio para medir el nivel de apertura bucal.
    (MAR = ||p2-p4|| / (2 * ||p1-p3||))
    p1, p3: puntos horizontales de la boca
    p2, p4: puntos verticales de la boca
    Umbral clave: MAR > 0.60 mantenido en el tiempo indica bostezo.
    """
    left_corner, right_corner, upper_lip, lower_lip = [landmarks_pixel[i] for i in mouth_index]
    vertical = _eucledian(upper_lip, lower_lip)
    horizontal = _eucledian(left_corner, right_corner)
    return vertical / horizontal

"""
DETECCIÓN FACIAL (MTCNN) Y LANDMARKS (MediaPipe)
Esta función recibe un frame, utiliza MTCNN para localizar el rostro
de forma robusta, y luego aplica MediaPipe Face Mesh sobre el rostro 
recortado para obtener las coordenadas 3D de los puntos faciales.
"""
def _detect_face_and_landmarks(frame_bgr, mtcnn_detector, face_landmarker):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = mtcnn_detector.detect_faces(rgb)
    if not detections:
        return None, None
    
    detector = max(detections, key=lambda d: d["box"][2] * d["box"][3])
    x, y, w, h = detector["box"]
    x, y = max(0, x), max(0, y)

    pad = int(FACE_PAD_RATIO * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_bgr.shape[1], x + w + pad)
    y2 = min(frame_bgr.shape[0], y + h + pad)

    face_crop = frame_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, None
    
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return face_crop, None
    
    h_crop, w_crop = face_crop.shape[:2]
    landmarks_pixel = [(lm.x * w_crop, lm.y * h_crop) for lm in result.face_landmarks[0]]
    return face_crop, landmarks_pixel

"""
EXTRACCIÓN DE REGIONES DE INTERÉS (ROI) DE OJOS Y BOCA
Calcula el bounding box a partir de los puntos de referencia faciales,
añade un margen de tolerancia (padding) y extrae de la imagen original
exclusivamente la región anatómica relevante, eliminando ruido de fondo.
"""
def _extract_rois(face_crop, landmarks_pixel, region_index, margin_ratio):
    points = np.array([landmarks_pixel[i] for i in region_index])
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    w, h = x_max - x_min, y_max - y_min
    margin_x, margin_y = w * margin_ratio, h * margin_ratio

    x1 = max(0, int(x_min - margin_x))
    y1 = max(0, int(y_min - margin_y))
    x2 = min(face_crop.shape[1], int(x_max + margin_x))
    y2 = min(face_crop.shape[0], int(y_max + margin_y))

    return face_crop[y1:y2, x1:x2]

"""
ETIQUETADO AUTOMÁTICO (CLASIFICACIÓN MULTICLASE)
Asigna de forma automática la etiqueta (alerta, bostezo o microsueño) 
a cada frame del video evaluando la secuencia temporal de EAR y MAR.
Evita la necesidad de anotación manual del dataset.
"""
def _assing_labels(metadata: list[dict]) -> list[str]:
    n = len(metadata)
    labels = ["alerta"] * n

    i = 0
    while i < n:
        if metadata[i]["ear"] < EAR_THRESHOLD:
            j = i
            while j < n and metadata[j]["ear"] < EAR_THRESHOLD:
                j += 1
            if (j - i) >= MICROSLEEP_CONSEC:
                for k in range(i, j):
                    labels[k] = "microsueno"
            
            i = j
        else:
            i += 1
    
    for k, m in enumerate(metadata):
        if labels[k] == "alerta" and m["mar"] > MAR_THRESHOLD:
            labels[k] = "bostezo"
    
    return labels

"""
Proceso del video recorriendo frame a frame,
deteccion de rostro con MTCNN y landmarks con MediaPipe,
calculo de EAR/MAR acumulando con el crop y los landmarks.
Asigna las etiquetas y extrae los ROIs de ojos y boca segun
la etiqueta final
"""
def process_video(video_path: Path, output_dir: Path, mtcnn_detector, face_landmarker) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    video_name = video_path.stem

    metadata: list[dict] = []
    crops: list[np.ndarray] = []
    landmarks_list: list[list[tuple[float, float]]] = []
    eye_region = RIGHT_EYE_REGION + LEFT_EYE_REGION

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % FRAME_STEP == 0:
            face_crop, lm_px = _detect_face_and_landmarks(frame, mtcnn_detector, face_landmarker)
            if face_crop is not None and lm_px is not None:
                ear = (_eye_aspect_ratio(lm_px, RIGHT_EYE_EAR) + _eye_aspect_ratio(lm_px, LEFT_EYE_EAR)) / 2.0
                mar = _mouth_aspect_ratio(lm_px, MOUTH_MAR_INDEX)

                metadata.append({"frame_index": frame_index, "ear": ear, "mar": mar})
                crops.append(face_crop)
                landmarks_list.append(lm_px)
        
        frame_index += 1
    
    cap.release()

    counts = {c: 0 for c in CLASSES}
    if not metadata:
        print(f"Warning: No se detectaron rostros en el video {video_name}")
        return counts
    
    labels = _assing_labels(metadata)

    for meta, label, face_crop, lm_px in zip(metadata, labels, crops, landmarks_list):
        eye_roi = _extract_rois(face_crop, lm_px, eye_region, EYE_MARGIN)
        mouth_roi = _extract_rois(face_crop, lm_px, MOUTH_REGION, MOUTH_MARGIN)

        if eye_roi.size == 0 or mouth_roi.size == 0:
            continue

        if IMG_OUT_SIZE is not None:
            eye_roi = cv2.resize(eye_roi, IMG_OUT_SIZE)
            mouth_roi = cv2.resize(mouth_roi, IMG_OUT_SIZE)

        class_dir = output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{video_name}_f{meta['frame_index']:06d}.jpg"
        cv2.imwrite(str(class_dir / f"eyes_{base_name}"), eye_roi)
        cv2.imwrite(str(class_dir / f"mouth_{base_name}"), mouth_roi)
        
        counts[label] += 1
    
    return counts

import concurrent.futures
import multiprocessing as mp_core

# Globals for the worker process
_mtcnn_detector = None
_face_landmarker = None

def init_worker(model_asset_path):
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    global _mtcnn_detector
    global _face_landmarker
    
    from mtcnn import MTCNN
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    
    _mtcnn_detector = MTCNN()
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
    _face_landmarker = vision.FaceLandmarker.create_from_options(options)

def process_video_worker(args):
    video_path, output_dir = args
    global _mtcnn_detector, _face_landmarker
    return video_path.name, process_video(video_path, output_dir, _mtcnn_detector, _face_landmarker)

def main(videos_dir: Path, output_dir: Path, model_path: Path):
    MAX_VIDEOS = None
    START_INDEX = 65
    
    video_files = sorted(videos_dir.rglob("*.mp4"))
    
    video_files = video_files[START_INDEX:]
    
    if MAX_VIDEOS:
        video_files = video_files[:MAX_VIDEOS]
        print(f"Procesando {MAX_VIDEOS} videos a partir del índice {START_INDEX}...")
    else:
        print(f"Procesando los {len(video_files)} videos restantes a partir del índice {START_INDEX}...")

    total = {c: 0 for c in CLASSES}
    
    args_list = [(v, output_dir) for v in video_files]
    
    # Usar 'spawn' para evitar problemas con CUDA y TF al crear subprocesos
    ctx = mp_core.get_context('spawn')
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=3, mp_context=ctx, initializer=init_worker, initargs=(model_path,)) as executor:
        futures = {executor.submit(process_video_worker, args): args for args in args_list}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(video_files), desc="Procesando videos paralelos"):
            try:
                name, counts = future.result()
                for c in CLASSES:
                    total[c] += counts[c]
                tqdm.write(f"{name}: {counts}")
            except Exception as exc:
                tqdm.write(f"{futures[future][0].name} generó una excepción: {exc}")

    print("\nResumen total:")
    for c in CLASSES:
        print(f"{c:12s}: {total[c]} pares")

if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent

    main(
        videos_dir = base / "data" / "raw" / "NITYMED_videos" / "Yawning",
        output_dir = base / "data" / "processed" / "nitymed_frames",
        model_path = base / "models" / "face_landmarker.task"
    )