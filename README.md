# Proyecto de Detección de Somnolencia (Bostezo y Microsueño)

Sistema automatizado basado en Inteligencia Artificial para la detección y clasificación en tiempo real de estados de alerta, bostezos y microsueños a través del análisis dual de imágenes faciales (ojos y boca).

## Arquitectura del Modelo
El núcleo del proyecto es una red neuronal **Dual-Stream ResNet50V2**. Esta arquitectura procesa simultáneamente dos flujos de información espacial (la región ocular y la región bucal), extrae sus características principales usando transfer learning (pesos de ImageNet) y concatena ambos flujos para una clasificación final altamente precisa.
---

## Fases del Pipeline (`pipeline/phases/`)

### 1. Extracción de ROIs (`roi_extraction.py`)
Procesa masivamente los videos crudos del dataset NITYMED.
- Utiliza **MTCNN** para ubicar el bounding box del rostro.
- Utiliza **MediaPipe Face Mesh** para ubicar los landmarks faciales (malla 3D).
- Calcula métricas dinámicas como **EAR** (Eye Aspect Ratio) y **MAR** (Mouth Aspect Ratio) a lo largo del tiempo para etiquetar automáticamente los frames como `alerta`, `bostezo` o `microsueno`.
- Acelera el procesamiento usando `ProcessPoolExecutor` para aprovechar los hilos de CPU y VRAM dinámica.

### 2. Balanceo y Aumento de Datos (`augmentation.py`)
Prepara el dataset físico para que la red neuronal aprenda sin sesgos poblacionales mediante una estrategia híbrida:
- **Undersampling:** Reduce masiva y aleatoriamente las clases dominantes (ej. alertas infinitas en video) a un objetivo fijo de 5,000 pares.
- **Oversampling (Aumento Dinámico):** Si una clase tiene menos datos de los requeridos, le inyecta micro-variaciones (rotaciones de ±15°, brillo ±40%, zoom ±15%) durante la partición de entrenamiento.
- Divide automáticamente el dataset en `train`, `val` y `test`.

### 3. Entrenamiento (`training.py`)
Orquesta el flujo de tensores hacia la arquitectura Dual-Stream.
- Utiliza `tf.data.Dataset` con prefetching para inyectar las imágenes directamente a la GPU sin cuellos de botella.
- **Optimizador:** Adam (`1e-3`).
- **Callbacks:** `EarlyStopping` (para evitar sobreajuste), `ReduceLROnPlateau` y `SaveBestModel`.
- Guarda automáticamente los pesos del mejor epoch en formato Keras 3 (`best_model_weights.weights.h5`).

### 4. Inferencia en Video (`inference.py`)
Toma el modelo entrenado y lo aplica a videos sin procesar en el mundo real.
- Analiza frame a frame recreando el pipeline de extracción (MTCNN + MediaPipe -> Crop Ojos/Boca).
- Superpone el resultado semántico (Verde = Alerta, Amarillo = Bostezo, Rojo = Microsueño) junto con su porcentaje de confianza en la esquina superior del video.
- Exporta el video anotado final usando el códec `mp4v` (Compatible nativamente con VLC Media Player).

---

## Estructura de Directorios

```text
ProyectoTesis/
├── data/
│   ├── raw/
│   │   └── NITYMED_videos/       # Videos originales
│   └── processed/
│       ├── nitymed_frames/       # Recortes crudos de extracción
│       └── nitymed_augmented/    # Dataset particionado y balanceado
│           ├── train/
│           ├── val/
│           └── test/
├── models/
│   ├── face_landmarker.task      # Modelo base de MediaPipe
│   └── drowsiness_v1/            # Pesos entrenados (Dual-Stream)
└── pipeline/
    └── phases/                   # Scripts principales documentados arriba
```

## Tecnologías y Requisitos

- **Entorno:** Python 3.11+
- **Deep Learning:** TensorFlow 2.16+ (Keras 3)
- **Computer Vision:** OpenCV (`cv2`), MediaPipe, MTCNN
- **Aceleración:** NVIDIA CUDA (Verificado en RTX 4050)

## Cómo Ejecutar

Para correr una prueba de inferencia sobre un video nuevo, utiliza el siguiente comando:

```bash
python pipeline/phases/inference.py \
  --input data/raw/ruta_a_tu_video.mp4 \
  --output data/processed/inference_result.mp4
```
