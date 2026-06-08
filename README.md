# Proyecto de Detección de Bostezos y Microsueño

Sistema automatizado para la detección y clasificación de bostezos y microsueño a través del análisis de imágenes faciales.

## Funcionalidades

### Extracción de Frames
- **`extractor.py`**
  - Extrae frames de videos almacenados en el dataset NITYMED
  - Genera imágenes por segundo desde cada video
  - Organiza las imágenes por clase (bostezos, microsueño) y género (hombres, mujeres)
  - Mapea automáticamente la estructura de carpetas del dataset

### Detección de Rostros
- **`face_detection.py`**
  - Detecta rostros en imágenes usando MTCNN
  - Identifica puntos de referencia facial (ojos, nariz, boca)
  - Dibuja rectángulos alrededor de los rostros detectados
  - Marca los puntos de referencia con círculos para validación visual
  - Guarda imágenes procesadas con anotaciones

### Corte de Rostros
- **`cut_faces.py`**
  - Extrae y recorta solo la región del rostro de cada imagen
  - Utiliza MTCNN para localización precisa de rostros
  - Organiza los rostros cortados mantieniendo la clasificación original
  - Genera dataset limpio de solo rostros

### Extracción de Regiones de Interés (ROI)
- **`extraction_roi.py`**
  - Extrae regiones específicas de los rostros (ojos y boca)
  - Aplica padding configurable para cada región
  - Genera dos tipos de ROI:
    - **Ojos**: región alrededor de los ojos (padding: 30px horizontal, 20px vertical)
    - **Boca**: región alrededor de la boca (padding: 25px en ambas direcciones)
  - Conserva la estructura de clases y géneros

### Procesamiento del Dataset UTA-RLDD
- **`extractor_uta.py`**
  - Procesa imágenes del dataset UTA-RLDD (Real-world with Lite and Dark)
  - Extrae ROI (ojos y boca) de imágenes de actividad normal y fatiga
  - Mapea clases ("active" → "normal", "fatigue" → "fatigue")
  - Organiza datos en estructura train/test/val
  - Aplica los mismos criterios de padding que el dataset principal

## Estructura de Datos Generada

```
dataset_images/        # Frames extraídos de videos
├── bostezo/
│   ├── hombres/
│   └── mujeres/
└── microsueno/
    ├── hombres/
    └── mujeres/

dataset_cut_faces/     # Rostros recortados
├── bostezo/
│   ├── hombres/
│   └── mujeres/
└── microsueno/
    ├── hombres/
    └── mujeres/

dataset_roi/           # Regiones de interés (ojos y boca)
├── bostezo/
│   ├── hombres/
│   └── mujeres/
└── microsueno/
    ├── hombres/
    └── mujeres/

dataset_uta_roi/       # ROI del dataset UTA-RLDD
├── train/
│   ├── normal/
│   └── fatigue/
├── test/
│   ├── normal/
│   └── fatigue/
└── val/
    ├── normal/
    └── fatigue/

dataset_faces/         # Rostros con anotaciones
```

## Tecnologías Utilizadas

- **OpenCV**: Procesamiento de imágenes y videos
- **MTCNN**: Detección de rostros y puntos de referencia
- **Kaggle Hub**: Descarga de datasets
- **Python 3.x**: Lenguaje de programación

## Requisitos

- Python 3.7+
- opencv-python
- mtcnn
- kagglehub
- Credenciales de Kaggle configuradas (para descarga de datasets)
