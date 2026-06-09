import cv2
import numpy as np
from pathlib import Path
from mtcnn import MTCNN

"""
Configuracion de umbrales de calidad de deteccion y escala de imagen para la deteccion de rostros.
"""
DETECTION_SCALE = 0.5
MIN_FACE_CONF = 0.95
MIN_EYES_W = 60
MIN_EYES_H = 20
MIN_MOUTH_W = 30
MIN_MOUTH_H = 20
MIN_PIXEL_VAR = 200
PAD_EYES_X = 30
PAD_EYES_Y = 20
PAD_MOUTH_X = 25
PAD_MOUTH_Y = 25

"""
Diccionario de Dataset con las rutas de los archivos de anotaciones y las carpetas de imagenes.
"""
DATASET_CONFIG = {
    "nitymed": {
        "extensions": ["*.jpg"],
        "class_mapping": None,
        "extract_eyes": ["microsueno"],
        "extract_mouth": ["bostezo"],
        "path_structure": "class/gender",
    },
    "uta": {
        "extensions": ["*.jpg", "*.png"],
        "class_mapping": {"active": "normal", "fatigue": "fatigue"},
        "extract_eyes": None,
        "extract_mouth": None,
        "path_structure": "split/class",
    },
}

def _scale_keypoints(keypoints: dict, scale: float) -> dict:
    return {k: (int(v[0] / scale), int(v[1] / scale)) for k, v in keypoints.items()}

def _crop_eyes(image, keypoints, width, height):
    le, re = keypoints['left_eye'], keypoints['right_eye']
    x_min = max(0, min(le[0], re[0]) - PAD_EYES_X)
    y_min = max(0, min(le[1], re[1]) - PAD_EYES_Y)
    x_max = min(width,  max(le[0], re[0]) + PAD_EYES_X)
    y_max = min(height, max(le[1], re[1]) + PAD_EYES_Y)
    crop = image[y_min:y_max, x_min:x_max]
    return crop, x_max - x_min, y_max - y_min

def _crop_mouth(image, keypoints, width, height):
    ml, mr = keypoints['mouth_left'], keypoints['mouth_right']
    x_min = max(0, min(ml[0], mr[0]) - PAD_MOUTH_X)
    y_min = max(0, min(ml[1], mr[1]) - PAD_MOUTH_Y)
    x_max = min(width,  max(ml[0], mr[0]) + PAD_MOUTH_X)
    y_max = min(height, max(ml[1], mr[1]) + PAD_MOUTH_Y + 20)
    crop = image[y_min:y_max, x_min:x_max]
    return crop, x_max - x_min, y_max - y_min

def _validate_crop(crop, w, h, min_w, min_h) -> tuple[bool, str]:
    if crop.size == 0 or w < min_w or h < min_h:
        return False, f"too small ({w}x{h})"
    if np.var(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) < MIN_PIXEL_VAR:
        return False, "low variance"
    return True, ""

def _get_dest_dir(img_path, config, output_folder) -> Path:
    mapping = config["class_mapping"]
    if config["path_structure"] == "class/gender":
        class_actual = img_path.parent.parent.name
        gender = img_path.parent.name
        return output_folder / class_actual / gender
    else:  # split/class
        original_class = img_path.parent.name
        original_split = img_path.parent.parent.name
        target_class   = mapping.get(original_class, original_class) if mapping else original_class
        return output_folder / original_split / target_class
    
"""
Extraccion de regiones de interes (ojos y boca) de las imagenes utilizando MTCNN para la deteccion de rostros y keypoints.
"""
def extraction_roi(dataset: str, input_folder: Path, output_folder: Path):
    config = DATASET_CONFIG[dataset]
    detector = MTCNN()

    image_paths = [
        p for sub in input_folder.glob("*/*") if sub.is_dir()
        for ext in config["extensions"]
        for p in sub.glob(ext)
    ]

    if not image_paths:
        print("No images found.")
        return

    print(f"[{dataset.upper()}] Processing {len(image_paths)} images...\n")

    eyes_ok = mouths_ok = eyes_skip = mouths_skip = 0

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARN] Could not read: {img_path.name}")
            continue

        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        small   = cv2.resize(image_rgb, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        results = detector.detect_faces(small)

        if not results:
            continue

        face = results[0]
        confidence = face['confidence']
        keypoints = _scale_keypoints(face['keypoints'], DETECTION_SCALE)

        if confidence < MIN_FACE_CONF:
            print(f"[SKIP] Low confidence ({confidence:.2f}): {img_path.name}")
            eyes_skip += 1; mouths_skip += 1
            continue

        dest_dir = _get_dest_dir(img_path, config, output_folder)
        dest_dir.mkdir(parents=True, exist_ok=True)

        current_class = img_path.parent.parent.name

        # Region de los Ojos
        extract_eyes = config["extract_eyes"]
        if extract_eyes is None or current_class in extract_eyes:
            crop, w, h = _crop_eyes(image, keypoints, width, height)
            valid, reason = _validate_crop(crop, w, h, MIN_EYES_W, MIN_EYES_H)
            if not valid:
                print(f"[SKIP] Eyes {reason}: {img_path.name}")
                eyes_skip += 1
            else:
                cv2.imwrite(str(dest_dir / f"eyes_{img_path.name}"), crop)
                eyes_ok += 1

        # Region de la Boca
        extract_mouth = config["extract_mouth"]
        if extract_mouth is None or current_class in extract_mouth:
            crop, w, h = _crop_mouth(image, keypoints, width, height)
            valid, reason = _validate_crop(crop, w, h, MIN_MOUTH_W, MIN_MOUTH_H)
            if not valid:
                print(f"[SKIP] Mouth {reason}: {img_path.name}")
                mouths_skip += 1
            else:
                cv2.imwrite(str(dest_dir / f"mouth_{img_path.name}"), crop)
                mouths_ok += 1

    print(f"\nEyes → extracted: {eyes_ok:4d} | skipped: {eyes_skip}")
    print(f"Mouth → extracted: {mouths_ok:4d} | skipped: {mouths_skip}")


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent

    extraction_roi(
        dataset = "nitymed",
        input_folder = base / "data" / "raw" / "NITYMED",
        output_folder = base / "data" / "processed" / "nitymed_roi"
    )

    # extraction_roi(
        # dataset = "uta",
        # input_folder = base / "data" / "raw" / "UTA-RLDD",
        # output_folder = base / "data" / "processed" / "uta_roi"
    # )