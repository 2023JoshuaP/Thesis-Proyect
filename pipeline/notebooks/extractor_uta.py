import cv2
from pathlib import Path
from mtcnn import MTCNN

def process_uta_images():
    current_dir = Path(__file__).parent
    input_folder = current_dir / "UTA-RLDD"
    output_folder = current_dir / "dataset_uta_roi"

    detector = MTCNN()

    class_mapping = {
        "active": "normal",
        "fatigue": "fatigue" 
    }

    # Tamaño mínimo aceptable para el crop de boca (ancho x alto en píxeles)
    MIN_MOUTH_W = 30
    MIN_MOUTH_H = 20
    # Confianza mínima de detección facial (0.0 - 1.0)
    MIN_FACE_CONFIDENCE = 0.95

    image_paths = []
    for subfolder in input_folder.glob("*/*"):
        if subfolder.is_dir():
            images = list(subfolder.glob("*.jpg")) + list(subfolder.glob("*.png"))
            image_paths.extend(images)

    if not image_paths:
        print("No images found in the input folder.")
        return

    print(f"\nStarting ROI extraction for all {len(image_paths)} images.\n")

    pad_eyes_x, pad_eyes_y = 30, 20
    pad_mouth_x, pad_mouth_y = 25, 25

    eyes_extracted = 0
    mouths_extracted = 0
    mouths_skipped = 0

    for img_path in image_paths:
        original_class = img_path.parent.name
        original_split = img_path.parent.parent.name
        target_class = class_mapping.get(original_class, original_class)

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read image {img_path.name}")
            continue

        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image_rgb)

        if not results:
            continue

        face = results[0]
        confidence = face['confidence']
        keypoints = face['keypoints']

        target_dir = output_folder / original_split / target_class
        target_dir.mkdir(parents=True, exist_ok=True)

        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']

        x_min_eye = max(0, min(left_eye[0], right_eye[0]) - pad_eyes_x)
        y_min_eye = max(0, min(left_eye[1], right_eye[1]) - pad_eyes_y)
        x_max_eye = min(width, max(left_eye[0], right_eye[0]) + pad_eyes_x)
        y_max_eye = min(height, max(left_eye[1], right_eye[1]) + pad_eyes_y)

        cropped_eyes = image[y_min_eye:y_max_eye, x_min_eye:x_max_eye]
        if cropped_eyes.size > 0:
            cv2.imwrite(str(target_dir / f"eyes_{img_path.name}"), cropped_eyes)
            eyes_extracted += 1

        # Validaciones de la región de boca
        left_mouth = keypoints['mouth_left']
        right_mouth = keypoints['mouth_right']

        y_min_mouth = max(0, min(left_mouth[1], right_mouth[1]) - pad_mouth_y)
        y_max_mouth = min(height, max(left_mouth[1], right_mouth[1]) + pad_mouth_y + 20)
        x_min_mouth = max(0, min(left_mouth[0], right_mouth[0]) - pad_mouth_x)
        x_max_mouth = min(width, max(left_mouth[0], right_mouth[0]) + pad_mouth_x)

        crop_w = x_max_mouth - x_min_mouth
        crop_h = y_max_mouth - y_min_mouth

        # Validación 1: confianza baja → keypoints poco confiables
        if confidence < MIN_FACE_CONFIDENCE:
            print(f"Skipping mouth (low confidence {confidence:.2f}): {img_path.name}")
            mouths_skipped += 1
            continue

        # Validación 2: crop demasiado pequeño → boca probablemente tapada
        if crop_w < MIN_MOUTH_W or crop_h < MIN_MOUTH_H:
            print(f"Skipping mouth (crop too small {crop_w}x{crop_h}): {img_path.name}")
            mouths_skipped += 1
            continue

        cropped_mouth = image[y_min_mouth:y_max_mouth, x_min_mouth:x_max_mouth]
        if cropped_mouth.size > 0:
            cv2.imwrite(str(target_dir / f"mouth_{img_path.name}"), cropped_mouth)
            mouths_extracted += 1

    print(f"\nEye regions extracted:   {eyes_extracted}")
    print(f"Mouth regions extracted: {mouths_extracted}")
    print(f"Mouth regions skipped:   {mouths_skipped}")

if __name__ == "__main__":
    process_uta_images()