import cv2
from pathlib import Path
from mtcnn import MTCNN

def extraction_roi():
    actual_directory = Path(__file__).parent
    input_folder = actual_directory / "dataset_images"
    output_folder = actual_directory / "dataset_roi"

    detector = MTCNN()

    limit_per_folder = 20
    path_images = []

    for subfolder in input_folder.glob("*/*"):
        if subfolder.is_dir():
            sample_images = list(subfolder.glob("*.jpg"))[:limit_per_folder]
            path_images.extend(sample_images)

    if not path_images:
        print("No images found in the input folder.")
        return
    
    # Parámetros de padding (margen en píxeles)
    pad_eyes_x, pad_eyes_y = 30, 20
    pad_mouth_x, pad_mouth_y = 25, 25

    eyes_extracted = 0
    mouth_extracted = 0

    for path_image in path_images:
        image = cv2.imread(str(path_image))
        if image is None:
            print(f"Could not read image: {path_image}")
            continue

        height_image, width_image, _ = image.shape 
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image_rgb)

        if results:
            keypoints = results[0]['keypoints']

            class_actual = path_image.parent.parent.name
            gender = path_image.parent.name

            path_destine_dir = output_folder / class_actual / gender
            path_destine_dir.mkdir(parents=True, exist_ok=True)

            # 1. EXTRACCIÓN DE OJOS
            if class_actual in ["microsueno", "normal"]:
                eye_left = keypoints['left_eye']
                eye_right = keypoints['right_eye']

                x_min = max(0, min(eye_left[0], eye_right[0]) - pad_eyes_x)
                y_min = max(0, min(eye_left[1], eye_right[1]) - pad_eyes_y)
                x_max = min(width_image, max(eye_left[0], eye_right[0]) + pad_eyes_x)
                y_max = min(height_image, max(eye_left[1], eye_right[1]) + pad_eyes_y)

                recorted_eyes = image[y_min:y_max, x_min:x_max]

                if recorted_eyes.size > 0:
                    file_name = f"eyes_{path_image.name}"
                    cv2.imwrite(str(path_destine_dir / file_name), recorted_eyes)
                    eyes_extracted += 1

            # 2. EXTRACCIÓN DE BOCA
            if class_actual in ["bostezo", "normal"]:
                mouth_left = keypoints['mouth_left']
                mouth_right = keypoints['mouth_right']

                y_min = max(0, min(mouth_left[1], mouth_right[1]) - pad_mouth_y)
                y_max = min(height_image, max(mouth_left[1], mouth_right[1]) + pad_mouth_y + 20)  
                x_min = max(0, min(mouth_left[0], mouth_right[0]) - pad_mouth_x)
                x_max = min(width_image, max(mouth_left[0], mouth_right[0]) + pad_mouth_x)

                recorted_mouth = image[y_min:y_max, x_min:x_max]

                if recorted_mouth.size > 0:
                    file_name = f"mouth_{path_image.name}"
                    cv2.imwrite(str(path_destine_dir / file_name), recorted_mouth)
                    mouth_extracted += 1

if __name__ == "__main__":
    extraction_roi()