import cv2
from pathlib import Path
from mtcnn import MTCNN

def cut_faces():
    actual_directory = Path(__file__).parent
    input_folder = actual_directory / "dataset_images"
    output_folder = actual_directory / "dataset_cut_faces"

    detector = MTCNN()

    path_images = list(input_folder.rglob("*.jpg"))

    if not path_images:
        print("No images found in the input folder.")
        return
    
    saved_faces = 0
    images_without_faces = 0

    for image_path in path_images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        width_image, height_image, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = detector.detect_faces(image_rgb)

        if results:
            face = results[0]
            x, y, width, height = face['box']

            x_initial = max(0, x)
            y_initial = max(0, y)
            x_final = min(width_image, x + width)
            y_final = min(height_image, y + height)

            face_image = image[y_initial:y_final, x_initial:x_final]

            if face_image.size > 0:
                subfolder_class = image_path.parent.parent.name
                subfolder_gender = image_path.parent.name

                output_destine_path = output_folder / subfolder_class / subfolder_gender
                output_destine_path.mkdir(parents=True, exist_ok=True)

                save_path = output_destine_path / image_path.name
                cv2.imwrite(str(save_path), face_image)
                saved_faces += 1
        else:
            images_without_faces += 1

if __name__ == "__main__":
    cut_faces()