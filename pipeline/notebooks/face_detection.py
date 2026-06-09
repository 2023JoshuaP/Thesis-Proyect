import cv2
from pathlib import Path
from mtcnn import MTCNN

def detect_faces_in_images():
    actual_directory = Path(__file__).parent
    input_folder = actual_directory / "dataset_images"
    output_folder = actual_directory / "dataset_faces"
    output_folder.mkdir(parents=True, exist_ok=True)

    detector = MTCNN()

    test_limit_images = 30

    for sub_class_folder in input_folder.glob("*/*"):
        if not sub_class_folder.is_dir():
            continue

        images = list(sub_class_folder.glob("*.jpg"))[:test_limit_images]

        if not images:
            print(f"No images found in {sub_class_folder}. Skipping.")
            continue

        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = detector.detect_faces(image_rgb)

            if results:
                face = results[0]
                x, y, width, height = face['box']

                x, y = max(0, x), max(0, y)

                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

                keypoints = face['keypoints']
                for point in keypoints.values():
                    cv2.circle(image, point, 3, (0, 0, 255), -1)
                
                output_name = f"{sub_class_folder.parent.name}_{sub_class_folder.name}_{image_path.name}"
                save_path = output_folder / output_name
                cv2.imwrite(str(save_path), image)
                print(f"Processed {image_path} -> {save_path}")
            else:
                print(f"No face detected in {image_path}. Skipping.")

if __name__ == "__main__":
    detect_faces_in_images()