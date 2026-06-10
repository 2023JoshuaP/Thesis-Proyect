import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import random

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

AUGMENTATIONS_PER_IMAGE = 2  # x3 total (1 original + 2 aumentadas)

"""
Rotar entre -15 y +15 grados
"""
def _augment_rotation(image: np.ndarray) -> np.ndarray:
    angle = random.uniform(-15, 15)
    h, w  = image.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

"""
Ajustar brillo multiplicando el canal V en HSV por un factor entre 0.6 y 1.4
"""
def _augment_brightness(image: np.ndarray) -> np.ndarray:
    factor = random.uniform(0.6, 1.4)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

"""
Escalar entre 85% y 115% manteniendo el tamaño original (recortando o rellenando)
"""
def _augment_scale(image: np.ndarray) -> np.ndarray:
    factor = random.uniform(0.85, 1.15)
    h, w   = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    resized = cv2.resize(image, (new_w, new_h))
    # Recortar o rellenar para mantener tamaño original
    canvas = np.zeros_like(image)
    y_off = max(0, (new_h - h) // 2)
    x_off = max(0, (new_w - w) // 2)
    canvas_y = max(0, (h - new_h) // 2)
    canvas_x = max(0, (w - new_w) // 2)
    crop_h = min(new_h, h)
    crop_w = min(new_w, w)
    canvas[canvas_y:canvas_y+crop_h, canvas_x:canvas_x+crop_w] = \
        resized[y_off:y_off+crop_h, x_off:x_off+crop_w]
    return canvas

# Flip horizontal
"""
Voltear horizontalmente (efecto espejo) para simular la variabilidad natural de las posiciones de la cabeza y la dirección de la mirada.
"""
def _augment_flip(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)  # 1 = flip horizontal (efecto espejo)


AUGMENT_POOL = [_augment_rotation, _augment_brightness, _augment_scale, _augment_flip]

def _apply_random_augmentation(image: np.ndarray) -> np.ndarray:
    """Aplica 1 o 2 transformaciones aleatorias combinadas."""
    transforms = random.sample(AUGMENT_POOL, k=random.randint(1, 2))
    for t in transforms:
        image = t(image)
    return image

# División y aumento
def split_and_augment(input_folder: Path, output_folder: Path):
    """
    1. Recopila imágenes por clase (bostezo / microsueno)
    2. Divide en train/val/test estratificado (70/15/15)
    3. Aplica aumento solo al conjunto de entrenamiento (x3)
    """
    # Recopilar todas las imágenes agrupadas por clase
    class_images: dict[str, list[Path]] = {}

    for class_dir in input_folder.iterdir():
        if not class_dir.is_dir():
            continue
        images = [
            p for gender_dir in class_dir.iterdir() if gender_dir.is_dir()
            for p in gender_dir.glob("*.jpg")
        ]
        if images:
            class_images[class_dir.name] = images
            print(f"{class_dir.name}] {len(images)} imágenes encontradas")

    if not class_images:
        print("No images found.")
        return

    total_train = total_val = total_test = total_aug = 0

    for class_name, images in class_images.items():
        random.seed(RANDOM_SEED)

        # División estratificada
        train_imgs, temp_imgs = train_test_split(
            images, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
        )

        splits = {
            "train": train_imgs,
            "val":   val_imgs,
            "test":  test_imgs,
        }

        for split_name, split_imgs in splits.items():
            dest_dir = output_folder / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_imgs:
                shutil.copy(img_path, dest_dir / img_path.name)

            # Aumento solo en train
            aug_count = 0
            if split_name == "train":
                for img_path in split_imgs:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    for i in range(AUGMENTATIONS_PER_IMAGE):
                        augmented = _apply_random_augmentation(image.copy())
                        aug_name  = f"aug{i+1}_{img_path.name}"
                        cv2.imwrite(str(dest_dir / aug_name), augmented)
                        aug_count += 1

            if split_name == "train":
                total_train += len(split_imgs)
                total_aug += aug_count
            elif split_name == "val":
                total_val += len(split_imgs)
            elif split_name == "test":
                total_test += len(split_imgs)

        print(f"\n[{class_name}]")
        print(f"Train: {len(train_imgs):4d} originales + {len(train_imgs) * AUGMENTATIONS_PER_IMAGE} aumentadas")
        print(f"Val: {len(val_imgs):4d}")
        print(f"Test: {len(test_imgs):4d}")

    print(f"\n{'─'*45}")
    print(f"Total train (con aumento): {total_train + total_aug}")
    print(f"Total val: {total_val}")
    print(f"Total test: {total_test}")


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent

    split_and_augment(
        input_folder  = base / "data" / "processed" / "nitymed_roi",
        output_folder = base / "data" / "processed" / "nitymed_augmented"
    )