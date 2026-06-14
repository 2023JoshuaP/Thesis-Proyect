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

def _augment_rotation(image: np.ndarray) -> np.ndarray:
    angle = random.uniform(-15, 15)
    h, w  = image.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def _augment_brightness(image: np.ndarray) -> np.ndarray:
    factor = random.uniform(0.6, 1.4)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def _augment_scale(image: np.ndarray) -> np.ndarray:
    factor = random.uniform(0.85, 1.15)
    h, w   = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    resized = cv2.resize(image, (new_w, new_h))
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

# ── Flip horizontal ──────────────────────────────────────────────────────────
def _augment_flip(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)
# ────────────────────────────────────────────────────────────────────────────

AUGMENT_POOL = [_augment_rotation, _augment_brightness, _augment_scale, _augment_flip]

def _apply_random_augmentation(image: np.ndarray) -> np.ndarray:
    transforms = random.sample(AUGMENT_POOL, k=random.randint(1, 2))
    for t in transforms:
        image = t(image)
    return image

def _get_base_name(img_path: Path) -> str:
    """Extrae el nombre base sin prefijo eyes_ o mouth_."""
    name = img_path.name
    for prefix in ("eyes_", "mouth_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

def split_and_augment(input_folder: Path, output_folder: Path):
    """
    1. Agrupa imágenes por nombre base (par eyes_ + mouth_ juntos)
    2. Divide pares en train/val/test estratificado (70/15/15)
    3. Aplica aumento solo al train (x3), manteniendo el par siempre junto
    """
    class_images: dict[str, dict[str, list[Path]]] = {}

    for class_dir in input_folder.iterdir():
        if not class_dir.is_dir():
            continue

        # Agrupar por nombre base → { "frame001.jpg": [eyes_frame001.jpg, mouth_frame001.jpg] }
        pairs: dict[str, list[Path]] = {}
        for gender_dir in class_dir.iterdir():
            if not gender_dir.is_dir():
                continue
            for img_path in gender_dir.glob("*.jpg"):
                base = _get_base_name(img_path)
                pairs.setdefault(base, []).append(img_path)

        # Solo pares completos (tienen eyes_ Y mouth_)
        complete_pairs = {k: v for k, v in pairs.items() if len(v) == 2}
        incomplete     = len(pairs) - len(complete_pairs)

        if incomplete > 0:
            print(f"[WARN] {incomplete} imágenes sin par completo en {class_dir.name} — descartadas")

        class_images[class_dir.name] = complete_pairs
        print(f"[{class_dir.name}] {len(complete_pairs)} pares completos encontrados")

    if not class_images:
        print("No images found.")
        return

    total_train = total_val = total_test = total_aug = 0

    for class_name, pairs in class_images.items():
        random.seed(RANDOM_SEED)

        base_names = list(pairs.keys())

        # Dividir por nombre base → el par siempre va junto
        train_bases, temp_bases = train_test_split(
            base_names, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
        )
        val_bases, test_bases = train_test_split(
            temp_bases, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
        )

        splits = {
            "train": train_bases,
            "val":   val_bases,
            "test":  test_bases,
        }

        for split_name, split_bases in splits.items():
            dest_dir = output_folder / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for base_name in split_bases:
                for img_path in pairs[base_name]:
                    shutil.copy(img_path, dest_dir / img_path.name)

            # Aumento solo en train — se aumenta el par completo con la misma transformación
            aug_count = 0
            if split_name == "train":
                for base_name in split_bases:
                    pair_imgs = pairs[base_name]
                    for i in range(AUGMENTATIONS_PER_IMAGE):
                        # Misma semilla para el par → misma transformación aplicada a ojos y boca
                        seed_i = RANDOM_SEED + hash(base_name) + i
                        random.seed(seed_i)
                        transforms = random.sample(AUGMENT_POOL, k=random.randint(1, 2))

                        for img_path in pair_imgs:
                            image = cv2.imread(str(img_path))
                            if image is None:
                                continue
                            augmented = image.copy()
                            for t in transforms:
                                augmented = t(augmented)
                            aug_name = f"aug{i+1}_{img_path.name}"
                            cv2.imwrite(str(dest_dir / aug_name), augmented)
                            aug_count += 1

            if split_name == "train":
                total_train += len(split_bases)
                total_aug   += aug_count
            elif split_name == "val":
                total_val += len(split_bases)
            elif split_name == "test":
                total_test += len(split_bases)

        print(f"\n[{class_name}]")
        print(f"  Train: {len(train_bases):4d} pares originales + {len(train_bases) * AUGMENTATIONS_PER_IMAGE * 2} imágenes aumentadas")
        print(f"  Val:   {len(val_bases):4d} pares")
        print(f"  Test:  {len(test_bases):4d} pares")

    print(f"\n{'─'*45}")
    print(f"Total train (con aumento): {total_train + total_aug}")
    print(f"Total val:                 {total_val}")
    print(f"Total test:                {total_test}")


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent

    split_and_augment(
        input_folder  = base / "data" / "processed" / "nitymed_roi",
        output_folder = base / "data" / "processed" / "nitymed_augmented"
    )