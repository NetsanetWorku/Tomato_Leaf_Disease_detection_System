"""
Data Preprocessing Module
Handles image loading, resizing, normalization, augmentation, and dataset splitting.
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ─── Constants ────────────────────────────────────────────────────────────────

IMAGE_SIZE = 128          # Target image size (128×128)
BATCH_SIZE = 32
DATASET_DIR = "dataset"   # Root folder containing class sub-folders

# Disease class names — must match folder names exactly in dataset/
CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

# Human-readable display names
DISPLAY_NAMES = {
    "Tomato_Bacterial_spot":                      "Bacterial Spot",
    "Tomato_Early_blight":                        "Early Blight",
    "Tomato_Late_blight":                         "Late Blight",
    "Tomato_Leaf_Mold":                           "Leaf Mold",
    "Tomato_Septoria_leaf_spot":                  "Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite":"Spider Mites",
    "Tomato__Target_Spot":                        "Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus":      "Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus":                "Mosaic Virus",
    "Tomato_healthy":                             "Healthy",
}

NUM_CLASSES = len(CLASS_NAMES)


# ─── Image Loading ─────────────────────────────────────────────────────────────

def load_dataset(dataset_dir: str = DATASET_DIR, allowed_classes: list = None):
    """
    Load all images from the dataset directory.
    Args:
        dataset_dir:     Root folder containing class sub-folders
        allowed_classes: If provided, only load these class folders (filters out Potato/Pepper etc.)
    Returns:
        images (np.ndarray): Array of shape (N, 128, 128, 3), float32, range [0,1]
        labels (np.ndarray): Integer class indices
        class_names (list): Ordered list of class folder names
    """
    images = []
    labels = []

    # Discover classes from folder names
    available = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    # Filter to only allowed classes if specified
    if allowed_classes is not None:
        class_names = [c for c in allowed_classes if c in available]
        missing = [c for c in allowed_classes if c not in available]
        if missing:
            print(f"WARNING: These classes not found in dataset: {missing}")
    else:
        class_names = available

    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    print(f"Found {len(class_names)} classes: {class_names}")

    for cls in class_names:
        cls_dir = os.path.join(dataset_dir, cls)
        files = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        print(f"  Loading {len(files)} images from '{cls}' ...")

        for fname in files:
            img_path = os.path.join(cls_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            images.append(img)
            labels.append(class_to_idx[cls])

    images = np.array(images, dtype="float32") / 255.0   # Normalize to [0, 1]
    labels = np.array(labels, dtype="int32")

    print(f"\nTotal images loaded: {len(images)}")
    return images, labels, class_names


# ─── Dataset Splitting ─────────────────────────────────────────────────────────

def split_dataset(images, labels, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Stratified split into train / validation / test sets.
    Default: 70% / 15% / 15%
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed
    )

    # Second split: val vs test
    val_fraction = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_fraction),
        stratify=y_temp,
        random_state=seed
    )

    print(f"\nDataset split:")
    print(f"  Train      : {len(X_train)} images ({train_ratio*100:.0f}%)")
    print(f"  Validation : {len(X_val)} images ({val_ratio*100:.0f}%)")
    print(f"  Test       : {len(X_test)} images ({test_ratio*100:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─── Data Generators ──────────────────────────────────────────────────────────

def get_data_generators(X_train, y_train, X_val, y_val, num_classes, batch_size=BATCH_SIZE):
    """
    Create Keras ImageDataGenerators with augmentation for training
    and simple normalization for validation.
    """
    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat   = to_categorical(y_val,   num_classes)

    # Training augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    )

    # Validation — no augmentation
    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow(X_train, y_train_cat, batch_size=batch_size, shuffle=True)
    val_gen   = val_datagen.flow(X_val,   y_val_cat,   batch_size=batch_size, shuffle=False)

    return train_gen, val_gen


# ─── Preprocessing a Single Image (for inference) ─────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image for model inference.
    Returns array of shape (1, 128, 128, 3).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)   # Shape: (1, 128, 128, 3)


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_sample_images(images, labels, class_names, n=10):
    """Display a grid of sample images with their class labels."""
    plt.figure(figsize=(20, 4))
    indices = np.random.choice(len(images), n, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(2, n // 2, i + 1)
        plt.imshow(images[idx])
        plt.title(DISPLAY_NAMES.get(class_names[labels[idx]], class_names[labels[idx]]),
                  fontsize=8)
        plt.axis("off")
    plt.suptitle("Sample Tomato Leaf Images", fontsize=14)
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=150)
    plt.show()
    print("Saved sample_images.png")


def plot_class_distribution(labels, class_names):
    """Bar chart of image counts per class."""
    counts = [np.sum(labels == i) for i in range(len(class_names))]
    display = [DISPLAY_NAMES.get(c, c) for c in class_names]

    plt.figure(figsize=(12, 5))
    bars = plt.bar(display, counts, color="steelblue", edgecolor="black")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.ylabel("Number of Images")
    plt.title("Class Distribution — PlantVillage Tomato Dataset")
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 str(count), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.show()
    print("Saved class_distribution.png")


# ─── Main (quick test) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    images, labels, class_names = load_dataset()
    plot_class_distribution(labels, class_names)
    plot_sample_images(images, labels, class_names)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(images, labels)
    print("\nPreprocessing complete.")
