"""
Model Training Script — tf.data pipeline version
Loads images on-the-fly (no full RAM preload), trains faster on CPU.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_custom_cnn, build_mobilenetv2, compile_model

# ─── Configuration ────────────────────────────────────────────────────────────

DATASET_DIR  = "dataset"
MODELS_DIR   = "models"
IMAGE_SIZE   = 128
BATCH_SIZE   = 32
EPOCHS       = 50
LEARNING_RATE= 1e-3
SEED         = 42

# Exact folder names in dataset/
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
NUM_CLASSES = len(CLASS_NAMES)


# ─── Build tf.data Dataset ────────────────────────────────────────────────────

def build_tf_dataset(dataset_dir, class_names, split="train",
                     train_ratio=0.70, val_ratio=0.15,
                     batch_size=BATCH_SIZE, seed=SEED):
    """
    Build a tf.data.Dataset that loads images on-the-fly.
    No full RAM preload — much faster startup and lower memory use.
    """
    all_paths, all_labels = [], []

    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"WARNING: {cls_dir} not found, skipping.")
            continue
        files = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        all_paths.extend(files)
        all_labels.extend([idx] * len(files))
        print(f"  {cls}: {len(files)} images")

    total = len(all_paths)
    print(f"\nTotal images found: {total}")

    # Shuffle with fixed seed for reproducibility
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)
    all_paths  = np.array(all_paths)[indices]
    all_labels = np.array(all_labels)[indices]

    # Split indices
    n_train = int(total * train_ratio)
    n_val   = int(total * val_ratio)

    if split == "train":
        paths, labels = all_paths[:n_train], all_labels[:n_train]
    elif split == "val":
        paths, labels = all_paths[n_train:n_train+n_val], all_labels[n_train:n_train+n_val]
    else:  # test
        paths, labels = all_paths[n_train+n_val:], all_labels[n_train+n_val:]

    print(f"  {split}: {len(paths)} images")

    # Save test paths/labels for evaluation
    if split == "test":
        np.save(os.path.join(MODELS_DIR, "test_paths.npy"),  paths)
        np.save(os.path.join(MODELS_DIR, "test_labels.npy"), labels)

    # Build tf.data pipeline
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        label = tf.one_hot(label, NUM_CLASSES)
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if split == "train":
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=1000, seed=seed)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ─── Callbacks ────────────────────────────────────────────────────────────────

def get_callbacks(model_name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.h5")
    return [
        ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-7, verbose=1),
    ], ckpt_path


# ─── Plot Training History ────────────────────────────────────────────────────

def plot_history(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"],     label="Train",      color="steelblue")
    axes[0].plot(history.history["val_accuracy"], label="Validation", color="darkorange")
    axes[0].set_title(f"{model_name} — Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train",      color="steelblue")
    axes[1].plot(history.history["val_loss"], label="Validation", color="darkorange")
    axes[1].set_title(f"{model_name} — Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(MODELS_DIR, f"{model_name}_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save class names
    with open(os.path.join(MODELS_DIR, "class_names.json"), "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)

    print("=" * 60)
    print("Building datasets (tf.data — no RAM preload) ...")
    print("=" * 60)

    train_ds = build_tf_dataset(DATASET_DIR, CLASS_NAMES, split="train")
    val_ds   = build_tf_dataset(DATASET_DIR, CLASS_NAMES, split="val")
    test_ds  = build_tf_dataset(DATASET_DIR, CLASS_NAMES, split="test")

    # ── Train Custom CNN ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Custom CNN")
    print("=" * 60)

    cnn = build_custom_cnn(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES
    )
    cnn = compile_model(cnn, learning_rate=LEARNING_RATE)
    cnn.summary()

    callbacks, _ = get_callbacks("custom_cnn")

    history = cnn.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    plot_history(history, "Custom_CNN")
    cnn.save(os.path.join(MODELS_DIR, "custom_cnn_final.h5"))
    print(f"Custom CNN saved → {MODELS_DIR}/custom_cnn_final.h5")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\nEvaluating on test set ...")
    loss, acc = cnn.evaluate(test_ds, verbose=1)
    print(f"Test Accuracy: {acc*100:.2f}%  |  Test Loss: {loss:.4f}")

    # ── Train MobileNetV2 ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training MobileNetV2 (Transfer Learning)")
    print("=" * 60)

    mob, base = build_mobilenetv2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES
    )
    mob = compile_model(mob, learning_rate=LEARNING_RATE)

    callbacks_mob, _ = get_callbacks("mobilenetv2")

    # Phase 1 — frozen base
    print("[Phase 1] Training head only ...")
    h1 = mob.fit(train_ds, epochs=20, validation_data=val_ds,
                 callbacks=callbacks_mob, verbose=1)

    # Phase 2 — fine-tune top layers
    print("[Phase 2] Fine-tuning top layers ...")
    base.trainable = True
    for layer in base.layers[:100]:
        layer.trainable = False

    mob.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_ft, _ = get_callbacks("mobilenetv2_finetune")
    h2 = mob.fit(train_ds, epochs=EPOCHS, validation_data=val_ds,
                 callbacks=callbacks_ft, verbose=1)

    mob.save(os.path.join(MODELS_DIR, "mobilenetv2_final.h5"))
    print(f"MobileNetV2 saved → {MODELS_DIR}/mobilenetv2_final.h5")

    loss2, acc2 = mob.evaluate(test_ds, verbose=1)
    print(f"MobileNetV2 Test Accuracy: {acc2*100:.2f}%")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Custom CNN  : {acc*100:.2f}%")
    print(f"MobileNetV2 : {acc2*100:.2f}%")
    print(f"Models saved in: {MODELS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
