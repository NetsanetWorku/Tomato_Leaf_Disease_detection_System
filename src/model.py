"""
CNN Model Architecture Module
Defines the custom CNN and MobileNetV2 transfer learning models.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2

# ─── Constants ────────────────────────────────────────────────────────────────

IMAGE_SIZE  = 128
NUM_CLASSES = 10


# ─── Custom CNN ───────────────────────────────────────────────────────────────

def build_custom_cnn(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=NUM_CLASSES):
    """
    Custom CNN architecture:
      3 × [Conv2D → BatchNorm → MaxPool → Dropout]
      Flatten → Dense(512) → Dropout → Dense(num_classes, softmax)

    Args:
        input_shape: Tuple (H, W, C)
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name="TomatoCNN")

    # ── Block 1 ──────────────────────────────────────────────────────────────
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                            input_shape=input_shape,
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Block 2 ──────────────────────────────────────────────────────────────
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Block 3 ──────────────────────────────────────────────────────────────
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.30))

    # ── Block 4 ──────────────────────────────────────────────────────────────
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.30))

    # ── Classifier Head ───────────────────────────────────────────────────────
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu",
                           kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.50))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


# ─── MobileNetV2 Transfer Learning ────────────────────────────────────────────

def build_mobilenetv2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=NUM_CLASSES):
    """
    MobileNetV2 transfer learning model.
    Base weights from ImageNet; top layers replaced for tomato classification.

    Training strategy:
      Phase 1 — freeze base, train head only
      Phase 2 — unfreeze top layers for fine-tuning (done in train.py)

    Returns:
        Compiled Keras model (Phase 1 ready)
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False   # Freeze for Phase 1

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="TomatoMobileNetV2")
    return model, base_model


# ─── Compile Helper ───────────────────────────────────────────────────────────

def compile_model(model, learning_rate=1e-3):
    """Compile model with Adam optimizer and categorical cross-entropy loss."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─── Summary ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Custom CNN Architecture")
    print("=" * 60)
    cnn = build_custom_cnn()
    compile_model(cnn)
    cnn.summary()

    print("\n" + "=" * 60)
    print("MobileNetV2 Transfer Learning Architecture")
    print("=" * 60)
    mob, _ = build_mobilenetv2()
    compile_model(mob)
    mob.summary()
