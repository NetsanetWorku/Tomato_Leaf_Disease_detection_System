"""
Model Evaluation Module
Generates accuracy, precision, recall, F1-score, confusion matrix,
and classification report for trained models.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_model_and_classes(models_dir: str = "models", model_filename: str = "custom_cnn_best.h5"):
    """Load a saved Keras model and class names list."""
    model_path = os.path.join(models_dir, model_filename)
    class_path  = os.path.join(models_dir, "class_names.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"class_names.json not found: {class_path}")

    model = tf.keras.models.load_model(model_path)
    with open(class_path) as f:
        class_names = json.load(f)

    print(f"Loaded model: {model_path}")
    print(f"Classes ({len(class_names)}): {class_names}")
    return model, class_names


def load_test_set(models_dir: str = "models"):
    """Load the saved test set arrays."""
    X_test = np.load(os.path.join(models_dir, "X_test.npy"))
    y_test = np.load(os.path.join(models_dir, "y_test.npy"))
    print(f"Test set: {X_test.shape[0]} images")
    return X_test, y_test


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, class_names, batch_size=32):
    """
    Run full evaluation on the test set.
    Returns predictions and prints all metrics.
    """
    print("\nRunning predictions on test set ...")
    y_pred_probs = model.predict(X_test, batch_size=batch_size, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Overall accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall Test Accuracy: {acc * 100:.2f}%")

    # Per-class report
    display_names = [_display(c) for c in class_names]
    report = classification_report(
        y_test, y_pred,
        target_names=display_names,
        digits=4
    )
    print("\nClassification Report:")
    print(report)

    return y_pred, y_pred_probs


def _display(class_name: str) -> str:
    """Convert folder name to human-readable label."""
    mapping = {
        "Tomato_Bacterial_spot":        "Bacterial Spot",
        "Tomato_Early_blight":          "Early Blight",
        "Tomato_Late_blight":           "Late Blight",
        "Tomato_Leaf_Mold":             "Leaf Mold",
        "Tomato_Septoria_leaf_spot":    "Septoria Leaf Spot",
        "Tomato_Spider_mites":          "Spider Mites",
        "Tomato_Target_Spot":           "Target Spot",
        "Tomato_Yellow_Leaf_Curl_Virus":"Yellow Leaf Curl Virus",
        "Tomato_mosaic_virus":          "Mosaic Virus",
        "Tomato_healthy":               "Healthy",
    }
    return mapping.get(class_name, class_name)


# ─── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, class_names, save_dir: str = "models"):
    """Plot and save a normalized confusion matrix heatmap."""
    display_names = [_display(c) for c in class_names]
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        linewidths=0.5
    )
    plt.title("Confusion Matrix (Normalized)", fontsize=14, pad=15)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Confusion matrix saved → {save_path}")


# ─── Per-Class Bar Chart ──────────────────────────────────────────────────────

def plot_per_class_metrics(y_test, y_pred, class_names, save_dir: str = "models"):
    """Bar chart comparing precision, recall, and F1-score per class."""
    from sklearn.metrics import precision_recall_fscore_support

    display_names = [_display(c) for c in class_names]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=list(range(len(class_names)))
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label="Precision", color="steelblue")
    ax.bar(x,         recall,    width, label="Recall",    color="darkorange")
    ax.bar(x + width, f1,        width, label="F1-Score",  color="seagreen")

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision, Recall, and F1-Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "per_class_metrics.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Per-class metrics chart saved → {save_path}")


# ─── Compare Two Models ───────────────────────────────────────────────────────

def compare_models(models_dir: str = "models"):
    """
    Load both custom CNN and MobileNetV2, evaluate on the same test set,
    and print a side-by-side comparison.
    """
    X_test, y_test = load_test_set(models_dir)

    results = {}
    for model_file, label in [
        ("custom_cnn_best.h5",   "Custom CNN"),
        ("mobilenetv2_best.h5",  "MobileNetV2"),
    ]:
        path = os.path.join(models_dir, model_file)
        if not os.path.exists(path):
            print(f"Skipping {label} — file not found: {path}")
            continue

        model, class_names = load_model_and_classes(models_dir, model_file)
        y_pred_probs = model.predict(X_test, batch_size=32, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        acc = accuracy_score(y_test, y_pred)

        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred, average="weighted")
        results[label] = {"accuracy": acc, "f1_weighted": f1}
        print(f"{label}: Accuracy={acc*100:.2f}%  Weighted F1={f1:.4f}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    models_dir = "models"

    # Evaluate Custom CNN
    model, class_names = load_model_and_classes(models_dir, "custom_cnn_best.h5")
    X_test, y_test = load_test_set(models_dir)

    y_pred, _ = evaluate_model(model, X_test, y_test, class_names)
    plot_confusion_matrix(y_test, y_pred, class_names, models_dir)
    plot_per_class_metrics(y_test, y_pred, class_names, models_dir)

    # Compare models if MobileNetV2 is also trained
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    compare_models(models_dir)


if __name__ == "__main__":
    main()
