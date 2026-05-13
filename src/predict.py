"""
Single Image Prediction Module
Load a trained model and predict the disease class for one image.
"""

import os
import json
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# ─── Constants ────────────────────────────────────────────────────────────────

IMAGE_SIZE = 128

DISEASE_INFO = {
    "Tomato_Bacterial_spot": {
        "display":     "Bacterial Spot",
        "description": "Small, dark, water-soaked spots that turn brown and may cause leaf drop.",
        "treatment":   "Apply copper-based bactericides. Remove infected plant debris. Avoid overhead irrigation.",
        "severity":    "Moderate"
    },
    "Tomato_Early_blight": {
        "display":     "Early Blight",
        "description": "Dark brown spots with concentric rings on older leaves, often surrounded by a yellow halo.",
        "treatment":   "Apply fungicides (chlorothalonil or mancozeb). Remove lower infected leaves. Rotate crops.",
        "severity":    "Moderate"
    },
    "Tomato_Late_blight": {
        "display":     "Late Blight",
        "description": "Large, dark, water-soaked lesions that can quickly destroy entire plants.",
        "treatment":   "Apply systemic fungicides immediately. Remove and destroy infected plants. Improve air circulation.",
        "severity":    "High"
    },
    "Tomato_Leaf_Mold": {
        "display":     "Leaf Mold",
        "description": "Pale greenish-yellow spots on upper surface with olive-green mold on lower surface.",
        "treatment":   "Improve ventilation. Apply fungicides. Reduce humidity in greenhouse conditions.",
        "severity":    "Moderate"
    },
    "Tomato_Septoria_leaf_spot": {
        "display":     "Septoria Leaf Spot",
        "description": "Small circular spots with dark borders and light gray centers, appearing first on lower leaves.",
        "treatment":   "Apply fungicides (chlorothalonil). Remove infected leaves. Avoid wetting foliage.",
        "severity":    "Moderate"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "display":     "Spider Mites",
        "description": "Stippling, discoloration, and bronzing of leaves caused by pest infestation.",
        "treatment":   "Apply miticides or insecticidal soap. Increase humidity. Introduce natural predators.",
        "severity":    "Moderate"
    },
    "Tomato__Target_Spot": {
        "display":     "Target Spot",
        "description": "Brown lesions with concentric rings resembling a target pattern on the leaf surface.",
        "treatment":   "Apply fungicides. Remove infected plant material. Ensure proper plant spacing.",
        "severity":    "Moderate"
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "display":     "Yellow Leaf Curl Virus",
        "description": "Yellowing and upward curling of leaves with stunted plant growth, spread by whiteflies.",
        "treatment":   "Control whitefly populations with insecticides. Remove infected plants. Use resistant varieties.",
        "severity":    "High"
    },
    "Tomato__Tomato_mosaic_virus": {
        "display":     "Mosaic Virus",
        "description": "Mosaic-like patterns of light and dark green on leaves with distortion and reduced fruit quality.",
        "treatment":   "Remove and destroy infected plants. Control aphid vectors. Disinfect tools regularly.",
        "severity":    "High"
    },
    "Tomato_healthy": {
        "display":     "Healthy",
        "description": "The leaf shows no signs of disease. The plant appears to be in good health.",
        "treatment":   "No treatment needed. Continue regular care and monitoring.",
        "severity":    "None"
    },
}


# ─── Prediction ───────────────────────────────────────────────────────────────

def load_model(models_dir: str = "models", model_filename: str = "custom_cnn_best.h5"):
    """Load the trained Keras model."""
    model_path = os.path.join(models_dir, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def load_class_names(models_dir: str = "models") -> list:
    """Load class names from JSON file."""
    path = os.path.join(models_dir, "class_names.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"class_names.json not found: {path}")
    with open(path) as f:
        return json.load(f)


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess a single image for inference."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)   # (1, 128, 128, 3)


def predict(image_path: str, model=None, class_names=None,
            models_dir: str = "models", model_filename: str = "custom_cnn_best.h5") -> dict:
    """
    Predict the disease class for a single tomato leaf image.

    Args:
        image_path:     Path to the input image
        model:          Pre-loaded Keras model (optional, loaded if None)
        class_names:    List of class names (optional, loaded if None)
        models_dir:     Directory containing saved models
        model_filename: Filename of the model to load

    Returns:
        dict with keys: class_name, display_name, confidence,
                        description, treatment, severity, top3
    """
    if model is None:
        model = load_model(models_dir, model_filename)
    if class_names is None:
        class_names = load_class_names(models_dir)

    img_array = preprocess_image(image_path)
    probs = model.predict(img_array, verbose=0)[0]   # Shape: (num_classes,)

    pred_idx  = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    # Top-3 predictions
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {
            "class":      class_names[i],
            "display":    DISEASE_INFO.get(class_names[i], {}).get("display", class_names[i]),
            "confidence": round(float(probs[i]) * 100, 2)
        }
        for i in top3_idx
    ]

    info = DISEASE_INFO.get(pred_class, {})

    return {
        "class_name":   pred_class,
        "display_name": info.get("display", pred_class),
        "confidence":   round(confidence, 2),
        "description":  info.get("description", ""),
        "treatment":    info.get("treatment", ""),
        "severity":     info.get("severity", "Unknown"),
        "top3":         top3,
    }


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_prediction(image_path: str, result: dict):
    """Display the image alongside the prediction result."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Image
    axes[0].imshow(img)
    axes[0].set_title("Input Image", fontsize=12)
    axes[0].axis("off")

    # Top-3 bar chart
    labels = [r["display"] for r in result["top3"]]
    confs  = [r["confidence"] for r in result["top3"]]
    colors = ["seagreen" if i == 0 else "steelblue" for i in range(len(labels))]

    axes[1].barh(labels[::-1], confs[::-1], color=colors[::-1])
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Confidence (%)")
    axes[1].set_title("Top-3 Predictions", fontsize=12)
    for i, (label, conf) in enumerate(zip(labels[::-1], confs[::-1])):
        axes[1].text(conf + 0.5, i, f"{conf:.1f}%", va="center", fontsize=10)
    axes[1].grid(axis="x", alpha=0.3)

    severity_color = {"None": "green", "Moderate": "orange", "High": "red"}.get(
        result["severity"], "gray"
    )
    fig.suptitle(
        f"Prediction: {result['display_name']}  |  Confidence: {result['confidence']:.1f}%  |  Severity: {result['severity']}",
        fontsize=13, color=severity_color, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict tomato leaf disease from an image.")
    parser.add_argument("image", help="Path to the tomato leaf image")
    parser.add_argument("--models-dir",  default="models",              help="Directory with saved models")
    parser.add_argument("--model-file",  default="custom_cnn_best.h5",  help="Model filename")
    parser.add_argument("--visualize",   action="store_true",           help="Show prediction visualization")
    args = parser.parse_args()

    result = predict(args.image, models_dir=args.models_dir, model_filename=args.model_file)

    print("\n" + "=" * 50)
    print(f"  Disease   : {result['display_name']}")
    print(f"  Confidence: {result['confidence']:.2f}%")
    print(f"  Severity  : {result['severity']}")
    print(f"  Description: {result['description']}")
    print(f"  Treatment : {result['treatment']}")
    print("\n  Top-3 Predictions:")
    for i, r in enumerate(result["top3"], 1):
        print(f"    {i}. {r['display']} — {r['confidence']:.2f}%")
    print("=" * 50)

    if args.visualize:
        visualize_prediction(args.image, result)
