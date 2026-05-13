"""
Flask Web Application
AI-Based Tomato Leaf Disease Detection System
"""

import os
import uuid
import json
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np

from src.predict import predict, load_model, load_class_names, DISEASE_INFO

# ─── App Configuration ────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"]       = "tomato-disease-detection-2024"
app.config["UPLOAD_FOLDER"]    = os.path.join("static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB limit

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MODELS_DIR   = "models"
MODEL_FILE   = "custom_cnn_best.h5"

# ─── Load Model at Startup ────────────────────────────────────────────────────

model       = None
class_names = None

def load_resources():
    """Load model and class names once at startup."""
    global model, class_names
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        model = load_model(MODELS_DIR, MODEL_FILE)
        class_names = load_class_names(MODELS_DIR)
        print(f"Model loaded. Classes: {len(class_names)}")
    else:
        print(f"WARNING: Model not found at {model_path}. Train the model first.")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_upload_dir():
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Landing home page."""
    return render_template("home.html")


@app.route("/detect")
def detect():
    """Detection page with upload form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Handle image upload and return prediction result.
    Accepts multipart/form-data with field 'image'.
    Returns JSON with prediction details.
    """
    ensure_upload_dir()

    # Validate request
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload JPG, PNG, or BMP."}), 400

    # Check model is loaded
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 503

    # Save uploaded file with unique name
    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        result = predict(
            image_path=filepath,
            model=model,
            class_names=class_names,
            models_dir=MODELS_DIR,
            model_filename=MODEL_FILE
        )
        result["image_url"] = url_for("static", filename=f"uploads/{filename}")
        return jsonify(result)

    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/about")
def about():
    """About page with project information."""
    return render_template("about.html")


@app.route("/diseases")
def diseases():
    """Disease reference page."""
    return render_template("diseases.html", diseases=DISEASE_INFO)


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
        "classes":      len(class_names) if class_names else 0
    })


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_resources()
    app.run(debug=True, host="0.0.0.0", port=5000)
