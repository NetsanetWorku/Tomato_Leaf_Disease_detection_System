# AI-Based Tomato Leaf Disease Detection System

A deep learning system using CNN to detect tomato leaf diseases from images.

## Project Structure

```
tomato-disease-detection/
├── dataset/                  # Place PlantVillage dataset here
├── models/                   # Saved trained models
├── static/                   # Flask static files (CSS, JS, uploads)
├── templates/                # HTML templates
├── src/
│   ├── preprocess.py         # Data preprocessing & augmentation
│   ├── model.py              # CNN model architecture
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Model evaluation & metrics
│   └── predict.py            # Single image prediction
├── app.py                    # Flask web application
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download the PlantVillage dataset (tomato classes) from:
- Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease
- GitHub: https://github.com/spMohanty/PlantVillage-Dataset

Place the tomato folders inside `dataset/` directory:
```
dataset/
├── Tomato_Bacterial_spot/
├── Tomato_Early_blight/
├── Tomato_Late_blight/
├── Tomato_Leaf_Mold/
├── Tomato_Septoria_leaf_spot/
├── Tomato_Spider_mites/
├── Tomato_Target_Spot/
├── Tomato_Yellow_Leaf_Curl_Virus/
├── Tomato_mosaic_virus/
└── Tomato_healthy/
```

### 3. Train the Model
```bash
python src/train.py
```

### 4. Run the Web Application
```bash
python app.py
```
Then open http://localhost:5000 in your browser.

## Disease Classes (10 Classes)
| Class | Type |
|-------|------|
| Healthy | Normal |
| Early Blight | Fungal |
| Late Blight | Oomycete |
| Leaf Mold | Fungal |
| Septoria Leaf Spot | Fungal |
| Spider Mites | Pest |
| Target Spot | Fungal |
| Yellow Leaf Curl Virus | Viral |
| Mosaic Virus | Viral |
| Bacterial Spot | Bacteria |

## Model Architecture
- Custom CNN with Conv2D, BatchNorm, MaxPooling, Dropout layers
- Input: 128×128×3 RGB images
- Output: 10-class Softmax
- Transfer learning with MobileNetV2 (alternative)

## Expected Performance
- Overall Accuracy: ~93%
- Weighted F1-Score: ~0.93
