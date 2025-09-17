# ♻️ Material Recognizer Web App

A deep learning-based web application that classifies waste material images into five categories — **cardboard**, **glass**, **metal**, **paper**, and **plastic** — using **MobileNetV2 transfer learning** and a custom **Focal Loss** function to improve classification on imbalanced datasets.

---

## 📌 Project Highlights

- 🧠 **Transfer Learning** with `MobileNetV2` (pretrained on ImageNet)
- 📸 Supports image upload and prediction in real-time
- ⚙️ Custom **Focal Loss** for imbalanced material classes
- 🧼 Image preprocessing (crop, resize, normalize)
- 🎨 Beautiful frontend built with **HTML/CSS**
- 🚀 Deployed using **Flask**
- ✅ Model trained on a dataset of 2,390 images
- 🏷️ Classes: Cardboard, Glass, Metal, Paper, Plastic

---

## 📁 Project Structure

```

material\_recognizer\_ml/
├── app/
│   ├── static/
│   │   └── styles.css                # Stylish frontend UI
│   ├── templates/
│   │   ├── index.html               # Upload interface
│   │   └── result.html              # Prediction result display
│   ├── app.py                       # Flask backend with Focal Loss integration
│   └── utils.py                     # Image preprocessing logic
├── model/
│   ├── final\_material\_model.keras   # Saved trained MobileNetV2 model
│   └── final\_label\_encoder.pkl      # Label encoder to map class indices
├── 3mobilenetv2\_transfer\_learning.ipynb   # Full model training notebook
├── dataset-resized.zip             # (Optional) Preprocessed dataset

````

---

## 🧠 Model Architecture

- **Base Model**: `MobileNetV2` (weights = "imagenet", include_top=False)
- **Trainable Layers**: Fine-tuned with frozen base initially, then some layers unfrozen
- **Head Layers**:
  - `GlobalAveragePooling2D`
  - `Dense(128, activation='relu')`
  - `Dropout(0.5)`
  - `Dense(5, activation='softmax')`

---

## ⚙️ Custom Loss: Focal Loss

The project uses **Focal Loss** instead of CrossEntropy to handle **class imbalance** in the material dataset.

```python
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        ...
    def call(self, y_true, y_pred):
        ...
````

Used in training and also loaded during Flask model inference via:

```python
model = tf.keras.models.load_model("final_material_model.keras", custom_objects={"FocalLoss": FocalLoss})
```

---

## 🧪 Dataset Details

* Total Images: **2,390**
* Resolution: **96x96**
* Classes: **cardboard, glass, metal, paper, plastic**
* Split: 80% Train, 10% Validation, 10% Test

---

## 🔧 Training Configuration

| Parameter         | Value           |
| ----------------- | --------------- |
| Base Model        | MobileNetV2     |
| Loss Function     | FocalLoss       |
| Optimizer         | Adam            |
| Batch Size        | 32              |
| Epochs            | 25–30           |
| Augmentation      | Yes             |
| Dropout           | 0.5             |
| Learning Rate     | Default (0.001) |
| Accuracy Achieved | \~83%           |

---

## 🧰 Image Preprocessing (utils.py)

* Reads and decodes uploaded image
* Crops to center square
* Resizes to (96, 96)
* Normalizes pixel values to \[0, 1]

```python
def preprocess_image(file):
    ...
    return np.expand_dims(normalized, axis=0)
```

---

## 🌐 Web App Features

### Frontend (HTML + CSS)

* Upload interface (`index.html`)
* Result display (`result.html`)
* Responsive and modern design (`styles.css`)

### Backend (Flask)

* `app.py`:

  * Loads model and label encoder
  * Handles image upload and inference
  * Renders predictions with confidence percentage

---

## ▶️ How to Run the Project

### 🛠️ 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

<sub>📌 You can generate `requirements.txt` with `pip freeze > requirements.txt`.</sub>

---

### ▶️ 2. Run Flask App

```bash
cd app
python app.py
```

Then open in your browser: `http://localhost:5000`

---

## 🧪 Example Prediction Flow

1. Upload an image of a recyclable item (e.g., metal can).
2. Backend preprocesses and classifies using the MobileNetV2 model.
3. Result is shown with predicted class and confidence score.

---

## 📊 Evaluation

* Model tested on hold-out test set (10%)
* Achieved >83% classification accuracy
* High performance on majority classes (glass, plastic)

---

## 📦 Model Artifacts

* `final_material_model.keras`: Trained model saved using Keras format.
* `final_label_encoder.pkl`: Saved `LabelEncoder` object to reverse class indices.

---

---

## 🤝 Acknowledgements

* [TensorFlow](https://www.tensorflow.org/)
* [Keras Applications (MobileNetV2)](https://keras.io/api/applications/mobilenet/)
* [Focal Loss Paper](https://arxiv.org/abs/1708.02002)

---

## 🧾 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

