from flask import Flask, render_template, request
import tensorflow as tf
import joblib
from utils import preprocess_image
import os

from tensorflow.keras.losses import Loss
from keras.saving import register_keras_serializable

@register_keras_serializable()
class FocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss", reduction="sum_over_batch_size"):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.math.pow(1 - y_pred, self.gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)

app = Flask(__name__)

# Load model with custom loss
model = tf.keras.models.load_model(
    "C:/Users/www12/OneDrive/Documents/Projects/material_recognizer_ml/model/final_material_model.keras",
    custom_objects={"FocalLoss": FocalLoss}
)

label_encoder = joblib.load("C:/Users/www12/OneDrive/Documents/Projects/material_recognizer_ml/model/final_label_encoder.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded", 400

    image = request.files["image"]
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0]
    class_index = prediction.argmax()
    class_label = label_encoder.inverse_transform([class_index])[0]
    confidence = prediction[class_index] * 100

    return render_template("result.html", label=class_label, confidence=round(confidence, 2))

if __name__ == "__main__":
    app.run(debug=True)
