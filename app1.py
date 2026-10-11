from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import json
import numpy as np
from PIL import Image

app = Flask(__name__, template_folder="templates")

# --------------------------------------------------
# LOAD TRAINED AI MODEL
# --------------------------------------------------

MODEL_PATH = "model/fruit_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"

print("Loading AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("AI model loaded successfully!")
print("Classes:", class_names)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# PREDICTION API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ------------------------------------------
        # CHECK IMAGE
        # ------------------------------------------

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No image selected"
            }), 400

        # ------------------------------------------
        # READ IMAGE
        # ------------------------------------------

        image = Image.open(file).convert("RGB")

        # Resize according to trained model
        image = image.resize((224, 224))

        # Convert to NumPy
        image_array = np.array(image, dtype=np.float32)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # IMPORTANT:
        # Do NOT use preprocess_input here.
        # MobileNetV2 preprocessing is already inside
        # the trained model.

        # ------------------------------------------
        # AI PREDICTION
        # ------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )

        predicted_index = int(
            np.argmax(predictions[0])
        )

        # Keep confidence between 0 and 1
        confidence = float(
            predictions[0][predicted_index]
        )

        predicted_class = class_names[predicted_index]

        print(
            f"Prediction: {predicted_class} | "
            f"Confidence: {confidence * 100:.2f}%"
        )

        # ------------------------------------------
        # RETURN ACTUAL AI RESULT
        # ------------------------------------------

        return jsonify({
            "success": True,
            "class_name": predicted_class,
            "confidence": confidence
        })

    except Exception as e:

        print("Prediction Error:", str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# --------------------------------------------------
# RUN FLASK SERVER
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True
    )