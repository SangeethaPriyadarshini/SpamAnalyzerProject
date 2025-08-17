import pickle
import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io


model = tf.keras.models.load_model("spam_lstm_model.h5", compile=False)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect keras.src.* -> tensorflow.keras.*
        if module.startswith("keras.src."):
            module = module.replace("keras.src.", "tensorflow.keras.")
        return super().find_class(module, name)

def load_tokenizer(filename):
    with open(filename, "rb") as f:
        return SafeUnpickler(f).load()

# Load tokenizer safely
tokenizer = load_tokenizer("tokenizer.pkl")


# Flask App

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        message = request.form["message"]

        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([message])
        padded = pad_sequences(sequence, maxlen=100)

        # Predict
        prediction = model.predict(padded)[0][0]

        # Threshold = 0.5
        if prediction >= 0.5:
            label = "Spam"
            confidence = round(prediction * 100, 2)
        else:
            label = "Ham"
            confidence = round((1 - prediction) * 100, 2)

        return render_template(
            "index.html",
            prediction_text=f"Message is {label} with {confidence}% confidence."
        )

# -------------------------
# Run Flask App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
