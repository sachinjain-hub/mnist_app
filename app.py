print(">>> app.py loaded")

import numpy as np
import cv2
import gradio as gr
import tensorflow as tf

# 1. Load the saved model
print(">>> Loading model...")
model = tf.keras.models.load_model("mnist_ann.keras")
print(">>> Model loaded successfully")

# 2. Prediction function
def predict_digit(img):
    """
    img: Numpy array (H, W, 3) from Gradio
    """
    if img is None:
        return -1

    # Convert RGB to Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # If you're drawing black digit on white background, invert:
    img = 255 - img

    # Optional: threshold to clean up
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Normalize (0–1), same as MNIST training
    img = img.astype("float32") / 255.0

    # Reshape for ANN model (1, 28, 28)
    img = img.reshape(1, 28, 28)

    # Predict
    y_prob = model.predict(img)
    digit = int(np.argmax(y_prob, axis=1)[0])

    return digit

# 3. Gradio interface
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="numpy", label="Draw or upload a digit"),
    outputs=gr.Number(label="Predicted Digit"),
    title="MNIST Digit Classifier",
    description="Draw a digit (0–9) or upload an image."
)

if __name__ == "__main__":
    print(">>> Launching Gradio...")
    demo.launch()
    print(">>> Gradio stopped")
