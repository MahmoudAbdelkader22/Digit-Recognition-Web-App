import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

model = tf.keras.models.load_model("digit_recognition_model.keras")

def process_image(img_pil):
    image = img_pil.convert("L").resize((28, 28))
    arr = np.array(image)
    if arr.mean() > 127:
        image = ImageOps.invert(image)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array, image

st.set_page_config(page_title="Digit Recognition Web App")
st.title("Handwritten Digit Recognition")
st.write("Upload an image or draw a digit to predict it:")

input_mode = st.sidebar.radio("Choose input mode:", ["Upload Image", "Draw Digit"])

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload a digit image (PNG/JPG/JPEG):")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", width=150)
        img_array, processed_img = process_image(image)
        prediction = model.predict(img_array)
        st.image(processed_img, caption="Processed Image (28x28)", width=150)
        st.success(f"Predicted Digit: {np.argmax(prediction)}")
    else:
        st.info("Please upload a digit image.")
elif input_mode == "Draw Digit":
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    if st.button("Predict"):
        if canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data[:, :, 0].astype("uint8"))
            img_array, processed_img = process_image(img)
            prediction = model.predict(img_array)
            st.image(processed_img.resize((140, 140)), caption="Processed", width=100)
            st.success(f"Predicted Digit: {np.argmax(prediction)}")
