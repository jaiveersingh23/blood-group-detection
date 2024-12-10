# app.py

import streamlit as st
from PIL import Image
from src.predict import predict_blood_group

# Streamlit UI
st.title("Blood Group Prediction from Fingerprint")
st.write("Upload a fingerprint image, and the model will predict the blood group.")

# File uploader
uploaded_file = st.file_uploader("Choose a fingerprint image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict blood group
    if st.button("Predict Blood Group"):
        image.save("temp.jpg")  # Save temporarily for prediction
        prediction = predict_blood_group("temp.jpg")
        st.write(f"Predicted Blood Group: **{prediction}**")
