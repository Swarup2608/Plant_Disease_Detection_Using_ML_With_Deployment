import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


# Loading the diseased plant model
model = load_model("plant_disease.h5")
class_names = [ 'Tomato Bacterial Spot','Corn Common Rust', 'Potato Early Blight']

# Creating the app 
st.title("Plant Disease Prediction")
st.markdown("Upload the image of the Plant Leaf : ")

# Uploading the dog image
image = st.file_uploader("Choose an image : ",type=["png","jpeg","jpg"])
submit = st.button("Predict the Diseased plant for the image")


if submit:
    if image is not None:
        # Converting the image to byte code
        file_by = np.asarray(bytearray(image.read()),dtype=np.uint8)
        # Opening the image
        open_cv = cv2.imdecode(file_by,1)
        
        # Display the image uploaded
        st.image(open_cv,channels="BGR")
        # Resizing the image
        open_cv = cv2.resize(open_cv,(256,256))
        # Convert the image into 4 dimension 
        open_cv.shape = (1,256,256,3)
        Y_pred = model.predict(open_cv)
        st.title("The plant leaf of the uploaded image is : "+class_names[np.argmax(Y_pred)])


