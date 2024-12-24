import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import datetime

# Load the models
plant_disease_model = load_model('plant_disease_model.h5')
crop_yield_model_pipeline = joblib.load('crop_yield_model.pkl')

# Name of Classes for plant disease detection
PLANT_CLASSES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Load the crop yield dataset
data = pd.read_csv('crop_yield.csv')

# Function to predict plant disease
def predict_disease(image):
    opencv_image = cv2.resize(image, (256, 256))
    opencv_image = np.expand_dims(opencv_image, axis=0)
    Y_pred = plant_disease_model.predict(opencv_image)
    return PLANT_CLASSES[np.argmax(Y_pred)]

# Function to predict crop yield
def predict_yield(data):
    return crop_yield_model_pipeline.predict(data)

# Streamlit app layout
st.set_page_config(page_title="Smart Agri", page_icon="🌾", layout="wide")

# Title with white background and dark green text
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://www.w3schools.com/w3images/forestbridge.jpg');
        background-size: cover;
    }
    .css-1d391kg {
        background-color: #388E3C !important;
        padding: 20px;
        text-align: center;
        font-size: 36px;
        color: white;
        font-weight: bold;
    }
    .stText {
        color: #388E3C;
        font-size: 18px;
    }
    .stButton>button {
        background-color: #388E3C;
        color: white;
        border-radius: 5px;
    }
    .prediction-result {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        color: #388E3C;
        font-size: 18px;
        margin-top: 20px;
    }
    .info-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        color: #388E3C;
        font-size: 16px;
        margin-top: 20px;
    }
    .stSubtitle, .stText, .stHeader {
        font-size: 20px;
        color: white;
        background-color: #388E3C;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True
)

# Add a title in a dark green background with white text
st.markdown('<div class="css-1d391kg">WELCOME TO SMART AGRI</div>', unsafe_allow_html=True)

# Toggle between the two functionalities using selectbox
app_mode = st.selectbox("Select Task", ["Plant Disease Detection", "Crop Yield Prediction"])

if app_mode == "Plant Disease Detection":
    st.markdown("<div class='stText'>Upload an image of the plant leaf to detect disease.</div>", unsafe_allow_html=True)
    plant_image = st.file_uploader("Choose an image...", type="jpg")
    if plant_image is not None:
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels="BGR")
        
        if st.button('Predict Disease'):
            result = predict_disease(opencv_image)
            st.markdown(f"""
            <div class="prediction-result">
            This is a {result.split('-')[0]} leaf with {result.split('-')[1]} disease.
            </div>
            """, unsafe_allow_html=True)
    
elif app_mode == "Crop Yield Prediction":
    st.markdown("<div class='stText'>Provide crop information to predict yield.</div>", unsafe_allow_html=True)
    
    # Crop yield input fields
    crop = st.selectbox('Select Crop', data['Crop'].unique())
    crop_year = st.number_input('Crop Year', min_value=1997, max_value=2040, step=1)
    season = st.selectbox('Season', ['Kharif', 'Rabi', 'Whole Year', 'Summer', 'Winter', 'Autumn'])
    state = st.selectbox('State', data['State'].unique())
    area = st.number_input('Area (in hectares)', min_value=0.0, value=100.0)
    production = st.number_input('Production (in tons)', min_value=0.0, value=100.0)
    annual_rainfall = st.number_input('Annual Rainfall (in mm)', min_value=0.0, value=1000.0)
    fertilizer = st.number_input('Fertilizer (in kg/ha)', min_value=0.0, value=1000.0)
    pesticide = st.number_input('Pesticide (in kg/ha)', min_value=0.0, value=10.0)

    if st.button('Predict Yield'):
        input_data = pd.DataFrame({
            'Crop': [crop], 'Crop_Year': [crop_year], 'Season': [season], 
            'State': [state], 'Area': [area], 'Production': [production],
            'Annual_Rainfall': [annual_rainfall], 'Fertilizer': [fertilizer],
            'Pesticide': [pesticide]
        })
        
        try:
            prediction = predict_yield(input_data)
            st.markdown(f"""
            <div class="prediction-result">
            Estimated Crop Yield: {prediction[0]:.2f} tons/ha
            </div>
            """, unsafe_allow_html=True)
            
            # Solutions and tips section
            st.subheader("Solutions to Improve Crop Yield", anchor="solutions")
            st.markdown("""
            <div class="info-box">
            **Improving Crop Yield:**
            - **Water Management:** Ensure proper irrigation systems to avoid water stress during critical growth stages.
            - **Fertilizer Application:** Apply fertilizers based on soil health and crop requirement. Over-fertilization can harm the soil.
            - **Pest Control:** Use effective pest management practices to reduce damage to crops.
            - **Climate Considerations:** Choose the right crop for the local climate and season, and consider climate change impacts.
            - **Crop Rotation:** Practice crop rotation to maintain soil fertility and reduce pest buildup.
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Suggested Crop Types for Specific Conditions", anchor="crop_types")
            st.markdown("""
            <div class="info-box">
            - **For High Rainfall Areas:** Rice, Sugarcane, Coconut.
            - **For Drier Regions:** Sorghum, Millet, Chickpeas.
            - **For Moderate Rainfall:** Wheat, Maize, Barley.
            - **For Cold Regions:** Barley, Rye, Peas.
            </div>
            """, unsafe_allow_html=True)
        
        except ValueError as e:
            st.error(f"Error: {e}")

# Footer with resources and feedback
st.subheader('Resources', anchor="resources")
st.markdown("""
- [Government Schemes](https://pib.gov.in/PressReleaseIframePage.aspx?PRID=2002012)
- [Best Practices](https://upagripardarshi.gov.in/Index.aspx)
- [Weather Forecast](https://www.accuweather.com/)
- [Market Prices](https://agmarknet.gov.in/PriceAndArrivals/CommodityDailyStateWise.aspx)
""")

st.subheader('Provide Feedback', anchor="feedback")
feedback = st.text_area("Your Feedback", "")
if st.button('Submit Feedback'):
    st.success("Thank you for your feedback!")

