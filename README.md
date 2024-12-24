# 🌾 Smart Agric: Intelligent Agriculture Management

Smart Agric is an intelligent agriculture management system that integrates cutting-edge AI technologies to help farmers and agricultural stakeholders. The platform features two primary functionalities: **Plant Disease Detection** , **Crop Yield Prediction** and **Irrigation management**

---

## 🚀 Features

1. **Plant Disease Detection**
   - Upload an image of a plant leaf to detect diseases.
   - Identifies diseases for crops like Tomato, Potato, and Corn.
   
2. **Crop Yield Prediction**
   - Predicts crop yield based on factors such as area, production, rainfall, and more.
   - Provides solutions and suggestions to improve yield.
3. **Irrigation Management**  
   - Offers intelligent recommendations for irrigation scheduling based on:  
   - Soil moisture levels  
   - Weather forecasts  
   - Crop type and growth stage  
   - Helps optimize water usage, reduce waste, and ensure healthy crop growth.  

4. **User-Friendly Interface**
   - Interactive web application built with Streamlit.
   - Recommendations and insights provided to users for better decision-making.


---

## Project Structure 📂

The project comprises essential components:

- `Plant_Disease_Detection.ipynb`: Jupyter Notebook with the code for model training.
- `Crop_Yield_Analysis.ipynb`: Jupyter Notebook with the code for model training.
- `app.py`: Streamlit web application for Smart Agriculture management.
- `plant_disease_model.h5`: Pre-trained model weights.
- `crop_yield_model.pkl`: Pre-trained model weights.
- `irrigationTuning.ipynb`: Pre-trained model weights.
- `requirements.txt`: List of necessary Python packages.

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Rimadaqch/Smart_Agric.git
   cd Smart_Agric
2. **Install dependencies**
   pip install -r requirements.txt
3.**Run the App**
   streamlit run app.py

## 🧪 How It Works

### Plant Disease Detection
- Upload a leaf image to detect diseases.
- Leverages a pre-trained deep learning model.

### Crop Yield Prediction
- Provide input details like area, production, and environmental factors.
- Predicts crop yield and offers data-driven improvement tips.

---

## 📊 Dataset Details

### Plant Disease Detection
- Trained on public datasets for plant leaf images.
- Covers diseases in Tomato, Potato, and Corn.

### Crop Yield Prediction
- Based on `crop_yield.csv`, containing historical data of crops, production, rainfall, etc.
### Irrigation Management  
- Integrates soil and weather data for precise irrigation planning ,`data.csv`.  
- Data sources include meteorological records and agricultural studies.
  

## 🔧 Technologies Used

- **Programming Language**: Python  
- **Framework**: Streamlit  
- **Machine Learning Libraries**: Keras, TensorFlow, scikit-learn, PyTorch, pandas, numpy  
- **Visualization Tools**: Matplotlib, Seaborn  
- **Image Processing**: OpenCV, PIL  


  

