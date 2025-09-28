# 🌱 AgriGuard - Plant Disease Detection

## 📌 Overview
**AgriGuard** is a **Plant Disease Detection Web App** that leverages **Deep Learning (TensorFlow/Keras)** to classify plant leaf images into healthy or diseased categories.  
The project provides an interactive **Streamlit** interface for farmers, students, and researchers to quickly upload a plant leaf image and get real-time predictions.

⚡ Model is trained on a custom dataset of plant leaf images using Convolutional Neural Networks (CNNs).

🗓️ **Note:** This project was developed during **March 2025 – July 2025** as part of my learning journey in Deep Learning and Web App development.

---

## 🚀 Features
- Train and evaluate a CNN model on plant disease datasets  
- Save and load trained models (`.h5` format)  
- Interactive **Streamlit app** for image uploads and predictions  
- Real-time classification with confidence scores  
- Modular design:  
  - `train_model.py` → training pipeline  
  - `main.py` → Streamlit app  

---

## 🛠️ Tech Stack
- **Python 3.x**  
- **TensorFlow / Keras**  
- **Streamlit**  
- **NumPy, Pandas**  
- **Matplotlib / Seaborn** (for visualizations)  

---

## 📂 Project Structure
📦 AgriGuard
┣ 📜 main.py # Streamlit web app
┣ 📜 train_model.py # Model training script
┣ 📜 trained_model.h5 # Saved CNN model
┣ 📜 requirements.txt # Dependencies
┣ 📜 README.md # Documentation
┗ 📂 dataset/ # Training & testing images (not uploaded to GitHub if large)


---

## ⚙️ Installation & Usage

### 1️⃣ Clone the repository
git clone https://github.com/lilpookie404/AgriGuard.git
cd AgriGuard

### 2️⃣ Install dependencies
pip install -r requirements.txt
or manually install:
pip install streamlit tensorflow numpy pandas matplotlib seaborn

### 3️⃣ Train the model (optional, if you want to retrain)
python train_model.py

### 4️⃣ Run the Streamlit app
streamlit run main.py

### 5️⃣ Upload a plant leaf image
Go to the web app (http://localhost:8501 by default)
Upload a leaf image
Get the prediction result (healthy/diseased + confidence score)

---

## 📊 Model Performance
Training accuracy: ~96.42%
Validation accuracy: ~93.55%
Tested on plant disease dataset with N classes

---

## 🔮 Future Improvements
Extend dataset with more plant species and diseases
Add Grad-CAM visualizations for explainable AI
Deploy the app on Streamlit Cloud / Heroku / Azure
Mobile app integration for field usage

---

## 👩‍💻 Author
Vaishnavi Awadhiya
3rd Year Project | Built with ❤️ using Deep Learning + Streamlit