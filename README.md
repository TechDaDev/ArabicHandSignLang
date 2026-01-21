# 🤟 Arabic Hand Sign Language Recognition AI

### **College of Artificial Intelligence / Department of Biomedical Applications**
### **كلية الذكاء الاصطناعي / قسم التطبيقات الطبية الحيوية**

---

## 📝 Project Description
This project is a high-precision real-time interpretation system for Arabic Sign Language. It leverages **MediaPipe** for hand landmark extraction and a **Multi-Layer Perceptron (MLP)** neural network to classify hand gestures into Arabic letters. 

The system was developed and benchmarked against multiple machine learning algorithms (SVM, Random Forest, XGBoost, KNN) to ensure the highest possible interpretation accuracy.

## 🚀 Key Features
- **Real-time Recognition**: Live camera stream interpretation with hand skeleton overlay.
- **Premium Streamlit UI**: A modern, glassmorphism-style web interface for easy interaction.
- **High Accuracy**: Powered by a 96.36% accurate MLP model.
- **Confidence Tracking**: Live probability bar showing the system's certainty for each prediction.
- **Institutional Branding**: Integration of College of AI / Biomedical Applications department info.

## 📁 Project Structure
```text
.
├── models/             # Pre-trained model artifacts (MLP, Scaler, Encoder)
├── reports/            # Detailed accuracy reports for all benchmarked models
├── scripts/            # CLI recognition and utility scripts
├── dataset/            # Normalized hand landmark CSV data
├── logo/               # Institutional branding assets
├── streamlit_app.py    # Main Premium Web Application
├── train_model.py      # Multi-model benchmarking & training script
├── requirements.txt    # Python dependencies
└── .gitignore          # Git exclusion rules
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TechDaDev/ArabicHandSignLang.git
   cd ArabicHandSignLang
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 1. Run the Web Application (Recommended)
Launch the interactive dashboard to see the real-time recognition in action:
```bash
streamlit run streamlit_app.py
```

### 2. Retrain/Benchmark Models
If you wish to re-evaluate the algorithms or train on new data:
```bash
python train_model.py
```
This will generate performance reports in the `reports/` folder and save the best model to `models/`.

### 3. Run CLI Recognition
For a lightweight OpenCV-based window without the web UI:
```bash
python scripts/realtime_recognition.py
```

## 📊 Model Performance
The current production model is a **Multi-Layer Perceptron (MLP)**. Here is how it compared to other tested algorithms:

| Algorithm | Accuracy |
| :--- | :--- |
| **MLP (Production)** | **96.36%** |
| SVM | 94.26% |
| XGBoost | 90.55% |
| Random Forest | 87.11% |
| KNN | 74.37% |

## 🧪 Technologies Used
- **Language**: Python 3.12
- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: Scikit-learn, XGBoost, Joblib
- **Web Framework**: Streamlit

---
*Developed for the College of Artificial Intelligence - Department of Biomedical Applications.*
