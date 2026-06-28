# 🔍 Duplicate Question Detector

An end-to-end Machine Learning web application that identifies whether two questions have the same intent. Built to solve the **Quora Question Pairs** problem, this system helps platforms reduce redundancy by flagging duplicate queries in real-time.

## 🚀 Live Demo
[**Click here to try the live app**]
(https://duplicate-question-detector-hgln.onrender.com)
## 📋 Project Overview
This project uses a **Random Forest Classifier** combined with **TF-IDF Vectorization** and custom feature engineering to detect duplicate questions with high accuracy. It processes text inputs, extracts over 6,000 features, and returns a prediction with a confidence score in under a second.

## ✨ Key Features
- **Hybrid Feature Engineering**: Combines **15 handcrafted features** (lexical overlap, token statistics, stopwords analysis) with **6,000 NLP features** from TF-IDF.
- **High Accuracy**: Achieves ~88% accuracy on test data by capturing both syntactic and semantic similarities.
- **Interactive Dashboard**: User-friendly **Streamlit** interface for real-time predictions.
- **Cloud Deployed**: Hosted on **Render** with automated CI/CD via GitHub.
- **Large Model Support**: Utilizes **Git LFS** to manage model files >100MB efficiently.

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Machine Learning**: Scikit-Learn (Random Forest, TF-IDF), NLTK
- **Data Processing**: Pandas, NumPy
- **Frontend**: Streamlit
- **Deployment**: Render / Streamlit Community Cloud
- **Version Control**: Git, GitHub (with Git LFS)

## 📦 Installation & Local Setup

### Prerequisites
- Python 3.8+
- Git & Git LFS installed

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/VinayTodkar/Duplicate-question-detector.git
   cd Duplicate-question-detector

Install dependencies:
pip install -r requirements.txt

Download NLTK Data:
python -c "import nltk; nltk.download('stopwords')"

Run the App:
streamlit run app.py

The app will open in your browser at http://localhost:8501.
📊 How It Works
Input: User enters two questions.
Preprocessing: Text is lowercased, stripped, and tokenized.
Feature Extraction:
Base Features: Word counts, common word ratios, first/last word matches.
Vector Features: TF-IDF transformation (3000 features per question).
Prediction: The Random Forest model classifies the pair as Duplicate or Not Duplicate.
📁 Project Structure
.
├── app.py              # Streamlit dashboard & logic
├── model.pkl           # Trained Random Forest model (Git LFS)
├── cv.pkl              # TF-IDF Vectorizer (Git LFS)
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
├── .gitattributes      # Git LFS configuration
└── README.md           # Project documentation

📈 Performance
Accuracy: ~88%
Precision: ~85%
Recall: ~82%
Prediction Time: < 1 second
🤝 Contributing
Contributions are welcome! Feel free to fork the repo and submit a pull request.

📄 License
This project is open-source and available under the MIT License.

👤 Author
Vinay Todkar
GitHub | LinkedIn

Built with ❤️ using Python & Streamlit
