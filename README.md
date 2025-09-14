Spam Email Classifier
A machine learning system that accurately identifies spam emails with 97.1% accuracy, combining traditional Logistic Regression with Google's Gemini AI for enhanced detection and explainable insights.

Dual Analysis: Machine Learning + Gemini AI integration
High Accuracy: 97.1% test accuracy on SpamAssassin dataset
Real-time Processing: Instant email classification
Visual Analytics: Probability scoring and feature importance
Web Interface: Streamlit-based responsive dashboard
 Tech Stack
Python, Scikit-learn, Logistic Regression
Google Gemini API for AI enhancement
Streamlit for web interface
TF-IDF Vectorization with 5,000 features
Matplotlib & WordCloud for visualizations
Project Structure:
spam_classifier/
├── data/                    # Email datasets
├── models/                  # Trained models
├── src/
│   ├── main.py             # Training script
│   └── app.py              # Web application
└── requirements.txt        # Dependencies 
Official SpamAssassin Public Corpus
Main Download Page:
https://spamassassin.apache.org/old/publiccorpus/
