import streamlit as st
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import os
import joblib
import requests
import json
import time
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Spam Classifier Pro",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .spam-prediction {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
    .ham-prediction {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    .feature-positive {
        color: #ff4b4b;
        font-weight: bold;
    }
    .feature-negative {
        color: #4caf50;
        font-weight: bold;
    }
    .probability-bar {
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .api-card {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .success-badge {
        background-color: #4caf50;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        margin-left: 10px;
    }
    .error-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        margin-left: 10px;
    }
    .gemini-analysis {
        background-color: #e6f7ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4285f4;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and vectorizer
@st.cache_resource
def load_model():
    try:
        # Try multiple possible locations for the model files
        possible_paths = [
            # Path where your training script saved them
            "C:/Users/Divya ch/Documents/models/logistic_regression_model.pkl",
            # Path in your project directory
            "./models/logistic_regression_model.pkl",
            # Relative path from src
            "../models/logistic_regression_model.pkl"
        ]
        
        model_path = None
        vectorizer_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                vectorizer_path = path.replace("logistic_regression_model.pkl", "tfidf_vectorizer.pkl")
                break
        
        if model_path is None:
            st.error("Model files not found. Please check the file paths.")
            return None, None
        
        # First try with pickle
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
        except:
            # If pickle fails, try with joblib
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            
        return model, vectorizer
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Gemini API Functions
def test_gemini_connection(api_key: str) -> bool:
    """Test connection to Gemini API"""
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': api_key
        }
        payload = {
            "contents": [{
                "parts": [{"text": "Test connection"}]
            }]
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Connection test failed: {str(e)}")
        return False

def analyze_with_gemini(api_key: str, email_text: str) -> dict:
    """Analyze email content using Gemini API"""
    try:
        # Create a prompt for spam detection
        prompt = f"""
        Analyze this email and determine if it's spam or legitimate (ham). 
        Consider these factors:
        1. Suspicious phrases or urgency language
        2. Grammar and spelling quality
        3. Request for personal information or money
        4. Unprofessional formatting
        5. Suspicious links or attachments
        6. Unusual sender addresses
        7. Too-good-to-be-true offers
        
        Email content: {email_text[:4000]}  # Limit length to avoid token limits
        
        Provide your analysis in this JSON format:
        {{
            "is_spam": boolean,
            "confidence_score": float (0.0 to 1.0),
            "explanation": string,
            "key_indicators": list of strings,
            "risk_level": "low", "medium", or "high"
        }}
        """
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': api_key
        }
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1024,
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            # Extract the text response
            if 'candidates' in result and len(result['candidates']) > 0:
                text_response = result['candidates'][0]['content']['parts'][0]['text']
                
                # Try to parse JSON from the response
                try:
                    # Extract JSON from the response (Gemini might add some text around)
                    json_start = text_response.find('{')
                    json_end = text_response.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = text_response[json_start:json_end]
                        return json.loads(json_str)
                    else:
                        # Fallback: create a response from the text
                        is_spam = any(word in text_response.lower() for word in ['spam', 'malicious', 'suspicious'])
                        return {
                            "is_spam": is_spam,
                            "confidence_score": 0.8 if is_spam else 0.2,
                            "explanation": text_response,
                            "key_indicators": ["AI analysis completed"],
                            "risk_level": "high" if is_spam else "low"
                        }
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    is_spam = any(word in text_response.lower() for word in ['spam', 'malicious', 'suspicious'])
                    return {
                        "is_spam": is_spam,
                        "confidence_score": 0.8 if is_spam else 0.2,
                        "explanation": text_response,
                        "key_indicators": ["AI analysis completed"],
                        "risk_level": "high" if is_spam else "low"
                    }
            else:
                st.error("Unexpected response format from Gemini API")
                return None
        else:
            st.error(f"Gemini API request failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"AI analysis failed: {str(e)}")
        return None

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to get top influential words
def get_top_features(vectorizer, model, n=20):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Create a DataFrame for easier manipulation
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    
    # Get top spam and ham features
    top_spam = feature_importance.nlargest(n, 'coefficient')
    top_ham = feature_importance.nsmallest(n, 'coefficient')
    
    return top_spam, top_ham

# Main application
def main():
    st.title("📧 Spam Classifier Pro with Gemini AI")
    st.markdown("Enhanced email classification with machine learning and Google's Gemini AI")
    
    # Load model
    model, vectorizer = load_model()
    
    # Create sidebar
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.radio("Choose mode:", 
                               ["Email Classification", "Gemini API Setup", "Model Analysis"])
    
    if app_mode == "Email Classification":
        render_classification(model, vectorizer)
    elif app_mode == "Gemini API Setup":
        render_gemini_setup()
    elif app_mode == "Model Analysis":
        if model and vectorizer:
            render_model_analysis(model, vectorizer)
        else:
            st.error("Please load the model first")

def render_classification(model, vectorizer):
    st.header("Email Classification")
    
    # Gemini API integration
    st.subheader("AI Enhancement with Gemini")
    use_gemini = st.checkbox("Enable Gemini AI Analysis", value=False)
    
    gemini_api_key = None
    
    if use_gemini:
        # ⬇️⬇️⬇️ ADD YOUR API KEY RIGHT HERE ⬇️⬇️⬇️
        gemini_api_key="AIzaSyDyLm9JSacomRguw25ciLx14RdI5o_90BU"
       #gemini_api_key = st.text_input("Gemini API Key:", type="password",
                                     #help="Get your free API key from Google AI Studio")
        
        if gemini_api_key:
            if st.button("Test Gemini Connection"):
                if test_gemini_connection(gemini_api_key):
                    st.success("✅ Gemini connection successful!")
                else:
                    st.error("❌ Connection failed. Check your API key.")
    
    # Email input
    st.subheader("Email Input")
    email_text = st.text_area("Paste email content here:", height=200,
                             placeholder="Type or paste email content here...")
    
    if st.button("Classify Email", type="primary") and email_text:
        # Traditional ML classification
        processed_text = preprocess_text(email_text)
        X = vectorizer.transform([processed_text])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        # Gemini AI analysis if enabled
        gemini_analysis = None
        if use_gemini and gemini_api_key:
            with st.spinner("Analyzing with Gemini AI..."):
                gemini_analysis = analyze_with_gemini(gemini_api_key, email_text)
        
        # Display results
        display_results(prediction, probability, gemini_analysis, email_text, model, vectorizer)

def render_gemini_setup():
    st.header("Gemini API Setup Guide")
    st.info("""
    **Enhance your spam classifier with Google's Gemini AI!** 
    Gemini offers free API access with generous limits for testing and development.
    """)
    
    with st.expander("Step-by-Step Guide to Getting Your Free API Key"):
        st.markdown("""
        ### Step 1: Login to your Google account
        Visit [Google AI Studio](https://ai.google.dev/) and sign in with your Google account.
        
        ### Step 2: Create an API Key
        1. Click on "Get API Key" in Google AI Studio
        2. Review and accept the terms of service
        3. Click "Create API Key" in a new or existing project
        
        ### Step 3: Copy Your API Key
        Once generated, copy your API key and store it securely.
        
        ### Step 4: Use in Your Application
        Paste your API key in the Classification tab to enable Gemini AI analysis.
        """)
    
    st.subheader("Why Use Gemini AI?")
    st.markdown("""
    - **Enhanced Accuracy**: Gemini can detect subtle patterns missed by traditional ML
    - **Context Understanding**: Better comprehension of email context and intent
    - **Multi-language Support**: Effective across multiple languages
    - **Explainable AI**: Provides reasoning for its classification decisions
    """)
    
    st.subheader("Pricing Information")
    st.markdown("""
    Gemini API offers:
    - **Free Tier**: Generous free usage limits for testing and development
    - **Pay-as-you-go**: Affordable pricing for production usage
    - **Rate Limits**: Reasonable limits for different usage tiers
    
    Check the [Google AI Studio pricing page](https://ai.google.dev/pricing) for current details.
    """)

def render_model_analysis(model, vectorizer):
    st.header("Model Analysis")
    
    # Get top features
    top_spam, top_ham = get_top_features(vectorizer, model, 15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Spam Indicators")
        for _, row in top_spam.iterrows():
            st.markdown(f'<span class="feature-positive">{row["feature"]}: +{row["coefficient"]:.4f}</span>', 
                       unsafe_allow_html=True)
    
    with col2:
        st.subheader("Top Ham Indicators")
        for _, row in top_ham.iterrows():
            st.markdown(f'<span class="feature-negative">{row["feature"]}: {row["coefficient"]:.4f}</span>', 
                       unsafe_allow_html=True)
    
    # Visualization
    st.subheader("Feature Importance Visualization")
    top_features = pd.concat([top_spam, top_ham])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red' if coef > 0 else 'green' for coef in top_features['coefficient']]
    y_pos = np.arange(len(top_features))
    
    ax.barh(y_pos, top_features['coefficient'], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Top Spam and Ham Indicators')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    st.pyplot(fig)

def display_results(prediction, probability, gemini_analysis=None, email_text=None, model=None, vectorizer=None):
    st.header("Classification Results")
    
    # ML Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Machine Learning Analysis")
        if prediction == 1:
            st.markdown('<div class="spam-prediction"><h3>🚫 SPAM DETECTED</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ham-prediction"><h3>✅ LEGITIMATE EMAIL</h3></div>', unsafe_allow_html=True)
        
        spam_percent = probability[1] * 100
        ham_percent = probability[0] * 100
        
        st.metric("Spam Probability", f"{spam_percent:.1f}%")
        st.metric("Ham Probability", f"{ham_percent:.1f}%")
    
    with col2:
        st.subheader("Confidence Visualization")
        fig, ax = plt.subplots(figsize=(8, 1))
        ax.barh([0], [spam_percent], color='red', alpha=0.7, label='Spam')
        ax.barh([0], [ham_percent], left=[spam_percent], color='green', alpha=0.7, label='Ham')
        ax.set_xlim(0, 100)
        ax.set_yticklabels([])
        ax.set_xlabel('Probability (%)')
        ax.legend()
        st.pyplot(fig)
    
    # Show influential words from ML model
    if email_text and model and vectorizer:
        st.subheader("Key Words Analysis")
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        
        # Get indices of words in the email
        processed_text = preprocess_text(email_text)
        email_words = processed_text.split()
        influential_words = []
        
        for word in set(email_words):
            if word in feature_names:
                idx = np.where(feature_names == word)[0][0]
                influential_words.append((word, coefficients[idx]))
        
        # Sort by absolute value of coefficient
        influential_words.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Display top 10
        if influential_words:
            cols = st.columns(2)
            for i, (word, coef) in enumerate(influential_words[:10]):
                with cols[i % 2]:
                    if coef > 0:
                        st.markdown(f'<span class="feature-positive">{word} (+{coef:.3f})</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="feature-negative">{word} ({coef:.3f})</span>', unsafe_allow_html=True)
        else:
            st.info("No significant influential words found in this email.")
    
    # Gemini Analysis Results if available
    if gemini_analysis:
        st.markdown("---")
        st.subheader("Gemini AI Analysis")
        
        ai_col1, ai_col2 = st.columns(2)
        
        with ai_col1:
            st.markdown('<div class="gemini-analysis">', unsafe_allow_html=True)
            st.info("AI Confidence Analysis")
            
            # Display risk level with appropriate color
            risk_level = gemini_analysis.get('risk_level', 'unknown').lower()
            risk_color = {
                'low': 'green',
                'medium': 'orange', 
                'high': 'red'
            }.get(risk_level, 'gray')
            
            st.metric("AI Spam Confidence", f"{gemini_analysis.get('confidence_score', 0) * 100:.1f}%")
            st.markdown(f"**Risk Level**: <span style='color: {risk_color}; font-weight: bold'>{risk_level.upper()}</span>", 
                       unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with ai_col2:
            st.info("AI Explanation")
            st.write(gemini_analysis.get('explanation', 'No explanation provided'))
            
            # Show key indicators if available
            if 'key_indicators' in gemini_analysis:
                st.write("**Key Indicators:**")
                for indicator in gemini_analysis['key_indicators']:
                    st.write(f"- {indicator}")
        
        # Compare results
        st.subheader("Result Comparison")
        comparison_data = {
            'Method': ['Machine Learning', 'Gemini AI'],
            'Spam Confidence': [f"{spam_percent:.1f}%", f"{gemini_analysis.get('confidence_score', 0) * 100:.1f}%"],
            'Verdict': ['SPAM' if prediction == 1 else 'HAM', 
                       'SPAM' if gemini_analysis.get('is_spam', False) else 'HAM']
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

if __name__ == "__main__":
    main()