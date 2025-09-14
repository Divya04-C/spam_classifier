import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class SpamClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = None
        self.df = None
        
    def debug_directory_scan(self):
        """Debug function to see what files are being found"""
        base_path = self.data_path
        
        print("🔍 DEBUG: Scanning directories...")
        print("=" * 50)
        
        for dir_name in ['easy_ham', 'easy_ham_2', 'hard_ham', 'spam', 'spam_2']:
            dir_path = os.path.join(base_path, dir_name)
            
            if os.path.exists(dir_path):
                try:
                    all_items = os.listdir(dir_path)
                    files = [f for f in all_items if os.path.isfile(os.path.join(dir_path, f))]
                    valid_files = [f for f in files if not f.startswith('cmds') and not f.startswith('.')]
                    
                    print(f"📁 {dir_name}: {len(valid_files)} files (out of {len(files)} total files)")
                    
                    if valid_files:
                        print(f"   First few: {valid_files[:3]}{'...' if len(valid_files) > 3 else ''}")
                        # Show file sizes
                        for f in valid_files[:2]:
                            file_path = os.path.join(dir_path, f)
                            size = os.path.getsize(file_path)
                            print(f"   {f}: {size} bytes")
                    else:
                        print(f"   ❌ No valid files found!")
                except PermissionError:
                    print(f"❌ Permission denied accessing {dir_path}")
                except Exception as e:
                    print(f"❌ Error scanning {dir_path}: {e}")
            else:
                print(f"❌ {dir_name}: Directory not found")
            print()
    
    def load_dataset(self):
        """Load emails from the SpamAssassin directory structure"""
        emails = []
        labels = []
        
        # Define directory mapping
        dir_mapping = {
            'easy_ham': 0,
            'easy_ham_2': 0, 
            'hard_ham': 0,
            'spam': 1,
            'spam_2': 1
        }
        
        total_files_processed = 0
        
        for dir_name, label in dir_mapping.items():
            dir_path = os.path.join(self.data_path, dir_name)
            
            if not os.path.exists(dir_path):
                print(f"⚠️ Warning: Directory {dir_path} not found. Skipping...")
                continue
                
            print(f"📂 Loading {dir_name} files...")
            
            # Get all files in directory
            try:
                all_files = os.listdir(dir_path)
            except PermissionError:
                print(f"❌ Permission denied accessing {dir_path}")
                continue
            except Exception as e:
                print(f"❌ Error listing files in {dir_path}: {e}")
                continue
            
            # Filter files (not cmds, not hidden, and is file)
            valid_files = []
            for filename in all_files:
                file_path = os.path.join(dir_path, filename)
                if (os.path.isfile(file_path) and 
                    not filename.startswith('cmds') and 
                    not filename.startswith('.')):
                    valid_files.append(filename)
            
            print(f"   Found {len(valid_files)} valid email files")
            
            # Read each file
            dir_file_count = 0
            for filename in valid_files:
                file_path = os.path.join(dir_path, filename)
                try:
                    # Try multiple encodings
                    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                    content = None
                    
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                                content = f.read()
                            if content and content.strip():
                                break
                        except (UnicodeDecodeError, UnicodeError):
                            continue
                        except Exception as e:
                            print(f"   ⚠️ Error reading {filename} with {encoding}: {e}")
                            continue
                    
                    if content is None or not content.strip():
                        print(f"   ⚠️ Could not read {filename} or empty content")
                        continue
                    
                    emails.append(content)
                    labels.append(label)
                    dir_file_count += 1
                    total_files_processed += 1
                        
                except Exception as e:
                    print(f"   ❌ Error reading {filename}: {e}")
            
            print(f"   ✅ Successfully loaded {dir_file_count} files from {dir_name}")
        
        if not emails:
            print("❌ No emails were loaded. Please check your dataset.")
            return None
        
        self.df = pd.DataFrame({'email': emails, 'label': labels})
        print(f"\n📊 Dataset loaded: {len(self.df)} emails")
        print(f"   - SPAM: {self.df['label'].sum()} emails")
        print(f"   - HAM: {len(self.df) - self.df['label'].sum()} emails")
        
        return self.df
    
    def preprocess_text(self, text):
        """Clean and preprocess email text"""
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove email headers (simplified)
        text = re.sub(r'^.*?subject:.*?\n', '', text, flags=re.IGNORECASE)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_emails(self):
        """Preprocess all emails in the dataset"""
        if self.df is None or len(self.df) == 0:
            print("❌ No data to preprocess. Please load dataset first.")
            return None
        
        print("🔄 Preprocessing emails...")
        self.df['processed_email'] = self.df['email'].apply(self.preprocess_text)
        
        # Remove empty processed emails
        initial_count = len(self.df)
        self.df = self.df[self.df['processed_email'].str.strip().astype(bool)]
        removed_count = initial_count - len(self.df)
        
        if removed_count > 0:
            print(f"   Removed {removed_count} empty emails after preprocessing")
        
        print(f"✅ Preprocessing completed! {len(self.df)} emails remaining")
        return self.df
    
    def extract_features(self):
        """Convert text to TF-IDF features"""
        if self.df is None or len(self.df) == 0:
            print("❌ No data for feature extraction. Please preprocess first.")
            return None, None
        
        print("⚙️ Extracting features with TF-IDF...")
        X = self.vectorizer.fit_transform(self.df['processed_email'])
        y = self.df['label']
        print(f"✅ Feature extraction completed: {X.shape[1]} features")
        return X, y
    
    def train_models(self, test_size=0.2):
        """Train and evaluate both models"""
        X, y = self.extract_features()
        
        if X is None or y is None:
            print("❌ Cannot train models. No features available.")
            return None, None, None, None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print("\n" + "="*50)
        print("🤖 TRAINING MODELS")
        print("="*50)
        
        # Train Naive Bayes
        print("\n1. Training Naive Bayes...")
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)
        nb_pred = nb_model.predict(X_test)
        
        nb_accuracy = accuracy_score(y_test, nb_pred)
        print(f"   Naive Bayes Accuracy: {nb_accuracy:.4f}")
        print("   Classification Report:")
        print(classification_report(y_test, nb_pred))
        
        # Train Logistic Regression
        print("\n2. Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        lr_accuracy = accuracy_score(y_test, lr_pred)
        print(f"   Logistic Regression Accuracy: {lr_accuracy:.4f}")
        print("   Classification Report:")
        print(classification_report(y_test, lr_pred))
        
        # Save the best model
        if lr_accuracy > nb_accuracy:
            self.model = lr_model
            print("\n✅ Logistic Regression selected as best model")
        else:
            self.model = nb_model
            print("\n✅ Naive Bayes selected as best model")
        
        return nb_model, lr_model, X_test, y_test
    
    def save_models(self, model, model_name="best_model"):
        """Save trained models and vectorizer"""
        try:
            # Create models directory if it doesn't exist
            models_dir = '../models'
            os.makedirs(models_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(models_dir, f'{model_name}.pkl')
            joblib.dump(model, model_path)
            print(f"💾 Model saved to: {os.path.abspath(model_path)}")
            
            # Save vectorizer
            vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
            joblib.dump(self.vectorizer, vectorizer_path)
            print(f"💾 Vectorizer saved to: {os.path.abspath(vectorizer_path)}")
            
            # Verify files were saved
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                print("✅ All model files saved successfully!")
                print("📁 Files in models directory:")
                for f in os.listdir(models_dir):
                    size = os.path.getsize(os.path.join(models_dir, f))
                    print(f"   {f} ({size} bytes)")
            else:
                print("❌ Error: Model files were not created")
                
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            import traceback
            traceback.print_exc()
    
    def predict_email(self, email_text):
        """Predict if a new email is spam or ham"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Preprocess the email
        processed_text = self.preprocess_text(email_text)
        
        # Transform using the trained vectorizer
        features = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        result = {
            'prediction': 'SPAM' if prediction == 1 else 'HAM',
            'spam_probability': probability[1],
            'ham_probability': probability[0]
        }
        
        return result

def main():
    # ✅ USE ABSOLUTE PATH - This will definitely work!
    data_path = "C:/Users/Divya ch/Documents/spam_classifier/data/spamassassin-public-corpus"
    classifier = SpamClassifier(data_path)
    
    print("🚀 Starting Spam Classifier Training")
    print("="*50)
    
    # First, debug the directory structure
    classifier.debug_directory_scan()
    
    # Load dataset
    df = classifier.load_dataset()
    
    if df is None or len(df) == 0:
        print("❌ No data loaded. Exiting.")
        print("\n💡 Troubleshooting tips:")
        print("1. Check if data directory exists: C:/Users/Divya ch/Documents/spam_classifier/data/spamassassin-public-corpus/")
        print("2. Verify email files are in the subdirectories")
        print("3. Check file permissions")
        return
    
    # Preprocess emails
    classifier.preprocess_emails()
    
    # Train models
    nb_model, lr_model, X_test, y_test = classifier.train_models()
    
    if nb_model is not None and lr_model is not None:
        # Save the best model
        if accuracy_score(y_test, lr_model.predict(X_test)) > accuracy_score(y_test, nb_model.predict(X_test)):
            classifier.save_models(lr_model, "logistic_regression_model")
        else:
            classifier.save_models(nb_model, "naive_bayes_model")
        
        # Example prediction
        print("\n" + "="*50)
        print("🔮 EXAMPLE PREDICTION")
        print("="*50)
        
        test_email = """
        Subject: Win a FREE iPhone!
        
        Congratulations! You've been selected to win a FREE iPhone 15. 
        Click here to claim your prize: http://free-iphone-scam.com
        Limited time offer! Don't miss this opportunity!
        """
        
        prediction = classifier.predict_email(test_email)
        print(f"📧 Email: {test_email[:100]}...")
        print(f"🔍 Prediction: {prediction['prediction']}")
        print(f"📊 Spam Probability: {prediction['spam_probability']:.4f}")
        print(f"📊 Ham Probability: {prediction['ham_probability']:.4f}")
        
        print("\n🎉 Training completed successfully!")
    else:
        print("❌ Model training failed.")

if __name__ == "__main__":
    main()