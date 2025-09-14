import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def analyze_model_comprehensive():
    try:
        # Load model and vectorizer
        model = joblib.load('../models/logistic_regression_model.pkl')
        vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
        
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        
        # Feature importance analysis
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        
        # Top 15 spam indicators (positive coefficients)
        top_spam_indices = coefficients.argsort()[-15:][::-1]
        top_spam_features = feature_names[top_spam_indices]
        top_spam_weights = coefficients[top_spam_indices]
        
        # Top 15 ham indicators (negative coefficients)
        top_ham_indices = coefficients.argsort()[:15]
        top_ham_features = feature_names[top_ham_indices]
        top_ham_weights = coefficients[top_ham_indices]
        
        print("\n🔝 Top 15 Spam Indicators:")
        for i, (feature, weight) in enumerate(zip(top_spam_features, top_spam_weights), 1):
            print(f"   {i:2d}. {feature:20s}: {weight:+.4f}")
        
        print("\n🔝 Top 15 Ham Indicators:")
        for i, (feature, weight) in enumerate(zip(top_ham_features, top_ham_weights), 1):
            print(f"   {i:2d}. {feature:20s}: {weight:+.4f}")
        
        # Create visualizations - use default style instead
        plt.style.use('default')
        sns.set_style("whitegrid")
        
        # Plot 1: Top spam features
        plt.figure(figsize=(12, 10))
        
        # Top spam features
        plt.subplot(2, 1, 1)
        colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(top_spam_features)))
        bars = plt.barh(range(len(top_spam_features)), top_spam_weights, color=colors)
        plt.yticks(range(len(top_spam_features)), top_spam_features)
        plt.xlabel('Feature Weight (Importance for Spam)')
        plt.title('Top 15 Spam Indicators')
        plt.gca().invert_yaxis()
        
        # Top ham features
        plt.subplot(2, 1, 2)
        colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(top_ham_features)))
        bars = plt.barh(range(len(top_ham_features)), top_ham_weights, color=colors)
        plt.yticks(range(len(top_ham_features)), top_ham_features)
        plt.xlabel('Feature Weight (Importance for Ham)')
        plt.title('Top 15 Ham Indicators')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('../models/feature_importance_comprehensive.png', dpi=300, bbox_inches='tight')
        print("\n📊 Comprehensive feature importance plot saved to models/feature_importance_comprehensive.png")
        
        # Distribution of coefficients
        plt.figure(figsize=(10, 6))
        plt.hist(coefficients, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Feature Weight')
        plt.ylabel('Frequency')
        plt.title('Distribution of Feature Weights')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.savefig('../models/weight_distribution.png', dpi=300, bbox_inches='tight')
        print("📊 Weight distribution plot saved to models/weight_distribution.png")
        
        # Show some statistics
        print(f"\n📈 Model Statistics:")
        print(f"   Total features: {len(coefficients):,}")
        print(f"   Positive weights (spam indicators): {sum(coefficients > 0):,}")
        print(f"   Negative weights (ham indicators): {sum(coefficients < 0):,}")
        print(f"   Zero weights: {sum(coefficients == 0):,}")
        print(f"   Max weight: {coefficients.max():.4f}")
        print(f"   Min weight: {coefficients.min():.4f}")
        print(f"   Mean weight: {coefficients.mean():.4f}")
        
        plt.show()  # Show the plots
        
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("Please make sure you've trained the model and the files exist in ../models/")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_model_comprehensive()