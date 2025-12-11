import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

# --- Configuration ---
DATA_FILE = "medical_classification_data.csv"
MODEL_OUTPUT = "medical_intent_classifier.pkl"

def train_and_evaluate_classifier(data_file: str, model_output: str):
    # 1. Load Data
    df = pd.read_csv(data_file)
    X = df['text']
    y = df['label']
    
    # 2. Encode Labels
    # Convert string labels (PATHOGEN, MEDICINE) into numerical format (0, 1, 2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Store classes for later display
    class_names = le.classes_
    print(f"Encoded Classes: {list(class_names)}")

    # 3. Split Data
    # Use 80% for training and 20% for testing (6 samples for test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded # Crucial for small datasets to maintain label balance
    )
    
    print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 4. Feature Extraction (TF-IDF)
    # TF-IDF converts text into numerical feature vectors
    # max_features is a good way to limit complexity on small data
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    

    # 5. Train Model (Logistic Regression)
    # A fast and reliable classifier for text data
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test_vec)
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    # Display the F1-Score, Precision, and Recall for each class
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("\n" + "="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    # Display confusion matrix to visualize correct vs. incorrect predictions
    print(confusion_matrix(y_test, y_pred))
    

    # 7. Save Model and Vectorizer (for later use in a prediction script)
    # joblib is standard for saving scikit-learn pipelines
    joblib.dump({
        'model': model, 
        'vectorizer': vectorizer, 
        'le': le
    }, model_output)
    
    print(f"\nModel and vectorizer saved to {model_output}")

if __name__ == "__main__":
    train_and_evaluate_classifier(DATA_FILE, MODEL_OUTPUT)