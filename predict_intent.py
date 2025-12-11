import joblib
import pandas as pd

# --- Configuration ---
MODEL_FILE = "medical_intent_classifier.pkl"

def load_and_predict():
    """
    Loads the saved model artifacts and runs a loop for real-time predictions.
    """
    try:
        # Load the model, vectorizer, and label encoder
        artifacts = joblib.load(MODEL_FILE)
        model = artifacts['model']
        vectorizer = artifacts['vectorizer']
        le = artifacts['le']
        
        print("=" * 50)
        print("Medical Intent Classifier Loaded (TF-IDF + Logistic Regression)")
        print("Enter a sentence to predict its primary medical category.")
        print("Type 'quit' or 'exit' to stop.")
        print("=" * 50)

    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_FILE}' not found. Did you run classifier_train.py?")
        return

    # Start the prediction loop
    while True:
        # Get user input
        user_input = input("\n[INPUT TEXT] > ")
        
        if user_input.lower() in ['quit', 'exit']:
            break

        if not user_input.strip():
            continue

        try:
            # 1. Vectorize the input text using the trained vectorizer
            # Note: We use .transform(), NOT .fit_transform()
            input_vec = vectorizer.transform([user_input])
            
            # 2. Predict the encoded label (e.g., 0, 1, or 2)
            encoded_prediction = model.predict(input_vec)
            
            # 3. Inverse transform to get the human-readable label (e.g., MEDICINE)
            predicted_label = le.inverse_transform(encoded_prediction)[0]
            
            # Optional: Get the prediction probability score
            prediction_proba = model.predict_proba(input_vec)
            confidence = np.max(prediction_proba)

            print("-" * 50)
            print(f"✅ Predicted Intent: **{predicted_label}** (Confidence: {confidence:.2f})")
            print("-" * 50)

        except Exception as e:
            print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    import numpy as np # Import numpy only if needed for confidence score
    load_and_predict()