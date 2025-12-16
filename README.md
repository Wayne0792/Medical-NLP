# 🏥 Medical NLP: Intent Classification & Entity Recognition

This repository showcases two distinct Natural Language Processing (NLP) projects designed to automate the processing and triage of medical text, demonstrating proficiency in both traditional Machine Learning (ML) and modern Transformer architectures.

---

## 1️⃣ Project 1: Medical Intent Classification (Complete & Deployed)

This project focuses on **Text Classification**—determining the primary category (intent) of a sentence from patient notes or research papers. It is built to be fast, reliable, and deployable for initial document triage.

### 🎯 Goal

To classify an input sentence into one of three distinct medical categories: **`MEDICALCONDITION`**, **`MEDICINE`**, or **`PATHOGEN`**.

### ⚙️ Technology Stack

* **Language:** Python
* **Libraries:** scikit-learn, pandas, joblib
* **Model:** TF-IDF Vectorization $\rightarrow$ Logistic Regression

### 🚀 Results & Demonstration

The model was trained on a highly limited dataset (30 samples, demonstrating robustness under data scarcity). The final model artifacts (`medical_intent_classifier.pkl`) are saved for quick deployment.

**[Link to your prediction script output video or GIF here, e.g., on LinkedIn or YouTube]**

#### Diagnosis of Data Scarcity

The initial training revealed a common ML challenge: **class imbalance**. Due to the low number of samples for the `PATHOGEN` class (only 5 samples), the model demonstrated a bias toward the majority class (`MEDICALCONDITION`).

| Input Text | Expected Label | Model Prediction | Confidence |
| :--- | :--- | :--- | :--- |
| "A large amount of **Salmonella** was found in the sample." | `PATHOGEN` | `MEDICALCONDITION` | 47% |

This result highlights that while the model works, the **next steps** would require **Data Augmentation** or the use of **Class Weighting** (e.g., `class_weight='balanced'` in Logistic Regression) to ensure equitable performance across all minority classes. 

### 📁 Key Files for Project 1

* `classify_prep.py`: Data cleaning and conversion from NER format to Sentence/Label classification format.
* `classifier_train.py`: Trains the TF-IDF vectorizer and the Logistic Regression model. Outputs the model artifacts and evaluation metrics.
* `predict_intent.py`: Deployment script to load the saved model (`.pkl`) and accept real-time user input.

---

## 2️⃣ Project 2: BioBERT Named Entity Recognition (Advanced Setup)

This project demonstrates proficiency in setting up a cutting-edge deep learning architecture for the more complex task of **Named Entity Recognition (NER)**—identifying and classifying specific spans of text.

### 🎯 Goal

To fine-tune the BioBERT Transformer model to recognize and tag specific entities (`MEDICALCONDITION`, `MEDICINE`, `PATHOGEN`) within sentences.

### ⚙️ Advanced Technical Skills Demonstrated

This project successfully navigates the complex setup required for modern NLP:

1.  **Transfer Learning:** Utilizing the pre-trained weights of **BioBERT (dmis-lab/biobert-v1.1)**, a model pre-trained on biomedical texts. 
2.  **Configuration Management:** Managing the complex `config.cfg` file structure required by spaCy v3 to correctly connect the external Transformer component to the internal NER pipeline.
3.  **Dependency Handling:** Successful installation and integration of `spacy-transformers`, `transformers`, and `torch`.

### ⚠️ Note on Project Status

This component was initialized and configured for training, demonstrating the ability to set up advanced Transformer pipelines. Due to environment-specific conflicts encountered during the final validation step (related to package versioning), full training was temporarily halted to prioritize the completion of Project 1. The final, validated `config.cfg` required for BioBERT training is available in the project folder.

---

## 💻 Setup and Usage

To replicate the Intent Classification project:

1.  **Clone the Repository:**
    ```bash
    git clone [Your Repository URL Here]
    cd Medical-NLP-Portfolio
    ```
2.  **Setup Virtual Environment & Dependencies:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # pip install -r requirements.txt (Assuming you create one)
    pip install pandas scikit-learn joblib
    ```
3.  **Run the Prediction Demo:**
    ```bash
    python 01_Intent_Classifier/predict_intent.py
    ```
