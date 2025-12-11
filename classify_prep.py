import json
import pandas as pd
from collections import defaultdict

# --- Configuration ---
DATA_FILE = "Corona2.json"
OUTPUT_FILE = "medical_classification_data.csv"

def prepare_classification_data(data_file: str, output_file: str):
    """
    Loads the NER data and converts it into a sentence-level classification dataset.
    A sentence is labeled by the entity type it contains (e.g., if it has a 
    PATHOGEN, the whole sentence is labeled PATHOGEN).
    """
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{data_file}' was not found.")
        return

    # Use a list of dictionaries to build the final dataset
    classification_data = []

    # Iterate through each text example
    for example in data.get('examples', []):
        text = example['content']
        
        # We need a dictionary to collect all unique labels found in this text
        labels_found = defaultdict(int)
        
        # Check all annotations (entities) in the text
        for annotation in example.get('annotations', []):
            # The original tag names are our categories (e.g., 'pathogen', 'medicine')
            label = annotation['tag_name'].upper()
            labels_found[label] += 1

        # --- Decision Logic for Multi-Label Sentences ---
        # A single text can contain multiple entities (e.g., a MEDICINE and a PATHOGEN).
        # For simplicity in this initial classification project, we assign a single primary label.
        
        # Determine the primary label for the sentence
        primary_label = 'NO_ENTITY'
        
        # Priority can be assigned if needed, but here we just take the first unique label found
        if labels_found:
            # Get the first label found as the primary label
            primary_label = list(labels_found.keys())[0]

        # Append the result
        if primary_label != 'NO_ENTITY':
            classification_data.append({
                'text': text,
                'label': primary_label
            })

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(classification_data)
    
    # Optional: Display the label distribution
    print("--- Label Distribution ---")
    print(df['label'].value_counts())
    
    # Save the cleaned data
    df.to_csv(output_file, index=False)
    print(f"\nSuccessfully created classification dataset: {output_file}")
    print(f"Total labeled samples: {len(df)}")

if __name__ == "__main__":
    prepare_classification_data(DATA_FILE, OUTPUT_FILE)