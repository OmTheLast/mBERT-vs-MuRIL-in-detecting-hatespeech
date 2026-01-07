# 1. Upload the file manually
import pandas as pd
import os

# 2. Load the dataset directly from the parent directory
dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'combined_hate_speech_dataset.csv')
print(f"Loading dataset from: {dataset_path}")

# 3. Read the file into Pandas
# We use 'try-except' because Hinglish text sometimes uses weird encodings
try:
    df = pd.read_csv(dataset_path, encoding='utf-8')
except:
    df = pd.read_csv(dataset_path, encoding='latin-1')

# 4. Inspect the structure (The most important part)
print("\n--- DATASET INFO ---")
print("Columns:", list(df.columns))
print("Total Rows:", len(df))

# 5. Check the Labels
# We need to know if the labels are '0/1', 'Yes/No', or 'Hate/Non-Hate'
# We assume the label column is the last one, but we print all unique values to be sure.
print("\n--- SAMPLE DATA ---")
print(df.head(3))

import re  # 're' stands for Regular Expressions (a pattern matching tool)

# 1. Keep only the columns we need
# (Dropping 'source', 'profanity_score' etc. as they are not needed for training)
df = df[['text', 'hate_label']]

# 2. Check the Balance (Crucial)
# If you have 29,000 '0's and only 500 '1's, the model will cheat by always guessing '0'.
print("--- Class Distribution ---")
print(df['hate_label'].value_counts())

# 3. The Cleaning Function
def clean_hinglish_text(text):
    # Ensure text is a string (handles rare empty rows)
    text = str(text).lower()

    # Remove URLs (http://...)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove User Mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # Remove special characters (keep only words and numbers)
    # We keep English letters and simple punctuation.
    # Note: If your dataset had Hindi Script (Devanagari), we would change this.
    # Since it is Romanized Hinglish ("kya hua"), standard cleanup works well.
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# 4. Apply the cleaning
print("\nCleaning text... this might take 10 seconds...")
df['clean_text'] = df['text'].apply(clean_hinglish_text)

# 5. Show Before vs After
print("\n--- Cleaning Results ---")
print("Original:", df['text'].iloc[5])
print("Cleaned: ", df['clean_text'].iloc[5])