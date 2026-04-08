"""
Sentiment Analysis on Product Reviews
Step 2: Data Cleaning & Preprocessing
Author: Data Science Project
Date: February 2026
"""

import pandas as pd
import numpy as np
import re
import string
import sys
import os

# Natural Language Processing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Set output encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Create directory if it doesn't exist
os.makedirs(DATA_PROCESSED, exist_ok=True)

print("="*80)
print("STEP 2: DATA CLEANING & PREPROCESSING")
print("="*80)
print()

# Download required NLTK data
print("Downloading required NLTK resources...")
print("-"*80)
nltk_resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
for resource in nltk_resources:
    try:
        nltk.download(resource, quiet=True)
    except:
        pass
print("[+] NLTK resources ready!")
print()

# ============================================================================
# LOAD THE CLEANED DATA FROM STEP 1
# ============================================================================
print("Loading data from Step 1...")
print("-"*80)

input_file = os.path.join(DATA_PROCESSED, 'cleaned_data_step1.csv')
df = pd.read_csv(input_file, encoding='utf-8')
print(f"[+] Loaded dataset with {len(df):,} rows and {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")
print()

print("Initial dataset preview:")
print(df.head())
print()

# ============================================================================
# STEP 2.1: HANDLE MISSING VALUES
# ============================================================================
print("STEP 2.1: Handle Missing Values")
print("-"*80)

print("Checking for missing values...")
missing_before = df.isnull().sum()
print(missing_before)
print()

# Remove rows with empty or null review text
initial_count = len(df)
df = df.dropna(subset=['review_text'])

# Also remove rows where review_text is empty string or whitespace only
df = df[df['review_text'].str.strip() != '']

rows_removed = initial_count - len(df)
print(f"Rows before: {initial_count:,}")
print(f"Rows after: {len(df):,}")
print(f"Rows removed: {rows_removed:,}")
print()

if rows_removed == 0:
    print("[+] No missing or empty values found - data is already clean!")
else:
    print(f"[+] Removed {rows_removed:,} rows with missing/empty review text")
print()

print("Why this step?")
print("-> Missing or empty text cannot be analyzed for sentiment")
print("-> Ensures every row has meaningful content for preprocessing")
print()

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def convert_to_lowercase(text):
    """Convert text to lowercase"""
    return text.lower()

def remove_punctuation_numbers_special(text):
    """Remove punctuation, numbers, and special characters"""
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove any remaining special characters
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def remove_extra_whitespaces(text):
    """Remove extra whitespaces"""
    return ' '.join(text.split())

def remove_stopwords_func(text, stop_words):
    """Remove stopwords from text"""
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def tokenize_text(text):
    """Tokenize text into words"""
    return word_tokenize(text)

def lemmatize_tokens(tokens, lemmatizer):
    """Apply lemmatization to tokens"""
    return [lemmatizer.lemmatize(token) for token in tokens]

def tokens_to_string(tokens):
    """Convert tokens back to string"""
    return ' '.join(tokens)

# ============================================================================
# STEP 2.2: CONVERT TO LOWERCASE
# ============================================================================
print("STEP 2.2: Convert to Lowercase")
print("-"*80)

df['processed_text'] = df['review_text'].apply(convert_to_lowercase)

print("Example transformations:")
for i in range(min(3, len(df))):
    print(f"Original:  '{df['review_text'].iloc[i]}'")
    print(f"Lowercase: '{df['processed_text'].iloc[i]}'")
    print()

print("Why this step?")
print("-> Ensures consistency: 'Good', 'good', 'GOOD' are treated as same word")
print("-> Reduces vocabulary size and improves model performance")
print()

# ============================================================================
# STEP 2.3: REMOVE PUNCTUATION, NUMBERS, SPECIAL CHARACTERS
# ============================================================================
print("STEP 2.3: Remove Punctuation, Numbers, and Special Characters")
print("-"*80)

df['processed_text'] = df['processed_text'].apply(remove_punctuation_numbers_special)

print("Example transformations:")
sample_idx = df[df['review_text'].str.contains(r'[0-9!.,?]', na=False, regex=True)].index
for i in sample_idx[:3]:
    print(f"Before: '{df['review_text'].iloc[i]}'")
    print(f"After:  '{df['processed_text'].iloc[i]}'")
    print()

print("Why this step?")
print("-> Punctuation (!,?.') doesn't contribute to sentiment in basic analysis")
print("-> Numbers are usually not meaningful for sentiment (e.g., '5' in text)")
print("-> Special characters (@#$%) add noise to the data")
print()

# ============================================================================
# STEP 2.4: REMOVE EXTRA WHITESPACES
# ============================================================================
print("STEP 2.4: Remove Extra Whitespaces")
print("-"*80)

df['processed_text'] = df['processed_text'].apply(remove_extra_whitespaces)

print("Example:")
print("Before: 'good    product     nice'")
print("After:  'good product nice'")
print()

print("Why this step?")
print("-> Multiple spaces can interfere with tokenization")
print("-> Normalizes text format for consistent processing")
print()

# ============================================================================
# STEP 2.5: REMOVE STOPWORDS
# ============================================================================
print("STEP 2.5: Remove Stopwords")
print("-"*80)

stop_words = set(stopwords.words('english'))
print(f"Total stopwords in English: {len(stop_words)}")
print(f"Sample stopwords: {list(stop_words)[:20]}")
print()

df['processed_text'] = df['processed_text'].apply(lambda x: remove_stopwords_func(x, stop_words))

print("Example transformations:")
for i in range(min(3, len(df))):
    print(f"Original:  '{df['review_text'].iloc[i]}'")
    print(f"No stops:  '{df['processed_text'].iloc[i]}'")
    print()

print("Why this step?")
print("-> Stopwords (the, is, and, a, etc.) are very common but carry little meaning")
print("-> Removing them reduces noise and focuses on meaningful words")
print("-> Reduces dimensionality for more efficient processing")
print()

# ============================================================================
# STEP 2.6: TOKENIZATION
# ============================================================================
print("STEP 2.6: Tokenize Text into Words")
print("-"*80)

df['tokens'] = df['processed_text'].apply(tokenize_text)

print("Example tokenization:")
for i in range(min(3, len(df))):
    print(f"Text:   '{df['processed_text'].iloc[i]}'")
    print(f"Tokens: {df['tokens'].iloc[i]}")
    print()

print("Why this step?")
print("-> Tokenization splits text into individual words (tokens)")
print("-> Essential for word-level analysis and lemmatization")
print("-> Enables counting word frequencies and patterns")
print()

# ============================================================================
# STEP 2.7: LEMMATIZATION
# ============================================================================
print("STEP 2.7: Apply Lemmatization")
print("-"*80)

lemmatizer = WordNetLemmatizer()
print("Using WordNet Lemmatizer from NLTK")
print()

df['lemmatized_tokens'] = df['tokens'].apply(lambda x: lemmatize_tokens(x, lemmatizer))

print("Example lemmatization:")
print("running -> run, better -> good, cars -> car")
print()

print("Sample transformations:")
for i in range(min(3, len(df))):
    print(f"Before lemmatization: {df['tokens'].iloc[i]}")
    print(f"After lemmatization:  {df['lemmatized_tokens'].iloc[i]}")
    print()

print("Why this step?")
print("-> Lemmatization reduces words to their base/dictionary form")
print("-> 'running', 'runs', 'ran' all become 'run'")
print("-> Reduces vocabulary size while preserving meaning")
print("-> Better than stemming as it produces valid words")
print()

# ============================================================================
# STEP 2.8: CREATE FINAL CLEANED TEXT COLUMN
# ============================================================================
print("STEP 2.8: Create Final Cleaned Text Column")
print("-"*80)

df['cleaned_text'] = df['lemmatized_tokens'].apply(tokens_to_string)

print("[+] Created 'cleaned_text' column with fully preprocessed text")
print()

print("Original review_text column preserved for reference!")
print()

# ============================================================================
# COMPARISON: BEFORE vs AFTER
# ============================================================================
print("="*80)
print("BEFORE vs AFTER COMPARISON")
print("="*80)
print()

print("Sample transformations showing the complete preprocessing pipeline:")
print("-"*80)

for i in range(min(5, len(df))):
    print(f"\nExample {i+1}:")
    print(f"  ORIGINAL: '{df['review_text'].iloc[i]}'")
    print(f"  CLEANED:  '{df['cleaned_text'].iloc[i]}'")
    print(f"  Tokens:   {df['lemmatized_tokens'].iloc[i]}")

print()

# ============================================================================
# STATISTICS & SUMMARY
# ============================================================================
print("="*80)
print("PREPROCESSING STATISTICS")
print("="*80)
print()

# Calculate average token counts
df['original_word_count'] = df['review_text'].str.split().str.len()
df['cleaned_word_count'] = df['cleaned_text'].str.split().str.len()

print("Word Count Statistics:")
print("-"*80)
print(f"Average words in original text: {df['original_word_count'].mean():.2f}")
print(f"Average words in cleaned text:  {df['cleaned_word_count'].mean():.2f}")
print(f"Average reduction: {(df['original_word_count'].mean() - df['cleaned_word_count'].mean()):.2f} words")
print(f"Reduction percentage: {((df['original_word_count'].mean() - df['cleaned_word_count'].mean()) / df['original_word_count'].mean() * 100):.2f}%")
print()

print("Token Statistics:")
print("-"*80)
df['token_count'] = df['lemmatized_tokens'].apply(len)
print(f"Average tokens per review: {df['token_count'].mean():.2f}")
print(f"Min tokens: {df['token_count'].min()}")
print(f"Max tokens: {df['token_count'].max()}")
print()

# ============================================================================
# SAVE PREPROCESSED DATA
# ============================================================================
print("Saving preprocessed data...")
print("-"*80)

# Save with all columns for reference
output_file = os.path.join(DATA_PROCESSED, 'preprocessed_data_step2.csv')
df_to_save = df[['review_text', 'rating', 'cleaned_text']]
df_to_save.to_csv(output_file, index=False, encoding='utf-8')

print(f"[+] Saved preprocessed data to: {output_file}")
print(f"[+] Columns saved: review_text (original), rating, cleaned_text (preprocessed)")
print(f"[+] Total rows: {len(df_to_save):,}")
print()

# Also save the full dataframe with all intermediate steps for analysis
full_output_file = os.path.join(DATA_PROCESSED, 'preprocessed_data_step2_full.csv')
df.to_csv(full_output_file, index=False, encoding='utf-8')
print(f"[+] Full dataset (with intermediate steps) saved to: {full_output_file}")
print()

# ============================================================================
# STEP 2 COMPLETION SUMMARY
# ============================================================================
print("="*80)
print("STEP 2 COMPLETION SUMMARY")
print("="*80)
print(f"""
Data Cleaning & Preprocessing has been successfully completed!

All 8 preprocessing steps performed:
[+] 1. Handled missing values - Removed {rows_removed:,} rows with empty/null text
[+] 2. Converted to lowercase - Ensured text consistency
[+] 3. Removed punctuation, numbers, special characters - Cleaned noise
[+] 4. Removed extra whitespaces - Normalized text format
[+] 5. Removed stopwords - Eliminated {len(stop_words)} common words
[+] 6. Tokenized text - Split into individual words
[+] 7. Applied lemmatization - Reduced words to base forms
[+] 8. Created cleaned_text column - Original text preserved

Final Dataset:
- Rows: {len(df):,}
- Columns: review_text (original), rating, cleaned_text (preprocessed)
- Average word reduction: {((df['original_word_count'].mean() - df['cleaned_word_count'].mean()) / df['original_word_count'].mean() * 100):.2f}%
- Output file: {output_file}

The preprocessed data is now ready for Step 3: Feature Extraction!
""")
print("="*80)
