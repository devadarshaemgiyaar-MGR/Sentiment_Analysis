"""
Sentiment Analysis on Product Reviews
Step 3: Feature Extraction
Author: Data Science Project
Date: February 2026
"""

import pandas as pd
import numpy as np
import sys
import os
import pickle

# Machine Learning library
from sklearn.feature_extraction.text import TfidfVectorizer

# Set output encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Create directories if they don't exist
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("="*80)
print("STEP 3: FEATURE EXTRACTION")
print("="*80)
print()

# ============================================================================
# LOAD PREPROCESSED DATA FROM STEP 2
# ============================================================================
print("Loading preprocessed data from Step 2...")
print("-"*80)

input_file = os.path.join(DATA_PROCESSED, 'preprocessed_data_step2.csv')
df = pd.read_csv(input_file, encoding='utf-8')

print(f"[+] Loaded dataset with {len(df):,} rows and {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")
print()

print("Sample data:")
print(df.head(10))
print()

# ============================================================================
# STEP 3.1: EXTRACT AND CLEAN TEXT FOR FEATURE EXTRACTION
# ============================================================================
print("STEP 3.1: Extract and Clean Text Column")
print("-"*80)

# Check for missing or empty values in cleaned_text
print("Checking for invalid values in cleaned_text...")
initial_count = len(df)

# Remove rows with NaN or empty strings in cleaned_text
df = df.dropna(subset=['cleaned_text'])
df = df[df['cleaned_text'].str.strip() != '']

rows_removed = initial_count - len(df)
print(f"Rows before cleaning: {initial_count:,}")
print(f"Rows after cleaning: {len(df):,}")
print(f"Rows removed: {rows_removed:,}")
print()

if rows_removed > 0:
    print(f"[!] Removed {rows_removed:,} rows with NaN or empty cleaned_text")
else:
    print("[+] No invalid values found")
print()

# Use the cleaned_text column from preprocessing
cleaned_texts = df['cleaned_text'].values

print(f"Total reviews for feature extraction: {len(cleaned_texts):,}")
print()

print("Sample cleaned texts:")
for i in range(min(5, len(cleaned_texts))):
    print(f"{i+1}. '{cleaned_texts[i]}'")
print()

print("Why this step?")
print("-> Cleaned text is the input for TF-IDF vectorization")
print("-> Each review will be converted to a numerical feature vector")
print("-> Empty or NaN values would cause errors in vectorization")
print()

# ============================================================================
# STEP 3.2: APPLY TF-IDF VECTORIZATION
# ============================================================================
print("STEP 3.2: Apply TF-IDF Vectorization")
print("-"*80)

print("What is TF-IDF?")
print("-"*80)
print("""
TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic
that reflects how important a word is to a document in a collection.

Components:
1. TF (Term Frequency): How often a word appears in a document
   - Higher frequency = more important to that specific document

2. IDF (Inverse Document Frequency): How rare a word is across all documents
   - Rare words get higher scores (more discriminative)
   - Common words get lower scores (less useful for classification)

Formula: TF-IDF = TF × IDF

Why use TF-IDF?
-> Converts text into numerical features that ML algorithms can process
-> Gives higher weight to distinctive/meaningful words
-> Reduces weight of very common words (even after stopword removal)
-> Creates sparse matrix representation (memory efficient)
-> Better than simple word counts (Bag of Words) for sentiment analysis
""")
print()

# Initialize TF-IDF Vectorizer
print("Initializing TF-IDF Vectorizer...")
print("-"*80)

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,      # Keep top 5000 most important features
    min_df=2,               # Word must appear in at least 2 documents
    max_df=0.9,             # Ignore words appearing in >90% of documents
    ngram_range=(1, 2),     # Use unigrams and bigrams
    sublinear_tf=True       # Apply sublinear tf scaling (1 + log(tf))
)

print("TF-IDF Vectorizer Parameters:")
print(f"  - max_features: 5000 (top 5000 most important words/phrases)")
print(f"  - min_df: 2 (word must appear in at least 2 reviews)")
print(f"  - max_df: 0.9 (ignore very common words in >90% reviews)")
print(f"  - ngram_range: (1, 2) (use single words and 2-word phrases)")
print(f"  - sublinear_tf: True (use log scaling for term frequency)")
print()

print("Fitting TF-IDF Vectorizer and transforming text...")
print("-"*80)

# Fit and transform the cleaned text
X = tfidf_vectorizer.fit_transform(cleaned_texts)

print(f"[+] TF-IDF transformation complete!")
print()

# ============================================================================
# STEP 3.3: FEATURE MATRIX INFORMATION
# ============================================================================
print("STEP 3.3: Feature Matrix (X) Information")
print("-"*80)

print(f"Feature Matrix Shape: {X.shape}")
print(f"  - Number of samples (reviews): {X.shape[0]:,}")
print(f"  - Number of features (words/phrases): {X.shape[1]:,}")
print(f"  - Matrix type: {type(X)}")
print(f"  - Matrix format: Sparse (CSR - Compressed Sparse Row)")
print(f"  - Sparsity: {(1.0 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%")
print()

print("Why sparse matrix?")
print("-> Most reviews use only a small subset of total vocabulary")
print("-> Sparse format saves memory (stores only non-zero values)")
print(f"-> Memory saved: ~{(1.0 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.1f}%")
print()

# Get feature names (vocabulary)
feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"Total vocabulary size: {len(feature_names):,} unique terms")
print()

print("Sample feature names (top 20):")
print(feature_names[:20])
print()

# ============================================================================
# STEP 3.4: SAMPLE FEATURE VECTORS
# ============================================================================
print("STEP 3.4: Sample Feature Vectors")
print("-"*80)

print("Examining feature vectors for first 3 reviews:")
print()

for i in range(min(3, X.shape[0])):
    print(f"Review {i+1}: '{cleaned_texts[i]}'")
    print(f"  Feature vector shape: {X[i].shape}")
    print(f"  Non-zero features: {X[i].nnz}")
    
    # Get non-zero features for this review
    row = X[i].toarray().flatten()
    non_zero_indices = np.nonzero(row)[0]
    
    if len(non_zero_indices) > 0:
        print(f"  Top features with TF-IDF scores:")
        # Get top 5 features by score
        top_indices = non_zero_indices[np.argsort(row[non_zero_indices])[-5:][::-1]]
        for idx in top_indices:
            print(f"    - '{feature_names[idx]}': {row[idx]:.4f}")
    print()

# ============================================================================
# STEP 3.5: PREPARE TARGET VARIABLE (y)
# ============================================================================
print("STEP 3.5: Prepare Target Variable (y)")
print("-"*80)

print("Mapping ratings to sentiment categories:")
print("  - Positive: Ratings 4-5")
print("  - Neutral:  Rating 3")
print("  - Negative: Ratings 1-2")
print()

def map_rating_to_sentiment(rating):
    """
    Map numerical rating to sentiment category
    """
    # Convert to string and try to extract numeric value
    rating_str = str(rating).strip()
    
    # Try to convert to float
    try:
        rating_num = float(rating_str)
        
        if rating_num >= 4:
            return 'positive'
        elif rating_num == 3:
            return 'neutral'
        elif rating_num <= 2:
            return 'negative'
        else:
            return 'unknown'
    except:
        # For non-numeric ratings (product names, etc.)
        return 'unknown'

# Apply mapping to create sentiment labels
df['sentiment'] = df['rating'].apply(map_rating_to_sentiment)

print("Sentiment distribution after mapping:")
print(df['sentiment'].value_counts())
print()

# Remove 'unknown' sentiments if any
initial_count = len(df)
unknown_mask = (df['sentiment'] != 'unknown').values

# Filter both dataframe and feature matrix using boolean mask
df = df[unknown_mask].reset_index(drop=True)
X = X[unknown_mask]

if len(df) < initial_count:
    print(f"[!] Removed {initial_count - len(df):,} rows with invalid ratings")
    print()

# Create target variable (y)
y = df['sentiment'].values

print(f"Target variable (y) shape: {y.shape}")
print(f"Target variable type: {type(y)}")
print()

print("Sample target values:")
print(y[:10])
print()

# ============================================================================
# STEP 3.6: SENTIMENT DISTRIBUTION ANALYSIS
# ============================================================================
print("STEP 3.6: Sentiment Distribution Analysis")
print("-"*80)

sentiment_counts = pd.Series(y).value_counts()
sentiment_percentages = pd.Series(y).value_counts(normalize=True) * 100

print("Detailed sentiment distribution:")
for sentiment in ['positive', 'neutral', 'negative']:
    if sentiment in sentiment_counts.index:
        count = sentiment_counts[sentiment]
        percentage = sentiment_percentages[sentiment]
        print(f"  {sentiment.capitalize():10s}: {count:6,} reviews ({percentage:5.2f}%)")

print()

print("Class balance analysis:")
max_count = sentiment_counts.max()
min_count = sentiment_counts.min()
imbalance_ratio = max_count / min_count
print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print(f"  [!] Warning: Dataset is imbalanced (ratio > 3:1)")
    print(f"  Consider using class weights or resampling techniques in modeling")
else:
    print(f"  [+] Dataset is reasonably balanced")
print()

# ============================================================================
# STEP 3.7: SAVE FEATURE MATRIX AND TARGET VARIABLE
# ============================================================================
print("STEP 3.7: Save Feature Matrix and Target Variable")
print("-"*80)

# Save feature matrix (sparse format for memory efficiency)
feature_matrix_file = os.path.join(DATA_PROCESSED, 'feature_matrix_X.npz')
np.savez_compressed(feature_matrix_file, 
                    data=X.data, 
                    indices=X.indices, 
                    indptr=X.indptr, 
                    shape=X.shape)
print(f"[+] Feature matrix saved: {feature_matrix_file}")

# Save target variable
target_file = os.path.join(DATA_PROCESSED, 'target_variable_y.npy')
np.save(target_file, y)
print(f"[+] Target variable saved: {target_file}")

# Save TF-IDF vectorizer for future use
vectorizer_file = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
with open(vectorizer_file, 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"[+] TF-IDF vectorizer saved: {vectorizer_file}")

# Save feature names
feature_names_file = os.path.join(DATA_PROCESSED, 'feature_names.npy')
np.save(feature_names_file, feature_names)
print(f"[+] Feature names saved: {feature_names_file}")
print()

# Also save a dataset with features for reference
df_final = df[['review_text', 'rating', 'sentiment']].copy()
final_output_file = os.path.join(DATA_PROCESSED, 'data_with_features_step3.csv')
df_final.to_csv(final_output_file, index=False, encoding='utf-8')
print(f"[+] Final dataset with sentiment labels saved: {final_output_file}")
print()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("="*80)
print("FEATURE EXTRACTION SUMMARY")
print("="*80)
print()

print("TF-IDF Vectorization Statistics:")
print("-"*80)
print(f"  Input: {len(cleaned_texts):,} cleaned reviews")
print(f"  Vocabulary size: {len(feature_names):,} unique terms")
print(f"  Feature matrix shape: {X.shape[0]:,} × {X.shape[1]:,}")
print(f"  Matrix sparsity: {(1.0 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%")
print(f"  Non-zero elements: {X.nnz:,}")
print()

print("Target Variable Statistics:")
print("-"*80)
print(f"  Total samples: {len(y):,}")
print(f"  Positive samples: {sentiment_counts.get('positive', 0):,} ({sentiment_percentages.get('positive', 0):.2f}%)")
print(f"  Neutral samples: {sentiment_counts.get('neutral', 0):,} ({sentiment_percentages.get('neutral', 0):.2f}%)")
print(f"  Negative samples: {sentiment_counts.get('negative', 0):,} ({sentiment_percentages.get('negative', 0):.2f}%)")
print()

print("Top 10 Most Important Features (highest average TF-IDF scores):")
print("-"*80)
# Calculate mean TF-IDF score for each feature
mean_tfidf = np.asarray(X.mean(axis=0)).flatten()
top_indices = np.argsort(mean_tfidf)[-10:][::-1]

for i, idx in enumerate(top_indices, 1):
    print(f"  {i:2d}. '{feature_names[idx]}': {mean_tfidf[idx]:.6f}")
print()

# ============================================================================
# STEP 3 COMPLETION SUMMARY
# ============================================================================
print("="*80)
print("STEP 3 COMPLETION SUMMARY")
print("="*80)
print(f"""
Feature Extraction has been successfully completed!

All required steps performed:
[+] 1. Loaded cleaned review text from Step 2
[+] 2. Applied TF-IDF vectorization with optimized parameters
[+] 3. Created feature matrix (X) with shape {X.shape[0]:,} × {X.shape[1]:,}
[+] 4. Converted reviews to numerical feature vectors
[+] 5. Prepared target variable (y) by mapping ratings to sentiment
[+] 6. Displayed feature matrix shape and sample values

Final Outputs:
- Feature matrix (X): {X.shape[0]:,} samples, {X.shape[1]:,} features
- Target variable (y): {len(y):,} sentiment labels
- Vocabulary: {len(feature_names):,} unique terms
- Files saved in: {DATA_PROCESSED}

Class Distribution:
- Positive: {sentiment_counts.get('positive', 0):,} reviews ({sentiment_percentages.get('positive', 0):.2f}%)
- Neutral: {sentiment_counts.get('neutral', 0):,} reviews ({sentiment_percentages.get('neutral', 0):.2f}%)
- Negative: {sentiment_counts.get('negative', 0):,} reviews ({sentiment_percentages.get('negative', 0):.2f}%)

The feature-engineered data is now ready for Step 4: Model Training!
""")
print("="*80)
