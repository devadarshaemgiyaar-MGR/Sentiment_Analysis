"""
Sentiment Analysis on Product Reviews
Step 1: Data Collection
Author: Data Science Project
Date: February 2026
"""

import pandas as pd
import numpy as np
import sys
import os

# Set output encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Create directories if they don't exist
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)

print("="*80)
print("STEP 1: DATA COLLECTION")
print("="*80)
print()

# ============================================================================
# 1. LOAD THE DATASET
# ============================================================================
print("1. Loading the Flipkart Product Review Dataset...")
print("-"*80)

dataset_path = os.path.join(DATA_RAW, 'Dataset-SA.csv')
df = pd.read_csv(dataset_path, encoding='utf-8', low_memory=False)
print("[+] Dataset loaded successfully!")
print()

# ============================================================================
# 2. DISPLAY FIRST FEW RECORDS
# ============================================================================
print("2. First 5 Records of the Dataset:")
print("-"*80)
print(df.head())
print()

# ============================================================================
# 3. TOTAL NUMBER OF ROWS AND COLUMNS
# ============================================================================
print("3. Dataset Dimensions:")
print("-"*80)
print(f"Total Number of Rows: {df.shape[0]:,}")
print(f"Total Number of Columns: {df.shape[1]}")
print()

# ============================================================================
# 4. LIST AND EXPLAIN ALL COLUMN NAMES
# ============================================================================
print("4. Column Names and Data Types:")
print("-"*80)
print(df.dtypes)
print()

print("Explanation of Each Column:")
print("-"*80)
column_explanations = {
    'product_name': 'Name of the product being reviewed',
    'product_price': 'Price of the product',
    'Rate': 'Customer rating/score given to the product (TARGET for sentiment)',
    'Review': 'Detailed review text written by customer',
    'Summary': 'Brief summary of the review',
    'Sentiment': 'Pre-labeled sentiment (positive/negative/neutral)'
}

for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")
    print(f"   - Data Type: {df[col].dtype}")
    print(f"   - Non-Null Count: {df[col].count():,}")
    print(f"   - Explanation: {column_explanations.get(col, 'Additional product information')}")
    print(f"   - Sample Value: {str(df[col].iloc[0])[:80] if len(df) > 0 else 'N/A'}")
    print()

# ============================================================================
# 5. IDENTIFY RELEVANT COLUMNS FOR SENTIMENT ANALYSIS
# ============================================================================
print("5. Columns Relevant for Sentiment Analysis:")
print("-"*80)

print("[+] Review Text Column: 'Review'")
print("  -> This contains the detailed customer feedback/review text")
print("  -> This is our PRIMARY text data for sentiment analysis")
print()

print("[+] Rating Column: 'Rate'")
print("  -> This contains the numerical rating (1-5 stars)")
print("  -> This serves as our sentiment indicator/target variable")
print()

print("[+] Summary Column: 'Summary' (Optional)")
print("  -> Contains brief review summaries")
print("  -> Can be used as additional text feature or combined with Review")
print()

print("Decision: We will use 'Review' as review_text and 'Rate' as rating")
print()

# ============================================================================
# 6. CHECK FOR MISSING/NULL VALUES
# ============================================================================
print("6. Missing/Null Values Analysis:")
print("-"*80)

missing_data = pd.DataFrame({
    'Column Name': df.columns,
    'Missing Count': df.isnull().sum().values,
    'Missing Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
})

print(missing_data.to_string(index=False))
print()

total_missing = df.isnull().sum().sum()
print(f"Total Missing Values in Dataset: {total_missing:,}")
print()

# Missing values in key columns
print("Key Observations:")
print(f"- Review column has {df['Review'].isnull().sum():,} missing values ({(df['Review'].isnull().sum()/len(df)*100):.2f}%)")
print(f"- Rate column has {df['Rate'].isnull().sum():,} missing values ({(df['Rate'].isnull().sum()/len(df)*100):.2f}%)")
print()

# ============================================================================
# 7. SELECT ONLY REQUIRED COLUMNS
# ============================================================================
print("7. Selecting Required Columns for Sentiment Analysis:")
print("-"*80)

# Select Review and Rate columns
df_final = df[['Review', 'Rate']].copy()

# Rename columns for standardization
df_final.columns = ['review_text', 'rating']

print("[+] Selected columns: 'Review' and 'Rate'")
print("[+] Renamed to: 'review_text' and 'rating'")
print()

# Remove rows with missing values in selected columns
initial_rows = len(df_final)
df_final = df_final.dropna()
rows_dropped = initial_rows - len(df_final)

print(f"Rows before removing null values: {initial_rows:,}")
print(f"Rows after removing null values: {len(df_final):,}")
print(f"Rows dropped: {rows_dropped:,}")
print()

# ============================================================================
# 8. FINAL DATASET STRUCTURE
# ============================================================================
print("8. Final Dataset Structure:")
print("-"*80)

print("Shape:", df_final.shape)
print(f"Rows: {df_final.shape[0]:,}, Columns: {df_final.shape[1]}")
print()

print("Column Information:")
print(df_final.info())
print()

print("Sample Records from Final Dataset:")
print(df_final.head(10))
print()

print("Statistical Summary of Review Text Length:")
df_final['review_length'] = df_final['review_text'].astype(str).str.len()
print(df_final['review_length'].describe())
df_final = df_final.drop('review_length', axis=1)
print()

print("Rating Distribution:")
rating_dist = df_final['rating'].value_counts().sort_index()
# Display rating distribution safely
print(f"Total Unique Ratings: {len(rating_dist)}")
print("Top 10 Most Common Ratings:")
for idx, (rating, count) in enumerate(rating_dist.head(10).items(), 1):
    print(f"  {idx}. Rating '{rating}': {count:,} reviews")
print()

print("Rating Statistics:")
print(f"- Unique Ratings: {df_final['rating'].nunique()}")
print(f"- Most Common Rating: {df_final['rating'].mode()[0]}")
print()

# Save the final dataset
output_file = os.path.join(DATA_PROCESSED, 'cleaned_data_step1.csv')
df_final.to_csv(output_file, index=False, encoding='utf-8')
print(f"[+] Final dataset saved as: {output_file}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("STEP 1 COMPLETION SUMMARY")
print("="*80)
print(f"""
Data Collection has been successfully completed for the Sentiment Analysis project.

Key Achievements:
- Loaded dataset with {df.shape[0]:,} records and {df.shape[1]} columns
- Identified and selected relevant columns for sentiment analysis
- Removed records with missing values ({rows_dropped:,} rows dropped)
- Final dataset contains {df_final.shape[0]:,} clean records
- Dataset saved as '{output_file}' for further processing

Next Steps:
The collected data is now ready for Step 2: Data Preprocessing and Cleaning.
""")
print("="*80)
