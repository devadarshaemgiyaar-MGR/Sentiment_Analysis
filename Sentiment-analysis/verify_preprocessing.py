import pandas as pd

# Load the preprocessed data
df = pd.read_csv('preprocessed_data_step2.csv')

print("="*80)
print("PREPROCESSED DATA VERIFICATION")
print("="*80)
print()

print("Dataset Shape:", df.shape)
print("Columns:", list(df.columns))
print()

print("First 15 rows with comparison:")
print("-"*80)
for i in range(min(15, len(df))):
    print(f"{i+1}. Rating: {df['rating'].iloc[i]}")
    print(f"   Original: \"{df['review_text'].iloc[i]}\"")
    print(f"   Cleaned:  \"{df['cleaned_text'].iloc[i]}\"")
    print()

print("="*80)
print("More Complex Examples (longer reviews):")
print("-"*80)

# Find longer reviews for better demonstration
longer_reviews = df[df['review_text'].str.len() > 20].head(10)
for idx, row in longer_reviews.iterrows():
    print(f"Original: \"{row['review_text']}\"")
    print(f"Cleaned:  \"{row['cleaned_text']}\"")
    print()
