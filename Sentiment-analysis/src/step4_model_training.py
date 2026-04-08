"""
Sentiment Analysis on Product Reviews
Step 4: Model Training
Author: Data Science Project
Date: February 2026
"""

import pandas as pd
import numpy as np
import sys
import os
import pickle
from scipy.sparse import csr_matrix

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Set output encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)

print("="*80)
print("STEP 4: MODEL TRAINING")
print("="*80)
print()

# ============================================================================
# LOAD FEATURE MATRIX AND TARGET VARIABLE FROM STEP 3
# ============================================================================
print("Loading feature matrix and target variable from Step 3...")
print("-"*80)

# Load feature matrix (X)
feature_matrix_file = os.path.join(DATA_PROCESSED, 'feature_matrix_X.npz')
X_data = np.load(feature_matrix_file)
X = csr_matrix((X_data['data'], X_data['indices'], X_data['indptr']), 
               shape=X_data['shape'])

print(f"[+] Feature matrix loaded: {X.shape}")
print(f"    - Samples: {X.shape[0]:,}")
print(f"    - Features: {X.shape[1]:,}")
print()

# Load target variable (y)
target_file = os.path.join(DATA_PROCESSED, 'target_variable_y.npy')
y = np.load(target_file, allow_pickle=True)

print(f"[+] Target variable loaded: {y.shape}")
print(f"    - Total labels: {len(y):,}")
print()

print("Sample data verification:")
print(f"  X type: {type(X)}")
print(f"  X shape: {X.shape}")
print(f"  y type: {type(y)}")
print(f"  y shape: {y.shape}")
print(f"  Unique classes: {np.unique(y)}")
print()

# ============================================================================
# STEP 4.1: SPLIT DATASET INTO TRAINING AND TESTING SETS
# ============================================================================
print("STEP 4.1: Split Dataset into Training and Testing Sets")
print("-"*80)

# Split: 80% training, 20% testing
# Use stratify to maintain class distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% for testing
    random_state=42,      # For reproducibility
    stratify=y            # Maintain class distribution
)

print("Dataset split configuration:")
print(f"  Train-Test Split Ratio: 80% - 20%")
print(f"  Random State: 42 (for reproducibility)")
print(f"  Stratified: Yes (maintains class distribution)")
print()

print("Training Set:")
print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  Training samples: {X_train.shape[0]:,} ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
print()

print("Testing Set:")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_test shape: {y_test.shape}")
print(f"  Testing samples: {X_test.shape[0]:,} ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
print()

# Display class distribution in training and testing sets
print("Class distribution in Training Set:")
unique_train, counts_train = np.unique(y_train, return_counts=True)
for label, count in zip(unique_train, counts_train):
    print(f"  {label.capitalize():10s}: {count:6,} samples ({count/len(y_train)*100:5.2f}%)")
print()

print("Class distribution in Testing Set:")
unique_test, counts_test = np.unique(y_test, return_counts=True)
for label, count in zip(unique_test, counts_test):
    print(f"  {label.capitalize():10s}: {count:6,} samples ({count/len(y_test)*100:5.2f}%)")
print()

print("Why this split?")
print("-> 80-20 split is standard for medium-sized datasets")
print("-> Stratified split maintains class distribution (important for imbalanced data)")
print("-> Random state ensures reproducible results")
print()

# ============================================================================
# STEP 4.2: SELECT MACHINE LEARNING CLASSIFIER
# ============================================================================
print("STEP 4.2: Select Machine Learning Classifier")
print("-"*80)

print("Selected Algorithm: Multinomial Naive Bayes")
print()

print("What is Multinomial Naive Bayes?")
print("-"*80)
print("""
Multinomial Naive Bayes is a probabilistic classifier based on Bayes' Theorem.
It's specifically designed for discrete features like word counts or TF-IDF scores.

Key Characteristics:
1. Based on Bayes' Theorem with "naive" independence assumption
2. Assumes features (words) are independent given the class
3. Works with discrete/count data (perfect for text)
4. Calculates probability: P(class|features) = P(features|class) × P(class) / P(features)

Formula:
  P(sentiment|review) = P(words|sentiment) × P(sentiment) / P(words)

The classifier predicts the sentiment with the highest probability.
""")
print()

print("Why Multinomial Naive Bayes for Text Classification?")
print("-"*80)
print("""
Advantages:
1. FAST: Training and prediction are very fast, even on large datasets
2. EFFICIENT: Works well with sparse matrices (our TF-IDF features)
3. PROVEN: Excellent performance on text classification tasks
4. SIMPLE: Easy to understand and interpret
5. ROBUST: Handles high-dimensional data well (we have 868 features)
6. BASELINE: Industry standard baseline for text classification

Best Suited For:
- Text classification (sentiment analysis, spam detection)
- Document categorization
- TF-IDF or count-based features
- When dataset is not extremely large (our 144K samples are perfect)

Limitations (acknowledged):
- Assumes feature independence (not always true, but works well in practice)
- Sensitive to feature scaling (not an issue with TF-IDF)
""")
print()

# ============================================================================
# STEP 4.3: INITIALIZE AND TRAIN THE CLASSIFIER
# ============================================================================
print("STEP 4.3: Train the Classifier")
print("-"*80)

print("Initializing Multinomial Naive Bayes classifier...")
print()

# Initialize the classifier
classifier = MultinomialNB(alpha=1.0)

print("Classifier Parameters:")
print(f"  Algorithm: Multinomial Naive Bayes")
print(f"  Alpha (smoothing parameter): {classifier.alpha}")
print(f"    -> Alpha=1.0 means Laplace smoothing (handles zero probabilities)")
print()

print("Training the model...")
print(f"  Training on {X_train.shape[0]:,} samples with {X_train.shape[1]:,} features")
print()

# Train the classifier
classifier.fit(X_train, y_train)

print("[+] Model training completed successfully!")
print()

# ============================================================================
# STEP 4.4: MODEL TRAINING CONFIRMATION
# ============================================================================
print("STEP 4.4: Training Confirmation")
print("-"*80)

print("Model training details:")
print(f"  Model Type: {type(classifier).__name__}")
print(f"  Number of classes: {len(classifier.classes_)}")
print(f"  Classes: {classifier.classes_}")
print(f"  Number of features: {classifier.n_features_in_}")
print(f"  Training samples used: {X_train.shape[0]:,}")
print()

print("Model is now trained and ready for:")
print("  - Making predictions on new data")
print("  - Evaluation on test set (Step 5)")
print("  - Deployment for real-world use")
print()

# ============================================================================
# STEP 4.5: SAVE THE TRAINED MODEL
# ============================================================================
print("STEP 4.5: Save Trained Model")
print("-"*80)

# Save the trained model
model_file = os.path.join(MODELS_DIR, 'sentiment_classifier_nb.pkl')
with open(model_file, 'wb') as f:
    pickle.dump(classifier, f)

print(f"[+] Trained model saved: {model_file}")
print()

# Also save train-test split for evaluation in Step 5
split_data_file = os.path.join(DATA_PROCESSED, 'train_test_split.npz')
np.savez_compressed(split_data_file,
                    X_train_data=X_train.data,
                    X_train_indices=X_train.indices,
                    X_train_indptr=X_train.indptr,
                    X_train_shape=X_train.shape,
                    X_test_data=X_test.data,
                    X_test_indices=X_test.indices,
                    X_test_indptr=X_test.indptr,
                    X_test_shape=X_test.shape,
                    y_train=y_train,
                    y_test=y_test)

print(f"[+] Train-test split data saved: {split_data_file}")
print()

print("Saved files:")
print(f"  1. {model_file}")
print(f"     -> Trained Multinomial Naive Bayes classifier")
print(f"     -> Can be loaded with pickle for predictions")
print()
print(f"  2. {split_data_file}")
print(f"     -> Training and testing sets")
print(f"     -> For model evaluation in next steps")
print()

# ============================================================================
# HOW TO LOAD AND USE THE TRAINED MODEL
# ============================================================================
print("How to load and use the trained model:")
print("-"*80)
print("""
import pickle
import numpy as np
from scipy.sparse import csr_matrix

# Load the trained model
with open('models/sentiment_classifier_nb.pkl', 'rb') as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Example: Predict sentiment for new review
new_review = "This product is amazing!"
new_features = vectorizer.transform([new_review])
prediction = model.predict(new_features)
print(f"Predicted sentiment: {prediction[0]}")

# Get probability scores
probabilities = model.predict_proba(new_features)
print(f"Confidence scores: {probabilities[0]}")
""")
print()

# ============================================================================
# STEP 4 COMPLETION SUMMARY
# ============================================================================
print("="*80)
print("STEP 4 COMPLETION SUMMARY")
print("="*80)
print(f"""
Model Training has been successfully completed!

All required steps performed:
[+] 1. Loaded feature matrix (X) and target variable (y) from Step 3
[+] 2. Split dataset: 80% training ({X_train.shape[0]:,} samples), 20% testing ({X_test.shape[0]:,} samples)
[+] 3. Selected Multinomial Naive Bayes classifier (optimal for text/TF-IDF)
[+] 4. Trained the classifier on {X_train.shape[0]:,} training samples
[+] 5. Saved trained model to: {model_file}

Model Specifications:
- Algorithm: Multinomial Naive Bayes
- Training samples: {X_train.shape[0]:,} reviews
- Features: {X_train.shape[1]:,} TF-IDF features
- Classes: {len(classifier.classes_)} sentiments ({', '.join(classifier.classes_)})
- Smoothing parameter (alpha): {classifier.alpha}

Dataset Split:
- Training: {X_train.shape[0]:,} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)
- Testing: {X_test.shape[0]:,} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)
- Stratified: Yes (maintains class distribution)

Saved Outputs:
- Trained model: models/sentiment_classifier_nb.pkl
- Train-test split: data/processed/train_test_split.npz

The trained model is ready for evaluation and deployment!
Next Step: Step 5 - Model Evaluation (NOT performed as per instructions)
""")
print("="*80)
print()

print("✓ STEP 4: MODEL TRAINING - SUCCESSFULLY COMPLETED")
print()
