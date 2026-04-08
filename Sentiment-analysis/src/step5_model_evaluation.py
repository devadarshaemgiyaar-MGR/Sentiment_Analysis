"""
Sentiment Analysis on Product Reviews
Step 5: Model Testing & Performance Evaluation
Author: Data Science Project
Date: February 2026
"""

import pandas as pd
import numpy as np
import sys
import os
import pickle
from scipy.sparse import csr_matrix

# Machine Learning evaluation libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# Set output encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

print("="*80)
print("STEP 5: MODEL TESTING & PERFORMANCE EVALUATION")
print("="*80)
print()

# ============================================================================
# LOAD TRAINED MODEL FROM STEP 4
# ============================================================================
print("Loading trained model from Step 4...")
print("-"*80)

model_file = os.path.join(MODELS_DIR, 'sentiment_classifier_nb.pkl')
with open(model_file, 'rb') as f:
    model = pickle.load(f)

print(f"[+] Model loaded: {type(model).__name__}")
print(f"    - Model type: Multinomial Naive Bayes")
print(f"    - Number of classes: {len(model.classes_)}")
print(f"    - Classes: {model.classes_}")
print()

# ============================================================================
# LOAD TEST DATASET FROM STEP 4
# ============================================================================
print("Loading test dataset from Step 4...")
print("-"*80)

split_file = os.path.join(DATA_PROCESSED, 'train_test_split.npz')
data = np.load(split_file, allow_pickle=True)

# Reconstruct test sparse matrix
X_test = csr_matrix((data['X_test_data'], data['X_test_indices'], 
                     data['X_test_indptr']), shape=data['X_test_shape'])
y_test = data['y_test']

print(f"[+] Test data loaded successfully")
print(f"    - X_test shape: {X_test.shape}")
print(f"    - y_test shape: {y_test.shape}")
print(f"    - Test samples: {X_test.shape[0]:,}")
print()

print("Test set class distribution:")
unique_test, counts_test = np.unique(y_test, return_counts=True)
for label, count in zip(unique_test, counts_test):
    print(f"  {label.capitalize():10s}: {count:6,} samples ({count/len(y_test)*100:5.2f}%)")
print()

# ============================================================================
# STEP 5.1: MAKE PREDICTIONS ON TEST DATASET
# ============================================================================
print("STEP 5.1: Make Predictions on Test Dataset")
print("-"*80)

print(f"Predicting sentiment labels for {X_test.shape[0]:,} test samples...")
print()

# Predict sentiment labels
y_pred = model.predict(X_test)

print(f"[+] Predictions completed!")
print(f"    - Predictions shape: {y_pred.shape}")
print(f"    - Unique predicted classes: {np.unique(y_pred)}")
print()

print("Sample predictions (first 10):")
print("  Actual   | Predicted")
print("  " + "-"*25)
for i in range(min(10, len(y_test))):
    match = "✓" if y_test[i] == y_pred[i] else "✗"
    print(f"  {y_test[i]:8s} | {y_pred[i]:9s}  {match}")
print()

print("Predicted class distribution:")
unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
for label, count in zip(unique_pred, counts_pred):
    print(f"  {label.capitalize():10s}: {count:6,} predictions ({count/len(y_pred)*100:5.2f}%)")
print()

# ============================================================================
# STEP 5.2: CALCULATE EVALUATION METRICS
# ============================================================================
print("STEP 5.2: Calculate Evaluation Metrics")
print("-"*80)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"1. Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print()

print("   What is Accuracy?")
print("   -> Percentage of correct predictions out of total predictions")
print(f"   -> Formula: (Correct Predictions) / (Total Predictions)")
print(f"   -> Interpretation: Model correctly predicted {accuracy*100:.2f}% of test samples")
print()

# Calculate Precision (weighted average for multi-class)
precision = precision_score(y_test, y_pred, average='weighted')
print(f"2. Precision (weighted): {precision:.4f} ({precision*100:.2f}%)")
print()

print("   What is Precision?")
print("   -> Of all samples predicted as a class, how many were actually that class?")
print("   -> Formula: True Positives / (True Positives + False Positives)")
print("   -> Interpretation: When model predicts a sentiment, it's correct {:.2f}% of the time".format(precision*100))
print()

# Calculate Recall (weighted average for multi-class)
recall = recall_score(y_test, y_pred, average='weighted')
print(f"3. Recall (weighted): {recall:.4f} ({recall*100:.2f}%)")
print()

print("   What is Recall?")
print("   -> Of all samples that are actually a class, how many did we correctly identify?")
print("   -> Formula: True Positives / (True Positives + False Negatives)")
print("   -> Interpretation: Model finds {:.2f}% of all actual instances of each sentiment".format(recall*100))
print()

# Calculate F1-Score (weighted average for multi-class)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"4. F1-Score (weighted): {f1:.4f} ({f1*100:.2f}%)")
print()

print("   What is F1-Score?")
print("   -> Harmonic mean of Precision and Recall (balanced measure)")
print("   -> Formula: 2 × (Precision × Recall) / (Precision + Recall)")
print("   -> Interpretation: Overall model quality considering both precision and recall")
print("   -> Range: 0 (worst) to 1 (best)")
print()

# ============================================================================
# DETAILED METRICS PER CLASS
# ============================================================================
print("Detailed Metrics Per Class:")
print("-"*80)

# Calculate per-class metrics
precision_per_class = precision_score(y_test, y_pred, average=None, labels=model.classes_)
recall_per_class = recall_score(y_test, y_pred, average=None, labels=model.classes_)
f1_per_class = f1_score(y_test, y_pred, average=None, labels=model.classes_)

print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 50)
for i, class_name in enumerate(model.classes_):
    print(f"{class_name.capitalize():<12} {precision_per_class[i]:<12.4f} "
          f"{recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f}")
print()

# ============================================================================
# STEP 5.3: GENERATE AND DISPLAY CONFUSION MATRIX
# ============================================================================
print("STEP 5.3: Generate and Display Confusion Matrix")
print("-"*80)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

print("Confusion Matrix:")
print()
print("   " + "Predicted →")
print("   " + " " * 15 + "  ".join([f"{c[:8]:>8s}" for c in model.classes_]))
print("Actual ↓")

for i, class_name in enumerate(model.classes_):
    row_label = f"{class_name.capitalize():<12}"
    row_values = "  ".join([f"{cm[i][j]:>8,}" for j in range(len(model.classes_))])
    print(f"{row_label} {row_values}")
print()

print("How to read the Confusion Matrix:")
print("  - Rows: Actual sentiment labels (ground truth)")
print("  - Columns: Predicted sentiment labels (model output)")
print("  - Diagonal (top-left to bottom-right): Correct predictions")
print("  - Off-diagonal: Misclassifications")
print()

# Analyze confusion matrix
print("Confusion Matrix Analysis:")
print("-"*80)

total_samples = cm.sum()
correct_predictions = np.trace(cm)  # Sum of diagonal
incorrect_predictions = total_samples - correct_predictions

print(f"Total test samples: {total_samples:,}")
print(f"Correct predictions: {correct_predictions:,} ({correct_predictions/total_samples*100:.2f}%)")
print(f"Incorrect predictions: {incorrect_predictions:,} ({incorrect_predictions/total_samples*100:.2f}%)")
print()

# Per-class accuracy
print("Per-class analysis:")
for i, class_name in enumerate(model.classes_):
    class_total = cm[i].sum()
    class_correct = cm[i][i]
    class_accuracy = class_correct / class_total if class_total > 0 else 0
    
    print(f"\n{class_name.capitalize()}:")
    print(f"  - Total actual samples: {class_total:,}")
    print(f"  - Correctly predicted: {class_correct:,} ({class_accuracy*100:.2f}%)")
    print(f"  - Misclassified: {class_total - class_correct:,} ({(1-class_accuracy)*100:.2f}%)")
    
    # Show main misclassifications
    if class_total > 0:
        for j, pred_class in enumerate(model.classes_):
            if i != j and cm[i][j] > 0:
                print(f"    • Predicted as {pred_class}: {cm[i][j]:,} samples")

print()

# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================
print("Complete Classification Report:")
print("-"*80)

report = classification_report(y_test, y_pred, target_names=model.classes_, digits=4)
print(report)
print()

# ============================================================================
# STEP 5.4: INTERPRETATION OF RESULTS
# ============================================================================
print("="*80)
print("STEP 5.4: Interpretation of Evaluation Results")
print("="*80)
print()

print("Model Performance Summary:")
print("-"*80)
print(f"""
Overall Metrics:
• Accuracy:  {accuracy*100:.2f}% - Model is correct {accuracy*100:.2f}% of the time
• Precision: {precision*100:.2f}% - When model makes a prediction, it's right {precision*100:.2f}% of the time
• Recall:    {recall*100:.2f}% - Model identifies {recall*100:.2f}% of all actual instances
• F1-Score:  {f1*100:.2f}% - Balanced measure of overall model quality
""")

print("Performance Analysis:")
print("-"*80)

# Determine performance level
if accuracy >= 0.90:
    performance = "Excellent"
    interpretation = "exceptional performance"
elif accuracy >= 0.80:
    performance = "Good"
    interpretation = "strong performance"
elif accuracy >= 0.70:
    performance = "Acceptable"
    interpretation = "reasonable performance"
else:
    performance = "Needs Improvement"
    interpretation = "suboptimal performance"

print(f"• Overall Performance: {performance}")
print(f"• The model shows {interpretation} on the test dataset.")
print()

# Class-specific insights
print("Class-Specific Insights:")
best_class_idx = np.argmax(f1_per_class)
worst_class_idx = np.argmin(f1_per_class)

print(f"• Best performing class: {model.classes_[best_class_idx].capitalize()} "
      f"(F1={f1_per_class[best_class_idx]:.4f})")
print(f"• Weakest performing class: {model.classes_[worst_class_idx].capitalize()} "
      f"(F1={f1_per_class[worst_class_idx]:.4f})")
print()

# Common misclassifications
print("Common Misclassification Patterns:")
max_misclass = 0
misclass_pair = (None, None)
for i in range(len(model.classes_)):
    for j in range(len(model.classes_)):
        if i != j and cm[i][j] > max_misclass:
            max_misclass = cm[i][j]
            misclass_pair = (model.classes_[i], model.classes_[j])

if misclass_pair[0] is not None:
    print(f"• Most common confusion: {misclass_pair[0].capitalize()} "
          f"→ {misclass_pair[1].capitalize()} ({max_misclass:,} cases)")
print()

# Model suitability
print("Model Suitability for Deployment:")
print("-"*80)
if accuracy >= 0.75:
    print("✓ Model performance is suitable for deployment")
    print("✓ Accuracy exceeds typical threshold for production use")
    if accuracy >= 0.85:
        print("✓ High confidence in model predictions")
else:
    print("⚠ Model may need improvement before deployment")
    print("  Consider: feature engineering, hyperparameter tuning, or different algorithms")
print()

# ============================================================================
# SAVE EVALUATION RESULTS
# ============================================================================
print("Saving evaluation results...")
print("-"*80)

# Save metrics to file
results = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'confusion_matrix': cm.tolist(),
    'classification_report': report,
    'per_class_precision': precision_per_class.tolist(),
    'per_class_recall': recall_per_class.tolist(),
    'per_class_f1': f1_per_class.tolist()
}

results_file = os.path.join(DATA_PROCESSED, 'evaluation_results.pkl')
with open(results_file, 'wb') as f:
    pickle.dump(results, f)

print(f"[+] Evaluation results saved: {results_file}")
print()

# ============================================================================
# STEP 5 COMPLETION SUMMARY
# ============================================================================
print("="*80)
print("STEP 5 COMPLETION SUMMARY")
print("="*80)
print(f"""
Model Testing & Performance Evaluation completed successfully!

All required steps performed:
[+] 1. Loaded trained model and test dataset
[+] 2. Made predictions on {X_test.shape[0]:,} test samples
[+] 3. Calculated evaluation metrics:
       - Accuracy: {accuracy*100:.2f}%
       - Precision: {precision*100:.2f}%
       - Recall: {recall*100:.2f}%
       - F1-Score: {f1*100:.2f}%
[+] 4. Generated and analyzed confusion matrix
[+] 5. Provided detailed interpretation of results

Model Performance: {performance}
• The Multinomial Naive Bayes classifier achieved {accuracy*100:.2f}% accuracy
• The model demonstrates {interpretation.lower()} for sentiment classification
• Evaluation results saved for future reference

Test Dataset:
• Total samples tested: {X_test.shape[0]:,}
• Correct predictions: {correct_predictions:,} ({correct_predictions/total_samples*100:.2f}%)
• Incorrect predictions: {incorrect_predictions:,} ({incorrect_predictions/total_samples*100:.2f}%)

Next Steps (NOT performed as per instructions):
• Hyperparameter tuning
• Model comparison
• Cross-validation
• Deployment preparation
""")
print("="*80)
print()

print("✓ STEP 5: MODEL TESTING & PERFORMANCE EVALUATION - SUCCESSFULLY COMPLETED")
print()
