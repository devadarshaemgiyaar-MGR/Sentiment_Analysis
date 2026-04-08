# Step 1: Data Collection - Project Documentation
## Sentiment Analysis on Product Reviews

---

## Overview
This document summarizes the completion of **Step 1: Data Collection** for the Sentiment Analysis project using Flipkart product review dataset.

---

## Dataset Information

### Original Dataset
- **Filename**: `Dataset-SA.csv`
- **Total Records**: 205,052 rows
- **Total Columns**: 6 columns
- **File Size**: ~33.4 MB

### Column Structure

| # | Column Name | Data Type | Non-Null Count | Description |
|---|-------------|-----------|----------------|-------------|
| 1 | product_name | object | 205,052 | Name of the product being reviewed |
| 2 | product_price | object | 205,052 | Price of the product |
| 3 | Rate | object | 205,052 | Customer rating/score (1-5 stars) - **TARGET VARIABLE** |
| 4 | Review | object | 180,388 | Detailed review text written by customer - **PRIMARY TEXT DATA** |
| 5 | Summary | object | 205,041 | Brief summary of the review |
| 6 | Sentiment | object | 205,052 | Pre-labeled sentiment (positive/negative/neutral) |

---

## Actions Performed in Step 1

### 1. Dataset Loading ✓
- Successfully loaded the CSV file using Pandas
- UTF-8 encoding used to handle special characters
- Low memory mode enabled for efficient processing

### 2. Data Exploration ✓
- Displayed first 5 records to understand data structure
- Examined all columns and their data types
- Identified sample values from each column

### 3. Dimensional Analysis ✓
- **Total Rows**: 205,052
- **Total Columns**: 6

### 4. Column Analysis ✓
Each column was analyzed for:
- Data type
- Non-null count
- Purpose and relevance to sentiment analysis
- Sample values

### 5. Relevant Column Identification ✓
For sentiment analysis, we identified:

**Primary Columns:**
- **Review**: Contains detailed customer feedback (our main text data for analysis)
- **Rate**: Contains numerical ratings 1-5 (our sentiment indicator/target)

**Optional Columns:**
- **Summary**: Brief review summaries (can be used as supplementary text features)

### 6. Missing Value Analysis ✓

| Column Name | Missing Count | Missing Percentage |
|-------------|---------------|-------------------|
| product_name | 0 | 0.00% |
| product_price | 0 | 0.00% |
| Rate | 0 | 0.00% |
| **Review** | **24,664** | **12.03%** |
| Summary | 11 | 0.01% |
| Sentiment | 0 | 0.00% |

**Key Findings:**
- Total missing values: 24,675
- Review column has 12.03% missing data
- Rate column has no missing values (complete data)

### 7. Column Selection ✓
- Selected columns: `Review` and `Rate`
- Renamed to: `review_text` and `rating` (for standardization)
- Removed rows with missing values in selected columns

**Data Cleaning Results:**
- Rows before cleaning: 205,052
- Rows after cleaning: 180,388
- **Rows dropped: 24,664** (12.03% of original data)

### 8. Final Dataset Structure ✓

**Dimensions:**
- Rows: **180,388**
- Columns: **2** (`review_text`, `rating`)

**Review Text Statistics:**
- Average review length: 12.62 characters
- Minimum length: 2 characters
- Maximum length: 140 characters
- Median length: 12 characters

**Rating Distribution:**
| Rating | Count | Percentage |
|--------|-------|------------|
| 1 star | 18,294 | 10.14% |
| 2 stars | 5,451 | 3.02% |
| 3 stars | 14,024 | 7.77% |
| 4 stars | 36,969 | 20.49% |
| 5 stars | 105,647 | 58.57% |

**Note**: 3 anomalous entries contain product names instead of ratings (data quality issue identified)

**Most Common Rating**: 5 stars (58.57% of reviews)

---

## Output Files Generated

1. **step1_data_collection.py**
   - Complete Python script for Step 1
   - Includes all 8 data collection tasks
   - Well-commented and structured code

2. **cleaned_data_step1.csv**
   - Final cleaned dataset
   - 180,388 rows × 2 columns
   - Contains: `review_text`, `rating`
   - Ready for preprocessing and model training

---

## Code Summary

```python
# Key steps in the code:
1. Load CSV with Pandas (utf-8 encoding)
2. Display first 5 records
3. Show dataset dimensions (205,052 × 6)
4. List and explain all columns
5. Identify relevant columns (Review, Rate)
6. Check missing values (24,664 nulls in Review)
7. Select required columns and remove nulls
8. Display final structure (180,388 × 2)
```

---

## Step 1 Completion Summary for Viva/Project Review

**Data Collection** is the foundational step in any machine learning project where we gather and understand the raw data. In this step, we successfully loaded the Flipkart product review dataset containing over 200,000 customer reviews. 

We performed comprehensive exploratory analysis to understand the dataset structure, identifying 6 columns including product information, customer reviews, ratings, and pre-labeled sentiments. Through careful examination, we determined that the **Review** column (containing customer feedback text) and the **Rate** column (containing 1-5 star ratings) are most relevant for our sentiment analysis task.

We conducted thorough missing value analysis and discovered that approximately 12% of reviews had null values. To ensure data quality, we removed these incomplete records, resulting in a clean dataset of **180,388 reviews** with two essential columns: review text and rating.

The rating distribution analysis revealed that the dataset is imbalanced, with 58.57% of reviews being 5-star ratings, indicating predominantly positive sentiment. This insight is crucial for selecting appropriate machine learning techniques in later stages.

The final cleaned dataset has been saved as `cleaned_data_step1.csv` and is now ready for the next phase: **Data Preprocessing and Cleaning**, where we will perform text normalization, tokenization, and feature extraction.

**Key Takeaways:**
- Removed 24,664 rows with missing review text
- Retained 180,388 clean records (87.97% of original data)
- Identified class imbalance (majority 5-star reviews)
- Standardized column names for consistency
- Created a baseline understanding of data quality and distribution

---

## Next Steps (NOT performed in Step 1)

Step 2 onwards will include:
- Text preprocessing (lowercasing, punctuation removal)
- Tokenization and stop word removal
- Feature extraction (TF-IDF, Bag of Words, Word Embeddings)
- Train-test split
- Model training and evaluation

---

**Completion Status**: ✅ Step 1: Data Collection COMPLETE
**Date**: February 6, 2026
**Dataset Ready**: Yes - `cleaned_data_step1.csv`
