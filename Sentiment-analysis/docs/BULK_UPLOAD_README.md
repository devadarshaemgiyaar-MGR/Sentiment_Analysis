# Bulk Review Upload – Implementation Documentation

## 1. Overview
The Bulk Review Upload feature is designed to process large datasets of product reviews efficiently. It allows users to upload a CSV file containing hundreds or thousands of reviews and receive a comprehensive sentiment analysis dashboard in seconds. This module bridges the gap between individual manual testing and large-scale industrial data analysis.

## 2. System Architecture Flow
```text
[ CSV FILE UPLOAD ]
        |
        v
[ VALIDATION LOGIC ] (Check column: 'review_text')
        |
        v
[ BATCH PREPROCESSING ] (Lowercasing, Tokenization, Lemmatization)
        |
        v
[ BATCH TF-IDF TRANSFORMATION ] (Feature Extraction)
        |
        v
[ MODEL PREDICTION ] (Naive Bayes Classifier)
        |
        v
[ DASHBOARD GENERATION ] (Metric Metrics, Charts, Tables)
        |
        v
[ EXPORT ENGINE ] (CSV & PDF Report Generation)
```

## 3. Implementation Steps
1.  **File Ingestion**: Utilize Streamlit's file uploader to accept `.csv` files.
2.  **Schema Verification**: Parse the data frame to ensure the mandatory 'review_text' column exists.
3.  **Preprocessing Pipeline**: Apply the standard NLP cleaning steps to the entire column using optimized lambda functions.
4.  **Vectorization**: Pass the cleaned text through the pre-trained TF-IDF vectorizer.
5.  **Classification**: Use the pre-trained Naive Bayes model to predict sentiment (Positive, Neutral, Negative).
6.  **Visualization**: Compute counts and distribution to render Matplotlib charts and summary metrics.

## 4. CSV File Structure Requirement
To ensure successful processing, the uploaded file must meet one primary requirement:
-   **Mandatory Column**: `review_text`
-   **Format**: Plain text format within the column.
-   **Encoding**: UTF-8 recommended.

## 5. Validation Logic
The system implements strict validation to prevent application crashes:
-   **Error Handling**: If the 'review_text' column is missing, an error message is displayed, and processing is halted.
-   **Empty State Management**: The system ignores empty rows or null values within the CSV to maintain data integrity in the final summary.

## 6. Batch Processing Logic
Unlike single-review mode, this module uses **Vectorized Batch Transformation**. Instead of looping through each row to predict sentiment, the system transforms the entire dataset into a sparse matrix in one operation and predicts the labels for all rows simultaneously. This minimizes computational overhead.

## 7. Performance Optimization Strategy
-   **Caching**: The ML model and TF-IDF vectorizer are stored in memory (`st.cache_resource`) to avoid reloading for every file upload.
-   **Vectorization over Loops**: Using Pandas `apply` and Scikit-learn's batch `transform` ensures the system remains responsive even with large files.

## 8. Output Generated
-   **Metric Cards**: High-level counts for Positive, Neutral, and Negative sentiments.
-   **Sentiment Proportions**: A Pie chart showing the percentage distribution.
-   **Comparison Chart**: A Bar chart for head-to-head volume comparison.
-   **Data Table**: A searchable and filterable table showing every review alongside its predicted sentiment.
-   **Downloadable Assets**: Final results available in CSV and professional PDF formats.

## 9. Design Objective
The core objective is **Scalability**. The interface is designed to make data-driven decision-making accessible to non-technical users by abstracting complex ML pipelines into a simple drag-and-drop experience.

## 10. Future Improvements
-   **Auto-Detection**: Implementing fuzzy logic to detect columns even if they aren't named exactly 'review_text' (e.g., 'Reviews', 'Comments').
-   **Asynchronous Processing**: Moving the heavy computation to background workers for extremely large datasets (100k+ rows).
