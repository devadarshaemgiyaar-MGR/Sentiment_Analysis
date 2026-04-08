# Sentiment Analysis System – Project Overview & Viva Guide

## 1. Project Overview
This project is an end-to-end **Sentiment Analysis System** designed for product reviews. It integrates traditional Natural Language Processing (NLP) with Machine Learning to provide instant insights from text and audio data. The application is built using a modern, performance-optimized Streamlit interface.

## 2. Objectives
-   To automate the classification of product reviews into Positive, Neutral, and Negative categories.
-   To handle multi-modal inputs (Text and Audio).
-   To provide scalable batch processing for enterprise datasets.
-   To deliver professional analytics through dynamic visualizations and PDF reports.

## 3. Technologies Used
-   **Frontend/UI**: Streamlit (Python-based interactive dashboard)
-   **Machine Learning**: Scikit-Learn (Naive Bayes Classifier)
-   **Natural Language Processing**: NLTK (Tokenization, Lemmatization, Stopwords)
-   **Feature Extraction**: TF-IDF Vectorizer
-   **Audio Processing**: SpeechRecognition (Google API), Pydub (FFmpeg-powered normalization)
-   **Reports**: ReportLab (PDF Generation), Matplotlib/Seaborn (Visualization)

## 4. Machine Learning Pipeline Explanation
1.  **Data Ingestion**: Loading raw text/audio.
2.  **Preprocessing**: Cleaning text by stripping special characters, lowercasing, and lemmatizing (reducing words to their root form).
3.  **Vectorization (TF-IDF)**: Converting text into numerical values based on term frequency and importance within the corpus.
4.  **Classification**: The **Multinomial Naive Bayes** model calculates the probability of the review belonging to each sentiment category and selects the highest one.

## 5. Feature Summary
-   **Single Review Mode**: Instant analysis of typed or dictated feedback.
-   **Bulk Upload Mode**: Drop a CSV file to analyze hundreds of reviews in seconds.
-   **Audio Review Mode**: Voice-to-sentiment engine with "Next Review" keyword segmentation.
-   **Export Engine**: Professional PDF reports with metadata and charts.

## 6. Optimization Techniques Used
-   **Resource Caching**: Using `st.cache_resource` to keep models in RAM, eliminating loading latency.
-   **Vectorized Predictions**: Processing arrays of reviews instead of looping through items sequentially.
-   **In-Memory Processing**: Handling all audio normalization and chunking in RAM (io.BytesIO) to avoid slow disk I/O.

## 7. Accuracy Improvement Techniques
-   **Negation Preservation**: Custom stopword filtering that retains words like "not" and "never".
-   **Audio Normalization**: Converting sound files to 16kHz mono PCM for clearer speech recognition.
-   **Recognizer Tuning**: Adjusting for ambient noise and dynamic energy levels to improve transcription precision.

## 8. Challenges Faced & Solutions
-   **Challenge**: Long audio files being truncated by the Speech-to-Text API.
    -   **Solution**: Implemented **Chunk-Based Transcription**, splitting audio into 15-second segments for reliable processing.
-   **Challenge**: Slow processing of large CSV files.
    -   **Solution**: Moved from loop-based logic to **Batch Matrix Transformations**.
-   **Challenge**: Loss of sentiment meaning during cleaning.
    -   **Solution**: Refined the preprocessing pipeline to exclude negation words from the stopword list.

## 9. Future Scope
-   **Deep Learning**: migrating to Transformer-based models (like BERT) for higher contextual accuracy.
-   **Real-time Streaming**: Processing live microphone input.
-   **Multi-language Support**: Expanding sentiment detection to regional and international languages.

## 10. Conclusion
This Sentiment Analysis System stands as a robust, production-ready solution for modern product feedback loops. By combining high-performance computing techniques with precise NLP tuning, it provides a seamless experience for both individual users and data analysts.
