# Audio Review Feature – Implementation Documentation

## 1. Overview
The Audio Review Feature allows users to capture product reviews via voice. This module uses advanced signal processing to transcribe audio and an NLP-based segmentation engine to handle multi-review recordings. This solves the challenge of transcribing long audio files which are typically truncated by standard speech-to-text APIs.

## 2. Complete Architecture Flow
```text
[ AUDIO FILE (.WAV/.MP3) ]
          |
          v
[ AUDIO NORMALIZATION ] (Mono, 16kHz, 16-bit PCM)
          |
          v
[ IN-MEMORY CHUNKING ] (15-second segments)
          |
          v
[ BATCH TRANSCRIPTION ] (Speech-to-Text via Google API)
          |
          v
[ TRANSCRIPT MERGING ] (Normalization & Cleanup)
          |
          v
[ KEYWORD SEGMENTATION ] (Regex-based "NEXT REVIEW" splitting)
          |
          v
[ SENTIMENT ANALYSIS ] (Batch TF-IDF + Classifier)
```

## 3. Audio Normalization Steps
To maximize transcription accuracy, every uploaded file undergoes a normalization pass:
-   **Channel Conversion**: Audio is converted to Mono to remove spatial noise.
-   **Resampling**: The frame rate is standardized to 16,000 Hz, the optimal input for the Speech Recognition engine.
-   **Bit Depth**: Standardized to 16-bit PCM for consistent signal interpretation.
-   **No Disk Latency**: All transformations are performed in-memory using byte buffers.

## 4. Chunk-Based Transcription Logic
Standard transcription services often truncate long audio (40-60 seconds). To solve this:
-   The system splits the normalized audio into fixed **15-second chunks**.
-   Each chunk is transcribed independently.
-   **Fault Tolerance**: If one chunk fails due to silence or noise, the system continues processing the remaining chunks instead of crashing.
-   Finally, the system merges all successful chunk transcripts into a single coherent text block.

## 5. Keyword Segmentation using "NEXT REVIEW"
The feature supports multi-review audio sessions where the speaker says **"NEXT REVIEW"** between different product feedback.
-   **Regex Processing**: A regular expression engine searches for the trigger phrase regardless of capitalization or extra spaces.
-   **Automatic Splitting**: The software segments the transcript into independent reviews.
-   **Fallback**: If the keyword is not found, the system treats the entire audio as one single, continuous review.

## 6. Negation Handling Improvement
Transcription sometimes misses subtle linguistic cues. To improve sentiment accuracy:
-   The preprocessing logic was adjusted to protect **negation keywords** (no, not, never, n't) from being removed by the stopword filter.
-   This ensures that phrases like "not happy" are correctly analyzed as Negative rather than "happy" (Positive).

## 7. Error Handling Strategy
-   **Ambient Noise**: The system uses a 0.4s calibration window for every chunk to adjust for background hum.
-   **API Safeguards**: Handles `UnknownValueError` (silence) and `RequestError` (network issues) gracefully without losing progress on previously transcribed chunks.

## 8. Performance Optimization
-   **Parallel Logic**: Chunks are processed sequentially but visualized with real-time progress bars.
-   **Batch Vectorization**: Once the transcript is segmented, all segments are vectorized and predicted in a single batch call to the ML model.

## 9. Output Generated
-   **Visual Transcript**: A collapsible view of the raw recognized text.
-   **Segmented Analytics**: Individual sentiment and confidence scores for every detected review in the audio.
-   **Session Dashboard**: Cumulative charts for the entire audio session.

## 10. Design Objective
The goal was to create a **"Hands-Free"** review analysis system that handles long-form verbal feedback without the technical limitations of standard transcription APIs.

## 11. Future Scope
-   **Keyword Customization**: Allowing users to define their own split markers.
-   **Speaker Diarization**: Detecting different speakers if multiple people are providing reviews in one audio file.
