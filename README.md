# MediVoice-AI
A voice-driven medical symptom analysis system that converts speech into structured data and predicts possible diseases using rule-based scoring and similarity matching.
# 🎤 Voice-Based Medical Diagnosis System

A Python-based intelligent system that analyzes medical symptoms from speech input and predicts possible diseases using a structured dataset.

## 🚀 Features
- 🎤 Speech-to-text symptom input
- 🧠 NLP-based symptom extraction and mapping
- 📊 Dataset-driven disease prediction (CSV-based)
- ⚖️ Hybrid approach using rule-based scoring and cosine similarity
- 🔊 Text-to-speech response for interactive feedback
- ⚡ Real-time analysis without model training

## 🧠 How It Works
1. Captures user speech input
2. Converts speech to text using speech recognition
3. Extracts symptoms using a custom NLP mapping system
4. Matches symptoms with a medical dataset
5. Ranks possible diseases using:
   - Match scoring
   - Similarity analysis
6. Outputs results via text and voice

## 📂 Tech Stack
- Python
- SpeechRecognition
- pyttsx3
- spaCy
- pandas, numpy
- scikit-learn

## ⚠️ Disclaimer
This project is for educational purposes only and should not be used as a substitute for professional medical advice.

## 🌟 Novelty
This system introduces a dynamic translation layer that converts unstructured speech into structured medical features and performs real-time diagnosis without requiring a trained machine learning model.
