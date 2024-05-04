# Text Classification Using Neural Networks

This repository contains a Python script for classifying text data using deep learning models in TensorFlow and Keras. The script utilizes natural language processing techniques, such as stemming and stop-word removal, and applies TF-IDF vectorization to prepare text data for classification.

## Project Overview

The script performs the following operations:
1. **Pre-processing**: Text data is stemmed and cleaned of stop words to reduce noise and dimensionality.
2. **Vectorization**: The cleaned text is converted into a numeric form using TF-IDF vectorization.
3. **Model Training**: A neural network model is trained using the vectorized text data.
4. **Evaluation**: The model is evaluated using the F1 score, precision, and recall metrics.
5. **Execution Time**: The script tracks the total execution time.

## Prerequisites

To run this script, you need the following libraries installed:
- pandas
- numpy
- scikit-learn
- nltk
- TensorFlow
- Keras

You can install these packages via pip:
```bash
pip install pandas numpy scikit-learn nltk tensorflow keras
