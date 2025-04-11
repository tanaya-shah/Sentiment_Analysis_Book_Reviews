# Sentiment_Analysis_Book_Reviews

## Project Overview

This project involved building a deep learning model to classify sentiment in Amazon Kindle book reviews. We developed a custom Transformer Encoder model and compared it with baseline models including logistic regression and pre-trained BERT. The goal was to accurately predict whether a review expressed positive or negative sentiment, enabling scalable customer feedback analysis.

## Tasks Completed

### 1. Data Preparation
Used a Kaggle dataset with 12,000 Kindle reviews.
Kept only relevant fields (reviewText, rating).
Encoded reviews as binary sentiment (1 = positive, 0 = negative).
Preprocessed text (lowercasing, punctuation removal, tokenization).
Padded/truncated sequences for uniform input size.

### 2. Modeling
Built a custom Transformer Encoder with:
Token + Positional embeddings
3 Transformer layers (2 attention heads)
Dropout and LayerNorm
Sigmoid output layer for binary classification
Tuned hyperparameters (learning rate, heads, depth, batch size).
Compared against:
Logistic Regression
Pre-trained BERT model

### 3. Attention Visualization
Extracted and averaged final-layer attention weights to analyze word importance.
Noted limitations in interpretability due to context-dependence and frequent function words.

## Results and Evaluation

Model	Accuracy	Loss
Transformer Encoder	83.19%	0.389
Logistic Regression	61.74%	36.7941
Pre-trained BERT	88.33%	0.4700
Custom Transformer was significantly better than logistic regression and nearly competitive with BERT, while being more computationally efficient.
Training included 64 hyperparameter combinations to find the optimal configuration.

## Deployment Ideas

Serve as an API for real-time sentiment scoring.
Embed in customer review dashboards for actionable feedback.

## Technologies Used

Python, PyTorch  
Transformers, BERT  
Grid Search for hyperparameter tuning  
Matplotlib/Seaborn for visualizations
