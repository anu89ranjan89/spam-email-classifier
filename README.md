# Spam Email Classifier

## Demo
https://spam-email-classifier-m58jhn64fj7x2wjqqe85ov.streamlit.app/

## Overview

This project is a Machine Learning-based Spam Email Classifier that predicts whether a message is Spam or Ham (Not Spam).

The model is trained using the SMS Spam Collection Dataset and uses TF-IDF Vectorization with the Multinomial Naive Bayes algorithm.

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* TF-IDF Vectorizer
* Multinomial Naive Bayes

## Streamlit Web Interface

The project also includes a Streamlit-based web application that allows users to enter messages and receive spam/ham predictions through a user-friendly interface.

## Workflow

1. Load and preprocess the dataset
2. Convert text into numerical features using TF-IDF
3. Split the dataset into training and testing sets
4. Train a Multinomial Naive Bayes classifier
5. Evaluate model performance
6. Predict spam/ham for new messages

## Features

* Spam Detection
* Ham Detection
* Model Persistence using Pickle
* Real-time Message Prediction

## Sample Predictions

Input:
Congratulations! You have won ₹50,000.

Output:
SPAM

Input:
Hey, can we meet tomorrow?

Output:
HAM

## Future Improvements

* Streamlit Web Interface
* Email Integration
* Deep Learning-based Classification
* Multi-language Support
