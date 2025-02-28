# SMS-Text-Classifier
This project is a machine learning-based classifier that categorizes SMS messages as either **spam** or **ham (not spam)**. The model uses Natural Language Processing (NLP) techniques to analyze and classify text messages.

## Features
- **Spam Detection** – Classifies SMS messages as spam or ham.
- **Machine Learning Model** – Uses NLP and supervised learning techniques.
- **Preprocessing Pipeline** – Tokenization, stopword removal, and vectorization.
- **Supports Multiple Algorithms** – Train using Naïve Bayes, Logistic Regression, or deep learning models.
- **Real-time Prediction** – Classifies messages instantly.
- **Dataset Support** – Compatible with datasets like the UCI SMS Spam Collection.

## Code Samples
python
# import libraries
try:
  # %tensorflow_version only exists in Colab.
  !pip install tf-nightly
except Exception:
  pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

python
# get data files
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"


## Installation & Usage
1. Clone the repository:
   bash
   git clone https://github.com/yourusername/sms-text-classifier.git
   
2. Install dependencies:
   bash
   pip install -r requirements.txt
   
3. Train the model:
   bash
   python train.py
   
4. Predict an SMS message:
   bash
   python predict.py --message "Congratulations! You've won a free prize. Call now!"
   
