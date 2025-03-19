# MLOPS
📌 Task Objective

The goal of this project is to build a machine learning model that accurately classifies SMS messages as either spam or not spam. The system applies natural language processing (NLP) techniques to process text messages and uses a Naive Bayes classifier for prediction.

🏗️ Project Overview

Dataset: SMS Spam Collection Dataset

Model: Multinomial Naive Bayes

Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency)

App Interface: Streamlit for easy user interaction

⚙️ Steps to Run the Project

Clone the Repository:

git clone https://github.com/murali954/spam-sms-detection.git
cd spam-sms-detection

Install Dependencies:

pip install pandas numpy scikit-learn streamlit

Place Dataset:

Download the dataset from Kaggle and place it at:

C:\Users\user\Downloads\spam.csv

Run the App:

streamlit run app.py

Use the App:

Enter an SMS message in the text box and click on the "Predict" button to see if it’s spam or not.

📊 Model Performance

Accuracy: ~96%

Classification Report: Available in the app output
