import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import streamlit as st

@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\user\Downloads\spam.csv", encoding='latin-1')
    data = data[['v1', 'v2']] 
    data.columns = ['label', 'message']
    return data

data = load_data()

X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)

st.write(f"Accuracy: {accuracy:.2f}")
st.write("Classification Report:")
st.table(pd.DataFrame(classification_report).transpose())

st.title("Spam SMS Detection")
user_input = st.text_area("Enter an SMS:")
if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    st.write("Prediction:", "Spam" if prediction == 'spam' else "Not Spam")

