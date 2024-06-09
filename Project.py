import pickle
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split





student_data = pickle.load(open('student_data.pkl', 'rb'))

def train_evaluate_model(student_data):
    # Drop rows with missing target values
    student_data.dropna(subset=['Grade'], inplace=True)

    # Separate features and target variable
    X = student_data.drop(['Grade'], axis=1)
    y = student_data['Grade']

    # Define categorical and numerical features
    categorical_features = ['Gender', 'Course', 'Year', 'Marital Status',
                            'How often have you felt down or hopeless?',
                            'How often have you had trouble falling or staying asleep, or slept too much?',
                            'How often have you felt tired or had little energy?',
                            'How often have you felt nervous, anxious, or on edge?',
                            'How often have you become easily annoyed or irritable?',
                            'How frequently do you experience panic attacks?',
                            'On average, how many hours per day do you spend on social media?',
                            'How many alcoholic drinks/beer do you consume in a typical week?',
                            'How many hours of sleep do you usually get on a typical night?',
                            'How many hours a day do you put aside for studying?']

    numerical_features = ['Age']

    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('num', 'passthrough')  # Passthrough numerical features without transformation
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Append the model to the preprocessing pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(random_state=42))])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    return model

# Load the model
model = train_evaluate_model(student_data)





# Streamlit UI
st.title('Grade Prediction System')

with st.form('input_form'):
    st.write('### Enter Student Information:')
    age = st.number_input('Age', min_value=0, max_value=100, value=20)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    course = st.text_input('Course')
    year = st.selectbox('Year', ['First Year', 'Second Year', 'Third Year', 'Fourth Year', 'Fifth Year', 'Post Graduate'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Widowed'])
    felt_down = st.selectbox('How often have you felt down or hopeless?', ['Never', 'Rarely', 'Sometimes', 'Occasionally', 'Frequently', 'Almost Constantly'])
    trouble_sleeping = st.selectbox('How often have you had trouble falling or staying asleep, or slept too much?', ['Never', 'Rarely', 'Sometimes', 'Occasionally', 'Frequently', 'Almost Constantly'])
    tiredness = st.selectbox('How often have you felt tired or had little energy?', ['Never', 'Rarely', 'Sometimes', 'Occasionally', 'Frequently', 'Almost Constantly'])
    nervousness = st.selectbox('How often have you felt nervous, anxious, or on edge?', ['Never', 'Rarely', 'Sometimes', 'Occasionally', 'Frequently', 'Almost Constantly'])
    irritability = st.selectbox('How often have you become easily annoyed or irritable?', ['Never', 'Rarely', 'Sometimes', 'Occasionally', 'Frequently', 'Almost Constantly'])
    panic_attacks = st.selectbox('How frequently do you experience panic attacks?', ['Never', 'Rarely', 'Sometimes', 'Occasionally', 'Frequently', 'Almost Constantly'])
    social_media_hours = st.selectbox('On average, how many hours per day do you spend on social media?', ['0-1 hour', '1-2 hours', '2-3 hours', '3-4 hours', '4-5 hours', '5-6 hours', '6-7 hours', '7-8 hours', '8+ hours'])
    alcoholic_drinks = st.selectbox('How many alcoholic drinks/beer do you consume in a typical week?', ['0', '1-2', '3-4', '5 to 6', '7 to 10', 'More than 10'])
    sleep_hours = st.selectbox('How many hours of sleep do you usually get on a typical night?', ['0-4 hours', '5-6 hours', '7-8 hours', '9+ hours'])
    study_hours = st.selectbox('How many hours a day do you put aside for studying?', ['0-1 hour', '1-2 hours', '2-3 hours', '3-4 hours', '4-5 hours', '5-6 hours', '6-7 hours', '7-8 hours', '8+ hours'])

    submit_button = st.form_submit_button(label='Predict Grade')

# Button to trigger grade prediction
if submit_button:
    # Prepare user input for prediction
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Course': [course],
        'Year': [year],
        'Marital Status': [marital_status],
        'How often have you felt down or hopeless?': [felt_down],
        'How often have you had trouble falling or staying asleep, or slept too much?': [trouble_sleeping],
        'How often have you felt tired or had little energy?': [tiredness],
        'How often have you felt nervous, anxious, or on edge?': [nervousness],
        'How often have you become easily annoyed or irritable?': [irritability],
        'How frequently do you experience panic attacks?': [panic_attacks],
        'On average, how many hours per day do you spend on social media?': [social_media_hours],
        'How many alcoholic drinks/beer do you consume in a typical week?': [alcoholic_drinks],
        'How many hours of sleep do you usually get on a typical night?': [sleep_hours],
        'How many hours a day do you put aside for studying?': [study_hours]
    })


# Predict the grade
    predicted_grade = model.predict(input_data)[0]  # Get the first element of the prediction array

    def classify_grade(predicted_grade):
        if 1.0 <= predicted_grade < 2.0:
            return 'First Class'
        elif 2.0 <= predicted_grade < 2.1:
            return 'Second Class Upper'
        elif 2.1 <= predicted_grade < 2.2:
            return 'Second Class Lower'
        elif 2.2 <= predicted_grade < 3.0:
            return 'Second Class Lower'  # Repeated for inclusivity
        elif 3.0 <= predicted_grade < 4.0:
            return 'Third Class'
        else:
            return 'Pass'

    # Classify the predicted grade
    grade_category = classify_grade(predicted_grade)

    # Show the grade classification only
    st.write(f'Grade Classification: {grade_category}')


