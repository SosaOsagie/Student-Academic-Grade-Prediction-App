from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)

# Load student data
student_data = pickle.load(open('C:/Users/HP/Desktop/Untitled Folder/student_data.pkl', 'rb'))

# Function to preprocess input data
def preprocess_input(input_data):
    # Convert input data into DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply one-hot encoding to categorical variables
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

    # One-hot encode categorical features
    encoder = OneHotEncoder(drop='first', sparse=False)
    input_df_encoded = pd.DataFrame(encoder.fit_transform(input_df[categorical_features]))
    input_df_encoded.columns = encoder.get_feature_names_out(categorical_features)

    # Combine encoded features with numerical features
    input_df_processed = pd.concat([input_df.drop(categorical_features, axis=1), input_df_encoded], axis=1)

    return input_df_processed

# Model training and evaluation logic
# Model training and evaluation logic
def train_evaluate_model(student_data):
    # Ensure 'Grade' column exists
    if 'Grade' not in student_data.columns:
        raise KeyError("The 'Grade' column is missing in the DataFrame.")

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

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # Return the trained model
    return model.predict(X)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_grade', methods=['GET', 'POST'])
def predict_grade():
    if request.method == 'POST':
        # Get form data
        input_data = {
            'Age': request.form['Age'],
            'Gender': request.form['Gender'],
            'Course': request.form['Course'],
            'Year': request.form['Year'],
            'Marital Status': request.form['Marital_Status'],
            'How often have you felt down or hopeless?': request.form['Felt_Down'],
            'How often have you had trouble falling or staying asleep, or slept too much?': request.form['Trouble_Sleeping'],
            'How often have you felt tired or had little energy?': request.form['Tiredness'],
            'How often have you felt nervous, anxious, or on edge?': request.form['Nervousness'],
            'How often have you become easily annoyed or irritable?': request.form['Irritability'],
            'How frequently do you experience panic attacks?': request.form['Panic_Attacks'],
            'On average, how many hours per day do you spend on social media?': request.form['Social_Media_Hours'],
            'How many alcoholic drinks/beer do you consume in a typical week?': request.form['Alcoholic_Drinks'],
            'How many hours of sleep do you usually get on a typical night?': request.form['Sleep_Hours'],
            'How many hours a day do you put aside for studying?': request.form['Study_Hours']
        }
        # Preprocess input data
        input_df_processed = preprocess_input(input_data)

        # Make prediction
        predicted_grade = train_evaluate_model(input_df_processed)

        # Render prediction template with predicted_grade
        return render_template('result.html', predicted_grade=predicted_grade)
    else:
        # Render input form template
        return render_template('input_form.html')

if __name__ == '__main__':
    app.run(debug=True)
