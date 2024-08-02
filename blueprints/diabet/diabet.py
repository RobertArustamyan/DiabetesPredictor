from flask import Blueprint, request, jsonify, render_template
import joblib
import pandas as pd

diabetes_bp = Blueprint('diabetes', __name__, template_folder='templates')


@diabetes_bp.route('/')
def index():
    return render_template('index.html')


@diabetes_bp.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        rf_model = joblib.load('random_forest_model.pkl')

        pregnancies = int(request.form['pregnancies'])
        pregnancies_binary = 1 if pregnancies >= 7 else 0

        data = {
            'Glucose': int(request.form['glucose']),
            'BloodPressure': int(request.form['bloodpressure']),
            'SkinThickness': int(request.form['skinthickness']),
            'Insulin': int(request.form['insulin']),
            'BMI': float(request.form['bmi']),
            'DiabetesPedigreeFunction': float(request.form['dpf']),
            'Age': int(request.form['age']),
            'Pregnancies_Binary': pregnancies_binary
        }

        # Create a DataFrame for prediction
        input_df = pd.DataFrame([data])

        # Predict using the model
        prediction = rf_model.predict(input_df)
        confidence = rf_model.predict_proba(input_df)[:, 1] * 100

        # Map prediction to outcome label
        outcome = "Diabetes" if prediction[0] == 1 else "No Diabetes"

        # Return result to template
        return render_template('index.html', prediction=outcome, confidence=confidence[0])
