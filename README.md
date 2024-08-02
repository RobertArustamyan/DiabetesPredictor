# Diabetes Prediction Project

## Overview

This project involves developing a machine learning model to predict diabetes using a dataset of medical features. The primary goal is to create an accurate and reliable model to identify individuals at risk of diabetes based on various health metrics. In addition to the model, a web application has been developed to allow users to input their health data and receive a prediction of their diabetes risk.

## Dataset

The dataset used in this project is the [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data), which includes the following features:
- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`
- `Outcome` (target variable: 0 for non-diabetic, 1 for diabetic)

## Web Application

A web application has been built using Flask, allowing users to input their health metrics and receive a diabetes prediction based on the trained Random Forest model. The application is user-friendly and provides both the predicted outcome (diabetic or non-diabetic) and the confidence level of the prediction.

### Features

- **User Input**: Users can input their health metrics such as glucose level, blood pressure, BMI, etc.
- **Prediction**: The application provides an instant prediction of whether the user is likely to be diabetic.
- **Confidence Level**: The application also shows the confidence level of the prediction.

## Models Used
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier** (selected model)
4. **Neural Network**
5. **Support Vector Classifier**

## Performance Metrics

The Random Forest model achieved the following performance metrics:

- **Accuracy**: 87.28%
- **Recall**: 81.08%
- **Precision**: 80.00%
- **F1 Score**: 80.54%
- **ROC AUC**: 94.56%

## Getting Started


### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction.git

2. Navigate to the project directory:
    ```bash
    cd diabetes-prediction
3. Install the required packages:
    ```bash
   pip install -r requirements.txt

### Running web application
1. Start flask server:
    ```bash
   python app.py
2. Open web browser `http://127.0.0.1:5000/diabetes/` to use application.
   