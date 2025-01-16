# app.py
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
import pickle
from main import main as training_pipeline

app = Flask(__name__)

def load_model():
    try:
        with open('artifacts/model_trainer/models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('artifacts/data_transformation/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'longitude': float(request.form['longitude']),
            'latitude': float(request.form['latitude']),
            'housing_median_age': float(request.form['housing_median_age']),
            'total_rooms': float(request.form['total_rooms']),
            'total_bedrooms': float(request.form['total_bedrooms']),
            'population': float(request.form['population']),
            'households': float(request.form['households']),
            'median_income': float(request.form['median_income']),
            'ocean_proximity': request.form['ocean_proximity']
        }
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Load model and preprocessor
        model, preprocessor = load_model()
        
        # Transform data
        X_transformed = preprocessor.transform(df)
        
        # Make prediction
        prediction = model.predict(X_transformed)[0]
        
        return jsonify({
            'prediction': f"${prediction:,.2f}",
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

@app.route('/train', methods=['POST'])
def train():
    try:
        training_pipeline()
        return jsonify({
            'message': 'Training completed successfully!',
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

if __name__ == '__main__':
    app.run(debug=True)