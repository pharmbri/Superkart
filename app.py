from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Define the paths to the serialized model, preprocessor, and feature names
model_path = 'tuned_random_forest_model.joblib'
preprocessor_path = 'preprocessor.joblib'
feature_names_path = 'all_feature_names.joblib'

# Load the model, preprocessor, and feature names
try:
    loaded_model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    all_feature_names = joblib.load(feature_names_path)
    print(f"Model, preprocessor, and feature names loaded successfully.")
except Exception as e:
    print(f"Error loading necessary components: {e}")
    loaded_model = None
    preprocessor = None
    all_feature_names = None


@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None or preprocessor is None or all_feature_names is None:
        return jsonify({'error': 'Server components not loaded'}), 500

    try:
        # Get data from POST request
        data = request.get_json(force=True)

        # Convert input data to DataFrame
        # Ensure the order of columns in input_df matches what the preprocessor expects
        # by creating a DataFrame from a dictionary of lists if single row, or list of dicts for multiple
        input_df = pd.DataFrame(data, index=[0])

        # Drop 'Product_Id' if present, as it's not used in prediction
        if 'Product_Id' in input_df.columns:
            input_df = input_df.drop(columns=['Product_Id'])

        # Preprocess the input data using the fitted preprocessor
        processed_input = preprocessor.transform(input_df)
        
        # Convert the processed data to a DataFrame with correct column names
        processed_input_df = pd.DataFrame(processed_input.toarray(), columns=all_feature_names)

        # Make prediction
        prediction = loaded_model.predict(processed_input_df)[0]

        return jsonify({'predicted_sales': prediction.item()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# To run the Flask app (for deployment, you would use a production-ready server like Gunicorn)
# This is typically run outside the notebook environment.
# For demonstration within Colab, you can use ngrok or similar to expose the local server.

print("""Flask app defined. To run:
  1. Save this code as a .py file (e.g., app.py).
  2. Run `python app.py` in your terminal.
  3. If in Colab and you want to expose it, use ngrok or a similar service.""")

# Example of how to run in Colab (for testing, requires ngrok installation and setup)
# from flask_ngrok import run_with_ngrok
# run_with_ngrok(app) # Starts ngrok when app is run
# app.run()
